"""Per-cube reward machine for the cube-sorting task (Iteration R4).

Implements the Mealy automaton ``{FREE, HELD, PLACED, LOST}`` and the
single per-step ``per_cube_reward`` function described in
:doc:`doc/reports/cube_sort_research/cube_sort_mdp_redesign_consolidated.md` §B.

This module replaces the env-level ``was_grasped`` latch (which was a
non-Markovian, non-potential reward transformation that produced the
"flinging" failure mode catalogued by Amodei2016 and forbidden by
Ng-Harada-Russell 1999).

Reward channels (all per-cube, summed then aggregated to (N,)):

- ``r_reach``:     ``-α · ‖cube − tcp‖`` for the single nearest active green cube.
- ``r_grasp``:     ``β · 𝟙[HELD ∧ green] + β₁ · 𝟙[FREE→HELD ∧ green]`` (one-shot edge).
- ``r_transport``: ``-γ · ‖cube − drum‖ · 𝟙[CURRENTLY held ∧ green]`` (the fling fix).
- ``r_place``:     ``K_PLACE · 𝟙[HELD→PLACED ∧ green]`` (one-shot).
- ``r_red``:       ``-K_RED · 𝟙[HELD ∧ red] - λ_red · ‖red − drum‖ · 𝟙[HELD ∧ red]``.
- ``r_red_drop``:  ``-K_RED_DROP · 𝟙[place_edge ∧ red]``.
- ``r_time``:      ``-c · #unplaced_green`` (potential-based, Ng1999-compliant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .grasp import is_holding
from .placement import (
    is_placed,
    reset_placement_state,
)

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Tunable constants (research §B "Suggested weights" + reward table).
# ---------------------------------------------------------------------------
ALPHA: float = 2.0      # reach distance penalty
BETA: float = 0.5       # held-and-green per-step reward
BETA1: float = 2.0      # one-shot grasp edge bonus
GAMMA: float = 3.0      # transport distance penalty
K_PLACE: float = 50.0   # one-shot placement reward (green)
K_RED: float = 5.0      # per-step penalty while holding a red cube
LAM_RED: float = 0.5    # transport-toward-drum penalty for red cubes
K_RED_DROP: float = 20.0  # one-shot penalty for red placed in drum
C_TIME: float = 0.02    # time-cost (per unplaced green per step)
LOST_Z_MARGIN: float = 0.02  # how far below belt counts as LOST

INF: float = 1e6

_STATE_ATTR = "_mdp_state"


# ---------------------------------------------------------------------------
# State buffers
# ---------------------------------------------------------------------------

def _init_buffers(env: "ManagerBasedRLEnv", num_cubes: int) -> dict:
    """Allocate the per-cube Mealy automaton state on ``env`` (lazy)."""
    state = getattr(env, _STATE_ATTR, None)
    if state is not None:
        return state
    device = env.device
    shape = (env.num_envs, num_cubes)

    def _z() -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.bool, device=device)

    state = dict(
        ever_held=_z(),
        ever_held_prev=_z(),
        held_prev=_z(),
        placed=_z(),
        placed_prev=_z(),
        lost=_z(),
    )
    setattr(env, _STATE_ATTR, state)
    return state


def reset_mdp_state(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None) -> None:
    """``EventTerm``-compatible reset hook.

    Clears the per-cube Mealy automaton state and the placement-dwell
    counter for the given env ids (or all envs if ``env_ids`` is None).
    Called once per episode reset.
    """
    state = getattr(env, _STATE_ATTR, None)
    if state is not None:
        if env_ids is None:
            for v in state.values():
                v.zero_()
        else:
            for v in state.values():
                v[env_ids] = False
    # Also clear the placement dwell counter / latch (R3).
    reset_placement_state(env, env_ids)
    # Also clear the is_holding dwell ring buffer (R2). Rebuilt on next call.
    if hasattr(env, "_hold_hist"):
        if env_ids is None:
            env._hold_hist.zero_()
            env._hold_hist_ptr = 0
        else:
            env._hold_hist[env_ids] = False


# ---------------------------------------------------------------------------
# Core per-cube reward
# ---------------------------------------------------------------------------

def _cube_role(env: "ManagerBasedRLEnv", target_label: int) -> torch.Tensor:
    """``(M,)`` int8 tensor: +1 for green/target, -1 for red/distractor."""
    cache_attr = "_per_cube_role_cache"
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return cached
    labels = env.cube_labels  # (M,) int32, populated by TensegrityCubeSortEnv
    role = torch.where(
        labels == target_label,
        torch.ones_like(labels),
        -torch.ones_like(labels),
    ).to(torch.int8)
    setattr(env, cache_attr, role)
    return role


def per_cube_reward(
    env: "ManagerBasedRLEnv",
    cubes_collection_name: str = "cubes",
    drum_name: str = "drum_target",
    robot_name: str = "robot",
    tcp_body_name: str = "tool_link_0",
    target_label: int = 0,
    z_belt: float = 0.83,
) -> torch.Tensor:
    """Single per-step reward term composed of all per-cube channels.

    Returns a ``(N_envs,)`` tensor — the sum of the seven per-cube reward
    channels plus the global potential-based time cost.

    Args:
        env: The ``ManagerBasedRLEnv`` instance.
        cubes_collection_name: Scene key of the ``RigidObjectCollection``.
        drum_name: Scene key of the drum target prim (``RigidObject`` or
            similar; world position read via ``shared_rew.get_world_pos``).
        robot_name: Scene key of the robot articulation.
        tcp_body_name: Body name to use as TCP for transport distance.
        target_label: Cube label that counts as "green" / target.
        z_belt: Belt-surface world Z (m). Used for the LOST transition.

    Returns:
        ``(N_envs,)`` per-step reward tensor.
    """
    # -- Geometry --------------------------------------------------------
    cubes = env.scene[cubes_collection_name]
    p = cubes.data.object_pos_w                                # (N, M, 3)
    num_cubes = p.shape[1]
    state = _init_buffers(env, num_cubes)

    robot = env.scene[robot_name]
    tcp_idx = robot.data.body_names.index(tcp_body_name)
    tcp = robot.data.body_pos_w[:, tcp_idx]                    # (N, 3)

    drum_obj = env.scene[drum_name]
    if hasattr(drum_obj, "data") and hasattr(drum_obj.data, "root_pos_w"):
        drum = drum_obj.data.root_pos_w                        # (N, 3)
    else:
        # XformPrim-style fallback (single drum across envs).
        drum = drum_obj.get_world_poses()[0]
        if drum.shape[0] == 1 and env.num_envs > 1:
            drum = drum.expand(env.num_envs, -1)

    d_hand = (p - tcp.unsqueeze(1)).norm(dim=-1)               # (N, M)
    d_drum = (p - drum.unsqueeze(1)).norm(dim=-1)              # (N, M)

    # -- Per-cube state transitions -------------------------------------
    held = is_holding(
        env,
        cubes_collection_name=cubes_collection_name,
        robot_name=robot_name,
        tcp_body_name=tcp_body_name,
    )                                                          # (N, M) bool

    state["ever_held"] = state["ever_held"] | held
    placed_now = is_placed(env, drum, held, cubes_collection_name=cubes_collection_name)
    new_placed = placed_now & state["ever_held"]
    place_edge = new_placed & (~state["placed_prev"])
    state["placed"] = state["placed"] | new_placed

    lost_now = (p[..., 2] < z_belt - LOST_Z_MARGIN) & (~state["placed"])
    state["lost"] = state["lost"] | lost_now

    active = (~state["placed"]) & (~state["lost"])
    role = _cube_role(env, target_label)                       # (M,) int8
    green = (role > 0).unsqueeze(0).expand_as(active)
    red = (role < 0).unsqueeze(0).expand_as(active)

    # -- Reach (top-1 nearest active green) -----------------------------
    cand = torch.where(active & green, d_hand, torch.full_like(d_hand, INF))
    nearest = cand.argmin(dim=-1, keepdim=True)
    reach_mask = torch.zeros_like(held).scatter_(1, nearest, True) & active & green
    r_reach = -ALPHA * d_hand * reach_mask.float()

    # -- Grasp (per-step + edge bonus) ----------------------------------
    grasp_edge = (
        held
        & (~state["held_prev"])
        & (~state["ever_held_prev"])
        & active
        & green
    )
    r_grasp = (
        BETA * (held & active & green).float()
        + BETA1 * grasp_edge.float()
    )

    # -- Transport (gated on currently-held — the fling fix) ------------
    r_transport = -GAMMA * d_drum * (held & active & green).float()

    # -- Placement (one-shot) -------------------------------------------
    r_place = K_PLACE * place_edge.float() * green.float()

    # -- Red penalties --------------------------------------------------
    held_red = (held & red).float()
    r_red = -K_RED * held_red - LAM_RED * d_drum * held_red    # (closer to drum → larger penalty)
    r_red_drop = -K_RED_DROP * (place_edge & red).float()

    # -- Time cost (potential-based, Ng1999) ----------------------------
    n_unplaced = (active & green).float().sum(dim=-1)          # (N,)
    r_time = -C_TIME * n_unplaced

    per_cube = r_reach + r_grasp + r_transport + r_place + r_red + r_red_drop
    step = per_cube.sum(dim=-1) + r_time

    # -- Update prev buffers -------------------------------------------
    state["held_prev"] = held.clone()
    state["ever_held_prev"] = state["ever_held"].clone()
    state["placed_prev"] = state["placed"].clone()
    return step

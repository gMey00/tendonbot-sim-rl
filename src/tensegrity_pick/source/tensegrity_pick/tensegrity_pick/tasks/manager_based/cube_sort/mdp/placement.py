"""Per-cube placement predicate for the cube-sorting task (Iteration R3).

Implements the sticky 5-conjunct ``is_placed`` predicate from
:doc:`doc/reports/cube_sort_research/cube_sort_mdp_redesign_consolidated.md` §C.

Conjuncts (must hold for ``DWELL_STEPS`` consecutive physics steps before
``is_placed`` latches True for the rest of the episode):

1. **In XY** — cube horizontal distance to drum centre < ``DRUM_R - CUBE_HALF``.
2. **In Z range** — cube z is between ``floor_z + CUBE_HALF - 0.01`` and
   ``floor_z + DRUM_H - CUBE_HALF``.
3. **Seated** — cube z within ``0.015 m`` of the drum floor (rules out
   bouncing in mid-air through the cylinder).
4. **At rest** — ``‖v_lin‖ < 0.05 m/s ∧ ‖v_ang‖ < 0.5 rad/s``
   (ManiSkill3 GPU-jitter thresholds, see §C).
5. **Not held** — ``¬is_holding[i]`` — rules out a cube being carried
   through the drum cylinder by the gripper.

The latch is **per-episode-per-cube**: once ``is_placed[e, i]`` flips
True, it stays True until ``reset_placement_state`` is called by the
env reset hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - type-only imports (avoid omni.timeline at import time)
    from isaaclab.assets import RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Tunable constants — see research §C "Recommendation".
# ---------------------------------------------------------------------------
DRUM_R: float = 0.2735
"""Drum inner radius (m). Matches `cube_sort_env_cfg.py` reset_cubes config."""

DRUM_H: float = 0.30
"""Drum height (m)."""

CUBE_HALF: float = 0.025
"""Cube half-edge (m). For 5 cm cubes."""

V_LIN_MAX: float = 0.05
"""Max linear speed (m/s) for the "at rest" conjunct (ManiSkill3 default)."""

V_ANG_MAX: float = 0.5
"""Max angular speed (rad/s) for the "at rest" conjunct (ManiSkill3 default —
explicitly chosen there as a GPU-jitter accommodation)."""

DWELL_STEPS: int = 5
"""Number of consecutive physics steps the frame predicate must hold.

≈ 0.08 s at 60 Hz, the literature consensus across ManiSkill3 / robosuite /
TossingBot. Report 1's 30-step / 0.5-s value is overly conservative.
"""

_DWELL_ATTR = "_place_dwell"
_LATCH_ATTR = "_place_latch"


def _ensure_buffers(
    env: "ManagerBasedRLEnv",
    num_cubes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the dwell counter and sticky latch lazily."""
    if not hasattr(env, _DWELL_ATTR):
        setattr(
            env,
            _DWELL_ATTR,
            torch.zeros((env.num_envs, num_cubes), dtype=torch.int32, device=device),
        )
    if not hasattr(env, _LATCH_ATTR):
        setattr(
            env,
            _LATCH_ATTR,
            torch.zeros((env.num_envs, num_cubes), dtype=torch.bool, device=device),
        )
    return getattr(env, _DWELL_ATTR), getattr(env, _LATCH_ATTR)


def reset_placement_state(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None) -> None:
    """Zero the dwell counter and latch for the given env ids (or all)."""
    dwell = getattr(env, _DWELL_ATTR, None)
    latch = getattr(env, _LATCH_ATTR, None)
    if dwell is None or latch is None:
        return  # nothing to reset; will be allocated on first call
    if env_ids is None:
        dwell.zero_()
        latch.zero_()
    else:
        dwell[env_ids] = 0
        latch[env_ids] = False


def is_placed(
    env: "ManagerBasedRLEnv",
    drum_center: torch.Tensor,
    held: torch.Tensor,
    cubes_collection_name: str = "cubes",
) -> torch.Tensor:
    """Compute the per-cube ``is_placed`` predicate (sticky).

    Args:
        env: The ``ManagerBasedRLEnv`` instance (used for buffer storage).
        drum_center: ``(N_envs, 3)`` world-frame position of the drum
            ORIGIN (= floor centre, since the drum cylinder is built with
            its base at the prim origin per ``in_upright_cylinder``).
        held: ``(N_envs, N_cubes)`` bool — output of :func:`is_holding`.
            Used to suppress placement while a cube is being transported.
        cubes_collection_name: Scene key of the cube
            ``RigidObjectCollection``. Default ``"cubes"``.

    Returns:
        ``(N_envs, N_cubes)`` bool tensor that latches True per cube once
        the 5-conjunct predicate has held for ``DWELL_STEPS`` consecutive
        steps. Latch is reset by :func:`reset_placement_state`.
    """
    cubes: "RigidObjectCollection" = env.scene[cubes_collection_name]
    p = cubes.data.object_pos_w           # (N, M, 3)
    vl = cubes.data.object_lin_vel_w      # (N, M, 3)
    va = cubes.data.object_ang_vel_w      # (N, M, 3)

    dxy = p[..., :2] - drum_center[:, None, :2]
    floor_z = drum_center[:, 2].unsqueeze(-1)                       # (N, 1)
    in_xy = dxy.norm(dim=-1) < (DRUM_R - CUBE_HALF)                 # (N, M)
    in_z = (p[..., 2] > floor_z + CUBE_HALF - 0.01) & (
        p[..., 2] < floor_z + DRUM_H - CUBE_HALF
    )
    sitting = p[..., 2] < floor_z + CUBE_HALF + 0.015
    v_ok = (vl.norm(dim=-1) < V_LIN_MAX) & (va.norm(dim=-1) < V_ANG_MAX)
    not_held = ~held

    frame_ok = in_xy & in_z & sitting & v_ok & not_held

    dwell, latch = _ensure_buffers(env, num_cubes=p.shape[1], device=p.device)
    # increment counter where the frame predicate holds, reset where it does not
    dwell.copy_(torch.where(frame_ok, dwell + 1, torch.zeros_like(dwell)))
    # latch on first time dwell crosses threshold; sticky thereafter
    latch.copy_(latch | (dwell >= DWELL_STEPS))
    return latch.clone()

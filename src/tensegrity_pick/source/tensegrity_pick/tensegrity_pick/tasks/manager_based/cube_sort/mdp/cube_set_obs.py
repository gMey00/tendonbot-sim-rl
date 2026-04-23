"""Cube-set observations for the DeepSets policy (R6).

These observation terms produce the flattened per-cube feature block
and the validity mask consumed by ``models/cube_set_policy.py``.

Per-cube feature vector (S = 15 dims, research §D Table):

  0..2   : (cube_pos − ee_pos) — 3
  3..5   : cube_lin_vel_w — 3
  6      : signed-distance-to-drum, normalised — 1
  7      : cube label one-hot bit (1.0 if green, else 0.0) — 1
  8      : active flag (1 if active in scene) — 1   (mirrors the mask)
  9      : currently-held flag (R2 ``is_holding`` per cube) — 1
  10     : ever-held flag (R4 ``_mdp_state['ever_held']``) — 1
  11     : placed flag (R3 sticky) — 1
  12     : (cube_z − belt_z), clamped to [-0.1, 0.5] — 1
  13..14 : drum-relative XY (cube − drum) — 2

Total: 15 floats per cube.

Privileged channels (critic-only path; flattened tail of the obs vector,
not consumed by the policy slice):

  per_cube_is_holding_priv   : N_max float ∈ {0, 1}
  per_cube_contact_mag_priv  : N_max float (||F_left|| + ||F_right||)

Total privileged: 2 * N_max.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv

# Maximum number of cube slots exposed to the encoder. Padded with zeros
# beyond the active cube count.  Must be ≥ NUM_CUBES_TOTAL.
N_MAX_CUBES: int = 16
SET_PER_CUBE_DIM: int = 15
PRIV_PER_CUBE_DIM: int = 2  # is_holding + contact_mag


def _zeros(env: "ManagerBasedRLEnv", *shape: int) -> torch.Tensor:
    return torch.zeros((env.num_envs, *shape), device=env.device)


def _ee_pos(env: "ManagerBasedRLEnv", ee_body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    idx = robot.data.body_names.index(ee_body_name)
    return robot.data.body_pos_w[:, idx] - env.scene.env_origins


def _drum_pos(env: "ManagerBasedRLEnv", drum_name: str) -> torch.Tensor:
    drum = env.scene[drum_name]
    if hasattr(drum, "data") and hasattr(drum.data, "root_pos_w"):
        return drum.data.root_pos_w - env.scene.env_origins
    return drum.get_world_poses()[0] - env.scene.env_origins


def _per_cube_held(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Best-effort current per-cube hold mask — uses R4 ``_mdp_state``
    when populated, else falls back to legacy proximity heuristic."""
    state = getattr(env, "_mdp_state", None)
    if state is not None and "held_prev" in state:
        # held_prev is the most recent observed held value (set at the end
        # of per_cube_reward each step).
        return state["held_prev"].float()
    if hasattr(env, "grasp_active_per_cube"):
        return env.grasp_active_per_cube.float()
    return _zeros(env, N_MAX_CUBES)


def _per_cube_ever_held(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = getattr(env, "_mdp_state", None)
    if state is not None:
        return state["ever_held"].float()
    return _zeros(env, N_MAX_CUBES)


def _per_cube_placed(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = getattr(env, "_mdp_state", None)
    if state is not None:
        return state["placed"].float()
    return _zeros(env, N_MAX_CUBES)


def _per_cube_contact_mag(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Sum of left- and right-pad force magnitudes per cube. (N, N_max)."""
    out = _zeros(env, N_MAX_CUBES)
    for sensor_name in ("contact_left", "contact_right"):
        sensor = env.scene.sensors.get(sensor_name) if hasattr(env.scene, "sensors") else None
        if sensor is None:
            try:
                sensor = env.scene[sensor_name]
            except KeyError:
                continue
        fm = getattr(sensor.data, "force_matrix_w", None)
        if fm is None:
            continue
        # fm shape: (N_envs, N_bodies=1, N_filters, 3)
        mag = fm.norm(dim=-1)[:, 0]  # (N, N_filters)
        n_fill = min(mag.shape[1], out.shape[1])
        out[:, :n_fill] = out[:, :n_fill] + mag[:, :n_fill]
    return out


def cube_features_flat(
    env: "ManagerBasedRLEnv",
    cubes_collection_name: str = "cubes",
    drum_name: str = "drum_target",
    ee_body_name: str = "tool_link_0",
    belt_height: float = 0.83,
) -> torch.Tensor:
    """Per-cube feature block flattened to ``(N_envs, N_max * S)``."""
    coll = env.scene[cubes_collection_name]
    p = coll.data.object_pos_w - env.scene.env_origins.unsqueeze(1)  # (N, M, 3)
    v = coll.data.object_lin_vel_w                                    # (N, M, 3)
    n_envs, m, _ = p.shape

    ee = _ee_pos(env, ee_body_name)                                   # (N, 3)
    drum = _drum_pos(env, drum_name)                                  # (N, 3)

    rel_ee = p - ee.unsqueeze(1)                                      # (N, M, 3)
    drum_rel_xy = p[..., :2] - drum.unsqueeze(1)[..., :2]             # (N, M, 2)
    sd_drum = drum_rel_xy.norm(dim=-1, keepdim=True)                  # (N, M, 1)
    z_above = (p[..., 2:3] - belt_height).clamp(-0.1, 0.5)            # (N, M, 1)

    labels = env.cube_labels                                          # (M,)
    target_label = int(getattr(env, "_target_label", 0))
    is_green = (labels == target_label).to(p.dtype)
    is_green = is_green.view(1, m, 1).expand(n_envs, m, 1)

    active_flag = _per_cube_active(env, m).unsqueeze(-1)              # (N, M, 1)
    held = _per_cube_held(env)[:, :m].unsqueeze(-1)
    ever = _per_cube_ever_held(env)[:, :m].unsqueeze(-1)
    placed = _per_cube_placed(env)[:, :m].unsqueeze(-1)

    feats = torch.cat(
        [rel_ee, v, sd_drum, is_green, active_flag, held, ever, placed, z_above, drum_rel_xy],
        dim=-1,
    )  # (N, M, 15)

    if m < N_MAX_CUBES:
        pad = torch.zeros(n_envs, N_MAX_CUBES - m, SET_PER_CUBE_DIM, device=p.device, dtype=p.dtype)
        feats = torch.cat([feats, pad], dim=1)
    elif m > N_MAX_CUBES:
        feats = feats[:, :N_MAX_CUBES]

    return feats.reshape(n_envs, N_MAX_CUBES * SET_PER_CUBE_DIM)


def cube_mask_flat(
    env: "ManagerBasedRLEnv",
    cubes_collection_name: str = "cubes",
) -> torch.Tensor:
    """Validity mask for each of the ``N_MAX_CUBES`` slots (1 = active)."""
    coll = env.scene[cubes_collection_name]
    m = coll.data.object_pos_w.shape[1]
    out = torch.zeros((env.num_envs, N_MAX_CUBES), device=env.device)
    n_fill = min(m, N_MAX_CUBES)
    out[:, :n_fill] = _per_cube_active(env, m)[:, :n_fill]
    return out


def _per_cube_active(env: "ManagerBasedRLEnv", m: int) -> torch.Tensor:
    """``(N, M)`` float mask: 1 where the cube is "in play" this episode.

    Uses the parking-position heuristic from the existing curriculum
    (cubes parked at world coordinates ≈ (100, 100, 1) are inactive).
    """
    coll = env.scene["cubes"]
    p = coll.data.object_pos_w
    parked = (p[..., 0].abs() > 50.0) | (p[..., 1].abs() > 50.0)
    return (~parked).float()


def gobj_sim(
    env: "ManagerBasedRLEnv",
    finger_joint_name: str = "finger_joint",
    closed_target: float = 0.7854,
    effort_thresh: float = 10.0,
) -> torch.Tensor:
    """Simulated Robotiq ``gOBJ`` register (R7, research §F).

    Returns ``(N, 1)`` float ∈ {0, 2, 3} per env per step:

    - ``2``: closing command + stalled position + stalled velocity +
      high effort  (object detected between the fingers)
    - ``3``: closing command + position has reached fully-closed target
      (no object detected)
    - ``0`` otherwise (open / mid-motion / no command)

    The returned dtype is float so that the value can be concatenated
    with the rest of the float observation vector without an explicit
    cast — Robotiq exposes only 4 levels (0/1/2/3) so the ordinal
    encoding is sufficient and avoids a one-hot expansion.
    """
    robot = env.scene["robot"]
    try:
        finger_id = robot.data.joint_names.index(finger_joint_name)
    except ValueError:
        return _zeros(env, 1)

    # Resolve the most recent gripper command (BinaryJointPositionAction
    # writes the scalar into ``robot.data.joint_pos_target`` each control
    # step, so reading that is robust to action-index ordering).
    cmd = robot.data.joint_pos_target[:, finger_id]
    pos = robot.data.joint_pos[:, finger_id]
    vel = robot.data.joint_vel[:, finger_id]
    eff = robot.data.applied_torque[:, finger_id]

    commanded_close = cmd > 0.70 * closed_target
    stalled_pos = pos < 0.90 * closed_target
    stalled_vel = vel.abs() < 0.01
    high_effort = eff.abs() > effort_thresh
    object_closing = commanded_close & stalled_pos & stalled_vel & high_effort
    reached_closed = (pos >= 0.90 * closed_target) & commanded_close

    g = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    g[reached_closed] = 3
    g[object_closing] = 2  # object_closing wins over reached_closed when both true
    return g.unsqueeze(-1).float()


def gripper_pad_force_mag(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_left",
) -> torch.Tensor:
    """Magnitude of the per-step net force on a finger pad. (N, 1)."""
    try:
        sensor = env.scene[sensor_name]
    except KeyError:
        return _zeros(env, 1)
    nf = getattr(sensor.data, "net_forces_w", None)
    if nf is None:
        return _zeros(env, 1)
    # nf shape: (N_envs, N_bodies=1, 3)
    return nf[:, 0].norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Privileged channels (critic-only)
# ---------------------------------------------------------------------------

def per_cube_is_holding_priv(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N, N_max)`` float — current per-cube ``is_holding`` value."""
    return _per_cube_held(env)[:, :N_MAX_CUBES]


def per_cube_contact_mag_priv(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N, N_max)`` float — sum-of-pad force magnitudes per cube."""
    return _per_cube_contact_mag(env)[:, :N_MAX_CUBES]


__all__ = [
    "N_MAX_CUBES",
    "SET_PER_CUBE_DIM",
    "PRIV_PER_CUBE_DIM",
    "cube_features_flat",
    "cube_mask_flat",
    "gobj_sim",
    "gripper_pad_force_mag",
    "per_cube_is_holding_priv",
    "per_cube_contact_mag_priv",
]

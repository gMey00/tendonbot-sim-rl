"""Reward, observation, and termination functions for the cube sorting task.

All functions operate on a *single* unified RigidObjectCollection and select
cubes by integer **label** (e.g. 0 = target/green, 1 = distractor/red).
This keeps the codebase generic: add a new category by defining a new label
constant — no new functions required.

7-phase reward pipeline (per research report):
  1. Reach      — tanh proximity  (dynamic fingertip → nearest target cube)
  2. Grasp      — closure × proximity
  3. Lift       — binary (velocity-gated) + height bonus
  4. Transport  — tanh XY distance to drum with urgency weighting
  5. Release    — gripper openness when target cube above drum
  6. Re-orient  — return toward belt when gripper empty and cubes remain
  (Event-based: placement bonus, miss penalty, red-grabbed penalty in env)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ...shared.gripper_cfg import (
    FINGER_JOINT_CLOSE_POS,
    BinCylinder,
    ConveyorBounds,
    dynamic_finger_tip_w,
    get_world_pos,
    grasp_center_w,
    in_upright_cylinder,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


PARKING_X_THRESHOLD = 50.0


# ---------------------------------------------------------------------------
# Collection helpers (label-aware, GPU-vectorised)
# ---------------------------------------------------------------------------

def _active_mask(env: "ManagerBasedRLEnv", collection_name: str) -> torch.Tensor:
    """(N, M) bool — cubes that are not parked."""
    pos = env.scene[collection_name].data.object_pos_w
    local_x = pos[..., 0] - env.scene.env_origins[:, None, 0]
    return local_x < PARKING_X_THRESHOLD


def _label_mask(env: "ManagerBasedRLEnv", label: int) -> torch.Tensor:
    """(M,) bool — cubes that match *label*, broadcasts to (N, M)."""
    return env.cube_labels == label


def _active_label_mask(
    env: "ManagerBasedRLEnv", collection_name: str, label: int,
) -> torch.Tensor:
    """(N, M) bool — active cubes of the given label."""
    return _active_mask(env, collection_name) & _label_mask(env, label)[None, :]


def _nearest_active_by_label(
    env: "ManagerBasedRLEnv",
    reference_pos: torch.Tensor,
    collection_name: str,
    label: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the nearest active cube of *label* per env.

    Returns ``(pos, idx, has_any)`` where
      pos:     (N, 3) world position
      idx:     (N,)   index into the collection
      has_any: (N,)   bool — False when no cubes of this label are active
    """
    coll = env.scene[collection_name]
    pos = coll.data.object_pos_w  # (N, M, 3)
    valid = _active_label_mask(env, collection_name, label)

    d2 = torch.sum((pos - reference_pos[:, None, :]) ** 2, dim=-1)
    d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))
    idx = torch.argmin(d2, dim=1)
    has_any = torch.isfinite(d2.min(dim=1).values)

    arange = torch.arange(pos.shape[0], device=pos.device)
    return pos[arange, idx], idx, has_any


# ---------------------------------------------------------------------------
# Observations (label-parameterised)
# ---------------------------------------------------------------------------

def nearest_cube_rel(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    collection_name: str,
    label: int,
) -> torch.Tensor:
    """Relative position of the nearest active cube of *label* w.r.t. grasp centre (N, 3)."""
    ee = grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    pos, _idx, has_any = _nearest_active_by_label(env, ee, collection_name, label)
    rel = pos - ee
    return torch.where(has_any[:, None], rel, torch.zeros_like(rel))


def nearest_cube_fingertip_rel(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    collection_name: str,
    label: int,
) -> torch.Tensor:
    """Relative position of the nearest cube of *label* w.r.t. dynamic fingertip (N, 3)."""
    robot: Articulation = env.scene[ee_cfg.name]
    tip = dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    pos, _idx, has_any = _nearest_active_by_label(env, tip, collection_name, label)
    rel = pos - tip
    return torch.where(has_any[:, None], rel, torch.zeros_like(rel))


def nearest_cube_velocity(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    collection_name: str,
    label: int,
) -> torch.Tensor:
    """Linear velocity of the nearest active cube of *label* (N, 3)."""
    ee = grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    _pos, idx, has_any = _nearest_active_by_label(env, ee, collection_name, label)
    vel = env.scene[collection_name].data.object_lin_vel_w
    arange = torch.arange(vel.shape[0], device=vel.device)
    nearest_vel = vel[arange, idx]
    return torch.where(has_any[:, None], nearest_vel, torch.zeros_like(nearest_vel))


# ---------------------------------------------------------------------------
# Task-context observations
# ---------------------------------------------------------------------------

def belt_speed_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Current belt speed per env (N, 1)."""
    extras = getattr(env, "extras", {})
    speed = extras.get("belt_speed", None)
    if speed is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return speed.unsqueeze(-1)


def time_fraction_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Fraction of episode time remaining [1→0] (N, 1)."""
    max_steps = env.max_episode_length
    remaining = (max_steps - env.episode_length_buf).float() / max(max_steps, 1)
    return remaining.unsqueeze(-1)


def targets_placed_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Count of target cubes successfully placed (N, 1)."""
    count = getattr(env, "_targets_placed", None)
    if count is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return count.float().unsqueeze(-1)


def targets_missed_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Count of target cubes missed (fell off belt) (N, 1)."""
    count = getattr(env, "_targets_missed", None)
    if count is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return count.float().unsqueeze(-1)


# ---------------------------------------------------------------------------
# Reward functions — 7-phase pipeline (label-parameterised)
# ---------------------------------------------------------------------------

# ── 1. Reach ──────────────────────────────────────────────────────────────

def cube_ee_distance(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    collection_name: str,
    label: int,
    std: float = 0.1,
) -> torch.Tensor:
    """Tanh proximity from dynamic fingertip to nearest active cube of *label*.

    Gated: only active when gripper is NOT holding a cube (grasp_active=False).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    pos, _idx, has_any = _nearest_active_by_label(env, tip, collection_name, label)
    distance = torch.norm(pos - tip, dim=-1)
    proximity = 1.0 - torch.tanh(distance / std)
    result = torch.where(has_any, proximity, torch.zeros_like(proximity))
    if hasattr(env, "grasp_active"):
        result = result * (~env.grasp_active).float()
    return result


# ── 2. Grasp ──────────────────────────────────────────────────────────────

def cube_grasp_reward(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    collection_name: str,
    label: int,
    std: float = 0.08,
) -> torch.Tensor:
    """Closure x proximity to nearest active cube of *label*."""
    robot: Articulation = env.scene[ee_cfg.name]
    tip = dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    pos, _idx, has_any = _nearest_active_by_label(env, tip, collection_name, label)
    distance = torch.norm(pos - tip, dim=-1)
    proximity = 1.0 - torch.tanh(distance / std)

    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
    reward = closure * proximity
    return torch.where(has_any, reward, torch.zeros_like(reward))


# ── 3. Lift ───────────────────────────────────────────────────────────────

def cube_is_lifted(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    collection_name: str,
    label: int,
    belt_height: float,
    minimal_height: float = 0.06,
    max_distance: float = 0.15,
    finger_cfg: SceneEntityCfg | None = None,
    min_closure: float = 0.20,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Binary 1.0 when any active cube of *label* is lifted near the gripper."""
    robot: Articulation = env.scene[ee_cfg.name]
    ee = grasp_center_w(robot, ee_cfg)
    coll = env.scene[collection_name]
    pos = coll.data.object_pos_w
    valid = _active_label_mask(env, collection_name, label)

    local_z = pos[..., 2] - env.scene.env_origins[:, None, 2]
    is_above = local_z > belt_height + minimal_height

    distance = torch.norm(pos - ee[:, None, :], dim=-1)
    is_near = distance < max_distance

    gate = valid & is_above & is_near
    if finger_cfg is not None:
        finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
        gate = gate & (finger_pos > min_closure)[:, None]
    if max_velocity is not None:
        cube_speed = torch.norm(coll.data.object_lin_vel_w, dim=-1)
        gate = gate & (cube_speed < max_velocity)

    return torch.any(gate, dim=1).to(torch.float32)


# ── 3b. Height bonus ─────────────────────────────────────────────────────

def cube_height_bonus(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    collection_name: str,
    label: int,
    belt_height: float,
    max_height: float = 0.30,
    max_distance: float = 0.15,
    finger_cfg: SceneEntityCfg | None = None,
    min_closure: float = 0.20,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Height reward for the highest lifted cube of *label* near the gripper."""
    robot: Articulation = env.scene[ee_cfg.name]
    ee = grasp_center_w(robot, ee_cfg)
    coll = env.scene[collection_name]
    pos = coll.data.object_pos_w
    valid = _active_label_mask(env, collection_name, label)

    local_z = pos[..., 2] - env.scene.env_origins[:, None, 2]
    height_above = torch.clamp(local_z - belt_height, min=0.0, max=max_height)
    normalized = height_above / max_height

    distance = torch.norm(pos - ee[:, None, :], dim=-1)
    gate = valid & (distance < max_distance)
    if finger_cfg is not None:
        finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
        gate = gate & (finger_pos > min_closure)[:, None]
    if max_velocity is not None:
        cube_speed = torch.norm(coll.data.object_lin_vel_w, dim=-1)
        gate = gate & (cube_speed < max_velocity)

    gated = torch.where(gate, normalized, torch.zeros_like(normalized))
    return gated.max(dim=1).values


# ── 4. Transport with urgency ────────────────────────────────────────────

def approach_target_tanh(
    env: "ManagerBasedRLEnv",
    collection_name: str,
    label: int,
    drum_name: str,
    belt_height: float,
    std: float = 0.20,
    lift_threshold: float = 0.02,
    urgency_alpha: float = 0.0,
    urgency_beta: float = 0.5,
    belt_start_x: float = -3.0,
    belt_end_x: float = 1.0,
) -> torch.Tensor:
    """Tanh XY proximity of the best-lifted cube of *label* to the drum.

    With urgency weighting: urgency = 1 + alpha * max(0, belt_progress - beta).
    A cube at 80% belt progress (alpha=4, beta=0.5) gets urgency 2.2x.
    Gated on ``was_grasped`` to prevent bumping cubes toward the drum.
    """
    coll = env.scene[collection_name]
    pos = coll.data.object_pos_w
    valid = _active_label_mask(env, collection_name, label)
    pos_d = get_world_pos(env.scene[drum_name], env=env)

    local_z = pos[..., 2] - env.scene.env_origins[:, None, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    d_xy = torch.sqrt(
        (pos[..., 0] - pos_d[:, None, 0]) ** 2
        + (pos[..., 1] - pos_d[:, None, 1]) ** 2
    )
    proximity = 1.0 - torch.tanh(d_xy / std)

    # Apply urgency weighting based on belt progress of the grasped cube
    if urgency_alpha > 0.0:
        local_x = pos[..., 0] - env.scene.env_origins[:, None, 0]
        belt_length = max(belt_end_x - belt_start_x, 0.01)
        progress = torch.clamp((local_x - belt_start_x) / belt_length, 0.0, 1.0)
        urgency = 1.0 + urgency_alpha * torch.clamp(progress - urgency_beta, min=0.0)
        proximity = proximity * urgency

    gated = torch.where(valid & is_lifted, proximity, torch.zeros_like(proximity))
    best = gated.max(dim=1).values

    if hasattr(env, "was_grasped"):
        best = best * env.was_grasped.to(torch.float32)
    return best


# ── 5. Release ────────────────────────────────────────────────────────────

def release_above_target(
    env: "ManagerBasedRLEnv",
    collection_name: str,
    label: int,
    drum_name: str,
    finger_cfg: SceneEntityCfg,
    belt_height: float,
    rim_clearance: float = 0.10,
    drum_radius: float = 0.2735,
) -> torch.Tensor:
    """Reward for opening the gripper when a cube of *label* is above the drum."""
    coll = env.scene[collection_name]
    pos = coll.data.object_pos_w
    valid = _active_label_mask(env, collection_name, label)
    pos_d = get_world_pos(env.scene[drum_name], env=env)

    d_xy = torch.sqrt(
        (pos[..., 0] - pos_d[:, None, 0]) ** 2
        + (pos[..., 1] - pos_d[:, None, 1]) ** 2
    )
    in_xy = d_xy < drum_radius

    local_z = pos[..., 2] - env.scene.env_origins[:, None, 2]
    above_rim = local_z > (belt_height + rim_clearance)

    any_above_drum = torch.any(valid & in_xy & above_rim, dim=1)

    robot: Articulation = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
    openness = 1.0 - closure

    gate = any_above_drum
    if hasattr(env, "was_grasped"):
        gate = gate & env.was_grasped
    return torch.where(gate, openness, torch.zeros_like(openness))


# ── 6. Re-orient (return to belt) ────────────────────────────────────────

def reorient_to_belt(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    collection_name: str,
    label: int,
    std: float = 0.3,
) -> torch.Tensor:
    """Pull gripper back toward next target cube after releasing.

    Active when: gripper is open (not grasping) AND target cubes remain on belt.
    Uses wider std than reach for stronger long-range gradient.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    pos, _idx, has_any = _nearest_active_by_label(env, tip, collection_name, label)
    distance = torch.norm(pos - tip, dim=-1)
    proximity = 1.0 - torch.tanh(distance / std)
    result = torch.where(has_any, proximity, torch.zeros_like(proximity))

    # Only active when NOT holding a cube
    if hasattr(env, "grasp_active"):
        result = result * (~env.grasp_active).float()

    # Only active after at least one successful grasp (was_grasped resets per episode)
    if hasattr(env, "_targets_placed"):
        has_placed = (env._targets_placed > 0).float()
        result = result * has_placed
    return result


# ── Penalties ─────────────────────────────────────────────────────────────

def cubes_in_target(
    env: "ManagerBasedRLEnv",
    collection_name: str,
    label: int,
    target_bin_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Count of active cubes of *label* inside the drum (metric/logging)."""
    coll = env.scene[collection_name]
    pos = coll.data.object_pos_w
    valid = _active_label_mask(env, collection_name, label)
    pos_t = get_world_pos(env.scene[target_bin_name], env=env)

    inside = in_upright_cylinder(pos, pos_t, bin_geom)
    return (inside & valid).to(torch.float32).sum(dim=1)


def cubes_missed(
    env: "ManagerBasedRLEnv",
    collection_name: str,
    label: int,
    x_passed_threshold: float,
    target_bin_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Count of cubes of *label* that passed the conveyor end without reaching the drum."""
    coll = env.scene[collection_name]
    pos = coll.data.object_pos_w
    valid = _active_label_mask(env, collection_name, label)
    pos_t = get_world_pos(env.scene[target_bin_name], env=env)

    local_x = pos[..., 0] - env.scene.env_origins[:, None, 0]
    passed = local_x > x_passed_threshold
    in_target = in_upright_cylinder(pos, pos_t, bin_geom)
    missed = valid & passed & ~in_target
    return missed.to(torch.float32).sum(dim=1)


def cubes_off_conveyor(
    env: "ManagerBasedRLEnv",
    collection_name: str,
    bounds: ConveyorBounds,
) -> torch.Tensor:
    """Penalty count for ALL active cubes (any label) knocked off the conveyor."""
    coll = env.scene[collection_name]
    pos = coll.data.object_pos_w
    active = _active_mask(env, collection_name)
    local = pos - env.scene.env_origins[:, None, :]
    out_y = (local[..., 1] < bounds.y_min) | (local[..., 1] > bounds.y_max)
    out_z = local[..., 2] < bounds.z_min
    off = active & (out_y | out_z)
    return off.to(torch.float32).sum(dim=1)

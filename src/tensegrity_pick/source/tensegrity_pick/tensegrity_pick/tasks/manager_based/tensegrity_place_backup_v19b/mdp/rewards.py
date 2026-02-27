# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward and MDP helper functions for the tensegrity place environment.

Single green / red cube placement task.  Cubes are individual RigidObjects
(not collections), so shapes are always (N, 3) for positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Geometry helpers (reused from cube_sorting)
# ---------------------------------------------------------------------------

# Vertical offset for approach target: rewards drive the EE to the CUBE TOP
# (center + half-size) rather than the cube centre.  This prevents the
# gripper from plunging into the conveyor belt during a top-down approach.
GRASP_APPROACH_OFFSET_Z = 0.025   # half of CUBE_SIZE_M (5 cm)

# Gripper geometry: the finger pad centre lies below the gripper base
# along the gripper's local -z axis.  All finger prim origins coincide
# with the gripper base, so the offset must be computed from the EE
# body quaternion.  Values from the Robotiq 2F-140 USD:
#   Open  → finger-base z=-0.150, tip z=-0.215 → pad centre -0.1825
#   Closed → finger-base z=-0.170, tip z=-0.235 → pad centre -0.2025
# Fixed average (2 cm max error) avoids the need for joint-state lookup.
GRASP_CENTER_LOCAL_Z = -0.1925


def _grasp_center_w(
    robot: Articulation, cfg: SceneEntityCfg
) -> torch.Tensor:
    """World position of the grasp centre between the finger pads.

    Computed by projecting a fixed local-z offset from the EE body
    (tool_link_0) using its world-frame quaternion.  This accounts for
    all finger prim origins sitting at the gripper base in the USD.
    """
    ee_pos = robot.data.body_pos_w[:, cfg.body_ids[0], :]   # (N, 3)
    ee_quat = robot.data.body_quat_w[:, cfg.body_ids[0], :]  # (N, 4) wxyz
    local_offset = ee_pos.new_tensor([0.0, 0.0, GRASP_CENTER_LOCAL_Z])  # (3,)
    world_offset = quat_apply(ee_quat, local_offset.expand_as(ee_pos))  # (N, 3)
    return ee_pos + world_offset


def _grasp_center_vel_w(
    robot: Articulation, cfg: SceneEntityCfg
) -> torch.Tensor:
    """Linear velocity at the grasp centre (EE body velocity).

    The offset is rigid relative to the EE, so the linear velocity
    contribution from angular velocity over ~19 cm is negligible at
    the rotational speeds encountered in this task.
    """
    return robot.data.body_lin_vel_w[:, cfg.body_ids[0], :]


def _grasp_target_w(
    cube: RigidObject,
    offset_z: float = GRASP_APPROACH_OFFSET_Z,
) -> torch.Tensor:
    """Cube position offset upward so rewards target the cube top."""
    pos = cube.data.root_pos_w.clone()
    pos[:, 2] += offset_z
    return pos


@dataclass
class BinCylinder:
    """Upright cylinder representing the target drum."""

    radius: float
    height: float


def _get_world_pos(
    obj: object,
    env: ManagerBasedRLEnv | None = None,
) -> torch.Tensor:
    """Get world position (N, 3) from either a RigidObject or XformPrimView.

    ``AssetBaseCfg`` prims produce a single-prim XFormPrimView whose
    ``get_world_poses()`` returns shape ``(1, 3)`` instead of ``(N, 3)``.
    When *env* is given we derive the local offset from env-0 and
    reconstruct the per-env world positions via ``env_origins + offset``.
    """
    if hasattr(obj, "data") and hasattr(obj.data, "root_pos_w"):
        return obj.data.root_pos_w
    pos, _ = obj.get_world_poses()
    if env is not None and pos.shape[0] == 1 and env.num_envs > 1:
        local_offset = pos[0] - env.scene.env_origins[0]
        pos = env.scene.env_origins + local_offset.unsqueeze(0)
    return pos


def _in_upright_cylinder(
    point: torch.Tensor, center: torch.Tensor, cyl: BinCylinder
) -> torch.Tensor:
    """Check whether points lie inside the cylinder.  Shapes: (N,3)."""
    rel = point - center
    r2 = rel[:, 0] ** 2 + rel[:, 1] ** 2
    inside_xy = r2 <= cyl.radius ** 2
    inside_z = (rel[:, 2] >= 0.0) & (rel[:, 2] <= cyl.height)
    return inside_xy & inside_z


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def ee_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp-centre world position (N,3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return _grasp_center_w(robot, asset_cfg)


def ee_lin_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp-centre world linear velocity (N,3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return _grasp_center_vel_w(robot, asset_cfg)


def cube_rel_pos(
    env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, cube_name: str
) -> torch.Tensor:
    """Relative position of a single cube w.r.t. the grasp centre (N,3).

    Returns zeros when the cube is parked (local x > 50).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)

    cube: RigidObject = env.scene[cube_name]
    pos = cube.data.root_pos_w  # (N, 3)

    # Detect parked cubes
    local_x = (pos[:, 0] - env.scene.env_origins[:, 0])
    active = local_x < 50.0

    rel = pos - ee
    return torch.where(active[:, None], rel, torch.zeros_like(rel))


def drum_rel_pos(
    env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, drum_name: str
) -> torch.Tensor:
    """Relative position of the drum target w.r.t. the grasp centre (N,3)."""
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    return _get_world_pos(env.scene[drum_name], env=env) - ee


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------

@dataclass
class SpawnBox:
    """Axis-aligned box in local env coordinates for cube spawning."""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    yaw_range: Tuple[float, float] = (-3.14159, 3.14159)


def reset_single_cube(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cube_name: str,
    spawn_box: SpawnBox,
    active: bool = True,
    parking_pose: Tuple[float, float, float] = (100.0, 100.0, 1.0),
) -> None:
    """Reset a single RigidObject cube — either place it in the spawn box or park it."""
    device = env.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    N = env_ids.numel()

    env_origins = env.scene.env_origins[env_ids]  # (N, 3)

    if active:
        x = torch.empty((N,), device=device).uniform_(*spawn_box.x_range)
        y = torch.empty((N,), device=device).uniform_(*spawn_box.y_range)
        z = torch.empty((N,), device=device).uniform_(*spawn_box.z_range)
        yaw = torch.empty((N,), device=device).uniform_(*spawn_box.yaw_range)

        qw = torch.cos(0.5 * yaw)
        qz = torch.sin(0.5 * yaw)
        quat = torch.stack(
            [qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1
        )
        local_pos = torch.stack([x, y, z], dim=-1)
        world_pos = local_pos + env_origins
    else:
        world_pos = (
            torch.tensor(parking_pose, device=device).unsqueeze(0).expand(N, -1)
            + env_origins
        )
        quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            .unsqueeze(0)
            .expand(N, -1)
        )

    pose = torch.cat([world_pos, quat], dim=-1)  # (N, 7)
    velocity = torch.zeros((N, 6), device=device)

    cube: RigidObject = env.scene[cube_name]
    cube.write_root_pose_to_sim(pose, env_ids=env_ids)
    cube.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def reset_place_cubes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    spawn_box: SpawnBox,
    green_name: str,
    red_name: str,
    parking_pose: Tuple[float, float, float] = (100.0, 100.0, 1.0),
) -> None:
    """Reset both cubes.  Red activation is curriculum-controlled via env.extras."""
    red_active = _is_red_active(env)
    reset_single_cube(env, env_ids, green_name, spawn_box, active=True, parking_pose=parking_pose)
    reset_single_cube(env, env_ids, red_name, spawn_box, active=red_active, parking_pose=parking_pose)


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

_CURRICULUM_KEY = "place_curriculum_step"
_RED_ACTIVE_KEY = "place_red_active"


def _is_red_active(env: ManagerBasedRLEnv) -> bool:
    """Check whether curriculum has enabled the red cube."""
    extras = getattr(env, "extras", None)
    if extras is not None and isinstance(extras, dict):
        return bool(extras.get(_RED_ACTIVE_KEY, False))
    return False


def activate_red_cube_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_steps: int,
) -> None:
    """Curriculum term: enable red cube after *num_steps* environment steps.

    This should be registered as a CurriculumTerm (mode handled by manager).
    """
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}

    total = env.common_step_counter  # global step counter
    env.extras[_RED_ACTIVE_KEY] = total >= num_steps


# ---------------------------------------------------------------------------
# Termination helpers
# ---------------------------------------------------------------------------

def green_in_drum(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """True for envs where the green cube is inside the target drum (early success)."""
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w  # (N, 3)
    pos_d = _get_world_pos(env.scene[drum_name], env=env)
    return _in_upright_cylinder(pos_g, pos_d, bin_geom)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def ee_to_green_l2(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
) -> torch.Tensor:
    """L2 distance from grasp centre to the green cube."""
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    return torch.norm(pos_g - ee, dim=-1)


def green_cube_in_target(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Returns 1.0 per env where the green cube is inside the drum, else 0.

    Gated on ``was_grasped`` when available to prevent reward hacking
    (agent pushing cube into drum without grasping).
    """
    inside = green_in_drum(env, green_name, drum_name, bin_geom).to(torch.float32)
    if hasattr(env, "was_grasped"):
        return inside * env.was_grasped.to(torch.float32)
    return inside


def red_cube_in_target(
    env: ManagerBasedRLEnv,
    red_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Penalty: returns 1.0 per env where the red cube is inside the drum."""
    red: RigidObject = env.scene[red_name]
    pos_r = red.data.root_pos_w
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    # Only count if the red cube is not parked
    local_x = pos_r[:, 0] - env.scene.env_origins[:, 0]
    active = local_x < 50.0

    inside = _in_upright_cylinder(pos_r, pos_d, bin_geom)
    return (inside & active).to(torch.float32)


def green_lifted_bonus(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    belt_height: float,
    lift_threshold: float = 0.08,
    proximity_threshold: float = 0.15,
) -> torch.Tensor:
    """Bonus when green cube is lifted above belt.

    With a sticky gripper this checks ``grasp_active`` directly.
    Falls back to proximity check for non-sticky envs.
    """
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    local_z = pos_g[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    if hasattr(env, "grasp_active"):
        return (is_lifted & env.grasp_active).to(torch.float32)

    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    d = torch.norm(pos_g - ee, dim=-1)
    is_close = d < proximity_threshold
    return (is_lifted & is_close).to(torch.float32)


def green_approaching_target(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    belt_height: float,
    lift_threshold: float = 0.05,
) -> torch.Tensor:
    """Dense shaping: XY distance from a lifted green cube to the drum.

    Returns 0 when cube is not lifted.
    """
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    local_z = pos_g[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    d_xy = torch.sqrt(
        (pos_g[:, 0] - pos_d[:, 0]) ** 2 + (pos_g[:, 1] - pos_d[:, 1]) ** 2
    )

    return torch.where(is_lifted, d_xy, torch.zeros_like(d_xy))


def approach_target_tanh(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    belt_height: float,
    std: float = 0.20,
    lift_threshold: float = 0.05,
) -> torch.Tensor:
    """Bounded tanh reward for XY proximity of a lifted cube to the drum.

    Returns ``1 - tanh(d_xy / std)`` when cube is lifted, else 0.
    Gated on ``was_grasped`` when available to prevent reward hacking.
    """
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    local_z = pos_g[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    d_xy = torch.sqrt(
        (pos_g[:, 0] - pos_d[:, 0]) ** 2 + (pos_g[:, 1] - pos_d[:, 1]) ** 2
    )
    proximity = 1.0 - torch.tanh(d_xy / std)

    gate = is_lifted
    if hasattr(env, "was_grasped"):
        gate = gate & env.was_grasped

    return torch.where(gate, proximity, torch.zeros_like(proximity))


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared difference between consecutive actions."""
    a = env.action_manager.action
    ap = env.action_manager.prev_action
    return torch.sum((a - ap) ** 2, dim=1)


# ---------------------------------------------------------------------------
# Metric / logging reward functions
# ---------------------------------------------------------------------------

def grasp_proximity_tanh(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    std: float = 0.05,
    approach_offset_z: float = GRASP_APPROACH_OFFSET_Z,
) -> torch.Tensor:
    """Tanh-shaped reward for grasp-centre proximity to the green cube *top*.

    Dense shaping that encourages approaching before grasping.  The target
    position is offset upward by ``approach_offset_z`` so the gripper stops
    at the correct grasping height instead of driving into the belt.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    target = _grasp_target_w(green, approach_offset_z)
    distance = torch.norm(target - ee, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def place_success_bonus(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Binary 1.0 when the green cube is inside the drum.

    With a small weight this serves as a TensorBoard success-rate metric.
    Gated on ``was_grasped`` to count only genuine placements.
    """
    inside = green_in_drum(env, green_name, drum_name, bin_geom).to(torch.float32)
    if hasattr(env, "was_grasped"):
        return inside * env.was_grasped.to(torch.float32)
    return inside


def green_grasp_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    belt_height: float,
    lift_threshold: float = 0.05,
    proximity_threshold: float = 0.10,
) -> torch.Tensor:
    """Binary 1.0 when the green cube is grasped and lifted.

    Uses ``grasp_active`` from the sticky gripper when available,
    otherwise falls back to proximity-based detection.
    """
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    local_z = pos_g[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    if hasattr(env, "grasp_active"):
        return (is_lifted & env.grasp_active).to(torch.float32)

    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    d = torch.norm(pos_g - ee, dim=-1)
    is_close = d < proximity_threshold
    return (is_lifted & is_close).to(torch.float32)


def joint_vel_l2_controlled(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 norm of controlled joint velocities, clamped per-joint."""
    asset = env.scene[asset_cfg.name]
    clamped = torch.clamp(asset.data.joint_vel[:, asset_cfg.joint_ids], -max_velocity, max_velocity)
    return torch.sum(clamped**2, dim=1)


def joint_vel_out_of_limit(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Termination: True when any joint velocity exceeds max_velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def gripper_open_penalty(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    belt_height: float,
    lift_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalise releasing the cube while it is lifted (dropping it).

    Returns 1.0 when the cube was lifted in the previous frame but is
    now falling (not grasped and above belt).
    """
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    local_z = pos_g[:, 2] - env.scene.env_origins[:, 2]
    was_lifted = local_z > (belt_height + lift_threshold)

    if hasattr(env, "grasp_active"):
        # Dropped = cube is above belt but no longer grasped
        return (was_lifted & ~env.grasp_active).to(torch.float32)

    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    d = torch.norm(pos_g - ee, dim=-1)
    dropped = was_lifted & (d > 0.20)
    return dropped.to(torch.float32)


def gripper_closing_near_cube(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    proximity_threshold: float = 0.10,
    finger_joint_name: str = "finger_joint",
) -> torch.Tensor:
    """Reward the agent for commanding grasp when near the green cube.

    Returns a smooth proximity signal (0–1) that is non-zero only when the
    agent commands the gripper to close (negative gripper action).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    distance = torch.norm(pos_g - ee, dim=-1)
    proximity = torch.clamp(1.0 - distance / proximity_threshold, min=0.0)

    # Only reward when the agent is commanding gripper close
    gripper_action = env.action_manager.action[:, -1]
    wants_close = (gripper_action < 0.0).to(torch.float32)
    return proximity * wants_close


def open_gripper_approach(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    std: float = 0.15,
) -> torch.Tensor:
    """Reward approaching the cube with an open gripper.

    Encourages the robot to keep the gripper open during approach,
    preventing premature closure that pushes the cube away.  The
    reward is ``(1 - tanh(d / std)) * wants_open``.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    distance = torch.norm(pos_g - ee, dim=-1)
    proximity = 1.0 - torch.tanh(distance / std)

    gripper_action = env.action_manager.action[:, -1]
    wants_open = (gripper_action >= 0.0).to(torch.float32)

    return proximity * wants_open


def phase_aware_gripper_reward(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    transition_distance: float = 0.10,
    transition_sharpness: float = 30.0,
    std: float = 0.15,
    approach_offset_z: float = GRASP_APPROACH_OFFSET_Z,
) -> torch.Tensor:
    """Two-phase gripper reward: approach open, grasp closed.

    Transitions smoothly from rewarding an open gripper (when far) to
    rewarding a closed gripper (when close) based on grasp-centre-to-
    cube-top distance.  The ``transition_distance`` marks the switch
    point.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    target = _grasp_target_w(green, approach_offset_z)
    distance = torch.norm(target - ee, dim=-1)

    proximity = 1.0 - torch.tanh(distance / std)

    close_weight = torch.sigmoid(
        (transition_distance - distance) * transition_sharpness
    )
    open_weight = 1.0 - close_weight

    gripper_action = env.action_manager.action[:, -1]
    wants_close = (gripper_action < 0.0).to(torch.float32)
    wants_open = 1.0 - wants_close

    return proximity * (open_weight * wants_open + close_weight * wants_close)


# Maximum finger_joint position for the Robotiq 2F-140 (fully closed).
FINGER_JOINT_MAX = 0.7854


def finger_grasp_reward(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    finger_cfg: SceneEntityCfg,
    proximity_threshold: float = 0.08,
) -> torch.Tensor:
    """Bridge reward for physically closing fingers around the cube.

    Returns the finger closure ratio (0–1) when the cube is within
    ``proximity_threshold`` of the grasp centre, else 0.  This bridges
    the gap between :func:`phase_aware_gripper_reward` (commands close)
    and :func:`green_lifted_bonus` (requires actual lift).  By rewarding
    the *actual* finger-joint position rather than the commanded action,
    the agent learns to hold the grasp long enough for the fingers to
    physically close around the cube.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    distance = torch.norm(pos_g - ee, dim=-1)
    is_close = distance < proximity_threshold

    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    finger_ratio = torch.clamp(finger_pos / FINGER_JOINT_MAX, 0.0, 1.0)

    return torch.where(is_close, finger_ratio, torch.zeros_like(distance))


def ee_to_green_distance_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
) -> torch.Tensor:
    """Raw distance from grasp centre to green cube (for TensorBoard metric logging)."""
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    return torch.norm(pos_g - ee, dim=-1)


def ee_height_match(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    std: float = 0.10,
    approach_offset_z: float = GRASP_APPROACH_OFFSET_Z,
) -> torch.Tensor:
    """Reward for matching grasp-centre height to the cube top.

    Returns ``1 - tanh(|gc_z - cube_top_z| / std)``.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    target = _grasp_target_w(green, approach_offset_z)
    height_error = torch.abs(ee[:, 2] - target[:, 2])
    return 1.0 - torch.tanh(height_error / std)


def belt_contact_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    belt_height: float,
    margin: float = 0.02,
    max_depth: float = 0.05,
) -> torch.Tensor:
    """Penalty proportional to how far gripper bodies go below the belt.

    Returns ``clamp(belt_height + margin - min_z, 0, max_depth)`` where
    ``min_z`` is the lowest body in the configured body set.  The depth
    cap prevents catastrophically large rewards during early exploration
    when the arm swings far below the belt.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    local_z = body_pos[:, :, 2] - env.scene.env_origins[:, 2:3]
    min_z = local_z.min(dim=1).values
    return torch.clamp(belt_height + margin - min_z, min=0.0, max=max_depth)


def belt_collision_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    belt_height: float,
    max_penetration: float = 0.05,
) -> torch.Tensor:
    """Terminate episodes where gripper bodies penetrate the belt surface.

    Returns a boolean tensor that is ``True`` for environments where the
    lowest tracked body is more than *max_penetration* below *belt_height*.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    local_z = body_pos[:, :, 2] - env.scene.env_origins[:, 2:3]
    min_z = local_z.min(dim=1).values
    return min_z < (belt_height - max_penetration)


def joint_effort_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold_ratio: float = 5.0,
) -> torch.Tensor:
    """Terminate when *any* tracked joint's *computed* torque is extreme.

    The PD controller *computed_torque* represents the torque the
    controller would like to apply; the physics engine clamps this to
    *applied_torque* within the effort limit.  When the computed torque
    far exceeds the limit (by *threshold_ratio*×), the mechanism is under
    extreme stress — equivalent to the physical robot breaking.

    A high threshold (default 5×) avoids false positives during normal
    PD clamping (which occurs routinely with high-stiffness actuators).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    computed = robot.data.computed_torque[:, asset_cfg.joint_ids]
    effort_limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    extreme = torch.abs(computed) > threshold_ratio * effort_limits
    return torch.any(extreme, dim=1)


def joint_torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Smooth penalty proportional to the fraction of effort limit used.

    Returns mean(|applied_torque| / effort_limit) across tracked joints,
    clipped to [0, 1].  Provides continuous gradient to avoid torque
    saturation rather than the hard termination alone.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    applied = robot.data.applied_torque[:, asset_cfg.joint_ids]
    effort_limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    fractions = torch.abs(applied) / (effort_limits + 1e-8)
    return torch.clamp(fractions, max=1.0).mean(dim=1)

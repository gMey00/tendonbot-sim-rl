"""Reward, observation, termination and reset functions for the shirt place task.

Follows the same 6-phase sequential reward structure as cube_place:
  1. Reach:      tanh proximity (dynamic fingertip → shirt proxy)
  2. Grasp:      closure × proximity (fingertip-based, at belt level)
  3. Lift:       binary + height bonus (velocity-gated, closure-gated)
  4. Transport:  tanh XY distance to drum (gated: was_grasped + z > belt)
  5. Release:    gripper openness when shirt above drum
  6. Success:    per-step bonus while shirt rests in drum

Key differences from cube_place:
  - Single object (shirt_proxy) instead of green+red cubes
  - No red cube curriculum or clearance rewards
  - Cloth-specific observations (keypoints, spread) to be added once
    ClothObject tensor API is wired up
  - Higher friction parameters for adhesion-based cloth grasping

The ``shirt_proxy`` is a temporary rigid-body stand-in for the cloth
centroid.  Once ClothObject.update() is implemented, observations and
rewards will read from the actual cloth state tensors instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Shared gripper geometry + observation/reward utilities
from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import (
    FINGER_JOINT_CLOSE_POS,
    FINGER_TIP_CLOSED_Z,
    FINGER_TIP_LOCAL_Z,
    FINGER_TIP_OPEN_Z,
    GRASP_CENTER_LOCAL_Z,
    BinCylinder,
    ConveyorBounds,
    SpawnBox,
    action_rate_l2,
    arm_velocity_bonus,
    base_velocity_l2,
    belt_collision_termination,
    belt_contact_penalty,
    drum_rel_pos,
    dynamic_finger_tip_w as _dynamic_finger_tip_w,
    ee_lin_vel_w,
    ee_pos_w,
    get_world_pos as _get_world_pos,
    grasp_center_w as _grasp_center_w,
    gripper_closure,
    gripper_torque_residual,
    in_upright_cylinder as _in_upright_cylinder,
    joint_torque_penalty,
    joint_vel_l2_controlled,
    joint_vel_out_of_limit,
)


# ---------------------------------------------------------------------------
# Observations (shirt-specific)
# ---------------------------------------------------------------------------

def shirt_rel_pos(
    env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, shirt_name: str
) -> torch.Tensor:
    """Relative position of shirt proxy w.r.t. the grasp centre (N, 3)."""
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    shirt: RigidObject = env.scene[shirt_name]
    return shirt.data.root_pos_w - ee


def fingertip_rel_shirt(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    shirt_name: str,
) -> torch.Tensor:
    """Relative position of shirt proxy w.r.t. dynamic fingertip (N, 3)."""
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    shirt: RigidObject = env.scene[shirt_name]
    return shirt.data.root_pos_w - tip


def shirt_velocity(env: ManagerBasedRLEnv, shirt_name: str) -> torch.Tensor:
    """Shirt proxy linear velocity (N, 3)."""
    shirt: RigidObject = env.scene[shirt_name]
    return shirt.data.root_lin_vel_w


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------

def reset_shirt(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    shirt_name: str,
    spawn_box: SpawnBox,
) -> None:
    """Reset the shirt proxy to a random position within the spawn box."""
    device = env.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    x = torch.empty((n,), device=device).uniform_(*spawn_box.x_range)
    y = torch.empty((n,), device=device).uniform_(*spawn_box.y_range)
    z = torch.empty((n,), device=device).uniform_(*spawn_box.z_range)
    yaw = torch.empty((n,), device=device).uniform_(*spawn_box.yaw_range)
    qw = torch.cos(0.5 * yaw)
    qz = torch.sin(0.5 * yaw)
    quat = torch.stack(
        [qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1
    )
    world_pos = torch.stack([x, y, z], dim=-1) + env_origins

    shirt: RigidObject = env.scene[shirt_name]
    shirt.write_root_pose_to_sim(
        torch.cat([world_pos, quat], dim=-1), env_ids=env_ids
    )
    shirt.write_root_velocity_to_sim(
        torch.zeros((n, 6), device=device), env_ids=env_ids
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def shirt_ee_distance(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    shirt_name: str,
    std: float = 0.1,
) -> torch.Tensor:
    """Tanh reward for dynamic-fingertip proximity to the shirt proxy.

    Gated: turns off once the shirt has been grasped (grasp_active=False).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    shirt: RigidObject = env.scene[shirt_name]
    distance = torch.norm(shirt.data.root_pos_w - tip, dim=-1)
    result = 1.0 - torch.tanh(distance / std)

    if hasattr(env, "grasp_active"):
        result = result * (~env.grasp_active).float()
    return result


def shirt_grasp_reward(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    shirt_name: str,
    std: float = 0.08,
) -> torch.Tensor:
    """Reward for closing the gripper near the shirt proxy.

    Returns ``closure_fraction * (1 - tanh(dist / std))``.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    shirt: RigidObject = env.scene[shirt_name]
    distance = torch.norm(shirt.data.root_pos_w - tip, dim=-1)
    proximity = 1.0 - torch.tanh(distance / std)

    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)

    return closure * proximity


def shirt_is_lifted(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    shirt_name: str,
    belt_height: float,
    minimal_height: float = 0.06,
    max_distance: float = 0.15,
    finger_cfg: SceneEntityCfg | None = None,
    min_closure: float = 0.20,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Binary reward: 1.0 when the shirt is above threshold AND near gripper."""
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    shirt: RigidObject = env.scene[shirt_name]
    shirt_pos = shirt.data.root_pos_w

    local_z = shirt_pos[:, 2] - env.scene.env_origins[:, 2]
    is_above = local_z > belt_height + minimal_height

    distance = torch.norm(shirt_pos - ee, dim=-1)
    is_near = distance < max_distance

    gate = is_above & is_near
    if finger_cfg is not None:
        finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
        gate = gate & (finger_pos > min_closure)
    if max_velocity is not None:
        shirt_speed = torch.norm(shirt.data.root_lin_vel_w, dim=-1)
        gate = gate & (shirt_speed < max_velocity)

    return torch.where(gate, 1.0, 0.0)


def shirt_height_bonus(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    shirt_name: str,
    belt_height: float,
    max_height: float = 0.30,
    max_distance: float = 0.15,
    finger_cfg: SceneEntityCfg | None = None,
    min_closure: float = 0.20,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Continuous reward proportional to shirt height above belt."""
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    shirt: RigidObject = env.scene[shirt_name]
    shirt_pos = shirt.data.root_pos_w

    local_z = shirt_pos[:, 2] - env.scene.env_origins[:, 2]
    height_above = torch.clamp(local_z - belt_height, min=0.0, max=max_height)
    normalized = height_above / max_height

    distance = torch.norm(shirt_pos - ee, dim=-1)
    gate = distance < max_distance
    if finger_cfg is not None:
        finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
        gate = gate & (finger_pos > min_closure)
    if max_velocity is not None:
        shirt_speed = torch.norm(shirt.data.root_lin_vel_w, dim=-1)
        gate = gate & (shirt_speed < max_velocity)

    return torch.where(gate, normalized, torch.zeros_like(normalized))


def shirt_approach_target(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    belt_height: float,
    std: float = 0.20,
    lift_threshold: float = 0.04,
) -> torch.Tensor:
    """Bounded tanh reward for XY proximity of a lifted shirt to the drum.

    Gated on ``was_grasped`` to prevent reward hacking.
    """
    shirt: RigidObject = env.scene[shirt_name]
    pos_s = shirt.data.root_pos_w
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    local_z = pos_s[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    d_xy = torch.sqrt(
        (pos_s[:, 0] - pos_d[:, 0]) ** 2 + (pos_s[:, 1] - pos_d[:, 1]) ** 2
    )
    proximity = 1.0 - torch.tanh(d_xy / std)

    gate = is_lifted
    if hasattr(env, "was_grasped"):
        gate = gate & env.was_grasped

    return torch.where(gate, proximity, torch.zeros_like(proximity))


def shirt_in_drum(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """True for envs where the shirt proxy is inside the target drum."""
    shirt: RigidObject = env.scene[shirt_name]
    pos_d = _get_world_pos(env.scene[drum_name], env=env)
    return _in_upright_cylinder(shirt.data.root_pos_w, pos_d, bin_geom)


def shirt_in_target(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Returns 1.0 when shirt is in drum, gated on was_grasped."""
    inside = shirt_in_drum(env, shirt_name, drum_name, bin_geom).to(torch.float32)
    if hasattr(env, "was_grasped"):
        return inside * env.was_grasped.to(torch.float32)
    return inside


def shirt_release_above_target(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    finger_cfg: SceneEntityCfg,
    belt_height: float,
    rim_clearance: float = 0.10,
    drum_radius: float = 0.2735,
) -> torch.Tensor:
    """Reward for opening gripper when shirt is above the drum."""
    shirt: RigidObject = env.scene[shirt_name]
    pos_s = shirt.data.root_pos_w
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    d_xy = torch.sqrt(
        (pos_s[:, 0] - pos_d[:, 0]) ** 2 + (pos_s[:, 1] - pos_d[:, 1]) ** 2
    )
    in_xy = d_xy < drum_radius

    local_z = pos_s[:, 2] - env.scene.env_origins[:, 2]
    above_rim = local_z > (belt_height + rim_clearance)

    robot: Articulation = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
    openness = 1.0 - closure

    gate = in_xy & above_rim
    if hasattr(env, "was_grasped"):
        gate = gate & env.was_grasped

    return torch.where(gate, openness, torch.zeros_like(openness))


# ---------------------------------------------------------------------------
# Metric-only rewards
# ---------------------------------------------------------------------------

def shirt_place_success_bonus(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    return shirt_in_target(env, shirt_name, drum_name, bin_geom)


def shirt_grasp_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    shirt_name: str,
    belt_height: float,
    lift_threshold: float = 0.06,
    proximity_threshold: float = 0.10,
) -> torch.Tensor:
    """Binary 1.0 when the shirt is grasped and lifted."""
    shirt: RigidObject = env.scene[shirt_name]
    local_z = shirt.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    if hasattr(env, "grasp_active"):
        return (is_lifted & env.grasp_active).to(torch.float32)

    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    is_close = torch.norm(shirt.data.root_pos_w - ee, dim=-1) < proximity_threshold
    return (is_lifted & is_close).to(torch.float32)


def ee_to_shirt_distance_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    shirt_name: str,
) -> torch.Tensor:
    """Raw distance from grasp centre to shirt proxy (for TensorBoard)."""
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    return torch.norm(env.scene[shirt_name].data.root_pos_w - ee, dim=-1)


def shirt_off_conveyor_penalty(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    bounds: ConveyorBounds,
) -> torch.Tensor:
    """Penalty (1.0) when shirt proxy is outside conveyor region."""
    shirt: RigidObject = env.scene[shirt_name]
    local = shirt.data.root_pos_w - env.scene.env_origins
    out_y = (local[:, 1] < bounds.y_min) | (local[:, 1] > bounds.y_max)
    out_z = local[:, 2] < bounds.z_min
    return (out_y | out_z).to(torch.float32)

"""Reward, observation, termination and reset functions for the place task.

Sequential reward structure for pick-and-place:
  1. Reach:      tanh proximity  (dynamic fingertip → nearest cube)
  2. Grasp:      closure × proximity (fingertip-based, at belt level)
  3. Lift:       binary + height bonus (velocity-gated, closure-gated)
  4. Transport:  tanh XY distance to drum (gated: was_grasped + z > belt + 0.02)
  5. Release:    gripper openness when cube above drum
  6. Success:    per-step bonus while correct colour cube rests in drum

Key design choices:
  - Reach/grasp use _dynamic_finger_tip_w instead of _grasp_center_w so
    the robot approaches from above (finger tips are the lowest point).
  - Reach targets the NEAREST active cube, making the reward general
    across green/red for the approach phase.
  - Transport gate (z > 0.82) is deliberately LOW so the approach reward
    stays active even when the cube dips during lateral arm extension.
  - Cubes knocked off the conveyor are penalised.
  - Episodes run the full duration without early success termination, so
    the green_in_target reward accumulates over remaining steps and
    clearly dominates indefinite holding.
  - Grasp detection remains closure-based (robust, matches Isaac Lab).
    Torque residual is exposed as an observation for the policy.

Cubes are individual RigidObjects (not collections), so shapes are
always (N, 3) for positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Shared gripper geometry — imported from shared/gripper_cfg.py
# ---------------------------------------------------------------------------
from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import (
    # Constants
    FINGER_TIP_OPEN_Z,
    FINGER_TIP_CLOSED_Z,
    FINGER_TIP_LOCAL_Z,
    GRASP_CENTER_LOCAL_Z,
    FINGER_JOINT_CLOSE_POS,
    # Dataclasses
    BinCylinder,
    SpawnBox,
    ConveyorBounds,
    # Geometry helpers (public names → aliased as private for internal use)
    project_local_z_offset as _project_local_z_offset,
    grasp_center_w as _grasp_center_w,
    finger_tip_w as _finger_tip_w,
    dynamic_finger_tip_w as _dynamic_finger_tip_w,
    get_world_pos as _get_world_pos,
    in_upright_cylinder as _in_upright_cylinder,
    # Observation functions (re-exported for place_env_cfg.py)
    ee_pos_w,
    ee_lin_vel_w,
    gripper_closure,
    gripper_torque_residual,
    drum_rel_pos,
    # Shared reward / penalty functions
    action_rate_l2,
    joint_vel_l2_controlled,
    base_velocity_l2,
    arm_velocity_bonus,
    belt_contact_penalty,
    joint_torque_penalty,
    # Shared termination functions
    joint_vel_out_of_limit,
    belt_collision_termination,
)


def _cube_is_active(
    cube: RigidObject, env: ManagerBasedRLEnv
) -> torch.Tensor:
    """Per-env bool: True when the cube is on the belt (not parked at x>50)."""
    local_x = cube.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return local_x < 50.0


def _nearest_active_cube_distance(
    env: ManagerBasedRLEnv,
    reference_pos: torch.Tensor,
    green_name: str,
    red_name: str,
) -> torch.Tensor:
    """Distance from ``reference_pos`` (N, 3) to the nearest active cube.

    Inactive (parked) cubes are excluded by assigning infinite distance.
    """
    green: RigidObject = env.scene[green_name]
    red: RigidObject = env.scene[red_name]
    green_active = _cube_is_active(green, env)
    red_active = _cube_is_active(red, env)

    INF = 1e6
    green_dist = torch.where(
        green_active,
        torch.norm(green.data.root_pos_w - reference_pos, dim=-1),
        torch.full_like(green_active, INF, dtype=reference_pos.dtype),
    )
    red_dist = torch.where(
        red_active,
        torch.norm(red.data.root_pos_w - reference_pos, dim=-1),
        torch.full_like(red_active, INF, dtype=reference_pos.dtype),
    )
    return torch.minimum(green_dist, red_dist)


# ---------------------------------------------------------------------------
# Observations (task-specific; shared obs imported from gripper_cfg above)
# ---------------------------------------------------------------------------


def cube_rel_pos(
    env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, cube_name: str
) -> torch.Tensor:
    """Relative position of a single cube w.r.t. the grasp centre (N, 3).

    Returns zeros when the cube is parked (local x > 50).
    """
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    cube: RigidObject = env.scene[cube_name]

    active = _cube_is_active(cube, env)

    rel = cube.data.root_pos_w - ee
    return torch.where(active[:, None], rel, torch.zeros_like(rel))


def fingertip_rel_cube(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    cube_name: str,
) -> torch.Tensor:
    """Relative position of a cube w.r.t. the dynamic fingertip (N, 3).

    The fingertip position is interpolated between open/closed based on
    the actual finger_joint state, giving the policy spatial awareness
    of where the contact surfaces are.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    cube: RigidObject = env.scene[cube_name]

    active = _cube_is_active(cube, env)

    rel = cube.data.root_pos_w - tip
    return torch.where(active[:, None], rel, torch.zeros_like(rel))


# drum_rel_pos, gripper_closure, gripper_torque_residual imported from shared/gripper_cfg.py


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------

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
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if active:
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
    else:
        world_pos = (
            torch.tensor(parking_pose, device=device).unsqueeze(0).expand(n, -1)
            + env_origins
        )
        quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            .unsqueeze(0)
            .expand(n, -1)
        )

    cube: RigidObject = env.scene[cube_name]
    cube.write_root_pose_to_sim(torch.cat([world_pos, quat], dim=-1), env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)


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
    """Curriculum term: enable red cube after *num_steps* environment steps."""
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    env.extras[_RED_ACTIVE_KEY] = env.common_step_counter >= num_steps


# ---------------------------------------------------------------------------
# Termination functions
# ---------------------------------------------------------------------------

def green_in_drum(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """True for envs where the green cube is inside the target drum."""
    green: RigidObject = env.scene[green_name]
    pos_d = _get_world_pos(env.scene[drum_name], env=env)
    return _in_upright_cylinder(green.data.root_pos_w, pos_d, bin_geom)


# joint_vel_out_of_limit, belt_collision_termination imported from shared/gripper_cfg.py


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def object_ee_distance(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    green_name: str,
    red_name: str,
    std: float = 0.1,
) -> torch.Tensor:
    """Tanh reward for dynamic-fingertip proximity to the nearest active cube.

    Uses ``_dynamic_finger_tip_w`` so that the reward guides the robot to
    approach from above (fingertips are the lowest point).  Targets the
    nearest active cube so the policy generalises across colours.

    Gated: only active when gripper is NOT holding a cube (grasp_active=False).
    This frees the reward budget for transport once a cube is grasped.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    distance = _nearest_active_cube_distance(env, tip, green_name, red_name)
    result = 1.0 - torch.tanh(distance / std)
    # Gate: turn off reach reward once holding a cube
    if hasattr(env, "grasp_active"):
        result = result * (~env.grasp_active).float()
    return result


def object_is_lifted(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    belt_height: float,
    minimal_height: float = 0.04,
    max_distance: float = 0.15,
    finger_cfg: SceneEntityCfg | None = None,
    min_closure: float = 0.20,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Binary reward: 1.0 when the cube is above threshold AND near the gripper.

    Gates:
      - Height: cube local z > belt_height + minimal_height
      - Proximity: cube within max_distance of grasp centre
      - Closure (optional): finger_joint > min_closure
      - Velocity (optional): cube speed < max_velocity (rejects bounces)
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    cube_pos = green.data.root_pos_w

    local_z = cube_pos[:, 2] - env.scene.env_origins[:, 2]
    is_above = local_z > belt_height + minimal_height

    distance = torch.norm(cube_pos - ee, dim=-1)
    is_near = distance < max_distance

    gate = is_above & is_near
    if finger_cfg is not None:
        finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
        gate = gate & (finger_pos > min_closure)
    if max_velocity is not None:
        cube_speed = torch.norm(green.data.root_lin_vel_w, dim=-1)
        gate = gate & (cube_speed < max_velocity)

    return torch.where(gate, 1.0, 0.0)


def cube_height_bonus(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    belt_height: float,
    max_height: float = 0.20,
    max_distance: float = 0.15,
    finger_cfg: SceneEntityCfg | None = None,
    min_closure: float = 0.20,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Continuous reward proportional to cube height above belt.

    Returns ``clamp(h / max_height, 0, 1)`` when all gates pass.

    Gates:
      - Proximity: cube within max_distance of grasp centre
      - Closure (optional): finger_joint > min_closure
      - Velocity (optional): cube speed < max_velocity (rejects bounces)
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    cube_pos = green.data.root_pos_w

    local_z = cube_pos[:, 2] - env.scene.env_origins[:, 2]
    height_above = torch.clamp(local_z - belt_height, min=0.0, max=max_height)
    normalized = height_above / max_height

    distance = torch.norm(cube_pos - ee, dim=-1)
    gate = distance < max_distance
    if finger_cfg is not None:
        finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
        gate = gate & (finger_pos > min_closure)
    if max_velocity is not None:
        cube_speed = torch.norm(green.data.root_lin_vel_w, dim=-1)
        gate = gate & (cube_speed < max_velocity)

    return torch.where(gate, normalized, torch.zeros_like(normalized))


def approach_target_tanh(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    belt_height: float,
    std: float = 0.20,
    lift_threshold: float = 0.04,
) -> torch.Tensor:
    """Bounded tanh reward for XY proximity of a lifted cube to the drum.

    Returns ``1 - tanh(d_xy / std)`` when cube is lifted, else 0.
    Gated on ``was_grasped`` to prevent reward hacking (pushing cube
    laterally toward the drum without grasping).
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


def green_cube_in_target(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Returns 1.0 when the green cube is inside the drum.

    Gated on ``was_grasped`` to prevent reward hacking.
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
    """Penalty: 1.0 when the red cube is inside the drum."""
    red: RigidObject = env.scene[red_name]
    pos_r = red.data.root_pos_w
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    local_x = pos_r[:, 0] - env.scene.env_origins[:, 0]
    active = local_x < 50.0

    return (_in_upright_cylinder(pos_r, pos_d, bin_geom) & active).to(torch.float32)


def red_clearance_from_drum(
    env: ManagerBasedRLEnv,
    red_name: str,
    drum_name: str,
    std: float = 0.4,
) -> torch.Tensor:
    """Reward for the red cube being far from the drum.

    Returns ``tanh(d_xy / std)`` — reward increases as the red cube
    is further from the drum in XY.  Only active when the red cube
    is not parked (local_x < 50).  Gated on ``!was_grasped`` so it
    only drives behaviour before the green cube has been grasped —
    once grasped, the agent should focus on transport + release.
    """
    red: RigidObject = env.scene[red_name]
    pos_r = red.data.root_pos_w
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    local_x = pos_r[:, 0] - env.scene.env_origins[:, 0]
    active = local_x < 50.0

    d_xy = torch.sqrt(
        (pos_r[:, 0] - pos_d[:, 0]) ** 2 + (pos_r[:, 1] - pos_d[:, 1]) ** 2
    )
    clearance = torch.tanh(d_xy / std)

    gate = active
    if hasattr(env, "was_grasped"):
        gate = gate & (~env.was_grasped)

    return torch.where(gate, clearance, torch.zeros_like(clearance))


# base_velocity_l2, action_rate_l2, joint_vel_l2_controlled,
# belt_contact_penalty, joint_torque_penalty imported from shared/gripper_cfg.py


# ---------------------------------------------------------------------------
# Metric-only rewards (tiny weight, for TensorBoard monitoring)
# ---------------------------------------------------------------------------

def place_success_bonus(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Binary 1.0 when the green cube is inside the drum (gated on was_grasped)."""
    return green_cube_in_target(env, green_name, drum_name, bin_geom)


def green_grasp_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    belt_height: float,
    lift_threshold: float = 0.04,
    proximity_threshold: float = 0.10,
) -> torch.Tensor:
    """Binary 1.0 when the green cube is grasped and lifted."""
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    local_z = pos_g[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    if hasattr(env, "grasp_active"):
        return (is_lifted & env.grasp_active).to(torch.float32)

    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    is_close = torch.norm(pos_g - ee, dim=-1) < proximity_threshold
    return (is_lifted & is_close).to(torch.float32)


# arm_velocity_bonus imported from shared/gripper_cfg.py


def grasp_reward(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    green_name: str,
    red_name: str,
    std: float = 0.08,
) -> torch.Tensor:
    """Reward for closing the gripper near the nearest active cube.

    Returns ``closure_fraction * (1 - tanh(dist / std))`` where dist is
    measured from the dynamic fingertip to the nearest active cube.
    Using fingertip instead of grasp centre encourages top-down approach.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    distance = _nearest_active_cube_distance(env, tip, green_name, red_name)
    proximity = 1.0 - torch.tanh(distance / std)

    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)

    return closure * proximity


def release_above_target(
    env: ManagerBasedRLEnv,
    green_name: str,
    drum_name: str,
    finger_cfg: SceneEntityCfg,
    belt_height: float,
    rim_clearance: float = 0.10,
    drum_radius: float = 0.2735,
) -> torch.Tensor:
    """Reward for opening the gripper when the cube is above the drum.

    Returns ``(1 - closure) * gate`` where gate requires the cube to
    be within the drum radius in XY and above the rim height.  Gated
    on ``was_grasped`` to prevent reward from random gripper opening
    without ever having grasped.
    """
    green: RigidObject = env.scene[green_name]
    pos_g = green.data.root_pos_w
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    d_xy = torch.sqrt(
        (pos_g[:, 0] - pos_d[:, 0]) ** 2 + (pos_g[:, 1] - pos_d[:, 1]) ** 2
    )
    in_xy = d_xy < drum_radius

    local_z = pos_g[:, 2] - env.scene.env_origins[:, 2]
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
# Observation: cube velocity
# ---------------------------------------------------------------------------

def cube_velocity(
    env: ManagerBasedRLEnv,
    cube_name: str,
) -> torch.Tensor:
    """Cube linear velocity (N, 3).  Returns zeros when parked.

    Gives the policy explicit awareness of whether the cube is moving
    (bouncing/falling) versus stationary (on belt, ready for grasp).
    """
    cube: RigidObject = env.scene[cube_name]
    active = _cube_is_active(cube, env)
    vel = cube.data.root_lin_vel_w
    return torch.where(active[:, None], vel, torch.zeros_like(vel))


def ee_to_green_distance_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
) -> torch.Tensor:
    """Raw distance from grasp centre to green cube (for TensorBoard logging)."""
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    return torch.norm(env.scene[green_name].data.root_pos_w - ee, dim=-1)


# ---------------------------------------------------------------------------
# Conveyor penalty
# ---------------------------------------------------------------------------

def cube_off_conveyor_penalty(
    env: ManagerBasedRLEnv,
    green_name: str,
    red_name: str,
    bounds: ConveyorBounds,
) -> torch.Tensor:
    """Penalty (1.0) when any active cube is outside the conveyor region.

    Checks Y bounds and Z threshold.  Returns per-env float in [0, 1].
    """
    penalty = torch.zeros(env.num_envs, device=env.device)
    for name in (green_name, red_name):
        cube: RigidObject = env.scene[name]
        active = _cube_is_active(cube, env)
        local = cube.data.root_pos_w - env.scene.env_origins
        out_y = (local[:, 1] < bounds.y_min) | (local[:, 1] > bounds.y_max)
        out_z = local[:, 2] < bounds.z_min
        penalty = torch.where(active & (out_y | out_z), 1.0, penalty)
    return penalty




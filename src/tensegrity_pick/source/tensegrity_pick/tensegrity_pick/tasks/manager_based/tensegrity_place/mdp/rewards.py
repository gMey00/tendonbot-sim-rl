"""Reward, observation, termination and reset functions for the place task.

Simplified reward structure modelled after IsaacLab's manipulation/lift:
  - Reaching:      tanh proximity  (grasp centre → cube)
  - Lifting:       binary          (local z above belt + threshold)
  - Goal tracking: tanh XY dist    (gated on lift + was_grasped)
  - Regularisation on ALL joints including gripper
  - NO gripper reward — the agent discovers grasping naturally

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
# Gripper geometry constants (Robotiq 2F-140, ceiling-mounted)
# ---------------------------------------------------------------------------
# IMPORTANT – CEILING-MOUNT ORIENTATION
# The robot is mounted upside-down on the ceiling.  In the tool_link_0
# local frame the +Z axis points DOWNWARD (toward the floor / cube) in
# world coordinates.  All offsets below are therefore POSITIVE so that
# ``quat_apply(ee_quat, [0, 0, +offset])`` yields a world position
# below tool_link_0 where the physical finger pads actually are.
#
# Empirical verification (measure_positions.py, 50 zero-action steps,
# base_z_joint = −0.25):
#   tool_link_0        z ≈ 1.032 (local)
#   grasp_center (+0.1925)  z ≈ 0.840   → 1.5 cm above cube centre
#   finger_tip_open  (+0.215)   z ≈ 0.817   → 0.8 cm below cube centre
#   finger_tip_closed (+0.235)  z ≈ 0.797   → 2.8 cm below cube centre
#   green cube centre           z ≈ 0.825
#   belt surface                z  = 0.800
#
# If you ever change the robot mount or gripper, re-run
# ``scripts/measure_positions.py`` and update these constants.
# ---------------------------------------------------------------------------

FINGER_TIP_OPEN_Z = 0.215       # tip offset when finger_joint = 0 (fully open)
FINGER_TIP_CLOSED_Z = 0.235     # tip offset when finger_joint = 0.7854 (closed)
FINGER_TIP_LOCAL_Z = 0.225      # static average for belt collision checks
GRASP_CENTER_LOCAL_Z = 0.1925   # average pad centre between open/closed
FINGER_JOINT_CLOSE_POS = 0.7854 # finger_joint target when fully closed


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class BinCylinder:
    """Upright cylinder representing the target drum."""

    radius: float
    height: float


@dataclass
class SpawnBox:
    """Axis-aligned box in local env coordinates for cube spawning."""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    yaw_range: Tuple[float, float] = (-3.14159, 3.14159)


def _project_local_z_offset(
    robot: Articulation, body_cfg: SceneEntityCfg, local_z: float
) -> torch.Tensor:
    """Project a local-z offset from a body frame into world coordinates.

    Returns the world position of a point that sits at (0, 0, local_z)
    in the frame of the body identified by ``body_cfg.body_ids[0]``.

    For the ceiling-mounted tensegrity robot, positive ``local_z``
    maps to a world position *below* the body (toward the floor/cube)
    because tool_link_0's local +Z axis points downward in world frame.
    """
    ee_pos = robot.data.body_pos_w[:, body_cfg.body_ids[0], :]
    ee_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]
    offset = ee_pos.new_tensor([0.0, 0.0, local_z])
    return ee_pos + quat_apply(ee_quat, offset.expand_as(ee_pos))


def _grasp_center_w(
    robot: Articulation, body_cfg: SceneEntityCfg
) -> torch.Tensor:
    """World position of the grasp centre between the finger pads."""
    return _project_local_z_offset(robot, body_cfg, GRASP_CENTER_LOCAL_Z)


def _finger_tip_w(
    robot: Articulation, body_cfg: SceneEntityCfg
) -> torch.Tensor:
    """World position of the finger tips (lowest point of the gripper)."""
    return _project_local_z_offset(robot, body_cfg, FINGER_TIP_LOCAL_Z)


def _dynamic_finger_tip_w(
    robot: Articulation,
    body_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World position of finger tips adjusted for actual finger joint state.

    Interpolates between open (-0.215) and closed (-0.235) tip offsets
    based on the finger_joint reading so the reference point tracks the
    physical gripper configuration.
    """
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
    tip_z = FINGER_TIP_OPEN_Z + closure * (FINGER_TIP_CLOSED_Z - FINGER_TIP_OPEN_Z)

    ee_pos = robot.data.body_pos_w[:, body_cfg.body_ids[0], :]
    ee_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]
    offset = torch.stack([
        torch.zeros_like(tip_z),
        torch.zeros_like(tip_z),
        tip_z,
    ], dim=-1)
    return ee_pos + quat_apply(ee_quat, offset)


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
    """Check whether points lie inside the cylinder.  Shapes: (N, 3)."""
    rel = point - center
    r2 = rel[:, 0] ** 2 + rel[:, 1] ** 2
    inside_xy = r2 <= cyl.radius ** 2
    inside_z = (rel[:, 2] >= 0.0) & (rel[:, 2] <= cyl.height)
    return inside_xy & inside_z


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def ee_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp-centre position relative to env origin (N, 3).

    Subtracting env_origins removes per-env world offsets so that the
    observation is invariant to env placement in the simulation grid.
    """
    pos_w = _grasp_center_w(env.scene[asset_cfg.name], asset_cfg)
    return pos_w - env.scene.env_origins


def ee_lin_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """EE body linear velocity (N, 3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]


def cube_rel_pos(
    env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, cube_name: str
) -> torch.Tensor:
    """Relative position of a single cube w.r.t. the grasp centre (N, 3).

    Returns zeros when the cube is parked (local x > 50).
    """
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    cube: RigidObject = env.scene[cube_name]
    pos = cube.data.root_pos_w

    local_x = pos[:, 0] - env.scene.env_origins[:, 0]
    active = local_x < 50.0

    rel = pos - ee
    return torch.where(active[:, None], rel, torch.zeros_like(rel))


def drum_rel_pos(
    env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, drum_name: str
) -> torch.Tensor:
    """Relative position of the drum target w.r.t. the grasp centre (N, 3)."""
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    return _get_world_pos(env.scene[drum_name], env=env) - ee


def fingertip_rel_cube(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    cube_name: str,
) -> torch.Tensor:
    """Relative position of the green cube w.r.t. the dynamic fingertip (N, 3).

    The fingertip position is interpolated between open/closed based on
    the actual finger_joint state, giving the policy spatial awareness
    of where the contact surfaces are.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    cube: RigidObject = env.scene[cube_name]
    pos = cube.data.root_pos_w

    local_x = pos[:, 0] - env.scene.env_origins[:, 0]
    active = local_x < 50.0

    rel = pos - tip
    return torch.where(active[:, None], rel, torch.zeros_like(rel))


def gripper_closure(
    env: ManagerBasedRLEnv, finger_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Normalized gripper closure fraction [0=open, 1=closed] (N, 1)."""
    robot: Articulation = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
    return closure.unsqueeze(-1)


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


def joint_vel_out_of_limit(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Termination: True when any joint velocity exceeds max_velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.any(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity,
        dim=1,
    )


def joint_effort_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold_ratio: float = 5.0,
) -> torch.Tensor:
    """Terminate when any tracked joint's computed torque exceeds the limit.

    The PD controller ``computed_torque`` is what the controller *wants* to
    apply; the physics engine clamps it to ``applied_torque`` within the
    effort limit.  When the computed torque exceeds the limit by
    ``threshold_ratio``x, the mechanism is under extreme structural stress.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    computed = robot.data.computed_torque[:, asset_cfg.joint_ids]
    limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.any(torch.abs(computed) > threshold_ratio * limits, dim=1)


def belt_collision_termination(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    belt_height: float,
    max_penetration: float = 0.05,
) -> torch.Tensor:
    """Terminate when the finger tips penetrate the belt surface.

    Uses _finger_tip_w (NOT grasp centre) because we want to detect
    the very bottom of the gripper touching the belt.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip_pos = _finger_tip_w(robot, ee_cfg)
    tip_z = tip_pos[:, 2] - env.scene.env_origins[:, 2]
    return tip_z < (belt_height - max_penetration)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def object_ee_distance(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    std: float = 0.1,
) -> torch.Tensor:
    """Tanh reward for grasp-centre proximity to the green cube.

    Mirrors IsaacLab's ``object_ee_distance``: measures the 3D distance
    from the grasp centre (fixed offset from tool_link_0) to the cube
    position and returns ``1 - tanh(distance / std)``.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    cube_pos = env.scene[green_name].data.root_pos_w
    distance = torch.norm(cube_pos - ee, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    belt_height: float,
    minimal_height: float = 0.04,
    max_distance: float = 0.15,
) -> torch.Tensor:
    """Binary reward: 1.0 when the cube is above threshold AND near the gripper.

    Extends IsaacLab's ``object_is_lifted`` with a proximity gate so that
    launching the cube via a bump does not yield reward — the gripper must
    remain close to the cube while it is lifted.  This avoids a reward
    exploit that does not exist in the Franka reference (where the arm
    physically cannot launch a 50 g cube).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    cube_pos = green.data.root_pos_w

    local_z = cube_pos[:, 2] - env.scene.env_origins[:, 2]
    is_above = local_z > belt_height + minimal_height

    distance = torch.norm(cube_pos - ee, dim=-1)
    is_near = distance < max_distance

    return torch.where(is_above & is_near, 1.0, 0.0)


def cube_height_bonus(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
    belt_height: float,
    max_height: float = 0.20,
    max_distance: float = 0.15,
) -> torch.Tensor:
    """Continuous reward proportional to cube height above belt, gated on proximity.

    Provides smooth gradient for lifting higher than the binary lift
    threshold, incentivising the agent to clear the drum rim before
    approaching in XY.  Returns ``clamp(h / max_height, 0, 1)`` when
    the gripper is within ``max_distance`` of the cube, else 0.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee = _grasp_center_w(robot, ee_cfg)
    green: RigidObject = env.scene[green_name]
    cube_pos = green.data.root_pos_w

    local_z = cube_pos[:, 2] - env.scene.env_origins[:, 2]
    height_above = torch.clamp(local_z - belt_height, min=0.0, max=max_height)
    normalized = height_above / max_height

    distance = torch.norm(cube_pos - ee, dim=-1)
    is_near = distance < max_distance

    return torch.where(is_near, normalized, torch.zeros_like(normalized))


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


def base_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 norm of base joint velocities.

    Penalises large/fast movements of the linear base so the policy
    learns to use the 3-DOF arm for most of the positioning work.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    base_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(base_vel ** 2, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared difference between consecutive actions (all dims)."""
    return torch.sum(
        (env.action_manager.action - env.action_manager.prev_action) ** 2,
        dim=1,
    )


def joint_vel_l2_controlled(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 norm of controlled joint velocities, clamped per-joint."""
    asset = env.scene[asset_cfg.name]
    clamped = torch.clamp(
        asset.data.joint_vel[:, asset_cfg.joint_ids], -max_velocity, max_velocity
    )
    return torch.sum(clamped ** 2, dim=1)


def belt_contact_penalty(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    belt_height: float,
    margin: float = 0.0,
    max_depth: float = 0.15,
) -> torch.Tensor:
    """Penalty proportional to how far the finger tips go below the belt.

    Uses _finger_tip_w (NOT grasp centre) because we want to penalise
    the very bottom of the gripper touching the belt.
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip_pos = _finger_tip_w(robot, ee_cfg)
    tip_z = tip_pos[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(belt_height + margin - tip_z, min=0.0, max=max_depth)


def joint_torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean fraction of effort limit used across tracked joints.

    Returns mean(|applied_torque| / effort_limit), clipped to [0, 1].
    """
    robot: Articulation = env.scene[asset_cfg.name]
    applied = robot.data.applied_torque[:, asset_cfg.joint_ids]
    limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    fractions = torch.abs(applied) / (limits + 1e-8)
    return torch.clamp(fractions, max=1.0).mean(dim=1)


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


def arm_velocity_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_velocity: float = 5.0,
) -> torch.Tensor:
    """Positive reward for arm joint velocity magnitude.

    Incentivises active use of the 3-DOF arm joints (elbow, wrist_y,
    wrist_x) for positioning rather than relying on the linear base.
    Returns the L2 norm of arm joint velocities, normalised by
    ``max_velocity`` and clamped to [0, 1].
    """
    robot: Articulation = env.scene[asset_cfg.name]
    arm_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.clamp(torch.norm(arm_vel, dim=1) / max_velocity, max=1.0)


def ee_to_green_distance_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    green_name: str,
) -> torch.Tensor:
    """Raw distance from grasp centre to green cube (for TensorBoard logging)."""
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    return torch.norm(env.scene[green_name].data.root_pos_w - ee, dim=-1)

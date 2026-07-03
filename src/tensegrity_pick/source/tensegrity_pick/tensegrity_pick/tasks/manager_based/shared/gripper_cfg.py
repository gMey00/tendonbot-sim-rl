"""Shared gripper geometry constants and helpers.

Constants and functions shared across all tasks using the ceiling-mounted
5-DOF tensegrity robot with Robotiq 2F-140 gripper.

Offsets are stored as frame-independent DISTANCES; the local-Z direction that
points toward the finger tips depends on the EE reference frame and is looked
up per body name in ``EE_FRAME_TIP_SIGN`` (tool_link_0: −Z, robotiq_base_link
/ end_effector_link: +Z — measured by
scripts/model_validation/check_f140_finger_tip.py).  The legacy negative
``*_Z`` constants keep the tool_link_0 convention for the tensegrity-only
tasks.  See also scripts/measure_positions.py for per-config measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Gripper geometry constants
# ---------------------------------------------------------------------------

# Distances (m) from the EE reference frame origin along the gripper axis
# toward the finger tips.  Frame-independent magnitudes — the DIRECTION of
# that axis in the EE frame's local Z depends on the frame convention, see
# ``EE_FRAME_TIP_SIGN`` below.
FINGER_TIP_OPEN_DIST = 0.215
FINGER_TIP_CLOSED_DIST = 0.235
FINGER_TIP_STATIC_DIST = 0.225   # static average for belt-collision checks
GRASP_CENTER_DIST = 0.1925       # centre between finger pads
FINGER_JOINT_CLOSE_POS = 0.7854

# Sign of the EE frame's local-Z direction that points toward the finger tips.
# Measured empirically (scripts/model_validation/check_f140_finger_tip.py,
# 2026-07-04): on every arm carrying the Robotiq 2F-140, ``tool_link_0`` and
# ``robotiq_base_link`` are CO-LOCATED with antiparallel Z axes —
# ``tool_link_0`` +Z points up the arm (fingers at −Z, the original
# tensegrity calibration), ``robotiq_base_link`` +Z points into the fingers.
# Kinova's ``end_effector_link`` is co-located with robotiq_base_link
# (+0.6 mm) and shares its direction.
EE_FRAME_TIP_SIGN = {
    "tool_link_0": -1.0,        # tensegrity arms
    "robotiq_base_link": 1.0,   # UR5e/UR10/Kinova F140 (+ present on tensegrity)
    "end_effector_link": 1.0,   # Kinova flange (F140 variants resolve robotiq_base_link first)
}


def tip_frame_sign(body_name: str) -> float:
    """Local-Z sign toward the finger tips for a known EE frame.

    Raises with a pointer to the calibration probe for unknown frames instead
    of silently projecting the grasp geometry to the wrong side.
    """
    try:
        return EE_FRAME_TIP_SIGN[body_name]
    except KeyError:
        raise KeyError(
            f"No finger-tip frame convention for EE body '{body_name}' — measure it "
            "with scripts/model_validation/check_f140_finger_tip.py and add it to "
            "gripper_cfg.EE_FRAME_TIP_SIGN."
        ) from None


# Legacy tool_link_0-frame constants (negative local Z), kept for the
# tensegrity-only tasks (cube_place/cube_sort/shirt_place) that import them.
FINGER_TIP_OPEN_Z = -FINGER_TIP_OPEN_DIST
FINGER_TIP_CLOSED_Z = -FINGER_TIP_CLOSED_DIST
FINGER_TIP_LOCAL_Z = -FINGER_TIP_STATIC_DIST
GRASP_CENTER_LOCAL_Z = -GRASP_CENTER_DIST


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BinCylinder:
    """Upright cylinder representing the target drum."""

    radius: float
    height: float


@dataclass
class SpawnBox:
    """Axis-aligned box in local env coordinates for spawning."""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    yaw_range: Tuple[float, float] = (-3.14159, 3.14159)


@dataclass
class ConveyorBounds:
    """Axis-aligned 2D bounds for the conveyor surface in local env coords."""

    y_min: float
    y_max: float
    z_min: float


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def project_local_z_offset(
    robot: Articulation, body_cfg: SceneEntityCfg, local_z: float,
) -> torch.Tensor:
    """Project a local-z offset from a body into world coordinates (N, 3)."""
    ee_pos = robot.data.body_pos_w[:, body_cfg.body_ids[0], :]
    ee_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]
    offset = ee_pos.new_tensor([0.0, 0.0, local_z])
    return ee_pos + quat_apply(ee_quat, offset.expand_as(ee_pos))


def _tip_sign(robot: Articulation, body_cfg: SceneEntityCfg) -> float:
    """Finger-direction sign for the EE body referenced by ``body_cfg``."""
    return tip_frame_sign(robot.body_names[body_cfg.body_ids[0]])


def grasp_center_w(robot: Articulation, body_cfg: SceneEntityCfg) -> torch.Tensor:
    """World position of the grasp centre between the finger pads (N, 3)."""
    return project_local_z_offset(
        robot, body_cfg, _tip_sign(robot, body_cfg) * GRASP_CENTER_DIST
    )


def finger_tip_w(robot: Articulation, body_cfg: SceneEntityCfg) -> torch.Tensor:
    """World position of the finger tips (lowest point of the gripper) (N, 3)."""
    return project_local_z_offset(
        robot, body_cfg, _tip_sign(robot, body_cfg) * FINGER_TIP_STATIC_DIST
    )


def dynamic_finger_tip_w(
    robot: Articulation,
    body_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World position of finger tips adjusted for actual finger joint state (N, 3)."""
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
    tip_z = _tip_sign(robot, body_cfg) * (
        FINGER_TIP_OPEN_DIST + closure * (FINGER_TIP_CLOSED_DIST - FINGER_TIP_OPEN_DIST)
    )

    ee_pos = robot.data.body_pos_w[:, body_cfg.body_ids[0], :]
    ee_quat = robot.data.body_quat_w[:, body_cfg.body_ids[0], :]
    offset = torch.stack([
        torch.zeros_like(tip_z),
        torch.zeros_like(tip_z),
        tip_z,
    ], dim=-1)
    return ee_pos + quat_apply(ee_quat, offset)


def get_world_pos(
    obj: object, env: "ManagerBasedRLEnv | None" = None,
) -> torch.Tensor:
    """Get world position (N, 3) from RigidObject or XformPrimView."""
    if hasattr(obj, "data") and hasattr(obj.data, "root_pos_w"):
        return obj.data.root_pos_w
    pos, _ = obj.get_world_poses()
    if env is not None and pos.shape[0] == 1 and env.num_envs > 1:
        local_offset = pos[0] - env.scene.env_origins[0]
        pos = env.scene.env_origins + local_offset.unsqueeze(0)
    return pos


def in_upright_cylinder(
    point: torch.Tensor, center: torch.Tensor, cyl: BinCylinder,
) -> torch.Tensor:
    """Check whether points lie inside the cylinder.

    Supports both (N, 3) and (N, M, 3) shapes — returns (N,) or (N, M).
    """
    rel = point - center if point.ndim == center.ndim else point - center[:, None, :]
    if rel.ndim == 2:
        r2 = rel[:, 0] ** 2 + rel[:, 1] ** 2
        inside_z = (rel[:, 2] >= 0.0) & (rel[:, 2] <= cyl.height)
    else:
        r2 = rel[..., 0] ** 2 + rel[..., 1] ** 2
        inside_z = (rel[..., 2] >= 0.0) & (rel[..., 2] <= cyl.height)
    return (r2 <= cyl.radius ** 2) & inside_z


# ---------------------------------------------------------------------------
# Common observations
# ---------------------------------------------------------------------------

def ee_pos_w(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Grasp-centre position relative to env origin (N, 3)."""
    pos_w = grasp_center_w(env.scene[asset_cfg.name], asset_cfg)
    return pos_w - env.scene.env_origins


def ee_lin_vel_w(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """EE body linear velocity (N, 3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]


def gripper_closure(
    env: "ManagerBasedRLEnv", finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Normalized gripper closure fraction [0=open, 1=closed] (N, 1)."""
    robot: Articulation = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    return torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0).unsqueeze(-1)


def gripper_torque_residual(
    env: "ManagerBasedRLEnv", finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Normalized finger_joint applied torque as fraction of effort limit (N, 1)."""
    robot: Articulation = env.scene[finger_cfg.name]
    applied = robot.data.applied_torque[:, finger_cfg.joint_ids[0]]
    limit = robot.data.joint_effort_limits[:, finger_cfg.joint_ids[0]]
    fraction = torch.abs(applied) / (limit + 1e-8)
    return torch.clamp(fraction, max=1.0).unsqueeze(-1)


def drum_rel_pos(
    env: "ManagerBasedRLEnv", ee_cfg: SceneEntityCfg, drum_name: str,
) -> torch.Tensor:
    """Relative position of the drum target w.r.t. the grasp centre (N, 3)."""
    ee = grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    return get_world_pos(env.scene[drum_name], env=env) - ee


# ---------------------------------------------------------------------------
# Common rewards
# ---------------------------------------------------------------------------

def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Squared difference between consecutive actions."""
    return torch.sum(
        (env.action_manager.action - env.action_manager.prev_action) ** 2,
        dim=1,
    )


def joint_vel_l2_controlled(
    env: "ManagerBasedRLEnv", max_velocity: float, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 norm of controlled joint velocities, clamped per-joint."""
    asset = env.scene[asset_cfg.name]
    clamped = torch.clamp(
        asset.data.joint_vel[:, asset_cfg.joint_ids], -max_velocity, max_velocity,
    )
    return torch.sum(clamped ** 2, dim=1)


def base_velocity_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 norm of base joint velocities."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(robot.data.joint_vel[:, asset_cfg.joint_ids] ** 2, dim=1)


def arm_velocity_bonus(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, max_velocity: float = 5.0,
) -> torch.Tensor:
    """Positive reward for arm joint velocity magnitude, normalised to [0, 1]."""
    robot: Articulation = env.scene[asset_cfg.name]
    arm_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.clamp(torch.norm(arm_vel, dim=1) / max_velocity, max=1.0)


def belt_contact_penalty(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    belt_height: float,
    margin: float = 0.0,
    max_depth: float = 0.15,
) -> torch.Tensor:
    """Penalty proportional to how far the finger tips go below the belt."""
    robot: Articulation = env.scene[ee_cfg.name]
    tip_pos = finger_tip_w(robot, ee_cfg)
    tip_z = tip_pos[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(belt_height + margin - tip_z, min=0.0, max=max_depth)


def joint_torque_penalty(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean fraction of effort limit used across tracked joints."""
    robot: Articulation = env.scene[asset_cfg.name]
    applied = robot.data.applied_torque[:, asset_cfg.joint_ids]
    limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    fractions = torch.abs(applied) / (limits + 1e-8)
    return torch.clamp(fractions, max=1.0).mean(dim=1)


# ---------------------------------------------------------------------------
# Common terminations
# ---------------------------------------------------------------------------

def joint_vel_out_of_limit(
    env: "ManagerBasedRLEnv", max_velocity: float, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """True when any joint velocity exceeds max_velocity."""
    asset = env.scene[asset_cfg.name]
    return torch.any(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity,
        dim=1,
    )


def belt_collision_termination(
    env: "ManagerBasedRLEnv",
    ee_cfg: SceneEntityCfg,
    belt_height: float,
    max_penetration: float = 0.05,
) -> torch.Tensor:
    """Terminate when the finger tips penetrate the belt surface."""
    robot: Articulation = env.scene[ee_cfg.name]
    tip_pos = finger_tip_w(robot, ee_cfg)
    tip_z = tip_pos[:, 2] - env.scene.env_origins[:, 2]
    return tip_z < (belt_height - max_penetration)

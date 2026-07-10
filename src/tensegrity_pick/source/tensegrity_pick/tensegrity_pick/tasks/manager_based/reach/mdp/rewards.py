# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_error_magnitude,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def current_end_effector_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current end-effector pose in the robot root frame: [x, y, z, qw, qx, qy, qz]."""
    asset: RigidObject = env.scene[asset_cfg.name]

    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    root_pos_w = asset.data.root_state_w[:, :3]
    root_quat_w = asset.data.root_state_w[:, 3:7]

    curr_pos_b, curr_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, curr_pos_w, curr_quat_w)
    return torch.cat((curr_pos_b, curr_quat_b), dim=1)


def pose_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pose error in world frame as [pos_err_x, pos_err_y, pos_err_z, rot_err_x, rot_err_y, rot_err_z]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_state_w[:, :3],
        asset.data.root_state_w[:, 3:7],
        des_pos_b,
        des_quat_b,
    )

    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    pos_error, rot_error = compute_pose_error(des_pos_w, des_quat_w, curr_pos_w, curr_quat_w)
    return torch.cat((pos_error, rot_error), dim=1)


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the orientation using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(error / std)


def pose_goal_reached(
    env: ManagerBasedRLEnv,
    position_threshold: float,
    orientation_threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Binary reward for simultaneous position and orientation success."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3],
        asset.data.root_state_w[:, 3:7],
        des_pos_b,
        des_quat_b,
    )
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    position_error = torch.norm(curr_pos_w - des_pos_w, dim=1)
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    reached = (position_error < position_threshold) & (orientation_error < orientation_threshold)
    return reached.float()


def orientation_reached(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Binary reward: 1.0 when the end-effector orientation is within *threshold* radians of the target.

    Orientation-only check (no position constraint).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return (error < threshold).float()


def goal_reached(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Binary reward: 1.0 when the end-effector is within *threshold* metres of the target.

    Useful for tracking success rate in TensorBoard via the episodic reward sum.
    Position-only check (no orientation constraint) so it fires early in training.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )

    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return (distance < threshold).float()


def joint_vel_l2_clamped(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """L2 norm of joint velocities, clamped per-joint to prevent outlier divergence.

    Unlike the standard ``joint_vel_l2``, individual joint velocities are clamped
    to ``[-max_velocity, max_velocity]`` before squaring.  This bounds the per-step
    contribution so that a single unstable environment cannot dominate the batch.
    """
    asset = env.scene[asset_cfg.name]
    clamped = torch.clamp(asset.data.joint_vel[:, asset_cfg.joint_ids], -max_velocity, max_velocity)
    return torch.sum(clamped**2, dim=1)


def linkage_integrity_penalty(
    env: ManagerBasedRLEnv,
    gap_threshold: float = 0.03,
    branch_angle_tolerance: float = 0.3,
    rod_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["rod_right_link"]),
    forearm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["forearm_link"]),
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["root_link"]),
    rod_joints_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["rod_left_joint", "rod_right_joint"]
    ),
) -> torch.Tensor:
    """Per-step 1.0 while the four-bar linkage is torn open or branch-flipped.

    Use as a *reward penalty* (large negative weight), NOT as a termination:
    the reach reward is net-negative per step, so a failure termination lets
    the policy end episodes early — during the 2026-07 retrain both physical
    variants immediately learned to break the linkage as a suicide exploit
    (episode length collapsed 180 → 5 steps).  A strong per-step penalty makes
    breaking strictly unprofitable with nothing to escape to, while the cable
    stops keep the state recoverable so the episode continues meaningfully.

    Detection logic is shared with
    :func:`~..mdp.terminations.linkage_closure_broken` (closure-anchor gap +
    parallelogram branch flip); see there for the threshold calibration.
    """
    from .terminations import linkage_closure_broken

    return linkage_closure_broken(
        env,
        gap_threshold=gap_threshold,
        branch_angle_tolerance=branch_angle_tolerance,
        rod_cfg=rod_cfg,
        forearm_cfg=forearm_cfg,
        root_cfg=root_cfg,
        rod_joints_cfg=rod_joints_cfg,
    ).float()

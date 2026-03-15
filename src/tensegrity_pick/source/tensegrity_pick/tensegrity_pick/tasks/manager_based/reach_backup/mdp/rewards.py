# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the tensegrity reach environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, max_error: float = 1.0
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm, clamped to a maximum.

    Computes the position error between the desired position (from the command)
    and the current position of the asset's body (in world frame).  The error is
    clamped to ``max_error`` so that a single physics-divergence event cannot
    produce an unbounded negative reward signal.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )

    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return torch.clamp(distance, max=max_error)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    Computes the position error and maps it with a tanh kernel for dense reward shaping.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b
    )

    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, max_error: float = 1.0
) -> torch.Tensor:
    """Penalize tracking orientation error using shortest-path quaternion distance.

    Computes the orientation error between the desired orientation (from the command,
    given in body frame) and the current orientation of the asset's body.
    The error is clamped to ``max_error`` for robustness against physics instability.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    return torch.clamp(quat_error_magnitude(curr_quat_w, des_quat_w), max=max_error)


def goal_reached(
    env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Binary reward: 1.0 when the end-effector is within *threshold* metres of the target.

    Useful for tracking success rate in TensorBoard via the episodic reward sum.
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

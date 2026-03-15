# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator that samples reachable poses via forward kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FKSampledPoseCommand(CommandTerm):
    """Generate reachable end-effector pose targets by sampling random joint angles
    and computing forward kinematics through the physics engine.

    Instead of sampling Cartesian targets that may be unreachable, this command
    samples joint positions within the robot's limits, temporarily writes them
    into the simulation, reads the resulting body pose via the FK update, and
    then restores the original joint state.

    The pose command is expressed in the robot's base (root) frame, consistent
    with :class:`UniformPoseCommand`.
    """

    cfg: FKSampledPoseCommandCfg

    def __init__(self, cfg: FKSampledPoseCommandCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx: int = self.robot.find_bodies(cfg.body_name)[0][0]

        # Resolve which joints to randomise (default: all joints)
        if cfg.joint_names is not None:
            self.joint_ids: list[int] = self.robot.find_joints(cfg.joint_names)[0]
        else:
            self.joint_ids = list(range(self.robot.num_joints))

        # Buffers ── (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # Reach-time tracking (per environment)
        self._step_counter = torch.zeros(self.num_envs, device=self.device)
        self._first_reach_step = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._has_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reach_time_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reach_time_variance"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "FKSampledPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tRandomised joint count: {len(self.joint_ids)}\n"
        return msg

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command (num_envs, 7): [pos_x, pos_y, pos_z, qw, qx, qy, qz]."""
        return self.pose_command_b

    # ── Implementation ────────────────────────────────────────────────────

    def _update_metrics(self) -> None:
        from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        # Track reach-time: mark first step where position error drops below threshold
        self._step_counter += 1.0
        just_reached = (~self._has_reached) & (self.metrics["position_error"] < self.cfg.success_threshold)
        self._first_reach_step[just_reached] = self._step_counter[just_reached]
        self._has_reached |= just_reached

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command term content and compute episode-level metrics."""
        if env_ids is None:
            env_ids = slice(None)

        step_dt = self._env.step_dt
        reached_mask = self._has_reached[env_ids]
        success_rate = reached_mask.float().mean().item()

        if reached_mask.any():
            reach_times = self._first_reach_step[env_ids][reached_mask] * step_dt
            reach_time_mean = reach_times.mean().item()
            reach_time_variance = reach_times.var().item() if reach_times.numel() > 1 else 0.0
        else:
            reach_time_mean = float("nan")
            reach_time_variance = float("nan")

        # Write metrics so the base-class reset() picks them up for logging
        self.metrics["success_rate"][env_ids] = reached_mask.float()
        self.metrics["reach_time_mean"][env_ids] = reach_time_mean
        self.metrics["reach_time_variance"][env_ids] = reach_time_variance

        # Reset per-env tracking buffers
        self._step_counter[env_ids] = 0.0
        self._first_reach_step[env_ids] = float("inf")
        self._has_reached[env_ids] = False

        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        joint_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]
        lower = joint_limits[:, 0]
        upper = joint_limits[:, 1]

        random_joint_positions = lower + (upper - lower) * torch.rand(
            len(env_ids), len(self.joint_ids), device=self.device
        )

        saved_joint_positions = self.robot.data.joint_pos[env_ids, :][:, self.joint_ids].clone()

        self.robot.write_joint_position_to_sim(random_joint_positions, joint_ids=self.joint_ids, env_ids=env_ids)

        body_pose = self.robot.data.body_link_pose_w[env_ids, self.body_idx]
        ee_pos_w = body_pose[:, :3]
        ee_quat_w = body_pose[:, 3:7]

        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        pos_b, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        self.pose_command_b[env_ids, :3] = pos_b
        self.pose_command_b[env_ids, 3:] = quat_unique(quat_b) if self.cfg.make_quat_unique else quat_b

        self.robot.write_joint_position_to_sim(saved_joint_positions, joint_ids=self.joint_ids, env_ids=env_ids)

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        from isaaclab.markers import VisualizationMarkers

        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


@configclass
class FKSampledPoseCommandCfg(CommandTermCfg):
    """Configuration for FK-based reachable pose command sampling."""

    class_type: type = FKSampledPoseCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    body_name: str = MISSING
    """Name of the end-effector body whose pose is the command target."""

    joint_names: list[str] | None = None
    """Joint names to randomise for FK sampling.

    When ``None`` (the default) **all** joints of the articulation are randomised.
    """

    make_quat_unique: bool = False
    """Whether to enforce a positive real part on the quaternion. Defaults to False."""

    success_threshold: float = 0.02
    """Position error (in metres) below which the target is considered reached. Defaults to 0.02."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    """Visualisation marker for the goal pose."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """Visualisation marker for the current body pose."""

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

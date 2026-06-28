# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .joint_pos_env_cfg import UR5eF140ReachEnvCfg
from tensegrity_pick.tasks.manager_based.reach.config.ik_reach_common import (
    apply_ik_rel_action,
)


@configclass
class UR5eF140IKRelReachEnvCfg(UR5eF140ReachEnvCfg):
    """Relative Differential-IK reach variant (Cartesian pose-delta action)."""

    def __post_init__(self):
        super().__post_init__()
        # Controlled joints were stored on the command by configure_f140_reach.
        apply_ik_rel_action(self, list(self.commands.ee_pose.joint_names))


@configclass
class UR5eF140IKRelReachEnvCfg_PLAY(UR5eF140IKRelReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True

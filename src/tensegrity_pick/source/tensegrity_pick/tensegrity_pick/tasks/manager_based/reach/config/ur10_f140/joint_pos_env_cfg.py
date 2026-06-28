# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from tensegrity_pick.robots.ur10_robot_cfg import UR10_GRIPPER_CFG, CONTROLLED_JOINT_NAMES
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg
from tensegrity_pick.tasks.manager_based.reach.config.f140_reach_common import (
    configure_f140_reach,
)


@configclass
class UR10F140ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_f140_reach(
            self,
            UR10_GRIPPER_CFG,
            list(CONTROLLED_JOINT_NAMES),
            joint_range_margin=0.1,
            clamp_fallback_range=3.0,
            clamp_max_range=3.0,
        )


@configclass
class UR10F140ReachEnvCfg_PLAY(UR10F140ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True

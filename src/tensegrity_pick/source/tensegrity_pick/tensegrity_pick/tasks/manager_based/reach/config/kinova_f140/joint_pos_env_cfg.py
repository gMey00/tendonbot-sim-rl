# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from tensegrity_pick.robots.kinova_gen3_robot_cfg import KINOVA_GEN3_GRIPPER_CFG, CONTROLLED_JOINT_NAMES
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg
from tensegrity_pick.tasks.manager_based.reach.config.f140_reach_common import (
    configure_f140_reach,
)


@configclass
class KinovaF140ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_f140_reach(
            self,
            KINOVA_GEN3_GRIPPER_CFG,
            list(CONTROLLED_JOINT_NAMES),
            joint_range_margin=0.1,
            clamp_fallback_range=6.2832,
            clamp_max_range=None,
        )
        # FK-target mode only (box mode ignores joint sampling): restrict FK
        # sampling to default ± 1.5 rad.  The Kinova's raw limits are full-circle
        # (continuous joints, 4.4-5.3 rad elbow ranges), so unrestricted FK
        # sampling spans the entire envelope — untrainable (iteration 19: 24-35 cm
        # plateau).  ±1.5 matches the UR arms' reset-clamped range, which trains.
        import os as _os
        self.commands.ee_pose.fk_sampling_half_range = float(
            _os.environ.get("REACH_FK_HALF_RANGE", "1.5") or "1.5"
        )


@configclass
class KinovaF140ReachEnvCfg_PLAY(KinovaF140ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True

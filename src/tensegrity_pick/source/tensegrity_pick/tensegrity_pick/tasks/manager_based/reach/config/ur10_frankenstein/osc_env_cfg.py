# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .joint_pos_env_cfg import UR10FrankensteinReachEnvCfg
from tensegrity_pick.tasks.manager_based.reach.config.osc_reach_common import (
    apply_osc_action,
)


@configclass
class UR10FrankensteinOSCReachEnvCfg(UR10FrankensteinReachEnvCfg):
    """Operational-Space-Control reach variant (task-space pose + variable stiffness)."""

    def __post_init__(self):
        super().__post_init__()
        # Controlled joints were stored on the command by configure_f140_reach.
        apply_osc_action(self, list(self.commands.ee_pose.joint_names))


@configclass
class UR10FrankensteinOSCReachEnvCfg_PLAY(UR10FrankensteinOSCReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True

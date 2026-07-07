# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e + Robotiq 2F-140 shirt-present environment configuration.

Second robot of the cloth-sorting pipeline; the showcase mounts the UR5e
yaw-rotated by +90° so its home pose faces the belt.
"""

from isaaclab.utils import configclass

from tensegrity_pick.robots.ur5e_robot_cfg import (
    CONTROLLED_JOINT_NAMES,
    TARGET_LINK_NAME,
    UR5E_GRIPPER_CFG,
)
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (
    SECOND_ROBOT_MOUNT_POS,
    SECOND_ROBOT_MOUNT_ROT_UR,
)
from tensegrity_pick.tasks.manager_based.shirt_present import mdp
from tensegrity_pick.tasks.manager_based.shirt_present.shirt_present_env_cfg import (
    ShirtPresentEnvCfg,
)


@configclass
class UR5eF140ShirtPresentEnvCfg(ShirtPresentEnvCfg):
    """UR5e (6-DOF) shirt-present task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = UR5E_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UR5E_GRIPPER_CFG.init_state.replace(
                pos=SECOND_ROBOT_MOUNT_POS,
                rot=SECOND_ROBOT_MOUNT_ROT_UR,
            ),
        )

        # EMA joint-position-to-limits (the reach-grid joint-space winner,
        # alpha = 0.2 -- reach/config/f140_reach_common.py).  Measured
        # justification for dropping the stub's delta-from-default action
        # (scale 0.5): the scripted baseline saturated it -- reaching the
        # lowest hanging point needs joint targets up to +-1.25 rad from the
        # ready pose (job 3809927: max |a| pinned at the +-2.5 clamp = +-1.25
        # rad; 0/16 reached at +-0.5 rad), far outside the +-1 action band a
        # Gaussian policy explores well.  To-limits mapping puts every
        # reachable posture inside [-1, 1] by construction.
        self.actions.arm_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=list(CONTROLLED_JOINT_NAMES),
            alpha=0.2,
        )

        self._set_robot_params(
            ee_body=TARGET_LINK_NAME,
            controlled_joints=list(CONTROLLED_JOINT_NAMES) + ["finger_joint"],
            arm_joints=list(CONTROLLED_JOINT_NAMES),
        )


@configclass
class UR5eF140ShirtPresentEnvCfg_PLAY(UR5eF140ShirtPresentEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

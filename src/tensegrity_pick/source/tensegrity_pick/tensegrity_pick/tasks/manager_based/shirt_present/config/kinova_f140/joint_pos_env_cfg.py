# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinova Gen3 + Robotiq 2F-140 shirt-present environment configuration.

Second robot of the cloth-sorting pipeline, standing on the pedestal at the
showcase pose (identical to the F140 reach workspace-analysis mount).
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots.kinova_gen3_robot_cfg import (
    CONTROLLED_JOINT_NAMES,
    KINOVA_GEN3_GRIPPER_CFG,
    TARGET_LINK_NAME,
)
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (
    SECOND_ROBOT_MOUNT_POS,
    SECOND_ROBOT_MOUNT_ROT,
)
from tensegrity_pick.tasks.manager_based.shirt_present import mdp
from tensegrity_pick.tasks.manager_based.shirt_present.shirt_present_env_cfg import (
    ShirtPresentEnvCfg,
)


@configclass
class KinovaF140ShirtPresentEnvCfg(ShirtPresentEnvCfg):
    """Kinova Gen3 (7-DOF) shirt-present task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = KINOVA_GEN3_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=KINOVA_GEN3_GRIPPER_CFG.init_state.replace(
                pos=SECOND_ROBOT_MOUNT_POS,
                rot=SECOND_ROBOT_MOUNT_ROT,
            ),
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(CONTROLLED_JOINT_NAMES),
            scale=0.5,
            use_default_offset=True,
        )

        self._set_robot_params(
            ee_body=TARGET_LINK_NAME,
            controlled_joints=list(CONTROLLED_JOINT_NAMES) + ["finger_joint"],
            arm_joints=list(CONTROLLED_JOINT_NAMES),
        )


@configclass
class KinovaF140ShirtPresentEnvCfg_PLAY(KinovaF140ShirtPresentEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

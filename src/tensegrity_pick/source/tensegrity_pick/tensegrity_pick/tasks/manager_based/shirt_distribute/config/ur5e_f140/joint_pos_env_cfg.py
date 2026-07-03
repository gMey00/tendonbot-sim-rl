# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e + Robotiq 2F-140 shirt-distribute environment configuration."""

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
from tensegrity_pick.tasks.manager_based.shirt_distribute import mdp
from tensegrity_pick.tasks.manager_based.shirt_distribute.shirt_distribute_env_cfg import (
    ShirtDistributeEnvCfg,
)


@configclass
class UR5eF140ShirtDistributeEnvCfg(ShirtDistributeEnvCfg):
    """UR5e (6-DOF) shirt-distribute task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = UR5E_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UR5E_GRIPPER_CFG.init_state.replace(
                pos=SECOND_ROBOT_MOUNT_POS,
                rot=SECOND_ROBOT_MOUNT_ROT_UR,
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
class UR5eF140ShirtDistributeEnvCfg_PLAY(UR5eF140ShirtDistributeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

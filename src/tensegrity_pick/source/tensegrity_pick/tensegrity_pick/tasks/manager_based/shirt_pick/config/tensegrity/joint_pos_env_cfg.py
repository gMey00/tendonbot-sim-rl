# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensegrity PD-driven shirt-pick environment configuration.

The retrieving robot of the cloth-sorting pipeline — hanging mount above the
belt, same pose and action wiring as the validated shirt_place tensegrity
variant (``shirt_place/config/tensegrity/joint_pos_env_cfg.py``).
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (
    RETRIEVE_MOUNT_HEIGHT_M,
    RETRIEVE_MOUNT_XY,
)
from tensegrity_pick.tasks.manager_based.shirt_pick import mdp
from tensegrity_pick.tasks.manager_based.shirt_pick.shirt_pick_env_cfg import ShirtPickEnvCfg

# ── Robot-specific constants ──────────────────────────────────────────────

EE_LINK = "tool_link_0"
BASE_JOINT_NAMES = ["base_y_joint", "base_z_joint"]
ARM_JOINT_NAMES = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]
CONTROLLED_JOINT_NAMES = BASE_JOINT_NAMES + ARM_JOINT_NAMES + ["finger_joint"]

_INITIAL_JOINT_POS = {
    "base_y_joint": 0.0,
    "base_z_joint": 0.0,
    "elbow_joint": 0.0,
    "wrist_y_joint": 0.0,
    "wrist_x_joint": 0.0,
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_outer_finger_joint": 0.0,
    "right_outer_finger_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "left_inner_finger_pad_joint": 0.0,
    "right_inner_finger_pad_joint": 0.0,
}


@configclass
class TensegrityShirtPickEnvCfg(ShirtPickEnvCfg):
    """Tensegrity 5-DOF PD shirt-pick task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = TENS_5DOF_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(RETRIEVE_MOUNT_XY[0], RETRIEVE_MOUNT_XY[1], RETRIEVE_MOUNT_HEIGHT_M),
                joint_pos=_INITIAL_JOINT_POS,
            ),
        )

        # Same action wiring as the validated shirt_place tensegrity variant.
        self.actions.base_delta = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=BASE_JOINT_NAMES,
            scale=0.50,
            use_default_offset=True,
            clip={"base_y_joint": (-0.5, 0.5), "base_z_joint": (-0.50, 0.0)},
        )
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            scale=1.0,
            use_default_offset=True,
            clip={
                "elbow_joint": (-1.5, 1.5),
                "wrist_y_joint": (-0.8, 0.8),
                "wrist_x_joint": (-0.8, 0.8),
            },
        )

        self._set_robot_params(
            ee_body=EE_LINK,
            controlled_joints=CONTROLLED_JOINT_NAMES,
            arm_joints=BASE_JOINT_NAMES + ARM_JOINT_NAMES,
        )


@configclass
class TensegrityShirtPickEnvCfg_PLAY(TensegrityShirtPickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

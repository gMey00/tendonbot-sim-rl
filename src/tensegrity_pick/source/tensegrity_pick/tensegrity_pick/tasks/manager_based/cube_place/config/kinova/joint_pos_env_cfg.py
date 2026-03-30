# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinova Gen3 cube place environment configuration.

Inherits from the robot-agnostic ``PlaceEnvCfg`` and fills in:
  - robot asset (Kinova Gen3 7-DOF + Robotiq 2F-140, ceiling-mounted)
  - arm action (7-DOF JointPositionActionCfg)
  - all MISSING body/joint names

Key differences from the tensegrity variant:
  - 7-DOF revolute arm (no prismatic base) → single arm_action
  - No base_velocity penalty (fixed-base robot)
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots import KINOVA_GEN3_GRIPPER_CFG
from tensegrity_pick.tasks.manager_based.cube_place import mdp
from tensegrity_pick.tasks.manager_based.cube_place.place_env_cfg import PlaceEnvCfg
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import KINOVA_MOUNT_HEIGHT_M

# ── Robot-specific constants ──────────────────────────────────────────────

EE_LINK = "end_effector_link"

KINOVA_ARM_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

CONTROLLED_JOINT_NAMES = KINOVA_ARM_JOINT_NAMES + ["finger_joint"]

_KINOVA_PLACE_INITIAL_JOINT_POS = {
    "joint_1": 0.0,
    "joint_2": 0.2618,
    "joint_3": 3.1416,
    "joint_4": -2.2689,
    "joint_5": 0.0,
    "joint_6": 0.9599,
    "joint_7": 3.1416,
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
class KinovaPlaceEnvCfg(PlaceEnvCfg):
    """Kinova Gen3 cube-placement task — fair comparison baseline."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # ── Robot ─────────────────────────────────────────────────────
        self.scene.robot = KINOVA_GEN3_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, KINOVA_MOUNT_HEIGHT_M),
                rot=(0.0, 0.0, 1.0, 0.0),
                joint_pos=_KINOVA_PLACE_INITIAL_JOINT_POS,
            ),
        )

        # ── Actions ───────────────────────────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=KINOVA_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )

        # ── Fill all MISSING body/joint names ─────────────────────────
        self._set_robot_params(
            ee_body=EE_LINK,
            controlled_joints=CONTROLLED_JOINT_NAMES,
            arm_joints=KINOVA_ARM_JOINT_NAMES,
        )


@configclass
class KinovaPlaceEnvCfg_PLAY(KinovaPlaceEnvCfg):
    """Smaller configuration for evaluation / play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.curriculum.activate_red.params["num_steps"] = 0

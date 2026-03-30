# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e cube place environment configuration.

Inherits from the robot-agnostic ``PlaceEnvCfg`` and fills in:
  - robot asset (UR10e + Robotiq 2F-140, ceiling-mounted)
  - arm action (6-DOF JointPositionActionCfg)
  - all MISSING body/joint names

Key differences from the tensegrity variant:
  - 6-DOF revolute arm (no prismatic base) → single arm_action
  - No base_velocity penalty (fixed-base robot)
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots import UR10E_GRIPPER_CFG
from tensegrity_pick.tasks.manager_based.cube_place import mdp
from tensegrity_pick.tasks.manager_based.cube_place.place_env_cfg import PlaceEnvCfg
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import UR10E_MOUNT_HEIGHT_M

# ── Robot-specific constants ──────────────────────────────────────────────

EE_LINK = "robotiq_base_link"

UR10E_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

CONTROLLED_JOINT_NAMES = UR10E_ARM_JOINT_NAMES + ["finger_joint"]

_UR10E_PLACE_INITIAL_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": 0.0,
    "elbow_joint": 0.0,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
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
class UR10ePlaceEnvCfg(PlaceEnvCfg):
    """UR10e cube-placement task — fair comparison baseline."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # ── Robot ─────────────────────────────────────────────────────
        self.scene.robot = UR10E_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, UR10E_MOUNT_HEIGHT_M),
                rot=(0.0, 0.0, 1.0, 0.0),
                joint_pos=_UR10E_PLACE_INITIAL_JOINT_POS,
            ),
        )

        # ── Actions ───────────────────────────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=UR10E_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )

        # ── Fill all MISSING body/joint names ─────────────────────────
        self._set_robot_params(
            ee_body=EE_LINK,
            controlled_joints=CONTROLLED_JOINT_NAMES,
            arm_joints=UR10E_ARM_JOINT_NAMES,
        )


@configclass
class UR10ePlaceEnvCfg_PLAY(UR10ePlaceEnvCfg):
    """Smaller configuration for evaluation / play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.curriculum.activate_red.params["num_steps"] = 0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensegrity PD-driven cube-placement environment configurations.

Inherits from the robot-agnostic ``PlaceEnvCfg`` and fills in:
  - robot asset (5-DOF tensegrity + Robotiq 2F-140)
  - arm action (base_delta + arm_delta JointPositionActionCfg)
  - all MISSING body/joint names
  - base_velocity penalty (tensegrity-only, prismatic base)
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG
from tensegrity_pick.tasks.manager_based.cube_place import mdp
from tensegrity_pick.tasks.manager_based.cube_place.mdp import rewards as task_rew
from tensegrity_pick.tasks.manager_based.cube_place.place_env_cfg import PlaceEnvCfg, PlaceEnvCfg_PLAY
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import TENSEGRITY_MOUNT_HEIGHT_M

# ── Robot-specific constants ──────────────────────────────────────────────

EE_LINK = "tool_link_0"
GRASP_BODIES = [EE_LINK]

BASE_JOINT_NAMES = ["base_y_joint", "base_z_joint"]
ARM_JOINT_NAMES = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]
CONTROLLED_JOINT_NAMES = BASE_JOINT_NAMES + ARM_JOINT_NAMES + ["finger_joint"]

_PLACE_INITIAL_JOINT_POS = {
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
class TensegrityPlaceEnvCfg(PlaceEnvCfg):
    """Tensegrity 5-DOF PD cube-placement task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # ── Robot ─────────────────────────────────────────────────────
        self.scene.robot = TENS_5DOF_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, TENSEGRITY_MOUNT_HEIGHT_M),
                joint_pos=_PLACE_INITIAL_JOINT_POS,
            ),
        )

        # ── Actions ───────────────────────────────────────────────────
        # Tensegrity has prismatic base + revolute arm + gripper.
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

        # ── Fill all MISSING body/joint names ─────────────────────────
        self._set_robot_params(
            ee_body=EE_LINK,
            controlled_joints=CONTROLLED_JOINT_NAMES,
            arm_joints=BASE_JOINT_NAMES + ARM_JOINT_NAMES,
        )

        # ── Tensegrity-specific: base velocity penalty ────────────────
        self.rewards.base_velocity = RewTerm(
            func=task_rew.base_velocity_l2,
            weight=-1.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=BASE_JOINT_NAMES),
            },
        )


@configclass
class TensegrityPlaceEnvCfg_PLAY(TensegrityPlaceEnvCfg):
    """Smaller configuration for evaluation / play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.curriculum.activate_red.params["num_steps"] = 0

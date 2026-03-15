# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG
from tensegrity_pick.robots.tensegrity_robot_cfg import TARGET_LINK_NAME_5DOF as TARGET_LINK_NAME
from tensegrity_pick.robots.tensegrity_robot_cfg import CONTROLLED_JOINT_NAMES_5DOF as CONTROLLED_JOINT_NAMES
from tensegrity_pick.tasks.manager_based.reach import mdp
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg


@configclass
class TensegrityReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # switch robot asset
        self.scene.robot = TENS_5DOF_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override commands
        self.commands.ee_pose.body_name = TARGET_LINK_NAME
        self.commands.ee_pose.joint_names = CONTROLLED_JOINT_NAMES
        # re-declare actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=CONTROLLED_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_orientation_tracking_fine_grained.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.pose_goal_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        # override events
        self.events.reset_robot_joints.params["asset_cfg"].joint_names = CONTROLLED_JOINT_NAMES




@configclass
class TensegrityReachEnvCfg_PLAY(TensegrityReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.assets.articulation import ArticulationCfg
from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG
from tensegrity_pick.robots.tensegrity_robot_cfg import TARGET_LINK_NAME_5DOF as TARGET_LINK_NAME
from tensegrity_pick.robots.tensegrity_robot_cfg import CONTROLLED_JOINT_NAMES_5DOF as CONTROLLED_JOINT_NAMES
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import TENSEGRITY_MOUNT_HEIGHT_M
from tensegrity_pick.tasks.manager_based.reach import mdp
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg


@configclass
class TensegrityReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Add robot asset
        self.scene.robot = TENS_5DOF_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, TENSEGRITY_MOUNT_HEIGHT_M)  # 0.15m from conveyor center
            )
        )
        # Stabilize ALL gripper joints for reach task (no grasping needed).
        # Default gripper has near-zero damping causing velocity divergence that
        # corrupts the PhysX solver and produces NaN across all joints.
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=100.0,
            armature=10.0,
        )
        self.scene.robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_joint", "right_inner_finger_joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=100.0,
            armature=10.0,
        )
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "right_outer_knuckle_joint",
            ],
            effort_limit_sim=1000.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=100.0,
            armature=10.0,
        )
        # override commands
        self.commands.ee_pose.body_name = TARGET_LINK_NAME
        self.commands.ee_pose.joint_names = CONTROLLED_JOINT_NAMES
        self.commands.ee_pose.joint_range_margin = 0.01
        # Use JointPositionToLimitsAction: maps policy output [-1,1] -> joint limits
        # This prevents PD-driven velocity divergence from out-of-limit targets
        self.actions.arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=CONTROLLED_JOINT_NAMES,
        )
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.position_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.orientation_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.pose_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        # Restrict joint velocity penalty to controlled arm joints (exclude passive gripper joints)
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
        # Filter observations to controlled joints only (passive gripper joints explode)
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)}
        # Filter termination velocity check to controlled joints only
        self.terminations.joint_vel_diverged.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
        # override events
        self.events.reset_robot_joints.params["asset_cfg"].joint_names = CONTROLLED_JOINT_NAMES




@configclass
class TensegrityReachEnvCfg_PLAY(TensegrityReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False

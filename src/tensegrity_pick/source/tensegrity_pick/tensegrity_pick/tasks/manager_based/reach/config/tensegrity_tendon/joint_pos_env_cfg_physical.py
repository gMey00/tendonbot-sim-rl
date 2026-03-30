# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach task environment configs for the *physical* tensegrity tendon model.

The physical model replaces the single ``elbow_joint`` with a four-bar
antiparallelogram linkage driven by body-force tendons.  The elbow action is
applied as cable forces at the physical attachment points rather than via the
constant Jacobian-transpose mapping used by the elbow_approx variant.

NOTE: The 5-DOF physical USD must be assembled in Robot Assembler before
these configs can be instantiated.
"""

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from tensegrity_pick.robots import PhysicalTendonEffortActionCfg
from tensegrity_pick.robots.tendon_robot_cfg import TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG
from tensegrity_pick.robots.tendon_robot_cfg import TARGET_LINK_NAME_5DOF_PHYSICAL as TARGET_LINK_NAME
from tensegrity_pick.robots.tendon_robot_cfg import CONTROLLED_JOINT_NAMES_5DOF_PHYSICAL as CONTROLLED_JOINT_NAMES
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import TENSEGRITY_MOUNT_HEIGHT_M
from tensegrity_pick.tasks.manager_based.reach import mdp
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg


@configclass
class PhysicalTendonReachActionsCfg:
    base_action = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
    )

    arm_tendon = PhysicalTendonEffortActionCfg(
        asset_name="robot",
        joint_names=["wrist_y_joint", "wrist_x_joint"],
    )


@configclass
class TensegrityReachPhysicalTendonEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Switch robot to the physical linkage 5-DOF arm
        self.scene.robot = TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, TENSEGRITY_MOUNT_HEIGHT_M)
            ),
        )
        # Stabilize ALL gripper joints for reach task (no grasping needed)
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
        self.commands.ee_pose.joint_range_margin = 0.25
        # re-declare actions for the physical tendon arm
        self.actions = PhysicalTendonReachActionsCfg()
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.position_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.orientation_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.pose_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        # Restrict joint velocity penalty to controlled joints
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
        # Filter observations to controlled joints only
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)}
        # Filter termination velocity check to controlled joints only
        self.terminations.joint_vel_diverged.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
        # override events
        self.events.reset_robot_joints.params["asset_cfg"].joint_names = CONTROLLED_JOINT_NAMES
        self.events.reset_robot_joints.params["position_range"] = (-0.25, 0.25)


@configclass
class TensegrityReachPhysicalTendonEnvCfg_PLAY(TensegrityReachPhysicalTendonEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False

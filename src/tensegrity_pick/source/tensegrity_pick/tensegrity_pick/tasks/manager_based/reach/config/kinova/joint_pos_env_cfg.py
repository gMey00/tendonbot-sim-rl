# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.actuators import ImplicitActuatorCfg

from tensegrity_pick.robots import KINOVA_GEN3_GRIPPER_CFG
from tensegrity_pick.robots.kinova_gen3_robot_cfg import TARGET_LINK_NAME, CONTROLLED_JOINT_NAMES
from tensegrity_pick.tasks.manager_based.reach import mdp
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import KINOVA_MOUNT_HEIGHT_M


@configclass
class KinovaReachEnvCfg(ReachEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        
        # switch robot asset
        self.scene.robot = KINOVA_GEN3_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Table mount: right-side up at optimal workspace height
        self.scene.robot.init_state.pos = (0.15, 0.0, KINOVA_MOUNT_HEIGHT_M)
        # Disable self-collisions for reach task (not needed, reduces solver work).
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        # Disable gravity for reach task: simplifies dynamics so agent
        # focuses purely on kinematics (matching UR10e/tensegrity configs).
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        # override commands
        self.commands.ee_pose.body_name = TARGET_LINK_NAME
        self.commands.ee_pose.joint_names = CONTROLLED_JOINT_NAMES
        # Narrow FK target sampling: Kinova's continuous joints (1,3,5,7) are
        # clamped to ±2π, creating a huge workspace. margin=0.40 restricts
        # targets to the inner 20 % of each joint range so targets stay
        # reachable from the starting configuration.
        self.commands.ee_pose.joint_range_margin = 0.40
        # Use EMA-smoothed JointPositionToLimits: maps policy output [-1,1] to
        # joint limits with exponential moving average (alpha=0.2) to prevent
        # rapid target oscillation that the PD controller can't track.
        # RelativeJointPositionAction caused chaotic oscillation; raw
        # JointPositionToLimitsAction was too coarse for PPO.  EMA smoothing
        # provides gradual target changes for fine-grained control.
        self.actions.arm_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=CONTROLLED_JOINT_NAMES,
            alpha=0.2,
        )
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        # Widen tanh reward std: default 0.1 provides zero gradient beyond
        # ~30 cm, but Kinova's large workspace means average error starts at
        # ~38 cm.  std=0.5 provides useful gradient up to ~1.5 m.
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.5
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.position_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.orientation_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.pose_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        # Restrict joint velocity penalty to controlled arm joints (exclude passive gripper joints)
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
        # Filter observations to controlled joints only (passive gripper joints can explode)
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)}
        # Filter termination velocity check to controlled joints only
        self.terminations.joint_vel_diverged.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
        # Switch from scale-based to offset-based reset: scale-based reset
        # multiplies defaults, so joints with default=0.0 (joint_1, joint_5)
        # never vary.  Offset-based adds a small random offset to every joint,
        # giving consistent starting diversity.
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.125, 0.125),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
            },
        )
        # Clamp infinite joint limits on continuous joints (1, 3, 5, 7) to
        # ±π from their default positions.  Without this,
        # EMAJointPositionToLimitsAction maps actions to [-inf, inf] → NaN.
        self.events.clamp_joint_limits = EventTerm(
            func=mdp.clamp_infinite_joint_limits,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "fallback_range": 6.2832},
        )
        # Stabilize gripper joints for reach task (no grasping needed).
        # Default passive joints (K=0, D=0) are freewheeling; adding stiffness
        # holds them at rest and prevents velocity drift.
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




@configclass
class KinovaReachEnvCfg_PLAY(KinovaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False

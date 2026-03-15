# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from tensegrity_pick.robots import TendonEffortActionCfg
from tensegrity_pick.robots.tendon_actuator import DEFAULT_JACOBIAN_TRANSPOSE
from tensegrity_pick.robots.tendon_robot_cfg import TENS_5DOF_GRIPPER_TENDON_CFG
from tensegrity_pick.robots.tendon_robot_cfg import TARGET_LINK_NAME_5DOF as TARGET_LINK_NAME
from tensegrity_pick.robots.tendon_robot_cfg import CONTROLLED_JOINT_NAMES_5DOF as CONTROLLED_JOINT_NAMES
from tensegrity_pick.tasks.manager_based.reach import mdp
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg
# from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import ROBOT_MOUNT_HEIGHT_M


# @configclass
# class TensegrityReachTendonSceneCfg(ReachSceneCfg):
#     robot = TENS_5DOF_GRIPPER_TENDON_CFG.replace(
#         prim_path="{ENV_REGEX_NS}/Robot",
#         init_state=ArticulationCfg.InitialStateCfg(
#             pos=(0.15, 0.0, ROBOT_MOUNT_HEIGHT_M),
#         ),
#     )


@configclass
class TendonReachActionsCfg:
    base_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
        scale=0.5,
        use_default_offset=True,
    )

    arm_tendon = TendonEffortActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        num_tendons=5,
        max_tension=500.0,
        jacobian_transpose=DEFAULT_JACOBIAN_TRANSPOSE,
    )


# @configclass
# class TendonReachObservationsCfg:
#     @configclass
#     class PolicyCfg(ObsGroup):
#         joint_pos = ObsTerm(
#             func=mdp.joint_pos_rel,
#             noise=Unoise(n_min=-0.002, n_max=0.002),
#             params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
#         )
#         joint_vel = ObsTerm(
#             func=mdp.joint_vel_rel,
#             noise=Unoise(n_min=-0.002, n_max=0.002),
#             params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
#         )
#         current_ee_pose = ObsTerm(
#             func=mdp.current_end_effector_pose,
#             params={"asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME])},
#         )
#         pose_error = ObsTerm(
#             func=mdp.pose_command_error,
#             params={"asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME]), "command_name": "ee_pose"},
#         )
#         pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
#         actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self) -> None:
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     policy: PolicyCfg = PolicyCfg()


@configclass
class TensegrityReachTendonEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # switch robot asset to the tendon-driven version of the 5-DOF arm
        self.scene.robot = TENS_5DOF_GRIPPER_TENDON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override commands
        self.commands.ee_pose.body_name = TARGET_LINK_NAME
        self.commands.ee_pose.joint_names = CONTROLLED_JOINT_NAMES
        # re-declare actions for the tendon-driven arm
        self.actions = TendonReachActionsCfg()
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_position_tracking_proximity.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.goal_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        # override events
        self.events.reset_robot_joints.params["asset_cfg"].joint_names = CONTROLLED_JOINT_NAMES



@configclass
class TensegrityReachTendonEnvCfg_PLAY(TensegrityReachTendonEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False

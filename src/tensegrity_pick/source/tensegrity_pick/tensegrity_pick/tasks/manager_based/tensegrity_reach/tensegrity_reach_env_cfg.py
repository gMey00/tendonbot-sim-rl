# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the tensegrity reach task.

The robot must reach a random end-effector position and orientation.
Target poses are sampled via forward kinematics so they are always reachable.
Preserves the original robot and base scene from ProjBaseSceneCfg.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

# Reuse the base scene (ground, lights, robot, conveyor, drum)
from ..tensegrity_pick.proj_base_scene_cfg import ProjBaseSceneCfg


##
# MDP settings
##

TARGET_LINK_NAME = "tool_link_0"  # end-effector link to track for the reach task
CONTROLLED_JOINT_NAMES = ["base_y_joint", "base_z_joint", "elbow_joint", "wrist_y_joint", "wrist_x_joint"]


@configclass
class ActionsCfg:
    """Joint position actions for the 5-DOF arm (no gripper needed for reach)."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=CONTROLLED_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    """FK-sampled pose command for the end-effector (always reachable)."""

    ee_pose = mdp.FKSampledPoseCommandCfg(
        asset_name="robot",
        body_name=TARGET_LINK_NAME,
        joint_names=CONTROLLED_JOINT_NAMES,
        resampling_time_range=(4.0, 4.0),
        success_threshold=0.05,
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the reach MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset events for the reach task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.75, 1.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the reach MDP.

    Weights are aligned with the Isaac Lab reference reach environment.
    """

    # -- task terms --
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME]), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_position_tracking_proximity = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME]), "std": 0.03, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME]), "command_name": "ee_pose"},
    )
    goal_reached = RewTerm(
        func=mdp.goal_reached,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME]),
            "threshold": 0.05,
            "command_name": "ee_pose",
        },
    )

    # -- regularisation terms --
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_clamped,
        weight=-0.0005,
        params={"max_velocity": 10.0, "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the reach MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_vel_diverged = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        time_out=True,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms that ramp up regularisation over training."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.01, "num_steps": 12000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.005, "num_steps": 12000}
    )


##
# Environment configuration
##


@configclass
class TensegrityReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the tensegrity reach end-effector pose tracking environment."""

    scene: ProjBaseSceneCfg = ProjBaseSceneCfg(num_envs=2000, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
        # PhysX stabilisation for cleaner velocity signals
        self.sim.physx.enable_stabilization = True


@configclass
class TensegrityReachEnvCfg_PLAY(TensegrityReachEnvCfg):
    """Smaller configuration for evaluation / play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False

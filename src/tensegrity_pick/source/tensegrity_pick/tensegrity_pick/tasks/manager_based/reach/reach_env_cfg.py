# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.devices import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ..shared.proj_base_scene_cfg import ProjBaseSceneCfg

from . import mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(ProjBaseSceneCfg):
    """Configuration for the reach scene.

    All non-robot objects (conveyors, drums) have collisions disabled so the
    robot can freely explore its full workspace without physical interference.
    The scene props remain visible as a reference for downstream tasks.
    """

    def __post_init__(self):
        super().__post_init__()
        # Make all scene props visual-only (penetrable)
        self.conveyor.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        self.conveyor_upstream.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        self.drum_target.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.FKSampledPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        joint_names=MISSING,
        # Resample only at episode reset: the cadence is set far larger than
        # episode_length_s (6.0s) so the mid-episode timer never fires.
        # One target per episode -> success_rate / reach_time mean exactly
        # "reached the target" and "time to that target".
        resampling_time_range=(1.0e9, 1.0e9),
        success_threshold=0.05,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            # params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            # params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        # current_ee_pose = ObsTerm(
        #     func=mdp.current_end_effector_pose,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME])},
        # )
        # pose_error = ObsTerm(
        #     func=mdp.pose_command_error,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[TARGET_LINK_NAME]), "command_name": "ee_pose"},
        # )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task terms --
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # -- success metrics (tiny weight → logged to TensorBoard, negligible effect on reward) --
    position_reached = RewTerm(
        func=mdp.goal_reached,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose", "threshold": 0.05},
    )
    orientation_reached = RewTerm(
        func=mdp.orientation_reached,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose", "threshold": 0.3},
    )
    pose_reached = RewTerm(
        func=mdp.pose_goal_reached,
        weight=1e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
            "position_threshold": 0.05,
            "orientation_threshold": 0.3,
        },
    )

    # -- regularisation terms --
    # Initial weights match Isaac Lab reference (-0.0001); curriculum ramps them up.
    # Ref: IsaacLab source/isaaclab_tasks/.../manipulation/reach/reach_env_cfg.py
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_vel_diverged = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        time_out=True,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms that ramp up regularisation over training."""

    # Final weights and ramp duration match Isaac Lab reference.
    # Ref: IsaacLab source/isaaclab_tasks/.../manipulation/reach/reach_env_cfg.py
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500},
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        # 6s gives a single FK-sampled target ample settling time while pinning
        # one target per episode (see ee_pose.resampling_time_range above).
        self.episode_length_s = 6.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        # self.scene.clone_in_fabric = True
        # self.sim.create_stage_in_memory = True
        self.sim.physx.enable_stabilization = True

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "gamepad": Se3GamepadCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    gripper_term=False,
                    sim_device=self.sim.device,
                ),
            },
        )

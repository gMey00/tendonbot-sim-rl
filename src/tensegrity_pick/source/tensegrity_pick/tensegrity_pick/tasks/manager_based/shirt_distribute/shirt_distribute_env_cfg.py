# shirt_distribute_env_cfg.py
#
# Robot-agnostic configuration for the shirt distribute task (pipeline task 3).
# Robot variants (the SECOND robot: UR/Kinova ± tensegrity wrist) inherit and
# call ``_set_robot_params()``.
#
# Goal-conditioned formulation (research report §4): the commanded bin's
# position enters the observation (``target_bin_rel``); the per-episode target
# is resampled by ``ShirtDistributeEnv`` so nearest ≠ correct.
#
# STUB status: the shirt starts on the belt edge (not yet grasped from the
# Task-2 terminal bank); the full reward structure (sparse landing bonus +
# shirt_place's anti-hover release design) is pipeline work in doc/TODO.md.

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..shared.cloth_object import apply_cloth_startup_event, disable_complex_colliders_event
from ..shared.cloth_sorting_scene_cfg import (
    CLOTH_SORTING_SHIRT_CFG,
    ClothSortingSceneCfg,
    configure_cloth_sim,
)
from . import mdp


##
# Scene
##


@configclass
class ShirtDistributeSceneCfg(ClothSortingSceneCfg):
    """Shirt-distribute scene: second robot + the three sorting drums."""

    holder_robot = None


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Robot-agnostic actions: variants fill ``arm_action``."""

    arm_action: ActionTerm = MISSING

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
        )
        # EE kinematics
        ee_pos_w = ObsTerm(
            func=mdp.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        # Shirt state
        shirt_rel = ObsTerm(
            func=mdp.shirt_rel_pos,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        shirt_vel = ObsTerm(func=mdp.shirt_velocity)
        # Goal: commanded bin relative to the shirt (label → bin position)
        target_bin_rel = ObsTerm(func=mdp.target_bin_rel_shirt)
        # Gripper + grasp state
        gripper_closure = ObsTerm(
            func=mdp.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        grasp_active = ObsTerm(func=mdp.grasp_active_obs)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Startup + reset events (cloth machinery validated in shirt_place)."""

    disable_conveyor_colliders = EventTerm(
        func=disable_complex_colliders_event,
        mode="prestartup",
        params={"asset_keys": ("Conveyor", "ConveyorUpstream")},
    )
    apply_cloth = EventTerm(
        func=apply_cloth_startup_event,
        mode="prestartup",
        params={"cloth_cfg": CLOTH_SORTING_SHIRT_CFG},
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "position_range": (-0.10, 0.10),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_gripper = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Stub rewards — full Task-3 structure is pipeline work (doc/TODO.md):
    sparse landing-in-correct-bin bonus + shirt_place's graded release event.
    dt-scaling lesson applies (one-shots need ~60× per-step weights)."""

    # Approach the shirt (grasp machinery handles the pick)
    reaching_shirt = RewTerm(
        func=mdp.ee_to_shirt_tanh,
        weight=2.0,
        params={"ee_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.3},
    )
    # Carry the shirt toward the commanded bin (goal-conditioned shaping)
    to_target_bin = RewTerm(
        func=mdp.shirt_to_target_bin_tanh,
        weight=4.0,
        params={"std": 0.4},
    )

    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_vel_diverged = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        time_out=True,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )


##
# Environment configuration
##


@configclass
class ShirtDistributeEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-agnostic base configuration for the shirt distribute task."""

    scene: ShirtDistributeSceneCfg = ShirtDistributeSceneCfg(num_envs=512, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self) -> None:
        self.episode_length_s = 6.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        configure_cloth_sim(self)

    def _set_robot_params(
        self,
        ee_body: str,
        controlled_joints: list[str],
        arm_joints: list[str],
    ) -> None:
        """Fill all ``MISSING`` body/joint names for a specific robot."""
        grasp = [ee_body]

        obs = self.observations.policy
        obs.joint_pos_rel.params["asset_cfg"].joint_names = controlled_joints
        obs.joint_vel_rel.params["asset_cfg"].joint_names = controlled_joints
        obs.ee_pos_w.params["asset_cfg"].body_names = grasp
        obs.shirt_rel.params["ee_cfg"].body_names = grasp

        self.rewards.reaching_shirt.params["ee_cfg"].body_names = grasp
        self.rewards.joint_vel.params["asset_cfg"].joint_names = controlled_joints

        self.events.reset_arm.params["asset_cfg"].joint_names = arm_joints
        self.terminations.joint_vel_diverged.params["asset_cfg"].joint_names = controlled_joints


@configclass
class ShirtDistributeEnvCfg_PLAY(ShirtDistributeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

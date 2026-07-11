# shirt_pick_env_cfg.py
#
# Robot-agnostic configuration for the shirt pick task (pipeline task 1).
# Robot variants inherit and call ``_set_robot_params()`` to fill the
# ``MISSING`` body/joint names — same pattern as reach / shirt_place.
#
# STUB status: observations / rewards / terminations are a minimal but
# runnable skeleton.  The full reward structure (stable-hold bonus, drop
# penalty, anti-hover lessons from shirt_place) is pipeline work tracked in
# doc/TODO.md.

from __future__ import annotations

from dataclasses import MISSING

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

from ..shared.cloth_object import apply_cloth_startup_event, disable_complex_colliders_event
from ..shared.cloth_sorting_scene_cfg import (
    CLOTH_SORTING_SHIRT_CFG,
    CONVEYOR_SURFACE_HEIGHT_M,
    ClothSortingSceneCfg,
    PRESENTATION_POS,
    configure_cloth_sim,
)
from ..shared.gripper_cfg import belt_collision_termination, belt_contact_penalty
from . import mdp
from .mdp import rewards as task_rew


##
# Scene
##


@configclass
class ShirtPickSceneCfg(ClothSortingSceneCfg):
    """Shirt-pick scene: retrieving robot only (no partner arm needed)."""

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

    # S2 grasp-point head (None = stage-1 behaviour; the head task variant
    # fills it via ``mdp.grasp_head.apply_grasp_head`` — see grasp_head.py).
    grasp_offset: ActionTerm | None = None


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
        # Shirt state (centroid proxy — camera-realistic terms are TODO)
        shirt_rel = ObsTerm(
            func=mdp.shirt_rel_pos,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        # Grasp target (highest point) relative to the dynamic finger tip —
        # the exact geometry the deterministic attach trigger uses.
        grasp_target_rel = ObsTerm(func=task_rew.grasp_target_rel_tip)
        shirt_vel = ObsTerm(func=mdp.shirt_velocity)
        # Presentation target relative to the shirt
        present_rel = ObsTerm(
            func=mdp.point_rel_shirt,
            params={"point": PRESENTATION_POS},
        )
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
    # The cloth reset is done by ``ClothSortingEnvBase._reset_cloth`` (owned by
    # the env, not an EventTerm — see shirt_place for the rationale).


@configclass
class RewardsCfg:
    """Sequential pick-and-present rewards (shirt_place design, adapted).

    dt-scaling (shirt_place lesson): Isaac Lab multiplies rewards by dt
    (1/60 s) — per-step weight w earns ≈ w × episode-seconds; a one-shot
    earns w/60, so the drop penalty is sized ~60× the per-step terms.

    No anti-hover fade is needed here: holding at the presentation pose IS
    the task, so the per-step ``presented`` bonus is the success reward.
    The fling hack is closed by grasp-gating the transport shaping.
    """

    # 1. Reach: tip → highest point (the deterministic attach geometry)
    reaching = RewTerm(func=task_rew.reaching_grasp_target, weight=2.0, params={"std": 0.25})
    # 2. Grasp: per-step while attached
    grasp_hold = RewTerm(func=task_rew.grasp_hold, weight=5.0)
    # 3. Lift: clamped height progress above the belt (grasp-gated)
    lift = RewTerm(func=task_rew.lift_progress, weight=8.0, params={"full_height": 0.30})
    # 4. Transport: grasp point → presentation pose (grasp-gated)
    to_presentation = RewTerm(func=task_rew.to_presentation, weight=15.0, params={"std": 0.35})
    # 5. Success: held AT the pose, slow — dominant per-step term
    presented = RewTerm(
        func=task_rew.presented,
        weight=30.0,
        params={"dist_threshold": 0.15, "vel_threshold": 0.20},
    )
    # 6. Failure: one-shot when an established grasp is lost (dt-scaled ≈ −2)
    drop = RewTerm(func=task_rew.drop_event, weight=-120.0)

    # Safety: finger tips pressing into the belt (positive depth → negative)
    belt_contact = RewTerm(
        func=belt_contact_penalty,
        weight=-10.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
        },
    )

    # Regularisation (curriculum ramps these up, shirt_place profile)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )


@configclass
class CurriculumCfg:
    """Ramp regularisation once the task is learned (shirt_place profile:
    ``num_steps`` counts trainer timesteps = env.step calls, NOT per-env
    samples)."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -3e-3, "num_steps": 2000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-3, "num_steps": 2000},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_vel_diverged = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        time_out=True,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )
    # Real failure: finger tips punched through the belt (0.12 m validated in
    # shirt_place — the fingers bend on contact, so small penetrations of the
    # virtual tip are normal).
    belt_collision = DoneTerm(
        func=belt_collision_termination,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "max_penetration": 0.12,
        },
    )


##
# Environment configuration
##


@configclass
class ShirtPickEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-agnostic base configuration for the shirt pick task."""

    scene: ShirtPickSceneCfg = ShirtPickSceneCfg(num_envs=512, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 7 s: the scripted baseline needs ~7.5 s for align→grasp→lift→carry
        # on crumpled piles; RL overlaps the phases but needs hold time on top
        # (present latch = 0.5 s stable hold).
        self.episode_length_s = 7.0
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

        self.rewards.belt_contact.params["ee_cfg"].body_names = grasp
        self.rewards.joint_vel.params["asset_cfg"].joint_names = controlled_joints

        self.events.reset_arm.params["asset_cfg"].joint_names = arm_joints
        self.terminations.joint_vel_diverged.params["asset_cfg"].joint_names = controlled_joints
        self.terminations.belt_collision.params["ee_cfg"].body_names = grasp


@configclass
class ShirtPickEnvCfg_PLAY(ShirtPickEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

# shirt_distribute_env_cfg.py
#
# Robot-agnostic configuration for the shirt distribute task (pipeline task 3).
# Robot variants (the SECOND robot: UR/Kinova ± tensegrity wrist) inherit and
# call ``_set_robot_params()``.
#
# Goal-conditioned formulation (research report §4): the commanded bin's
# position enters the observation (``target_bin_rel`` / ``target_bin_rel_ee``);
# the per-episode target is resampled by ``ShirtDistributeEnv`` so
# nearest ≠ correct.
#
# The episode starts with the shirt already hanging from the robot's own
# closed gripper at a sampled end-of-Task-2 holding pose (see
# ``shirt_distribute_env.py``); the reward structure is the shirt_place
# release-into-drum design retargeted to the commanded bin (graded one-shot
# release event, anti-hover clearance fade, release-required success,
# post-drop return-to-neutral — all dt-scaled: one-shots ~60× per-step terms).
#
# ACTION SPACE NOTE (measured need, 2026-07-04): the stub's
# ``JointPositionActionCfg(use_default_offset=True)`` commands
# ``default + scale·action`` — from the sampled far-from-default holding
# poses a ZERO action would yank the arm back to its default pose at full PD
# speed on the first step, whipping the held cloth.  The task therefore uses
# ``RelativeJointPositionActionCfg`` (target = current + scale·action): zero
# action = hold the sampled pose, which is the correct null behaviour for a
# task that starts mid-pipeline.  Scale 0.05 rad/step @ 60 Hz caps joint
# speed at ≈3 rad/s — near the UR5e wrist velocity limits.

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
    ClothSortingSceneCfg,
    configure_cloth_sim,
)
from ..shared.proj_base_scene_cfg import CONVEYOR_SURFACE_HEIGHT_M, DRUM_HEIGHT_M
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
    """Robot-agnostic actions: variants fill ``arm_action`` (relative joint
    positions — see the action-space note in the module docstring)."""

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
        # Lowest hanging point rel EE — the drape length the clearance
        # rewards key on (camera-trivial from depth, shirt_present pattern).
        shirt_lowest_rel = ObsTerm(
            func=mdp.shirt_lowest_point_rel_ee,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        # Goal: commanded bin relative to the shirt AND to the EE (the carry
        # is EE-centric; the shirt moves independently after release).
        target_bin_rel = ObsTerm(func=mdp.target_bin_rel_shirt)
        target_bin_rel_ee = ObsTerm(
            func=mdp.target_bin_rel_ee,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        # Gripper + grasp + phase state
        gripper_closure = ObsTerm(
            func=mdp.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        grasp_active = ObsTerm(func=mdp.grasp_active_obs)
        was_distributed = ObsTerm(func=mdp.was_distributed_obs)
        actions = ObsTerm(func=mdp.last_action)
        # Goal task-id one-hot (mode-collapse fix B3). MUST stay the LAST term:
        # the per-goal critic and PerGoalPPO recover the commanded bin from
        # ``obs[..., -3:].argmax(-1)`` (see mdp.rewards.target_bin_onehot).
        target_bin_onehot = ObsTerm(func=mdp.target_bin_onehot, params={"num_bins": 3})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Startup + reset events (cloth machinery validated in shirt_place).

    ``reset_arm`` writes default+noise but is immediately overridden by the
    env's holding-pose reset (``_reset_cloth`` runs after the event manager);
    kept as a harmless fallback so the manager pipeline stays standard.
    """

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
            "position_range": (0.0, 0.0),
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
    # LAST event term: writes the sampled holding pose + closed gripper so
    # (a) it overrides the default-pose events above and (b) it runs BEFORE
    # action_manager.reset() — absolute/EMA action terms snapshot the correct
    # pose into their buffers (enabler for EMAJointPositionToLimitsAction).
    reset_holding_pose = EventTerm(func=mdp.reset_holding_pose, mode="reset")


@configclass
class RewardsCfg:
    """shirt_place release design, goal-conditioned (dt-scaled: per-step
    weight w pays ≈ w × episode-seconds; one-shots pay w/60)."""

    # ── 1. Carry the held shirt toward the commanded bin ────────────
    approach_bin = RewTerm(
        func=mdp.approach_target_bin,
        weight=15.0,
        params={"std": 1.0},
    )
    approach_bin_fine = RewTerm(
        func=mdp.approach_target_bin,
        weight=6.0,
        params={"std": 0.20},
    )

    # ── 2. Whole shirt cleared above the commanded drum ─────────────
    clearance_over_bin = RewTerm(
        func=mdp.clearance_over_target_bin,
        weight=4.0,
        params={
            "drum_top": DRUM_HEIGHT_M,
            "rim_clearance": 0.04,
            "clear_margin": 0.05,
            "drum_radius": 0.32,
        },
    )

    # ── 3. Release: per-step openness hint + graded one-shot ────────
    release_hint = RewTerm(
        func=mdp.release_openness_over_target,
        weight=5.0,
        params={
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "drum_top": DRUM_HEIGHT_M,
            "rim_clearance": 0.02,
            "drum_radius": 0.2735,
        },
    )
    # One-shot: weight 240 ≈ 4 total at quality 1 (dt-scaling: /60), graded
    # ×(0.25..1) by centering × whole-shirt-lift (shirt_place design).
    release_event = RewTerm(func=mdp.release_event_bonus, weight=240.0)
    # One-shot: opening NOT over the commanded bin (incl. the instant drop at
    # episode start).  DISCOVERY CURRICULUM (iteration 4): starts mild (−5)
    # and ramps to −60 at 4000 steps — at a constant −60/−120 with σ = e⁻¹,
    # 2 of 4 seeds collapsed (release_rate < 0.01 for thousands of steps:
    # early penalty hits push the gripper-action mean negative and releases
    # stop being SAMPLED, extinguishing the exploration the penalty is meant
    # to shape).
    bad_release = RewTerm(func=mdp.bad_release_penalty, weight=-5.0)

    # ── 4. Success: released cloth in the commanded drum ────────────
    in_target_bin = RewTerm(func=mdp.fraction_in_target_bin, weight=120.0)
    # Wrong-bin landings are the sorting error the pipeline exists to avoid:
    # clearly worse than landing nowhere.
    in_wrong_bin = RewTerm(func=mdp.fraction_in_wrong_bin, weight=-60.0)
    dropped_on_floor = RewTerm(
        func=mdp.shirt_dropped_on_floor,
        weight=-5.0,
        params={"z_threshold": 0.70, "drum_radius": 0.40},
    )

    # ── 5. Anti-hover time cost + post-drop settling ────────────────
    carry_time = RewTerm(func=mdp.carry_time_penalty, weight=-1.5)
    # Calm-and-high instead of return-to-default (the UR5e default pose is
    # LOW from the pedestal mount — the neutral pull dove the arm into the
    # belt_collision cliff after every success; see settle_after_success).
    settle_after_success = RewTerm(
        func=mdp.settle_after_success,
        weight=200.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "vel_std": 1.0,
            "safe_tip_z": 0.95,
        },
    )

    # ── Regularisation (curriculum ramps, see CurriculumCfg) ────────
    # action_l2 (MAGNITUDE, not rate): with RELATIVE joint actions a constant
    # saturated action is a constant-velocity command with ZERO action-rate
    # penalty — run 1's deterministic policy parked at ±0.84 mean |action|
    # (joints pinned at limits) and landed 1 % while the stochastic policy
    # scored via noise.  Penalizing magnitude pulls the mean toward "hold
    # still" unless motion pays.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.1)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-3e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_controlled,
        weight=-3e-4,
        params={"max_velocity": 10.0, "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )
    # margin 0.05: the penalty gradient starts at tip z = 0.85, well before
    # the belt_collision termination cliff at 0.68 (iteration 3 — the noise
    # random-walk was absorbing episodes at the cliff with no prior warning).
    belt_contact = RewTerm(
        func=mdp.belt_contact_penalty,
        weight=-10.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "margin": 0.05,
            "max_depth": 0.15,
        },
    )


@configclass
class CurriculumCfg:
    """Ramp the smoothness penalties once behaviour has formed (shirt_place
    setting: num_steps counts env.step() calls == trainer timesteps)."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -3e-3, "num_steps": 2000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-3, "num_steps": 2000},
    )
    # Ramp the magnitude penalty after behaviour discovery (−0.4 ≈ −9/episode
    # at run-1's saturated |a| ≈ 0.7 — decisive against limit-parking, small
    # against a deliberate ~1 s carry).
    action_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_l2", "weight": -0.4, "num_steps": 4000},
    )
    # Full bad-release penalty only after the release behaviour has formed.
    bad_release = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "bad_release", "weight": -60.0, "num_steps": 4000},
    )


@configclass
class TerminationsCfg:
    """Episode runs its full duration after a success — an early
    settled-termination BACKFIRED in shirt_place (cutting the episode cut the
    total return below hover-to-timeout and PPO drifted back to hovering);
    instead ``in_target_bin`` + ``return_to_neutral`` keep paying post-drop,
    so dropping strictly dominates holding."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    joint_vel_diverged = DoneTerm(
        func=mdp.joint_vel_out_of_limit,
        time_out=True,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )
    # time_out=False (was True in shirt_place): a bootstrapped truncation
    # makes wandering into the cliff nearly FREE for PPO (the value estimate
    # substitutes the lost return) — with relative-action noise random-walking
    # the arm, the cliff absorbed up to 48 % of train3 episodes.  As a true
    # termination the lost return teaches avoidance.
    belt_collision = DoneTerm(
        func=mdp.belt_collision_termination,
        time_out=False,
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
class ShirtDistributeEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-agnostic base configuration for the shirt distribute task."""

    scene: ShirtDistributeSceneCfg = ShirtDistributeSceneCfg(num_envs=512, env_spacing=5.0)
    # Cached holding-pose bank (robot-specific; None = per-process sweep).
    # See ShirtDistributeEnv.pose_bank_path for why caching matters.
    pose_bank_path: str | None = None
    # B5 difficulty-proportional goal sampling (mode-collapse robustness lever).
    # OFF by default → uniform goal sampling (deployment distribution); a variant
    # cfg turns it on for TRAINING only, and evaluate_shirt_distribute.py forces
    # it off so per-bin rates are measured on the uniform distribution.
    adaptive_goal_sampling: bool = False
    goal_sampling_ema_alpha: float = 0.05
    goal_sampling_floor: float = 0.3
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 8 s: scripted-baseline carry times to the far drum ≈ 2–3 s + drop
        # settle; headroom for the post-drop neutral return.
        self.episode_length_s = 8.0
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
        obs.shirt_lowest_rel.params["ee_cfg"].body_names = grasp
        obs.target_bin_rel_ee.params["ee_cfg"].body_names = grasp

        self.rewards.joint_vel.params["asset_cfg"].joint_names = controlled_joints
        self.rewards.settle_after_success.params["asset_cfg"].joint_names = arm_joints
        self.rewards.belt_contact.params["ee_cfg"].body_names = grasp

        self.events.reset_arm.params["asset_cfg"].joint_names = arm_joints
        self.terminations.joint_vel_diverged.params["asset_cfg"].joint_names = controlled_joints
        self.terminations.belt_collision.params["ee_cfg"].body_names = grasp


@configclass
class ShirtDistributeEnvCfg_PLAY(ShirtDistributeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

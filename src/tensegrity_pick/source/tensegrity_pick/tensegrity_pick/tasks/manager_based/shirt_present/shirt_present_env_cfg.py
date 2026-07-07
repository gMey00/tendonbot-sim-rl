# shirt_present_env_cfg.py
#
# Robot-agnostic configuration for the shirt present task (pipeline task 2).
# Robot variants (the SECOND robot: UR5e / Kinova + F140) inherit and call
# ``_set_robot_params()``.
#
# MDP: naive two-grasp presentation heuristic — regrasp the hanging shirt at
# its LOWEST point (deterministic attachment grasp, slot 0) and stretch it
# between the two grasp points until the camera-plane silhouette coverage,
# tautness and stillness gates hold (see shirt_present_env.py for the success
# predicate and mdp/rewards.py for the reward rationale).

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
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

from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG

from ..shared.cloth_object import apply_cloth_startup_event, disable_complex_colliders_event
from ..shared.cloth_sorting_scene_cfg import (
    CLOTH_SORTING_SHIRT_CFG,
    ClothSortingSceneCfg,
    configure_cloth_sim,
)
from . import mdp
from .mdp import rewards as task_rew


##
# Scene
##


# Local presentation anchor (env-local) — mirrors ShirtPresentEnv.PRESENT_ANCHOR_LOCAL.
# Kept here (not imported) so the scene cfg has no import cycle with the env.
_PRESENT_ANCHOR = (0.50, 0.85, 1.20)
# Holder-arm rest EE offset below its mount (MEASURED, baseline job 3820849:
# the tensegrity 5-DOF arm's tool_link_0 sits 0.98 m below the mount at the
# straight-down joint pose below).  Mount so the gripper sits at the anchor.
_HOLDER_REST_DROP = 0.98


@configclass
class ShirtPresentSceneCfg(ClothSortingSceneCfg):
    """Shirt-present scene: second robot learns; retriever spawns passively.

    Finding #3 fix (visual): the passive holder is posed to GRIP the anchor
    patch instead of hanging in its far rest pose.  It is mounted directly
    above the (local) presentation anchor with the arm pointing straight down
    so its gripper sits at the grasp point.  The actual "hold" is still the
    static solver anchor (see ShirtPresentEnv); this only makes the retriever
    visually plausible.  NOTE: exact fingertip alignment is cosmetic and should
    be GUI-confirmed on a workstation (Alex cannot render); ``_HOLDER_REST_DROP``
    is the measured rest drop and can be nudged there.
    """

    holder_robot: ArticulationCfg = TENS_5DOF_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/HolderRobot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(_PRESENT_ANCHOR[0], _PRESENT_ANCHOR[1],
                 _PRESENT_ANCHOR[2] + _HOLDER_REST_DROP),
            # Straight-down arm pose so tool_link_0 reaches the anchor below.
            joint_pos={
                "base_y_joint": 0.0,
                "base_z_joint": 0.0,
                "elbow_joint": 0.0,
                "wrist_y_joint": 0.0,
                "wrist_x_joint": 0.0,
            },
        ),
    )


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
        # Hanging shirt state (centroid + regrasp target).  The targeted hem
        # corner AND the horizontal-pull goal are given relative to the DYNAMIC
        # finger tip — the exact geometry the deterministic attach trigger uses
        # (camera-trivial from depth) plus where to pull the grasped corner.
        shirt_rel = ObsTerm(
            func=mdp.shirt_rel_pos,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        hand_target_rel = ObsTerm(func=task_rew.hand_target_rel_tip)
        pull_target_rel = ObsTerm(func=task_rew.pull_target_rel_tip)
        shirt_vel = ObsTerm(func=mdp.shirt_velocity)
        # Task state: both grasps, tautness, camera-plane coverage — all
        # camera-derivable in principle (grasp points visible, garment flat
        # geometry known, coverage = segmentation-mask area ratio).
        grasp_active = ObsTerm(func=mdp.grasp_active_obs)
        holder_attached = ObsTerm(func=task_rew.holder_attached_obs)
        stretch_ratio = ObsTerm(func=task_rew.stretch_ratio_obs)
        coverage = ObsTerm(func=task_rew.coverage_obs)
        # Gripper state
        gripper_closure = ObsTerm(
            func=mdp.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
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
    # The cloth reset is done by ``ShirtPresentEnv._reset_cloth`` (owned by
    # the env, not an EventTerm — see shirt_place for the rationale).


@configclass
class RewardsCfg:
    """Sequential regrasp-and-stretch rewards (shirt_pick design, adapted).

    dt-scaling (shirt_place lesson): Isaac Lab multiplies rewards by dt
    (1/60 s) — per-step weight w earns ≈ w × episode-seconds; a one-shot
    earns w/60, so the drop penalty is sized ~60× the per-step terms.

    No anti-hover fade is needed: holding the stretched presentation IS the
    task, so the per-step ``presented`` bonus is the success reward.  The
    fling/bunching hacks are closed by gating stretch+coverage on BOTH
    attachments and by the silhouette (not bbox) coverage metric.
    """

    # 1. Reach: tip → targeted HEM CORNER, paid only with an OPEN gripper
    # pre-grasp (fixes the premature-close hack — see mdp/rewards.py).
    reaching = RewTerm(func=task_rew.reaching_target, weight=2.0, params={"std": 0.25})
    # 2. Grasp: per-step while the hand attachment holds
    grasp_hold = RewTerm(func=task_rew.grasp_hold, weight=5.0)
    # 3. Pull: DIRECT the second grasp to the horizontal-pull target (holder
    # height, offset along camera-plane x) — the study's taut horizontal chord.
    pull = RewTerm(func=task_rew.pulling_horizontal, weight=10.0, params={"std": 0.20})
    # 4. Stretch: clamped tautness progress (gated on both attachments)
    stretch = RewTerm(func=task_rew.stretch_progress, weight=4.0, params={"lo": 0.80, "hi": 1.02})
    # 5. Coverage: camera-plane silhouette coverage (gated on both attachments).
    # The hem<->hem geometry lifts scripted median coverage to 0.82, so this
    # term now has real headroom above the 0.65 gate.
    coverage = RewTerm(func=task_rew.coverage_reward, weight=14.0)
    # 6. Success: full presentation predicate — dominant per-step term
    presented = RewTerm(func=task_rew.presented, weight=30.0)
    # 7. Safety: tautness beyond the validated band (per-step, proportional).
    # Onset 1.10 (band upper 1.15): the study's ≤1.10 sweet spot / ≤1.15 hard
    # limit — penalise before the untested-stability region.
    overstretch = RewTerm(func=task_rew.overstretch_penalty, weight=-40.0, params={"limit": 1.10})
    # 8. Failure: one-shot when an established hand grasp is lost (dt-scaled
    # ≈ −4); sized ~60× per-step terms.
    drop = RewTerm(func=task_rew.drop_event, weight=-240.0)
    # 9. Anti-hack: penalise commanding the gripper closed while far from the
    # target and ungrasped (the premature-close behaviour, finding #1).
    early_close = RewTerm(func=task_rew.early_close_penalty, weight=-15.0, params={"clear_dist": 0.12})
    # 10. Cosmetic: mild penalty for the arm occluding the −Y camera view of the
    # cloth (finding #5).  Small — it fights the fixed base geometry and the
    # coverage metric cannot see occlusion.
    occlusion = RewTerm(func=task_rew.occlusion_penalty, weight=-2.0)

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


##
# Environment configuration
##


@configclass
class ShirtPresentEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-agnostic base configuration for the shirt present task."""

    scene: ShirtPresentSceneCfg = ShirtPresentSceneCfg(num_envs=512, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 8 s: reach-down (~1.5 s) + grasp + stretch (~1-2 s) + the 1 s
        # windowed present latch, with slack for the swinging hang; the
        # scripted baseline measures the actual phase timings.
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

        self.rewards.joint_vel.params["asset_cfg"].joint_names = controlled_joints

        self.events.reset_arm.params["asset_cfg"].joint_names = arm_joints
        self.terminations.joint_vel_diverged.params["asset_cfg"].joint_names = controlled_joints


@configclass
class ShirtPresentEnvCfg_PLAY(ShirtPresentEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

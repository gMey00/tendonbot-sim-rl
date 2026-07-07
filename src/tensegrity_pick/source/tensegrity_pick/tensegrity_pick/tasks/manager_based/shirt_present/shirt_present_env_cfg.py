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
    RETRIEVE_MOUNT_HEIGHT_M,
    RETRIEVE_MOUNT_XY,
    configure_cloth_sim,
)
from . import mdp
from .mdp import rewards as task_rew


##
# Scene
##


@configclass
class ShirtPresentSceneCfg(ClothSortingSceneCfg):
    """Shirt-present scene: second robot learns; retriever spawns passively.

    The passive holder is visual scenery — the actual "hold" is a static
    solver anchor at the presentation pose (see ShirtPresentEnv).
    TODO(pipeline): pose the holder arm at its Task-1 terminal configuration
    from the state bank and anchor the cloth to its fingertip.
    """

    holder_robot: ArticulationCfg = TENS_5DOF_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/HolderRobot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(RETRIEVE_MOUNT_XY[0], RETRIEVE_MOUNT_XY[1], RETRIEVE_MOUNT_HEIGHT_M),
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
        # Hanging shirt state (centroid + regrasp target).  The lowest point
        # is given relative to the DYNAMIC finger tip — the exact geometry the
        # deterministic attach trigger uses (camera-trivial from depth).
        shirt_rel = ObsTerm(
            func=mdp.shirt_rel_pos,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        lowest_point_rel = ObsTerm(func=task_rew.lowest_point_rel_tip)
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

    # 1. Reach: tip → lowest hanging point (the deterministic attach geometry)
    reaching = RewTerm(func=task_rew.reaching_lowest_point, weight=2.0, params={"std": 0.25})
    # 2. Grasp: per-step while the hand attachment holds
    grasp_hold = RewTerm(func=task_rew.grasp_hold, weight=5.0)
    # 3. Stretch: clamped tautness progress (gated on both attachments)
    stretch = RewTerm(func=task_rew.stretch_progress, weight=8.0, params={"lo": 0.50, "hi": 0.98})
    # 4. Coverage: camera-plane silhouette coverage (gated on both attachments)
    # 10 → 14 after run 3/4 diags: coverage was the weakest presented gate
    # (0.47–0.55 in-gate fraction while holding; some episodes hover just
    # under the 0.50 threshold).
    coverage = RewTerm(func=task_rew.coverage_reward, weight=14.0)
    # 5. Success: full presentation predicate — dominant per-step term
    presented = RewTerm(func=task_rew.presented, weight=30.0)
    # 6. Safety: tautness beyond the validated band (per-step, proportional).
    # Penalty onset 1.10 → 1.05: the reward plateau 0.97–1.10 had no gradient,
    # and diag'd failures held at ratio ~1.13 just past the predicate band
    # edge — starting the penalty at 1.05 creates a moat under it.
    overstretch = RewTerm(func=task_rew.overstretch_penalty, weight=-40.0, params={"limit": 1.05})
    # 7. Failure: one-shot when an established hand grasp is lost (dt-scaled
    # ≈ −4).  −120 → −240 after run-2 evals: grasp_rate 1.00 but drop_rate
    # 0.135–0.229 capped deterministic present_rate at 0.65–0.71 (the policy
    # kept flirting with the gripper-open threshold mid-hold).
    drop = RewTerm(func=task_rew.drop_event, weight=-240.0)

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

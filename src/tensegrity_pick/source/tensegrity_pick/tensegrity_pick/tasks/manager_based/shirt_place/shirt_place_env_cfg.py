"""Robot-agnostic base environment configuration for the shirt-placement task.

Sequential pick-and-place reward structure:
  1. Reach → 2. Grasp → 3. Lift → 4. Transport → 5. Release → 6. Success

Uses a rigid-body proxy (``shirt_proxy``) as a temporary stand-in for
the cloth centroid.  Once ``ClothObject`` tensor access is wired up,
observation/reward functions will read from actual cloth state tensors.

Differences from cube_place:
  - Single object (shirt_proxy) instead of green + red cubes
  - No red cube curriculum or clearance rewards
  - Cloth-specific observations (keypoints, spread) will be added once
    ClothObject is implemented

Robot-specific parameters (EE body, joint names, arm action) are filled
in by each variant config via ``ShirtPlaceEnvCfg._set_robot_params()`` in
``__post_init__``, following the same pattern as cube_place.
"""

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

from . import mdp
from .shirt_place_scene_cfg import ShirtPlaceSceneCfg, CONVEYOR_SURFACE_HEIGHT_M, SHIRT_CLOTH_CFG
from .mdp import rewards as task_rew
from ..shared.cloth_object import apply_cloth_startup_event, disable_complex_colliders_event

# ── Scene-level constants ─────────────────────────────────────────────

_SPAWN_BOX = task_rew.SpawnBox(
    x_range=(0.10, 0.20),
    y_range=(-0.10, 0.10),
    z_range=(CONVEYOR_SURFACE_HEIGHT_M + 0.03, CONVEYOR_SURFACE_HEIGHT_M + 0.05),
)

# Success cylinder spans the drum *interior depth* (rim ≈ 0.88 m), not just the
# bottom 0.30 m used for the rigid cube: a released cloth drapes throughout the
# drum and rarely reaches the very bottom, so a shallow cylinder reports 0 even
# when the shirt is clearly inside.
_BIN_GEOM = task_rew.BinCylinder(radius=0.547 * 0.5, height=0.85)

_CONVEYOR_BOUNDS = task_rew.ConveyorBounds(y_min=-0.4, y_max=0.4, z_min=0.70)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Robot-agnostic action specification.

    ``arm_action`` is ``MISSING`` — each variant fills it with the
    appropriate action term.
    """

    arm_action: ActionTerm = MISSING

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


@configclass
class ObservationsCfg:
    """Observation specification.

    ``body_names`` and ``joint_names`` that vary per robot are set to
    ``MISSING`` and filled by ``ShirtPlaceEnvCfg._set_robot_params()``.
    """

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
            func=task_rew.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        ee_vel_w = ObsTerm(
            func=task_rew.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )

        # Shirt-relative positions
        shirt_rel = ObsTerm(
            func=task_rew.shirt_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "shirt_name": "shirt_proxy",
            },
        )

        # Dynamic fingertip → shirt
        fingertip_shirt_rel = ObsTerm(
            func=task_rew.fingertip_rel_shirt,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "shirt_name": "shirt_proxy",
            },
        )

        # Gripper state
        gripper_closure = ObsTerm(
            func=task_rew.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        gripper_torque = ObsTerm(
            func=task_rew.gripper_torque_residual,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )

        # Shirt velocity
        shirt_vel = ObsTerm(
            func=task_rew.shirt_velocity,
            params={"shirt_name": "shirt_proxy"},
        )

        # Drum-relative position
        drum_rel = ObsTerm(
            func=task_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "drum_name": "drum_target",
            },
        )

        # Last actions
        actions = ObsTerm(func=mdp.last_action)

        # Task completion flag — lets the policy switch to return-to-neutral.
        was_placed = ObsTerm(func=task_rew.was_placed_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Startup + reset events."""

    # ── Prestartup: replace the conveyor's complex colliders with a box ─
    # PBD particle cloth tunnels through triangle-mesh / convex-hull colliders.
    # Disable them so the cloth rests on the simple ``conveyor_collider`` box.
    disable_conveyor_colliders = EventTerm(
        func=disable_complex_colliders_event,
        mode="prestartup",
        params={"asset_keys": ("Conveyor", "ConveyorUpstream")},
    )

    # ── Prestartup: spawn cloth mesh and apply PBD physics ────────────
    # Must be "prestartup" so it runs BEFORE sim.reset() — PhysX needs
    # cloth schemas in the USD before GPU physics initialisation.
    apply_cloth = EventTerm(
        func=apply_cloth_startup_event,
        mode="prestartup",
        params={"cloth_cfg": SHIRT_CLOTH_CFG},
    )

    # ── Reset events ────────────────────────────────────────────────
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

    # The shirt itself (PBD cloth) is laid flat at a randomized pose by
    # ``TensegrityShirtPlaceEnv._reset_cloth`` after ``super()._reset_idx`` — it
    # cannot be a reset EventTerm because the cloth view is owned by the env and
    # the kinematic proxy is synced to the cloth centroid afterwards.


@configclass
class RewardsCfg:
    """Reward terms for the shirt pick-and-place task.

    Sequential structure: reach → grasp → lift → transport → release → success.

    Weight hierarchy:
      Success (100) > Goal tracking (40) > Release (25) >
      Goal fine (10) > Reach fine (5) = Lift (5) = Height (5) >
      Grasp (3) > Reach coarse (2) > Arm util (0.5)
    """

    # ── 1a. Reach coarse (std=2.0) ──────────────────────────────────
    reaching_shirt = RewTerm(
        func=task_rew.shirt_ee_distance,
        weight=2.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "shirt_name": "shirt_proxy",
            "std": 2.0,
        },
    )

    # ── 1b. Reach fine (std=0.5) ────────────────────────────────────
    reaching_shirt_fine = RewTerm(
        func=task_rew.shirt_ee_distance,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "shirt_name": "shirt_proxy",
            "std": 0.5,
        },
    )

    # ── 2. Grasp: closure × proximity ───────────────────────────────
    grasping = RewTerm(
        func=task_rew.shirt_grasp_reward,
        weight=3.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "shirt_name": "shirt_proxy",
            "std": 0.08,
        },
    )

    # ── 3. Lift: binary ─────────────────────────────────────────────
    lifting_shirt = RewTerm(
        func=task_rew.shirt_is_lifted,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "shirt_name": "shirt_proxy",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "minimal_height": 0.06,
            "max_distance": 0.15,
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "max_velocity": 1.0,
        },
    )

    # ── 3b. Height bonus ────────────────────────────────────────────
    # max_height raised so lifting high enough to suspend the *whole* shirt above
    # the drum rim (grasp point ~1.3 m) is rewarded, not capped at 1.10 m.
    height_bonus = RewTerm(
        func=task_rew.shirt_height_bonus,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "shirt_name": "shirt_proxy",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            # 0.65 (was 0.50): headroom so the arm keeps a climb gradient up to a
            # grasp point of ~1.45 m — enough that the shirt hanging ~0.4 m below
            # the tip fully clears the 0.88 m drum rim before the drop.
            "max_height": 0.65,
            "max_distance": 0.15,
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "max_velocity": 1.0,
        },
    )

    # ── 4. Goal tracking coarse (std=1.0) ───────────────────────────
    # Weights cut hard (was 40/10): these are per-step *state* rewards paid while
    # the shirt simply hovers over the drum.  At 40+10 they exceeded the drop
    # reward (100×fraction at fraction≈0.05–0.3), so the optimal policy was to
    # hold the shirt over the drum forever and never release.  Now positioning
    # only guides; the drop (shirt_in_target) dominates.
    # ``lift_threshold`` raised 0.02 → 0.25 m: transport (XY-to-drum) only pays
    # once the grasp point is at a genuine *carry height* (≈1.05 m).  At 0.02 m
    # the policy unlocked the full XY reward after a 2 cm lift and simply dragged
    # the shirt across the belt to the drum; now it must lift the shirt clear
    # first, enforcing the pick → lift → carry → drop sequence the task wants.
    goal_tracking = RewTerm(
        func=task_rew.shirt_approach_target,
        weight=15.0,
        params={
            "shirt_name": "shirt_proxy",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "std": 1.0,
            "lift_threshold": 0.25,
        },
    )

    # ── 4b. Goal tracking fine (std=0.20) ───────────────────────────
    goal_tracking_fine = RewTerm(
        func=task_rew.shirt_approach_target,
        weight=6.0,
        params={
            "shirt_name": "shirt_proxy",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "std": 0.20,
            "lift_threshold": 0.25,
        },
    )

    # ── 4c. Suspend the whole shirt above the drum opening ──────────
    # Weight cut 12 → 4: at 12 this per-step "shirt cleared over the drum" reward
    # became a hover attractor — the policy parked the shirt above the drum and
    # never released (place_success stuck at 0 while clearance climbed).  It now
    # only *guides* to the drop pose; the release + drop must pay the real reward.
    clearance_over_drum = RewTerm(
        func=task_rew.shirt_clearance_over_drum,
        weight=4.0,
        params={
            "shirt_name": "shirt_proxy",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "rim_clearance": 0.08,
            # 0.20 (was 0.08): a smoother clearance gradient that *pulls* the
            # whole hanging shirt above the rim, instead of an all-or-nothing step
            # the policy could never reach (so it stayed 0 and gave no signal).
            "clear_margin": 0.20,
            "drum_radius": 0.32,
        },
    )

    # ── 5. Release above drum ───────────────────────────────────────
    # Small per-step openness shaping (was 25) — kept only as a gradient hint
    # toward opening once above the drum...
    release = RewTerm(
        func=task_rew.shirt_release_above_target,
        weight=5.0,
        params={
            "shirt_name": "shirt_proxy",
            "drum_name": "drum_target",
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "rim_clearance": 0.10,
            "drum_radius": 0.2735,
        },
    )

    # ...with the real payoff a one-time bonus for actually committing to the drop
    # (grasp opened centred over + above the drum rim).  Anti-hover: the drop is a
    # discrete rewarded event, not a state the policy can hover in.
    release_event = RewTerm(
        func=task_rew.release_event_bonus,
        weight=80.0,
    )

    # ── 5b. Anti-hover time cost ────────────────────────────────────
    # Bleeds reward while a *lifted* shirt is held without being placed, so
    # holding over the drum is worse than committing to the drop.
    carry_time = RewTerm(
        func=task_rew.carry_time_penalty,
        weight=-1.5,
        params={
            "shirt_name": "shirt_proxy",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "lift_threshold": 0.20,
        },
    )

    # ── 6. Success: shirt in drum (dominant terminal reward) ────────
    shirt_in_target = RewTerm(
        func=task_rew.shirt_in_target,
        weight=120.0,
        params={
            "shirt_name": "shirt_proxy",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )

    # ── 7. Return to neutral after placement ────────────────────────
    # Gated on was_placed — drives the arm back to its default pose once the
    # shirt is in the drum (prevents lingering over the rim).  Mirrors cube_place.
    return_to_neutral = RewTerm(
        func=task_rew.return_to_neutral,
        weight=100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "std": 0.25,
        },
    )

    # ── Regularisation ──────────────────────────────────────────────
    # Raised (was -1e-4) so the gripper carries the shirt smoothly and slowly —
    # fast jerky moves whip the welded cloth and exaggerate stretching.
    action_rate = RewTerm(
        func=task_rew.action_rate_l2,
        weight=-3e-4,
    )
    joint_vel = RewTerm(
        func=task_rew.joint_vel_l2_controlled,
        weight=-3e-4,
        params={
            "max_velocity": 10.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    # ── Arm utilization ─────────────────────────────────────────────
    # Cut (was 0.25): it rewarded raw arm velocity, encouraging the fast motion
    # that whips the cloth.  Kept small only to discourage a frozen arm.
    arm_utilization = RewTerm(
        func=task_rew.arm_velocity_bonus,
        weight=0.10,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "max_velocity": 5.0,
        },
    )

    # ── Belt contact penalty ────────────────────────────────────────
    belt_contact = RewTerm(
        func=task_rew.belt_contact_penalty,
        weight=-10.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "margin": 0.0,
            "max_depth": 0.15,
        },
    )

    # ── Joint torque penalty ────────────────────────────────────────
    joint_torque = RewTerm(
        func=task_rew.joint_torque_penalty,
        weight=-0.025,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    # ── Metric-only ─────────────────────────────────────────────────
    metric_place_success = RewTerm(
        func=task_rew.shirt_place_success_bonus,
        weight=0.01,
        params={
            "shirt_name": "shirt_proxy",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )
    metric_grasp_rate = RewTerm(
        func=task_rew.shirt_grasp_metric,
        weight=0.01,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "shirt_name": "shirt_proxy",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "lift_threshold": 0.06,
            "proximity_threshold": 0.10,
        },
    )
    metric_ee_distance = RewTerm(
        func=task_rew.ee_to_shirt_distance_metric,
        weight=-0.01,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "shirt_name": "shirt_proxy",
        },
    )

    # ── Off-conveyor penalty ────────────────────────────────────────
    shirt_off_conveyor = RewTerm(
        func=task_rew.shirt_off_conveyor_penalty,
        weight=-5.0,
        params={
            "shirt_name": "shirt_proxy",
            "bounds": _CONVEYOR_BOUNDS,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum: ramp regularisation penalties once behaviour stabilises.

    Mirrors cube_place — small action-rate / joint-velocity penalties grow over
    training so early exploration is unconstrained and late policies are smooth.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -2e-3, "num_steps": 150000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-3, "num_steps": 150000},
    )


@configclass
class TerminationsCfg:
    """Termination terms.

    Episode runs full duration — the shirt_in_target reward (weight=100)
    accumulates after a successful drop, making release optimal.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    joint_vel_diverged = DoneTerm(
        func=task_rew.joint_vel_out_of_limit,
        time_out=True,
        params={
            "max_velocity": 100.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    belt_collision = DoneTerm(
        func=task_rew.belt_collision_termination,
        time_out=True,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "max_penetration": 0.20,
        },
    )

    # NOTE: an earlier ``placed_settled`` termination (end the episode ~15 steps
    # after a drop) backfired — placing has higher *per-step* reward but ending the
    # episode early cut its *total* return below hovering-to-timeout, so PPO drifted
    # back to hovering (reward↑ while place_success↓).  Removed: a placed shirt now
    # keeps earning ``return_to_neutral`` (weight 100) for the full episode, so
    # dropping strictly dominates holding.  The carry-time penalty still pressures
    # the policy to place sooner.


##
# Environment configuration
##


@configclass
class ShirtPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-agnostic base configuration for the shirt-placement task.

    Each robot variant inherits from this class and calls
    ``_set_robot_params()`` inside ``__post_init__`` to fill in
    ``MISSING`` body/joint names and set the arm action term.
    """

    scene: ShirtPlaceSceneCfg = ShirtPlaceSceneCfg(num_envs=512, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # 60 Hz physics, decimation 1 → 60 Hz control.  Halves the per-step cloth
        # cost vs the 120 Hz / decimation-2 profile (the 11 k-particle PBD cloth
        # is the sim bottleneck); PBD is position-based and unconditionally
        # stable, and the softened solver + gentle deterministic grasp tolerate
        # the larger step.  (Use 120 Hz for final high-fidelity rendering.)
        self.decimation = 1
        self.episode_length_s = 5.0
        self.viewer.eye = (3.5, 3.5, 3.5)

        # ── Cloth solver profile ─────────────────────────────────────────
        # Full-fidelity stretch resistance: 16 position iterations enforce the
        # stiff (1e5) stretch constraints so the cloth stays inextensible when
        # the welded grasp is dragged, and self-collision keeps it from
        # collapsing into a filament.  (Earlier 6-iter / no-self-collision profile
        # was faster but let the cloth over-stretch into a strand.)  CCD stays off
        # (expensive; the gentle, low-speed motion does not tunnel).
        cp = SHIRT_CLOTH_CFG.pbd_params
        # 24 (was 16): stiffer inextensibility to curb the visible over-stretch on
        # pickup (PBD projects the 1e5 stretch constraints more times per step).
        # Throughput headroom exists (~2 it/s at 128 envs).
        cp.solver_position_iterations = 24
        cp.enable_ccd = False
        cp.global_self_collision = True
        # 60 Hz physics (see decimation note above) — halves cloth sim cost.
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_contact_count = 2**21
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.gpu_max_particle_contacts = 2**22
        self.sim.physx.friction_correlation_distance = 0.00625
        # PBD particle cloth requires a large collision stack.  Capped at the
        # signed-32-bit maximum: 2**31 overflows to a negative value, which
        # PhysX reads as a tiny stack and reports as "collisionStackSize buffer
        # overflow ... contacts dropped", silently degrading the cloth physics.
        self.sim.physx.gpu_collision_stack_size = 2**31 - 1

    # ------------------------------------------------------------------
    # Robot-parameter helper
    # ------------------------------------------------------------------

    def _set_robot_params(
        self,
        ee_body: str,
        controlled_joints: list[str],
        arm_joints: list[str],
    ) -> None:
        """Fill all ``MISSING`` body/joint names for a specific robot."""
        grasp = [ee_body]

        # -- Observations --
        obs = self.observations.policy
        obs.joint_pos_rel.params["asset_cfg"].joint_names = controlled_joints
        obs.joint_vel_rel.params["asset_cfg"].joint_names = controlled_joints
        obs.ee_pos_w.params["asset_cfg"].body_names = grasp
        obs.ee_vel_w.params["asset_cfg"].body_names = grasp
        obs.shirt_rel.params["ee_cfg"].body_names = grasp
        obs.fingertip_shirt_rel.params["ee_cfg"].body_names = grasp
        obs.drum_rel.params["ee_cfg"].body_names = grasp

        # -- Rewards --
        rew = self.rewards
        rew.reaching_shirt.params["ee_cfg"].body_names = grasp
        rew.reaching_shirt_fine.params["ee_cfg"].body_names = grasp
        rew.grasping.params["ee_cfg"].body_names = grasp
        rew.lifting_shirt.params["ee_cfg"].body_names = grasp
        rew.height_bonus.params["ee_cfg"].body_names = grasp
        rew.belt_contact.params["ee_cfg"].body_names = grasp
        rew.metric_grasp_rate.params["ee_cfg"].body_names = grasp
        rew.metric_ee_distance.params["ee_cfg"].body_names = grasp
        rew.joint_vel.params["asset_cfg"].joint_names = controlled_joints
        rew.arm_utilization.params["asset_cfg"].joint_names = arm_joints
        rew.joint_torque.params["asset_cfg"].joint_names = arm_joints
        rew.return_to_neutral.params["asset_cfg"].joint_names = arm_joints

        # -- Events --
        self.events.reset_arm.params["asset_cfg"].joint_names = arm_joints

        # -- Terminations --
        self.terminations.joint_vel_diverged.params["asset_cfg"].joint_names = controlled_joints
        self.terminations.belt_collision.params["ee_cfg"].body_names = grasp


@configclass
class ShirtPlaceEnvCfg_PLAY(ShirtPlaceEnvCfg):
    """Smaller configuration for evaluation / play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

    episode_length_s = 8.0

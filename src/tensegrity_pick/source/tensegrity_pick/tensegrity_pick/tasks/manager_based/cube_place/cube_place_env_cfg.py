# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the tensegrity place task.

Sequential pick-and-place reward structure:
  1. Reach → 2. Grasp → 3. Lift → 4. Transport → 5. Release → 6. Success

Critical reward balance:
  - Transport (weight=50) must dominate lift+height (weight=8 each) so the
    agent moves laterally toward the drum rather than holding the cube high.
  - Low transport gate (z > belt + 0.02) keeps the reward active during arm
    extension when the cube naturally dips.
  - Success reward (weight=100) accumulates over remaining episode steps,
    making drop-in-drum clearly optimal over indefinite holding.

Scene: 1 green + 1 red cube on a belt, target drum 0.85 m away in Y.
Conveyor is inactive.  Goal: place the green cube into the target drum.
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

from . import mdp
from .place_scene_cfg import PlaceSceneCfg, PlaceTendonSceneCfg, CONVEYOR_SURFACE_HEIGHT_M
from .mdp import rewards as task_rew

# ── Constants ─────────────────────────────────────────────────────────────
CONTROLLED_JOINT_NAMES = [
    "base_y_joint", "base_z_joint",
    "elbow_joint", "wrist_y_joint", "wrist_x_joint",
    "finger_joint",
]
EE_LINK = "tool_link_0"
GRASP_BODIES = [EE_LINK]

# Spawn box: cubes appear on a narrow line directly below the robot mount
# so the arm can always reach every spawn position.
_SPAWN_BOX = task_rew.SpawnBox(
    x_range=(0.10, 0.20),
    y_range=(-0.10, 0.10),
    z_range=(CONVEYOR_SURFACE_HEIGHT_M + 0.03, CONVEYOR_SURFACE_HEIGHT_M + 0.05),
)

# Drum geometry – the radius matches the physical drum (0.547 m diameter)
# but the height is reduced to 0.30 m so that only cubes actually dropped
# into the drum (settling at ground level z≈0.025) are counted as "inside".
_BIN_GEOM = task_rew.BinCylinder(radius=0.547 * 0.5, height=0.30)

# Conveyor surface bounds — cubes outside these are penalised.
_CONVEYOR_BOUNDS = task_rew.ConveyorBounds(y_min=-0.4, y_max=0.4, z_min=0.70)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint-position delta actions (base + arm + gripper)."""

    base_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
        scale=0.50,
        use_default_offset=True,
        clip={"base_y_joint": (-0.5, 0.5), "base_z_joint": (-0.50, 0.0)},
    )
    # scale=1.0 gives elbow ∈ [−1.0, 1.0] rad (within ±1.5 clip)
    # and wrist ∈ [−0.8, 0.8] rad (full range).  The arm must bridge
    # the 0.35 m gap from base_y_max (0.5) to drum (Y=0.85),
    # requiring at least elbow=-0.50 rad (see verify_actuation.py).
    arm_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        scale=1.0,
        use_default_offset=True,
        clip={
            "elbow_joint": (-1.5, 1.5),
            "wrist_y_joint": (-0.8, 0.8),
            "wrist_x_joint": (-0.8, 0.8),
        },
    )
    # Only the drive joint (finger_joint) is actively commanded.
    # All other gripper joints follow through the physical four-bar
    # linkage and mimic constraints defined in the Robotiq 2F-140 USD.
    # Commanding passive/auxiliary joints directly causes them to fight
    # the linkage mechanism and break the gripper.
    # BinaryJointPositionActionCfg already handles continuous tanh
    # outputs from SKRL PPO: negative → close (0.7854), positive/zero →
    # open (0.0).  This ensures the gripper defaults to open on reset
    # (raw_actions reset to 0.0 → open_command).
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


@configclass
class ObservationsCfg:
    """Observation specification — only controlled joints are observed."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception (controlled joints only)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )

        # EE kinematics
        ee_pos_w = ObsTerm(
            func=task_rew.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
        ee_vel_w = ObsTerm(
            func=task_rew.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )

        # Cube-relative positions
        green_rel = ObsTerm(
            func=task_rew.cube_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "cube_name": "green_cube",
            },
        )
        red_rel = ObsTerm(
            func=task_rew.cube_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "cube_name": "red_cube",
            },
        )

        # Dynamic fingertip-to-cube: accounts for actual gripper closure
        # so the policy knows where the contact surfaces are.
        fingertip_green_rel = ObsTerm(
            func=task_rew.fingertip_rel_cube,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "cube_name": "green_cube",
            },
        )
        fingertip_red_rel = ObsTerm(
            func=task_rew.fingertip_rel_cube,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "cube_name": "red_cube",
            },
        )

        # Normalized gripper closure [0=open, 1=closed]
        gripper_closure = ObsTerm(
            func=task_rew.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )

        # Normalized gripper torque [0=no effort, 1=at limit]
        gripper_torque = ObsTerm(
            func=task_rew.gripper_torque_residual,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )

        # Cube velocities — lets the policy distinguish stationary
        # cubes from bouncing / falling ones.
        green_cube_vel = ObsTerm(
            func=task_rew.cube_velocity,
            params={"cube_name": "green_cube"},
        )
        red_cube_vel = ObsTerm(
            func=task_rew.cube_velocity,
            params={"cube_name": "red_cube"},
        )

        # Drum-relative position
        drum_rel = ObsTerm(
            func=task_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "drum_name": "drum_target",
            },
        )

        # Last actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset events.  No conveyor motion events needed."""

    # Reset all prims to default state first (critical for broken gripper recovery).
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["base_y_joint", "base_z_joint",
                             "elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            ),
            "position_range": (-0.10, 0.10),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Gripper starts fully open (matching reference Lift task).
    reset_gripper = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_cubes = EventTerm(
        func=task_rew.reset_place_cubes,
        mode="reset",
        params={
            "spawn_box": _SPAWN_BOX,
            "green_name": "green_cube",
            "red_name": "red_cube",
            "parking_pose": (100.0, 100.0, 1.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the tensegrity pick-and-place task.

    Sequential structure: reach → grasp → lift → transport → release → success.

    Weight hierarchy (effective per-step after ×dt=0.02):
      Transport (50×0.02=1.00 max) > Success (100×0.02=2.00) >
      Release (10×0.02=0.20) > Lift (8×0.02=0.16) =
      Height (8×0.02=0.16) > Grasp (5×0.02=0.10) > Reach (1×0.02=0.02)

    The transport reward dominates lift/height once the cube is grasped,
    preventing a \"hold high and stay still\" local optimum.
    """

    # ── 1. Reach: tanh proximity (fingertip → nearest cube) ────────
    # Uses dynamic fingertip (not grasp centre) so the robot
    # approaches from above.  Targets the nearest active cube.
    reaching_object = RewTerm(
        func=task_rew.object_ee_distance,
        weight=1.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "green_name": "green_cube",
            "red_name": "red_cube",
            "std": 0.1,
        },
    )

    # ── 2. Grasp: closure × proximity ────────────────────────────────
    # Explicit reward for closing the gripper when positioned near the
    # cube.  Creates the gradient: approach → position → close fingers.
    # Fires at belt level (before any lift), teaching the agent to grasp
    # the cube while it is lying still.
    grasping = RewTerm(
        func=task_rew.grasp_reward,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "green_name": "green_cube",
            "red_name": "red_cube",
            "std": 0.08,
        },
    )

    # ── 3. Lift: binary bonus when cube is grasped and lifted ────────
    # max_velocity=1.0 rejects bounced cubes (high velocity at bounce
    # peak).  Only a genuinely grasped, controlled lift satisfies all
    # gates: above belt, near gripper, closed fingers, low velocity.
    # Weight balanced against transport to avoid hold-in-place optimum.
    lifting_object = RewTerm(
        func=task_rew.object_is_lifted,
        weight=8.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "green_name": "green_cube",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "minimal_height": 0.06,
            "max_distance": 0.15,
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "max_velocity": 1.0,
        },
    )

    # ── 3b. Height bonus: smooth gradient to lift above drum rim ─────
    # Bridges the binary lift and the approach gate so the agent has
    # gradient to keep lifting.  Weight balanced against transport to
    # prevent over-optimising vertical hold at the expense of lateral
    # movement toward the drum.
    height_bonus = RewTerm(
        func=task_rew.cube_height_bonus,
        weight=8.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "green_name": "green_cube",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "max_height": 0.30,
            "max_distance": 0.15,
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "max_velocity": 1.0,
        },
    )

    # ── 4. Goal tracking: coarse (std=1.0) — drives toward drum ──────
    # std=1.0 gives ~0.31 proximity at d_xy=0.85 m.
    # Transport signal (0.31/step) must dominate the hold-in-place
    # reward from lift+height (0.21/step combined).
    # lift_threshold=0.02: z>0.82 is trivially easy to maintain during
    # arm extension, preventing the "dip below gate" problem where the
    # cube temporarily drops during lateral motion and zeros out the
    # transport reward.
    goal_tracking = RewTerm(
        func=task_rew.approach_target_tanh,
        weight=50.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "std": 1.0,
            "lift_threshold": 0.02,
        },
    )

    # ── 4b. Goal tracking: fine (std=0.20) — precision near drum ─────
    # At d=0.27 m (drum edge) gives 0.24 proximity — a useful
    # fine-positioning signal that helps centre the cube above the
    # drum opening.
    goal_tracking_fine = RewTerm(
        func=task_rew.approach_target_tanh,
        weight=5.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "std": 0.20,
            "lift_threshold": 0.02,
        },
    )

    # ── 5. Release: reward opening gripper above the drum ────────────
    # Only fires when cube is within drum radius in XY, above the rim,
    # and was_grasped is true (preventing random gripper flapping).
    # Creates the gradient for the final phase: transport → open → drop.
    release = RewTerm(
        func=task_rew.release_above_target,
        weight=10.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "rim_clearance": 0.10,
            "drum_radius": 0.2735,
        },
    )

    # ── 6. Success: large per-step reward while cube is in drum ──────
    # Weight=100 gives 2.0 per step (after dt=0.02).  With episode NOT
    # terminating on success, this accumulates over remaining steps and
    # clearly dominates holding rewards (~0.8/step).  Gated on
    # was_grasped to prevent accidental/bumped placements.
    green_in_target = RewTerm(
        func=task_rew.green_cube_in_target,
        weight=100.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )

    # ── 7. Negative: red cube in drum ────────────────────────────────
    red_in_target = RewTerm(
        func=task_rew.red_cube_in_target,
        weight=-12.0,
        params={
            "red_name": "red_cube",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )

    # ── Regularisation (start small, ramped by curriculum) ───────────
    action_rate = RewTerm(
        func=task_rew.action_rate_l2,
        weight=-1e-4,
    )
    joint_vel = RewTerm(
        func=task_rew.joint_vel_l2_controlled,
        weight=-1e-4,
        params={
            "max_velocity": 10.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=CONTROLLED_JOINT_NAMES,
            ),
        },
    )

    # ── Base velocity: penalty to prefer arm over base ────────────
    base_velocity = RewTerm(
        func=task_rew.base_velocity_l2,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["base_y_joint", "base_z_joint"],
            ),
        },
    )

    # ── Arm utilization: preference for arm movement ─────────────────
    arm_utilization = RewTerm(
        func=task_rew.arm_velocity_bonus,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            ),
            "max_velocity": 5.0,
        },
    )

    # ── Belt contact: penalty for finger tips below belt ────────────
    belt_contact = RewTerm(
        func=task_rew.belt_contact_penalty,
        weight=-10.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "margin": 0.0,
            "max_depth": 0.15,
        },
    )

    # ── Joint torque: penalise high effort on arm joints ─────────────
    joint_torque = RewTerm(
        func=task_rew.joint_torque_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            ),
        },
    )

    # ── Metric-only (tiny weight, for TensorBoard analysis) ──────────
    metric_place_success = RewTerm(
        func=task_rew.place_success_bonus,
        weight=0.01,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )
    metric_grasp_rate = RewTerm(
        func=task_rew.green_grasp_metric,
        weight=0.01,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "green_name": "green_cube",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "lift_threshold": 0.06,
            "proximity_threshold": 0.10,
        },
    )
    metric_ee_distance = RewTerm(
        func=task_rew.ee_to_green_distance_metric,
        weight=-0.01,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "green_name": "green_cube",
        },
    )

    # ── Conveyor penalty: penalise cubes knocked off the belt ────────
    cube_off_conveyor = RewTerm(
        func=task_rew.cube_off_conveyor_penalty,
        weight=-5.0,
        params={
            "green_name": "green_cube",
            "red_name": "red_cube",
            "bounds": _CONVEYOR_BOUNDS,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms.

    The episode runs the full duration without early success termination.
    The green_in_target reward (weight=100) accumulates over remaining
    steps after a successful drop, making release clearly more valuable
    than holding above the drum indefinitely.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    joint_vel_diverged = DoneTerm(
        func=task_rew.joint_vel_out_of_limit,
        time_out=True,
        params={
            "max_velocity": 100.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
        },
    )

    # Terminate when the finger tips penetrate well below the belt.
    # The default tip z ≈ 0.807 is only 0.007 m above the belt (0.800).
    # max_penetration=0.20 puts the kill-line at z = 0.60 so early
    # exploration doesn't terminate most environments immediately.
    # The belt_contact penalty (weight=-10) provides a softer gradient
    # above this hard cap.
    belt_collision = DoneTerm(
        func=task_rew.belt_collision_termination,
        time_out=True,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "max_penetration": 0.20,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum: ramp up regularisation + introduce the red cube."""

    # Red cube introduced at 100k steps to let the green-only policy
    # master reach→grasp→lift→transport→place before adding the
    # distractor.  Introducing it too early causes severe reward crash.
    activate_red = CurrTerm(
        func=task_rew.activate_red_cube_curriculum,
        params={"num_steps": 100000},
    )

    # IsaacLab-style: ramp regularisation from near-zero to meaningful
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -2e-3, "num_steps": 200000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-3, "num_steps": 200000},
    )


##
# Environment configuration
##


@configclass
class TensegrityPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the simplified tensegrity cube-placement task."""

    scene: PlaceSceneCfg = PlaceSceneCfg(num_envs=8192, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # Match IsaacLab's lift task sim settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_contact_count = 2**21
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class TensegrityPlaceEnvCfg_PLAY(TensegrityPlaceEnvCfg):
    """Smaller configuration for evaluation / play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        # Force-activate red cube immediately in play mode so both
        # cubes are visible (curriculum threshold 0 = always active).
        self.curriculum.activate_red.params["num_steps"] = 0


# ══════════════════════════════════════════════════════════════════════════
# Tendon-driven variant
# ══════════════════════════════════════════════════════════════════════════
# The arm's 3 revolute joints (elbow, wrist_y, wrist_x) are actuated by
# 5 tendons instead of implicit PD drives.  Everything else (base,
# gripper, rewards, terminations, curriculum) is identical.

from tensegrity_pick.robots import TendonEffortActionCfg
from tensegrity_pick.robots.tendon_actuator import DEFAULT_JACOBIAN_TRANSPOSE


@configclass
class TendonActionsCfg:
    """Mixed action space: implicit PD for base, tendon efforts for arm,
    binary position for gripper."""

    base_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
        scale=0.50,
        use_default_offset=True,
        clip={"base_y_joint": (-0.5, 0.5), "base_z_joint": (-0.50, 0.0)},
    )

    arm_tendon = TendonEffortActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        num_tendons=5,
        max_tension=500.0,
        jacobian_transpose=DEFAULT_JACOBIAN_TRANSPOSE,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


@configclass
class TendonObservationsCfg:
    """Same as base observations but with tendon tensions appended."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        ee_pos_w = ObsTerm(
            func=task_rew.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
        ee_vel_w = ObsTerm(
            func=task_rew.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
        green_rel = ObsTerm(
            func=task_rew.cube_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "cube_name": "green_cube",
            },
        )
        red_rel = ObsTerm(
            func=task_rew.cube_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "cube_name": "red_cube",
            },
        )

        # Dynamic fingertip-to-cube (parity with PD observations)
        fingertip_green_rel = ObsTerm(
            func=task_rew.fingertip_rel_cube,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "cube_name": "green_cube",
            },
        )
        fingertip_red_rel = ObsTerm(
            func=task_rew.fingertip_rel_cube,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "cube_name": "red_cube",
            },
        )
        gripper_closure = ObsTerm(
            func=task_rew.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        gripper_torque = ObsTerm(
            func=task_rew.gripper_torque_residual,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )

        green_cube_vel = ObsTerm(
            func=task_rew.cube_velocity,
            params={"cube_name": "green_cube"},
        )
        red_cube_vel = ObsTerm(
            func=task_rew.cube_velocity,
            params={"cube_name": "red_cube"},
        )

        drum_rel = ObsTerm(
            func=task_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "drum_name": "drum_target",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TensegrityPlaceTendonEnvCfg(TensegrityPlaceEnvCfg):
    """Tensegrity place task driven by tendon tensions instead of joint
    position deltas.

    The arm's 3 revolute joints (elbow, wrist_y, wrist_x) are actuated by
    5 tendons: 2 antagonistic for the elbow and 3 at 120° for the 2-DOF
    wrist.  Everything else (base, gripper, rewards, terminations,
    curriculum) is identical to the base task.
    """

    scene: PlaceTendonSceneCfg = PlaceTendonSceneCfg(num_envs=4096, env_spacing=5.0)
    actions: TendonActionsCfg = TendonActionsCfg()
    observations: TendonObservationsCfg = TendonObservationsCfg()


@configclass
class TensegrityPlaceTendonEnvCfg_PLAY(TensegrityPlaceTendonEnvCfg):
    """Smaller evaluation / play configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.curriculum.activate_red.params["num_steps"] = 0

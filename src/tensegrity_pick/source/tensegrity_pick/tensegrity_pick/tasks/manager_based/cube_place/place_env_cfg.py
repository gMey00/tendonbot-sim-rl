# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agnostic base environment configuration for the cube-placement task.

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

Robot-specific parameters (EE body, joint names, arm action) are filled
in by each variant config via ``PlaceEnvCfg._set_robot_params()`` in
``__post_init__``, following the same pattern as the reach task.
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
from .place_scene_cfg import PlaceSceneCfg, CONVEYOR_SURFACE_HEIGHT_M
from .mdp import rewards as task_rew

# ── Scene-level constants (shared by all robot variants) ──────────────────

# Spawn box: cubes start in a narrow band and widen via curriculum.
# Initial y_range ±0.10 matches Iter 2 baseline (proven breakthrough at 12k).
# The widen_spawn curriculum linearly widens to ±0.30 (75% belt width).
# ±0.40 caused grasp_rate to drop to 55% (workspace limit); ±0.30 achieves 86%.
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
    """Robot-agnostic action specification.

    ``arm_action`` is ``MISSING`` — each variant fills it with the
    appropriate action term (JointPositionActionCfg for PD arms,
    TendonEffortActionCfg for tendon arms, etc.).

    Variants with a prismatic base (tensegrity) add ``base_delta``
    in their own ``__post_init__``.
    """

    arm_action: ActionTerm = MISSING

    # Shared across all robots: only the drive joint (finger_joint) is
    # actively commanded.  All other gripper joints follow through the
    # physical four-bar linkage and mimic constraints in the USD.
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
    ``MISSING`` and filled by ``PlaceEnvCfg._set_robot_params()``.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception (controlled joints only)
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

        # Cube-relative positions
        green_rel = ObsTerm(
            func=task_rew.cube_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "cube_name": "green_cube",
            },
        )
        red_rel = ObsTerm(
            func=task_rew.cube_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "cube_name": "red_cube",
            },
        )

        # Dynamic fingertip-to-cube: accounts for actual gripper closure
        # so the policy knows where the contact surfaces are.
        fingertip_green_rel = ObsTerm(
            func=task_rew.fingertip_rel_cube,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "cube_name": "green_cube",
            },
        )
        fingertip_red_rel = ObsTerm(
            func=task_rew.fingertip_rel_cube,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
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
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "drum_name": "drum_target",
            },
        )

        # Last actions
        actions = ObsTerm(func=mdp.last_action)

        # Task completion flag — lets the policy know when to return to neutral
        was_placed = ObsTerm(func=task_rew.was_placed_obs)

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
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
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
    """Reward terms for the cube pick-and-place task.

    Sequential structure: reach → grasp → lift → transport → release → success.

    Weight hierarchy (aligned with cube_sort):
      Success (100) > Goal tracking (40) > Release (25) >
      Goal fine (10) > Reach fine (5) = Lift (5) = Height (5) >
      Grasp (3) > Reach coarse (2) > Arm util (0.5)

    Reaching is gated on !grasp_active so it turns off once holding a
    cube, freeing reward budget for transport. Coarse std=2.0 provides
    gradient from the top-start position; fine std=0.5 guides final
    approach.

    Robot-specific ``body_names`` / ``joint_names`` are ``MISSING`` and
    filled by ``PlaceEnvCfg._set_robot_params()``.
    """

    # ── 1a. Reach coarse: long-range gradient (std=2.0) ─────────────
    # Large std provides gradient even from the top position (0.25 m away).
    # Gated on !grasp_active so it turns off once holding a cube.
    reaching_object = RewTerm(
        func=task_rew.object_ee_distance,
        weight=2.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "green_name": "green_cube",
            "red_name": "red_cube",
            "std": 2.0,
        },
    )

    # ── 1b. Reach fine: mid-range gradient (std=0.5) ─────────────────
    # Tighter gradient kicks in once close, guiding final approach.
    reaching_object_fine = RewTerm(
        func=task_rew.object_ee_distance,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "green_name": "green_cube",
            "red_name": "red_cube",
            "std": 0.5,
        },
    )

    # ── 2. Grasp: closure × proximity ────────────────────────────────
    grasping = RewTerm(
        func=task_rew.grasp_reward,
        weight=3.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "green_name": "green_cube",
            "red_name": "red_cube",
            "std": 0.08,
        },
    )

    # ── 3. Lift: binary bonus when cube is grasped and lifted ────────
    lifting_object = RewTerm(
        func=task_rew.object_is_lifted,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "green_name": "green_cube",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "minimal_height": 0.06,
            "max_distance": 0.15,
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "max_velocity": 1.0,
        },
    )

    # ── 3b. Height bonus: smooth gradient to lift above drum rim ─────
    height_bonus = RewTerm(
        func=task_rew.cube_height_bonus,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "green_name": "green_cube",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "max_height": 0.30,
            "max_distance": 0.15,
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "max_velocity": 1.0,
        },
    )

    # ── 4. Goal tracking: coarse (std=1.0) — drives toward drum ──────
    goal_tracking = RewTerm(
        func=task_rew.approach_target_tanh,
        weight=40.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "std": 1.0,
            "lift_threshold": 0.02,
        },
    )

    # ── 4b. Goal tracking: fine (std=0.20) — precision near drum ─────
    goal_tracking_fine = RewTerm(
        func=task_rew.approach_target_tanh,
        weight=10.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "std": 0.20,
            "lift_threshold": 0.02,
        },
    )

    # ── 5. Release: reward opening gripper above the drum ────────────
    release = RewTerm(
        func=task_rew.release_above_target,
        weight=25.0,
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
    green_in_target = RewTerm(
        func=task_rew.green_cube_in_target,
        weight=100.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )

    # ── 7a. Red clearance: reward for red cube being far from drum ───
    # Active only before green cube is grasped — encourages the agent
    # to push red aside first, then focus on green.
    red_clearance = RewTerm(
        func=task_rew.red_clearance_from_drum,
        weight=5.0,
        params={
            "red_name": "red_cube",
            "drum_name": "drum_target",
            "std": 0.4,
        },
    )

    # ── 7b. Red-green separation: push red away from green ─────────
    # Drives a push-aside phase before grasping, especially when red
    # is on top of green.  Only active before grasping (!was_grasped).
    red_green_separation = RewTerm(
        func=task_rew.red_green_separation,
        weight=3.0,
        params={
            "green_name": "green_cube",
            "red_name": "red_cube",
            "std": 0.15,
        },
    )

    # ── 8. Return to neutral after task completion ───────────────────
    # Active once the cube is placed in the drum (gated on was_placed).
    # Drives the arm back to its default (zero) joint positions to
    # prevent collisions with the drum rim.
    return_to_neutral = RewTerm(
        func=task_rew.return_to_neutral,
        weight=100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "std": 0.25,
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    # ── Arm utilization: preference for arm movement ─────────────────
    # joint_names filled per variant (UR10e: 6 rev, Kinova: 7 rev,
    # tensegrity: 3 rev arm joints).
    arm_utilization = RewTerm(
        func=task_rew.arm_velocity_bonus,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "max_velocity": 5.0,
        },
    )

    # ── Belt contact: penalty for finger tips below belt ────────────
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

    # ── Joint torque: penalise high effort on arm joints ─────────────
    joint_torque = RewTerm(
        func=task_rew.joint_torque_penalty,
        weight=-0.025,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
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
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
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
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "green_name": "green_cube",
        },
    )

    # ── Conveyor penalty: penalise GREEN cube knocked off the belt ────
    # Red cube is intentionally excluded — the agent may push red off
    # the belt during the push-aside phase and that’s acceptable.
    cube_off_conveyor = RewTerm(
        func=task_rew.cube_off_conveyor_penalty,
        weight=-5.0,
        params={
            "green_name": "green_cube",
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


@configclass
class CurriculumCfg:
    """Curriculum: narrow→wide spawn, ramp regularisation, introduce red cube."""

    widen_spawn = CurrTerm(
        func=task_rew.widen_spawn_curriculum,
        params={"initial_y": 0.10, "final_y": 0.30, "num_steps": 75000, "delay_steps": 25000},
    )

    activate_red = CurrTerm(
        func=task_rew.activate_red_cube_curriculum,
        params={"num_steps": 125000},  # delay red until agent stabilizes at ±0.30 (spawn done at 100k)
    )

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -2e-3, "num_steps": 150000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-3, "num_steps": 150000},
    )


##
# Environment configuration
##


@configclass
class PlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-agnostic base configuration for the cube-placement task.

    Each robot variant inherits from this class and calls
    ``_set_robot_params()`` inside ``__post_init__`` to fill in
    ``MISSING`` body/joint names and set the arm action term.
    """

    scene: PlaceSceneCfg = PlaceSceneCfg(num_envs=8192, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        # Sim settings (shared by all robots)
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
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024  # needs ≥32791; doubled for headroom
        self.sim.physx.friction_correlation_distance = 0.00625

    # ------------------------------------------------------------------
    # Robot-parameter helper
    # ------------------------------------------------------------------

    def _set_robot_params(
        self,
        ee_body: str,
        controlled_joints: list[str],
        arm_joints: list[str],
    ) -> None:
        """Fill all ``MISSING`` body/joint names for a specific robot.

        Called from each variant's ``__post_init__`` after ``super().__post_init__()``.

        Args:
            ee_body: End-effector body name (e.g. ``"tool_link_0"``).
            controlled_joints: All actively controlled joint names
                (arm + finger, used for observations and terminations).
            arm_joints: Arm-only joint names (used for arm_utilization,
                joint_torque, and reset_arm events).
        """
        grasp = [ee_body]

        # -- Observations --
        obs = self.observations.policy
        obs.joint_pos_rel.params["asset_cfg"].joint_names = controlled_joints
        obs.joint_vel_rel.params["asset_cfg"].joint_names = controlled_joints
        obs.ee_pos_w.params["asset_cfg"].body_names = grasp
        obs.ee_vel_w.params["asset_cfg"].body_names = grasp
        obs.green_rel.params["ee_cfg"].body_names = grasp
        obs.red_rel.params["ee_cfg"].body_names = grasp
        obs.fingertip_green_rel.params["ee_cfg"].body_names = grasp
        obs.fingertip_red_rel.params["ee_cfg"].body_names = grasp
        obs.drum_rel.params["ee_cfg"].body_names = grasp

        # -- Rewards --
        rew = self.rewards
        rew.reaching_object.params["ee_cfg"].body_names = grasp
        rew.reaching_object_fine.params["ee_cfg"].body_names = grasp
        rew.grasping.params["ee_cfg"].body_names = grasp
        rew.lifting_object.params["ee_cfg"].body_names = grasp
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
class PlaceEnvCfg_PLAY(PlaceEnvCfg):
    """Smaller base configuration for evaluation / play.

    Variants should inherit from their own train config instead, but
    this provides shared play-mode defaults.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        # Force-activate red cube immediately in play mode so both
        # cubes are visible (curriculum threshold 0 = always active).
        self.curriculum.activate_red.params["num_steps"] = 0

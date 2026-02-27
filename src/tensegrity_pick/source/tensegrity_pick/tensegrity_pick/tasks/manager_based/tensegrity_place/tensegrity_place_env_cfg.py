# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the tensegrity place task.

Simplified cube-placement modelled after IsaacLab's manipulation/lift:
  - Clean tanh-based reward structure with few terms.
  - No custom step() injection — all rewards via RewardManager.
  - Curriculum ramps up regularisation penalties.
  - 1 green + 1 red cube placed directly below the robot arm on the belt.
  - Conveyor is **inactive** (no belt motion).
  - Goal: place the green cube into the target drum.
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
from .place_scene_cfg import PlaceSceneCfg, CONVEYOR_SURFACE_HEIGHT_M
from .mdp import rewards as task_rew

# ── Constants ─────────────────────────────────────────────────────────────
CONTROLLED_JOINT_NAMES = [
    "base_y_joint", "base_z_joint",
    "elbow_joint", "wrist_y_joint", "wrist_x_joint",
    "finger_joint",
]
EE_LINK = "tool_link_0"
GRASP_BODIES = [EE_LINK]

# Spawn box: cubes appear directly below the robot mount (local coords)
_SPAWN_BOX = task_rew.SpawnBox(
    x_range=(0.00, 0.30),
    y_range=(-0.15, 0.15),
    z_range=(CONVEYOR_SURFACE_HEIGHT_M + 0.03, CONVEYOR_SURFACE_HEIGHT_M + 0.05),
)

# Drum geometry – the radius matches the physical drum (0.547 m diameter)
# but the height is reduced to 0.30 m so that only cubes actually dropped
# into the drum (settling at ground level z≈0.025) are counted as "inside".
_BIN_GEOM = task_rew.BinCylinder(radius=0.547 * 0.5, height=0.30)


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
    arm_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        scale=0.50,
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

        # Drum-relative position
        drum_rel = ObsTerm(
            func=task_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "drum_name": "drum_target",
            },
        )

        # Previous actions
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset events.  No conveyor motion events needed."""

    # Reset all prims to default state first (critical for broken gripper recovery)
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
    """Reward terms modelled after IsaacLab's manipulation/lift task.

    Structure: reach → lift (binary) → goal tracking + regularisation.
    NO gripper reward — the agent discovers grasping naturally because
    it is the only way to get the cube above the lift threshold.
    """

    # ── 1. Reach: tanh proximity (grasp centre → cube) ────────────────
    # std scaled for our 1.5 m workspace (Franka reference uses 0.1 at
    # ~0.5 m; tanh(1.5/0.1)≈1 gives zero gradient for us).
    reaching_object = RewTerm(
        func=task_rew.object_ee_distance,
        weight=1.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "green_name": "green_cube",
            "std": 1.0,
        },
    )

    # ── 2. Lift: binary bonus when cube is above belt ─────────────────
    # minimal_height=0.06 places the threshold at belt+0.06=0.86,
    # which is above the max spawn z (0.85).  This ensures the signal
    # is zero at rest and fires only for genuine lifts.
    lifting_object = RewTerm(
        func=task_rew.object_is_lifted,
        weight=15.0,
        params={
            "green_name": "green_cube",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "minimal_height": 0.06,
        },
    )

    # ── 3. Goal tracking: coarse (std=0.3) — drives toward drum ──────
    goal_tracking = RewTerm(
        func=task_rew.approach_target_tanh,
        weight=16.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "std": 0.3,
            "lift_threshold": 0.06,
        },
    )

    # ── 4. Goal tracking: fine (std=0.05) — precision near drum ──────
    goal_tracking_fine = RewTerm(
        func=task_rew.approach_target_tanh,
        weight=5.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "std": 0.05,
            "lift_threshold": 0.06,
        },
    )

    # ── 5. Success: large bonus for cube in drum ─────────────────────
    green_in_target = RewTerm(
        func=task_rew.green_cube_in_target,
        weight=40.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )

    # ── 6. Negative: red cube in drum ────────────────────────────────
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
    # ALL joints including gripper — matching the reference Lift task.
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

    # ── Belt contact: penalty for finger tips near/below belt ─────────
    belt_contact = RewTerm(
        func=task_rew.belt_contact_penalty,
        weight=-10.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "margin": 0.02,
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
        weight=0.01,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "green_name": "green_cube",
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Auto-terminate on successful placement for faster episode turnover.
    green_placed = DoneTerm(
        func=task_rew.green_in_drum,
        time_out=False,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )

    joint_vel_diverged = DoneTerm(
        func=task_rew.joint_vel_out_of_limit,
        time_out=True,
        params={
            "max_velocity": 100.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
        },
    )

    # Terminate when any arm/wrist joint's *computed* torque is
    # extremely high (5× effort limit).  Normal PD clamping (computed >
    # limit) happens routinely with high-stiffness actuators; only truly
    # extreme stress (e.g. jamming against the belt or self-collision)
    # produces 5× overshoot.
    joint_effort_saturated = DoneTerm(
        func=task_rew.joint_effort_exceeded,
        time_out=True,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            ),
            "threshold_ratio": 5.0,
        },
    )

    # Terminate when the finger tips penetrate >5 cm below the belt.
    # Uses EE body quaternion to project the finger tip offset (body
    # origins in the Robotiq USD sit at the gripper base, not the tips).
    belt_collision = DoneTerm(
        func=task_rew.belt_collision_termination,
        time_out=True,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "max_penetration": 0.05,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum: ramp up regularisation + introduce the red cube."""

    activate_red = CurrTerm(
        func=task_rew.activate_red_cube_curriculum,
        params={"num_steps": 30000},
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

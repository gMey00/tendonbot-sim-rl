"""Environment configuration for the tensegrity cube-sorting task.

7-phase reward structure (per research report) on an active conveyor:
  1. Reach → 2. Grasp → 3. Lift → 4. Transport (urgency) → 5. Release
  → 6. Re-orient → (Event-based: placement bonus, miss penalty, red-grabbed)

Scene: unified collection of 16 labelled cubes on a moving belt, target drum
beside the belt.  Labels decide sorting logic (0 = target, 1 = distractor).

Stage 1 curriculum: 1 green cube, 0 red cubes, belt speed 0.1-0.3 m/s.
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
from .cube_sorting_scene_cfg import (
    CubeSortingSceneCfg,
    CONVEYOR_START_X,
    CONVEYOR_END_X,
    BELT_HEIGHT_M,
    BELT_WIDTH_M,
    CONVEYOR_CENTER_Y,
    TARGET_LABEL,
    DISTRACTOR_LABEL,
)
from .mdp import cube_sorting_mdp as task_mdp
from .mdp import rewards as task_rew
from ..shared import gripper_cfg as shared_rew

# ── Constants ─────────────────────────────────────────────────────────────
CONTROLLED_JOINT_NAMES = [
    "base_y_joint", "base_z_joint",
    "elbow_joint", "wrist_y_joint", "wrist_x_joint",
    "finger_joint",
]
EE_LINK = "tool_link_0"
GRASP_BODIES = [EE_LINK]

CUBES_KEY = "cubes"

_SPAWN_BOX = task_mdp.SpawnBox(
    x_range=(CONVEYOR_START_X + 0.10, CONVEYOR_START_X + 1.80),
    y_range=(-0.20, 0.20),
    z_range=(BELT_HEIGHT_M + 0.03, BELT_HEIGHT_M + 0.05),
)

_BIN_GEOM = shared_rew.BinCylinder(radius=0.547 * 0.5, height=0.30)
_CONVEYOR_BOUNDS = shared_rew.ConveyorBounds(y_min=-0.4, y_max=0.4, z_min=0.70)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint-position delta actions (base + arm) + binary gripper."""

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
        scale=1.0,
        use_default_offset=True,
        clip={
            "elbow_joint": (-1.5, 1.5),
            "wrist_y_joint": (-0.8, 0.8),
            "wrist_x_joint": (-0.8, 0.8),
        },
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


@configclass
class ObservationsCfg:
    """Observation specification.

    Robot state + nearest-cube observations + task context.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ── Proprioception ────────────────────────────────────────────
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )

        # ── EE kinematics (grasp-centre based) ───────────────────────
        ee_pos_w = ObsTerm(
            func=shared_rew.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
        ee_vel_w = ObsTerm(
            func=shared_rew.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )

        # ── Nearest-cube relative positions ──────────────────────────
        nearest_target_rel = ObsTerm(
            func=task_rew.nearest_cube_rel,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "collection_name": CUBES_KEY,
                "label": TARGET_LABEL,
            },
        )
        nearest_distractor_rel = ObsTerm(
            func=task_rew.nearest_cube_rel,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "collection_name": CUBES_KEY,
                "label": DISTRACTOR_LABEL,
            },
        )

        # ── Dynamic fingertip-to-cube ────────────────────────────────
        fingertip_target_rel = ObsTerm(
            func=task_rew.nearest_cube_fingertip_rel,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "collection_name": CUBES_KEY,
                "label": TARGET_LABEL,
            },
        )
        fingertip_distractor_rel = ObsTerm(
            func=task_rew.nearest_cube_fingertip_rel,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
                "collection_name": CUBES_KEY,
                "label": DISTRACTOR_LABEL,
            },
        )

        # ── Gripper state ────────────────────────────────────────────
        gripper_closure = ObsTerm(
            func=shared_rew.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        gripper_torque = ObsTerm(
            func=shared_rew.gripper_torque_residual,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )

        # ── Cube velocities ──────────────────────────────────────────
        target_cube_vel = ObsTerm(
            func=task_rew.nearest_cube_velocity,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "collection_name": CUBES_KEY,
                "label": TARGET_LABEL,
            },
        )
        distractor_cube_vel = ObsTerm(
            func=task_rew.nearest_cube_velocity,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "collection_name": CUBES_KEY,
                "label": DISTRACTOR_LABEL,
            },
        )

        # ── Drum-relative position ───────────────────────────────────
        drum_rel = ObsTerm(
            func=shared_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "drum_name": "drum_target",
            },
        )

        # ── Task context (per research report Section 4) ─────────────
        belt_speed = ObsTerm(func=task_rew.belt_speed_obs)
        time_remaining = ObsTerm(func=task_rew.time_fraction_obs)
        placed_count = ObsTerm(func=task_rew.targets_placed_obs)
        missed_count = ObsTerm(func=task_rew.targets_missed_obs)

        # ── Last actions ─────────────────────────────────────────────
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset events + active conveyor.

    Stage 1 curriculum: 1 green cube, 0 red cubes, belt speed 0.1-0.3 m/s.
    """

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

    reset_gripper = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Stage 1: 1 green, 0 red (curriculum will increase these)
    reset_cubes = EventTerm(
        func=task_mdp.reset_cubes,
        mode="reset",
        params={
            "collection_name": CUBES_KEY,
            "spawn_box": _SPAWN_BOX,
            "active_per_label": {str(TARGET_LABEL): 1, str(DISTRACTOR_LABEL): 0},
            "parking_pose": (100.0, 100.0, 1.0),
        },
    )

    # Stage 1: slow belt speed 0.1-0.3 m/s
    sample_belt_speed = EventTerm(
        func=task_mdp.sample_and_store_belt_speed,
        mode="reset",
        params={"high": 0.3, "low": 0.1, "key": "belt_speed"},
    )

    apply_conveyor = EventTerm(
        func=task_mdp.apply_conveyor_velocity_to_cubes,
        mode="interval",
        interval_range_s=(0.001, 0.001),
        params={
            "collection_name": CUBES_KEY,
            "belt_speed": None,
            "belt_speed_key": "belt_speed",
            "belt_axis": "x",
            "belt_height": BELT_HEIGHT_M,
            "belt_center_y": CONVEYOR_CENTER_Y,
            "belt_half_width": BELT_WIDTH_M * 0.5,
            "on_belt_height_tol": 0.10,
            "lift_disable_height": 0.15,
        },
    )


@configclass
class RewardsCfg:
    """7-phase continuous rewards (per research report Section 3).

    Event-based rewards (placement, miss, red-grab) are in the env class.
    Weight hierarchy per report:
      Transport (30) > Release (8) > Lift (5) = Height (5) > Grasp (3) >
      Re-orient (2) > Reach (1)
    """

    # ── 1. Reach (gated: only when not holding) ──────────────────────
    reaching_object = RewTerm(
        func=task_rew.cube_ee_distance,
        weight=1.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "collection_name": CUBES_KEY,
            "label": TARGET_LABEL,
            "std": 0.1,
        },
    )

    # ── 2. Grasp ─────────────────────────────────────────────────────
    grasping = RewTerm(
        func=task_rew.cube_grasp_reward,
        weight=3.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "collection_name": CUBES_KEY,
            "label": TARGET_LABEL,
            "std": 0.08,
        },
    )

    # ── 3. Lift ──────────────────────────────────────────────────────
    lifting_object = RewTerm(
        func=task_rew.cube_is_lifted,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "collection_name": CUBES_KEY,
            "label": TARGET_LABEL,
            "belt_height": BELT_HEIGHT_M,
            "minimal_height": 0.06,
            "max_distance": 0.15,
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "max_velocity": 1.0,
        },
    )

    # ── 3b. Height bonus ─────────────────────────────────────────────
    height_bonus = RewTerm(
        func=task_rew.cube_height_bonus,
        weight=5.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "collection_name": CUBES_KEY,
            "label": TARGET_LABEL,
            "belt_height": BELT_HEIGHT_M,
            "max_height": 0.30,
            "max_distance": 0.15,
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "max_velocity": 1.0,
        },
    )

    # ── 4. Transport with urgency ────────────────────────────────────
    goal_tracking = RewTerm(
        func=task_rew.approach_target_tanh,
        weight=30.0,
        params={
            "collection_name": CUBES_KEY,
            "label": TARGET_LABEL,
            "drum_name": "drum_target",
            "belt_height": BELT_HEIGHT_M,
            "std": 1.0,
            "lift_threshold": 0.02,
            "urgency_alpha": 4.0,
            "urgency_beta": 0.5,
            "belt_start_x": CONVEYOR_START_X,
            "belt_end_x": CONVEYOR_END_X,
        },
    )

    # ── 4b. Transport (fine) ─────────────────────────────────────────
    goal_tracking_fine = RewTerm(
        func=task_rew.approach_target_tanh,
        weight=5.0,
        params={
            "collection_name": CUBES_KEY,
            "label": TARGET_LABEL,
            "drum_name": "drum_target",
            "belt_height": BELT_HEIGHT_M,
            "std": 0.20,
            "lift_threshold": 0.02,
            "urgency_alpha": 4.0,
            "urgency_beta": 0.5,
            "belt_start_x": CONVEYOR_START_X,
            "belt_end_x": CONVEYOR_END_X,
        },
    )

    # ── 5. Release ───────────────────────────────────────────────────
    release = RewTerm(
        func=task_rew.release_above_target,
        weight=8.0,
        params={
            "collection_name": CUBES_KEY,
            "label": TARGET_LABEL,
            "drum_name": "drum_target",
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "belt_height": BELT_HEIGHT_M,
            "rim_clearance": 0.10,
            "drum_radius": 0.2735,
        },
    )

    # ── 6. Re-orient (return to belt after placing) ──────────────────
    reorient = RewTerm(
        func=task_rew.reorient_to_belt,
        weight=2.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]),
            "collection_name": CUBES_KEY,
            "label": TARGET_LABEL,
            "std": 0.3,
        },
    )

    # ── Regularisation ───────────────────────────────────────────────
    action_rate = RewTerm(func=shared_rew.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=shared_rew.joint_vel_l2_controlled,
        weight=-1e-4,
        params={
            "max_velocity": 10.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
        },
    )

    # ── Base velocity penalty ────────────────────────────────────────
    base_velocity = RewTerm(
        func=shared_rew.base_velocity_l2,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["base_y_joint", "base_z_joint"],
            ),
        },
    )

    # ── Arm utilisation bonus ────────────────────────────────────────
    arm_utilization = RewTerm(
        func=shared_rew.arm_velocity_bonus,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            ),
            "max_velocity": 5.0,
        },
    )

    # ── Belt contact penalty ─────────────────────────────────────────
    belt_contact = RewTerm(
        func=shared_rew.belt_contact_penalty,
        weight=-10.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "belt_height": BELT_HEIGHT_M,
            "margin": 0.0,
            "max_depth": 0.15,
        },
    )

    # ── Joint torque penalty ─────────────────────────────────────────
    joint_torque = RewTerm(
        func=shared_rew.joint_torque_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            ),
        },
    )

    # ── Conveyor penalty ─────────────────────────────────────────────
    cube_off_conveyor = RewTerm(
        func=task_rew.cubes_off_conveyor,
        weight=-5.0,
        params={
            "collection_name": CUBES_KEY,
            "bounds": _CONVEYOR_BOUNDS,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    all_cubes_passed = DoneTerm(
        func=task_mdp.all_active_passed_x,
        params={
            "collection_name": CUBES_KEY,
            "x_threshold": CONVEYOR_END_X + 0.35,
            "parking_x_threshold": 50.0,
        },
    )

    joint_vel_diverged = DoneTerm(
        func=shared_rew.joint_vel_out_of_limit,
        time_out=True,
        params={
            "max_velocity": 100.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
        },
    )

    belt_collision = DoneTerm(
        func=shared_rew.belt_collision_termination,
        time_out=True,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "belt_height": BELT_HEIGHT_M,
            "max_penetration": 0.20,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum: ramp up regularisation over training."""

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
class TensegrityCubeSortEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the tensegrity cube-sorting task (Stage 1)."""

    scene: CubeSortingSceneCfg = CubeSortingSceneCfg(num_envs=8192, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 8.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_stabilization = True
        self.sim.physx.gpu_max_rigid_contact_count = 2**22
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class TensegrityCubeSortEnvCfg_PLAY(TensegrityCubeSortEnvCfg):
    """Smaller configuration for evaluation / play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

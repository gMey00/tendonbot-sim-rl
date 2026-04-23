"""Robot-agnostic base environment configuration for the cube-sorting task.

7-phase reward structure on an active conveyor:
  1. Reach → 2. Grasp → 3. Lift → 4. Transport (urgency) → 5. Release
  → 6. Re-orient → (Event-based: placement bonus, miss penalty, red-grabbed)

Scene: unified collection of 16 labelled cubes on a moving belt, target drum
beside the belt.  Labels decide sorting logic (0 = target, 1 = distractor).

Robot-specific parameters (EE body, joint names, arm action) are filled
in by each variant config via ``CubeSortEnvCfg._set_robot_params()`` in
``__post_init__``, following the same pattern as the reach and cube_place tasks.
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
from .mdp import cube_set_obs as task_per_cube_obs
from .mdp import cube_sorting_mdp as task_mdp
from .mdp import per_cube_state as task_per_cube
from .mdp import rewards as task_rew
from ..shared import gripper_cfg as shared_rew


# ── Episode Length ────────────────────────────────────────────────────────
ROBOT_X: float = 0.15
PROCESSING_MARGIN_S: float = 30.0


def compute_episode_length(
    belt_speed: float,
    spawn_x_farthest: float,
    conveyor_end_x: float = CONVEYOR_END_X,
    processing_margin_s: float = PROCESSING_MARGIN_S,
) -> float:
    """Compute minimum episode duration so all cubes pass and the robot
    can handle the last one.

    episode = belt_transit_time + processing_margin
    belt_transit_time = (conveyor_end_x - spawn_x_farthest) / belt_speed
    """
    belt_transit_s = (conveyor_end_x - spawn_x_farthest) / belt_speed
    return belt_transit_s + processing_margin_s


CUBES_KEY = "cubes"

# ── Belt / Spawn ──────────────────────────────────────────────────────────
BELT_SPEED: float = 0.30  # m/s — static for all envs

_SPAWN_BOX = task_mdp.SpawnBox(
    x_range=(-1.50, -0.30),
    y_range=(-0.20, 0.20),
    z_range=(BELT_HEIGHT_M + 0.03, BELT_HEIGHT_M + 0.05),
    min_x_spacing=0.20,
)

_EPISODE_LENGTH_S: float = compute_episode_length(
    belt_speed=BELT_SPEED,
    spawn_x_farthest=_SPAWN_BOX.x_range[0],
)

_CONVEYOR_BOUNDS = shared_rew.ConveyorBounds(y_min=-0.4, y_max=0.4, z_min=0.70)
_DRUM_GEOM = shared_rew.BinCylinder(radius=0.2735, height=0.30)

# ── k-nearest observation config ──────────────────────────────────────────
K_TARGET: int = 4  # observe up to 4 nearest target cubes
K_DISTRACTOR: int = 2  # observe up to 2 nearest distractor cubes


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

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


@configclass
class ObservationsCfg:
    """R6 observation specification.

    Layout (in concatenation order — DO NOT REORDER, the
    ``CubeSetLayout`` slice math depends on it):

    1. Non-set features (``non_set_dim`` floats):
       proprio + EE + gripper + drum + belt/time + smooth contact
       aggregates + last action.
    2. Per-cube features (``N_max × S`` floats), flattened.
    3. Per-cube validity mask (``N_max`` floats).
    4. Privileged channels (``priv_dim`` floats): per-cube ``is_holding``
       + per-cube contact magnitude. The policy ignores these (it only
       slices [0 : F + S*N + N]); the value head reads them.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ── 1. Non-set features ──────────────────────────────────────
        # Proprio
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
        )
        # EE
        ee_pos_w = ObsTerm(
            func=shared_rew.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        ee_vel_w = ObsTerm(
            func=shared_rew.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
        )
        # Gripper smooth signals
        gripper_closure = ObsTerm(
            func=shared_rew.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        gripper_torque = ObsTerm(
            func=shared_rew.gripper_torque_residual,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        # Drum
        drum_rel = ObsTerm(
            func=shared_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
                "drum_name": "drum_target",
            },
        )
        # Task context
        belt_speed = ObsTerm(func=task_rew.belt_speed_obs)
        time_remaining = ObsTerm(func=task_rew.time_fraction_obs)
        # R6 smooth contact aggregates (per-pad force magnitude)
        pad_force_left = ObsTerm(
            func=task_per_cube_obs.gripper_pad_force_mag,
            params={"sensor_name": "contact_left"},
        )
        pad_force_right = ObsTerm(
            func=task_per_cube_obs.gripper_pad_force_mag,
            params={"sensor_name": "contact_right"},
        )
        # R7 placeholder — gOBJ register, fixed at zero until R7 ships
        gobj_sim = ObsTerm(func=task_per_cube_obs.gobj_sim)
        # Last action
        actions = ObsTerm(func=mdp.last_action)

        # ── 2. Per-cube set features (flattened) ─────────────────────
        cube_features_flat = ObsTerm(
            func=task_per_cube_obs.cube_features_flat,
            params={
                "cubes_collection_name": CUBES_KEY,
                "drum_name": "drum_target",
                "ee_body_name": MISSING,
                "belt_height": BELT_HEIGHT_M,
            },
        )

        # ── 3. Per-cube validity mask ────────────────────────────────
        cube_mask_flat = ObsTerm(
            func=task_per_cube_obs.cube_mask_flat,
            params={"cubes_collection_name": CUBES_KEY},
        )

        # ── 4. Privileged channels (critic-only — see CubeSetValue) ──
        per_cube_is_holding_priv = ObsTerm(func=task_per_cube_obs.per_cube_is_holding_priv)
        per_cube_contact_mag_priv = ObsTerm(func=task_per_cube_obs.per_cube_contact_mag_priv)

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

    # R4: clear per-cube Mealy automaton + placement dwell + grasp dwell
    reset_mdp_state = EventTerm(
        func=task_per_cube.reset_mdp_state,
        mode="reset",
    )

    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=MISSING,
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

    # Stage 1b: 1 green, 1 red — learn place + distractor awareness
    reset_cubes = EventTerm(
        func=task_mdp.reset_cubes,
        mode="reset",
        params={
            "collection_name": CUBES_KEY,
            "spawn_box": _SPAWN_BOX,
            "active_per_label": {str(TARGET_LABEL): 2, str(DISTRACTOR_LABEL): 1},
            "parking_pose": (100.0, 100.0, 1.0),
        },
    )

    # Static belt speed for all envs
    sample_belt_speed = EventTerm(
        func=task_mdp.sample_and_store_belt_speed,
        mode="reset",
        params={"high": BELT_SPEED, "low": BELT_SPEED, "key": "belt_speed"},
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
            "stop_when_target_in_reach": True,
            "target_label": TARGET_LABEL,
            "workspace_x_min": ROBOT_X + 0.10,
            "workspace_x_max": ROBOT_X + 0.20,
        },
    )


@configclass
class RewardsCfg:
    """R4 per-cube reward machine + regularisation.

    Replaces the legacy 11 goal-related per-step terms (reach/grasp/lift/
    transport/release/reorient/etc.) with a single per-cube reward term
    that derives all signals from the Mealy automaton in
    ``mdp/per_cube_state.py``.

    Event-based bonuses (placement, miss, red-grab) remain in the env class.
    """

    # ── R4: per-cube reward machine (replaces all 11 legacy goal terms) ──
    per_cube_reward = RewTerm(
        func=task_per_cube.per_cube_reward,
        weight=1.0,
        params={
            "cubes_collection_name": CUBES_KEY,
            "drum_name": "drum_target",
            "robot_name": "robot",
            "tcp_body_name": MISSING,
            "target_label": TARGET_LABEL,
            "z_belt": BELT_HEIGHT_M,
        },
    )

    # ── Regularisation ───────────────────────────────────────────────
    action_rate = RewTerm(func=shared_rew.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=shared_rew.joint_vel_l2_controlled,
        weight=-1e-4,
        params={
            "max_velocity": 10.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    # ── Arm utilisation bonus ────────────────────────────────────────
    arm_utilization = RewTerm(
        func=shared_rew.arm_velocity_bonus,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "max_velocity": 5.0,
        },
    )

    # ── Belt contact penalty ─────────────────────────────────────────
    belt_contact = RewTerm(
        func=shared_rew.belt_contact_penalty,
        weight=-10.0,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "belt_height": BELT_HEIGHT_M,
            "margin": 0.0,
            "max_depth": 0.15,
        },
    )

    # ── Joint torque penalty ─────────────────────────────────────────
    joint_torque = RewTerm(
        func=shared_rew.joint_torque_penalty,
        weight=-0.025,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
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

    all_targets_placed = DoneTerm(
        func=task_mdp.all_targets_placed,
        time_out=True,
    )

    joint_vel_diverged = DoneTerm(
        func=shared_rew.joint_vel_out_of_limit,
        time_out=True,
        params={
            "max_velocity": 100.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    belt_collision = DoneTerm(
        func=shared_rew.belt_collision_termination,
        time_out=True,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "belt_height": BELT_HEIGHT_M,
            "max_penetration": 0.20,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum: ramp up regularisation over training."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -2e-3, "num_steps": 75000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-3, "num_steps": 75000},
    )


##
# Environment configuration
##


@configclass
class CubeSortEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-agnostic base configuration for the cube-sorting task.

    Each robot variant inherits from this class and calls
    ``_set_robot_params()`` inside ``__post_init__`` to fill in
    ``MISSING`` body/joint names and set the arm action term.
    """

    scene: CubeSortingSceneCfg = CubeSortingSceneCfg(num_envs=4096, env_spacing=5.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = _EPISODE_LENGTH_S
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
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
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

        # -- Observations (R6 layout) --
        obs = self.observations.policy
        obs.joint_pos_rel.params["asset_cfg"].joint_names = controlled_joints
        obs.joint_vel_rel.params["asset_cfg"].joint_names = controlled_joints
        obs.ee_pos_w.params["asset_cfg"].body_names = grasp
        obs.ee_vel_w.params["asset_cfg"].body_names = grasp
        obs.drum_rel.params["ee_cfg"].body_names = grasp
        obs.cube_features_flat.params["ee_body_name"] = ee_body

        # -- Rewards --
        rew = self.rewards
        rew.per_cube_reward.params["tcp_body_name"] = ee_body
        rew.belt_contact.params["ee_cfg"].body_names = grasp
        rew.joint_vel.params["asset_cfg"].joint_names = controlled_joints
        rew.arm_utilization.params["asset_cfg"].joint_names = arm_joints
        rew.joint_torque.params["asset_cfg"].joint_names = arm_joints

        # -- Events --
        self.events.reset_arm.params["asset_cfg"].joint_names = arm_joints

        # -- Terminations --
        self.terminations.joint_vel_diverged.params["asset_cfg"].joint_names = controlled_joints
        self.terminations.belt_collision.params["ee_cfg"].body_names = grasp

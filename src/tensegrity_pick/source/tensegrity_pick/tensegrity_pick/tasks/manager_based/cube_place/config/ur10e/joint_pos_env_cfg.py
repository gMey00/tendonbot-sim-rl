# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e cube place environment configuration.

Drop-in replacement for the tensegrity place task using a UR10e + Robotiq 2F-140.
Sequential reward structure is identical to enable fair comparison
(PG-6: Tendon Wrist Transfer Experiment).

Key differences from the tensegrity variant:
  - 6-DOF revolute arm (no prismatic base) → single JointPositionActionCfg
  - No base_velocity penalty (fixed-base robot)
  - arm_utilization covers all 6 arm joints
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from tensegrity_pick.robots import UR10E_GRIPPER_CFG
from tensegrity_pick.tasks.manager_based.cube_place import mdp
from tensegrity_pick.tasks.manager_based.cube_place.mdp import rewards as task_rew
from tensegrity_pick.tasks.manager_based.cube_place.place_scene_cfg import (
    PlaceSceneCfg,
    CONVEYOR_SURFACE_HEIGHT_M,
    CUBE_SIZE_M,
    SPAWN_HEIGHT_M,
)
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import UR10E_MOUNT_HEIGHT_M

# ── Constants ─────────────────────────────────────────────────────────────

UR10E_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

CONTROLLED_JOINT_NAMES = UR10E_ARM_JOINT_NAMES + ["finger_joint"]

EE_LINK = "robotiq_base_link"
GRASP_BODIES = [EE_LINK]

_SPAWN_BOX = task_rew.SpawnBox(
    x_range=(0.10, 0.20),
    y_range=(-0.10, 0.10),
    z_range=(CONVEYOR_SURFACE_HEIGHT_M + 0.03, CONVEYOR_SURFACE_HEIGHT_M + 0.05),
)

_BIN_GEOM = task_rew.BinCylinder(radius=0.547 * 0.5, height=0.30)

_CONVEYOR_BOUNDS = task_rew.ConveyorBounds(y_min=-0.4, y_max=0.4, z_min=0.70)


# ── Scene ─────────────────────────────────────────────────────────────────

_UR10E_PLACE_INITIAL_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": 0.0,
    "elbow_joint": 0.0,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_outer_finger_joint": 0.0,
    "right_outer_finger_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "left_inner_finger_pad_joint": 0.0,
    "right_inner_finger_pad_joint": 0.0,
}


@configclass
class UR10ePlaceSceneCfg(PlaceSceneCfg):
    """Place scene with UR10e + Robotiq 2F-140 instead of tensegrity."""

    robot = UR10E_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.15, 0.0, UR10E_MOUNT_HEIGHT_M),
            # 180° about Y to flip upside-down for ceiling mount.
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos=_UR10E_PLACE_INITIAL_JOINT_POS,
        ),
    )


# ── Actions ───────────────────────────────────────────────────────────────

@configclass
class UR10ePlaceActionsCfg:
    """Joint-position delta actions (6 arm joints + binary gripper)."""

    arm_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=UR10E_ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


# ── Observations ──────────────────────────────────────────────────────────

@configclass
class UR10ePlaceObservationsCfg:
    """Observation specification — UR10e controlled joints."""

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


# ── Events ────────────────────────────────────────────────────────────────

@configclass
class UR10eEventsCfg:
    """Reset events for the UR10e place task."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES),
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


# ── Rewards ───────────────────────────────────────────────────────────────

@configclass
class UR10eRewardsCfg:
    """Reward terms — identical structure/weights to tensegrity, UR10e joints.

    Omits base_velocity (no prismatic base).  arm_utilization covers all
    6 revolute joints.
    """

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

    green_in_target = RewTerm(
        func=task_rew.green_cube_in_target,
        weight=100.0,
        params={
            "green_name": "green_cube",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )

    red_in_target = RewTerm(
        func=task_rew.red_cube_in_target,
        weight=-12.0,
        params={
            "red_name": "red_cube",
            "drum_name": "drum_target",
            "bin_geom": _BIN_GEOM,
        },
    )

    action_rate = RewTerm(
        func=task_rew.action_rate_l2,
        weight=-1e-4,
    )
    joint_vel = RewTerm(
        func=task_rew.joint_vel_l2_controlled,
        weight=-1e-4,
        params={
            "max_velocity": 10.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
        },
    )

    # No base_velocity penalty — UR10e is a fixed-base robot.

    arm_utilization = RewTerm(
        func=task_rew.arm_velocity_bonus,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES),
            "max_velocity": 5.0,
        },
    )

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

    joint_torque = RewTerm(
        func=task_rew.joint_torque_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UR10E_ARM_JOINT_NAMES),
        },
    )

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

    cube_off_conveyor = RewTerm(
        func=task_rew.cube_off_conveyor_penalty,
        weight=-5.0,
        params={
            "green_name": "green_cube",
            "red_name": "red_cube",
            "bounds": _CONVEYOR_BOUNDS,
        },
    )


# ── Terminations ──────────────────────────────────────────────────────────

@configclass
class UR10eTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    joint_vel_diverged = DoneTerm(
        func=task_rew.joint_vel_out_of_limit,
        time_out=True,
        params={
            "max_velocity": 100.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES),
        },
    )

    belt_collision = DoneTerm(
        func=task_rew.belt_collision_termination,
        time_out=True,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
            "belt_height": CONVEYOR_SURFACE_HEIGHT_M,
            "max_penetration": 0.20,
        },
    )


# ── Curriculum ────────────────────────────────────────────────────────────

@configclass
class UR10eCurriculumCfg:

    activate_red = CurrTerm(
        func=task_rew.activate_red_cube_curriculum,
        params={"num_steps": 100000},
    )

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -2e-3, "num_steps": 200000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -2e-3, "num_steps": 200000},
    )


# ── Environment config ────────────────────────────────────────────────────

@configclass
class UR10ePlaceEnvCfg(ManagerBasedRLEnvCfg):
    """UR10e cube-placement task — fair comparison baseline."""

    scene: UR10ePlaceSceneCfg = UR10ePlaceSceneCfg(num_envs=8192, env_spacing=5.0)
    actions: UR10ePlaceActionsCfg = UR10ePlaceActionsCfg()
    observations: UR10ePlaceObservationsCfg = UR10ePlaceObservationsCfg()
    terminations: UR10eTerminationsCfg = UR10eTerminationsCfg()
    rewards: UR10eRewardsCfg = UR10eRewardsCfg()
    events: UR10eEventsCfg = UR10eEventsCfg()
    curriculum: UR10eCurriculumCfg = UR10eCurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 0.01
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
class UR10ePlaceEnvCfg_PLAY(UR10ePlaceEnvCfg):
    """Smaller configuration for evaluation / play."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.curriculum.activate_red.params["num_steps"] = 0

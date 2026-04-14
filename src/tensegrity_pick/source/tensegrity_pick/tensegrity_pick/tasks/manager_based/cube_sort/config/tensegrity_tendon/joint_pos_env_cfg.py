# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensegrity tendon-driven cube-sorting environment configurations.

The arm's 3 revolute joints (elbow, wrist_y, wrist_x) are actuated by
5 tendons instead of implicit PD drives.  Everything else (base,
gripper, rewards, terminations, curriculum) is identical to the PD
variant.

Inherits from the tensegrity PD variant and overrides:
  - robot asset (tendon-actuated 5-DOF)
  - actions (TendonEffortActionCfg for arm)
  - observations (adds tendon observations)

Also includes the physical tendon variant (body-force elbow).
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots import (
    TENS_5DOF_GRIPPER_TENDON_CFG,
    TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG,
    TendonEffortActionCfg,
    PhysicalTendonEffortActionCfg,
)
from tensegrity_pick.robots.tendon_actuator import (
    DEFAULT_JACOBIAN_TRANSPOSE,
    ELBOW_TENDON_FOREARM_OFFSETS,
    ELBOW_TENDON_ROOT_OFFSETS,
    WRIST_JACOBIAN_TRANSPOSE,
)
from tensegrity_pick.tasks.manager_based.cube_sort import mdp
from tensegrity_pick.tasks.manager_based.cube_sort.mdp import rewards as task_rew
from tensegrity_pick.tasks.manager_based.cube_sort.config.tensegrity.joint_pos_env_cfg import (
    TensegrityCubeSortEnvCfg,
    CONTROLLED_JOINT_NAMES,
    GRASP_BODIES,
    BASE_JOINT_NAMES,
    ARM_JOINT_NAMES,
    EE_LINK,
)
from tensegrity_pick.tasks.manager_based.cube_sort.cube_sorting_scene_cfg import (
    CUBES_KEY,
    TARGET_LABEL,
    DISTRACTOR_LABEL,
)
from tensegrity_pick.tasks.manager_based.shared import gripper_cfg as shared_rew
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import TENSEGRITY_MOUNT_HEIGHT_M

_TENDON_INITIAL_JOINT_POS = {
    "base_y_joint": 0.0,
    "base_z_joint": 0.0,
    "elbow_joint": 0.0,
    "wrist_y_joint": 0.0,
    "wrist_x_joint": 0.0,
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
class TendonActionsCfg:
    """Mixed action space: implicit PD for base, tendon efforts for arm,
    binary position for gripper."""

    base_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=BASE_JOINT_NAMES,
        scale=0.50,
        use_default_offset=True,
        clip={"base_y_joint": (-0.5, 0.5), "base_z_joint": (-0.50, 0.0)},
    )

    arm_tendon = TendonEffortActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
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
            func=shared_rew.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
        ee_vel_w = ObsTerm(
            func=shared_rew.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
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
        gripper_closure = ObsTerm(
            func=shared_rew.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        gripper_torque = ObsTerm(
            func=shared_rew.gripper_torque_residual,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
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
        drum_rel = ObsTerm(
            func=shared_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "drum_name": "drum_target",
            },
        )
        belt_speed = ObsTerm(func=task_rew.belt_speed_obs)
        time_remaining = ObsTerm(func=task_rew.time_fraction_obs)
        placed_count = ObsTerm(func=task_rew.targets_placed_obs)
        missed_count = ObsTerm(func=task_rew.targets_missed_obs)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TendonCubeSortEnvCfg(TensegrityCubeSortEnvCfg):
    """Tensegrity cube-sort task driven by tendon tensions instead of
    joint position deltas."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Override robot with tendon-actuated variant
        self.scene.robot = TENS_5DOF_GRIPPER_TENDON_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, TENSEGRITY_MOUNT_HEIGHT_M),
                joint_pos=_TENDON_INITIAL_JOINT_POS,
            ),
        )

        # Override actions and observations
        self.actions = TendonActionsCfg()
        self.observations = TendonObservationsCfg()


@configclass
class TendonCubeSortEnvCfg_PLAY(TendonCubeSortEnvCfg):
    """Smaller evaluation / play configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0


# ── Physical tendon model ─────────────────────────────────────────────────

CONTROLLED_JOINT_NAMES_PHYSICAL = [
    "base_y_joint", "base_z_joint",
    "rod_left_joint", "rod_right_joint", "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
    "finger_joint",
]

ARM_JOINT_NAMES_PHYSICAL = [
    "rod_left_joint", "rod_right_joint", "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
]

_PHYSICAL_INITIAL_JOINT_POS = {
    "base_y_joint": 0.0,
    "base_z_joint": 0.0,
    "rod_left_joint": 0.0,
    "rod_right_joint": 0.0,
    "coupler_left_joint": 0.0,
    "wrist_y_joint": 0.0,
    "wrist_x_joint": 0.0,
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
class PhysicalTendonActionsCfg:
    """Mixed action space: implicit PD for base, body-force tendons for arm,
    binary position for gripper."""

    base_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=BASE_JOINT_NAMES,
        scale=0.50,
        use_default_offset=True,
        clip={"base_y_joint": (-0.5, 0.5), "base_z_joint": (-0.50, 0.0)},
    )

    arm_tendon = PhysicalTendonEffortActionCfg(
        asset_name="robot",
        joint_names=["wrist_y_joint", "wrist_x_joint"],
        num_tendons=5,
        max_tension=500.0,
        root_body_name="root_link",
        forearm_body_name="forearm_link",
        elbow_tendon_root_offsets=ELBOW_TENDON_ROOT_OFFSETS,
        elbow_tendon_forearm_offsets=ELBOW_TENDON_FOREARM_OFFSETS,
        wrist_jacobian_transpose=WRIST_JACOBIAN_TRANSPOSE,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


@configclass
class PhysicalTendonObservationsCfg:
    """Observations for the physical tendon model."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES_PHYSICAL)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES_PHYSICAL)},
        )
        ee_pos_w = ObsTerm(
            func=shared_rew.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
        ee_vel_w = ObsTerm(
            func=shared_rew.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
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
        gripper_closure = ObsTerm(
            func=shared_rew.gripper_closure,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
        gripper_torque = ObsTerm(
            func=shared_rew.gripper_torque_residual,
            params={"finger_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"])},
        )
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
        drum_rel = ObsTerm(
            func=shared_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "drum_name": "drum_target",
            },
        )
        belt_speed = ObsTerm(func=task_rew.belt_speed_obs)
        time_remaining = ObsTerm(func=task_rew.time_fraction_obs)
        placed_count = ObsTerm(func=task_rew.targets_placed_obs)
        missed_count = ObsTerm(func=task_rew.targets_missed_obs)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PhysicalTendonCubeSortEnvCfg(TensegrityCubeSortEnvCfg):
    """Tensegrity cube-sort task driven by physical body-force tendons."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Override robot with physical tendon variant
        self.scene.robot = TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, TENSEGRITY_MOUNT_HEIGHT_M),
                joint_pos=_PHYSICAL_INITIAL_JOINT_POS,
            ),
        )

        # Override actions and observations
        self.actions = PhysicalTendonActionsCfg()
        self.observations = PhysicalTendonObservationsCfg()

        # Re-set robot params for physical joint names
        self._set_robot_params(
            ee_body=EE_LINK,
            controlled_joints=CONTROLLED_JOINT_NAMES_PHYSICAL,
            arm_joints=ARM_JOINT_NAMES_PHYSICAL,
        )


@configclass
class PhysicalTendonCubeSortEnvCfg_PLAY(PhysicalTendonCubeSortEnvCfg):
    """Smaller evaluation / play configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0

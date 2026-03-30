# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensegrity *physical* tendon-driven cube-placement environment configs.

The physical model replaces the single ``elbow_joint`` with a four-bar
antiparallelogram linkage driven by body-force tendons.  Wrist tendons
still use the constant Jacobian-transpose mapping.

NOTE: The 5-DOF physical USD must be assembled in Robot Assembler before
these configs can be instantiated.

Inherits from the tensegrity PD variant and overrides robot, actions
and observations for the physical tendon model.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots import TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG, PhysicalTendonEffortActionCfg
from tensegrity_pick.robots.tendon_actuator import (
    ELBOW_TENDON_FOREARM_OFFSETS,
    ELBOW_TENDON_ROOT_OFFSETS,
    WRIST_JACOBIAN_TRANSPOSE,
)
from tensegrity_pick.tasks.manager_based.cube_place import mdp
from tensegrity_pick.tasks.manager_based.cube_place.mdp import rewards as task_rew
from tensegrity_pick.tasks.manager_based.cube_place.config.tensegrity.joint_pos_env_cfg import (
    TensegrityPlaceEnvCfg,
    GRASP_BODIES,
    BASE_JOINT_NAMES,
    EE_LINK,
)
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import TENSEGRITY_MOUNT_HEIGHT_M

# Physical model controlled joint names (linkage joints replace elbow_joint)
CONTROLLED_JOINT_NAMES_PHYSICAL = [
    "base_y_joint", "base_z_joint",
    "rod_left_joint", "rod_right_joint", "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
    "finger_joint",
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

ARM_JOINT_NAMES_PHYSICAL = [
    "rod_left_joint", "rod_right_joint", "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
]


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
        root_offsets=ELBOW_TENDON_ROOT_OFFSETS,
        forearm_offsets=ELBOW_TENDON_FOREARM_OFFSETS,
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
    """Observations for the physical tendon model.

    Identical structure to elbow_approx but with physical joint names in
    the observation filters."""

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


@configclass
class TensegrityPlacePhysicalTendonEnvCfg(TensegrityPlaceEnvCfg):
    """Tensegrity place task driven by physical body-force tendons.

    The elbow is actuated by cable forces at the physical attachment
    points of the four-bar linkage.  The wrist uses J^T as before.
    """

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
class TensegrityPlacePhysicalTendonEnvCfg_PLAY(TensegrityPlacePhysicalTendonEnvCfg):
    """Smaller evaluation / play configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.curriculum.activate_red.params["num_steps"] = 0

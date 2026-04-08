# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensegrity tendon-driven cube-placement environment configurations.

The arm's 3 revolute joints (elbow, wrist_y, wrist_x) are actuated by
5 tendons instead of implicit PD drives.  Everything else (base,
gripper, rewards, terminations, curriculum) is identical to the PD
variant.

Inherits from the tensegrity PD variant and overrides:
  - robot asset (tendon-actuated 5-DOF)
  - actions (TendonEffortActionCfg for arm)
  - observations (adds tendon observations)
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots import TENS_5DOF_GRIPPER_TENDON_CFG, TendonEffortActionCfg
from tensegrity_pick.robots.tendon_actuator import DEFAULT_JACOBIAN_TRANSPOSE
from tensegrity_pick.tasks.manager_based.cube_place import mdp
from tensegrity_pick.tasks.manager_based.cube_place.mdp import rewards as task_rew
from tensegrity_pick.tasks.manager_based.cube_place.config.tensegrity.joint_pos_env_cfg import (
    TensegrityPlaceEnvCfg,
    CONTROLLED_JOINT_NAMES,
    GRASP_BODIES,
    BASE_JOINT_NAMES,
    ARM_JOINT_NAMES,
    EE_LINK,
)
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

        # Task completion flag — lets the policy know when to return to neutral
        was_placed = ObsTerm(func=task_rew.was_placed_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TensegrityPlaceTendonEnvCfg(TensegrityPlaceEnvCfg):
    """Tensegrity place task driven by tendon tensions instead of joint
    position deltas.

    The arm's 3 revolute joints (elbow, wrist_y, wrist_x) are actuated by
    5 tendons: 2 antagonistic for the elbow and 3 at 120 deg for the 2-DOF
    wrist.  Everything else (base, gripper, rewards, terminations,
    curriculum) is identical to the base task.
    """

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
class TensegrityPlaceTendonEnvCfg_PLAY(TensegrityPlaceTendonEnvCfg):
    """Smaller evaluation / play configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.curriculum.activate_red.params["num_steps"] = 0

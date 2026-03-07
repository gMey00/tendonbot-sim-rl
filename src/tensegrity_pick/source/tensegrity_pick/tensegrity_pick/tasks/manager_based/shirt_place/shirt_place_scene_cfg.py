# shirt_place_scene_cfg.py
#
# Scene for the shirt placement task.
# Extends ProjBaseSceneCfg (ground, lights, robot, conveyor, drum) and
# replaces cubes with cloth-simulated T-shirts.
#
# TODO: Replace RigidObjectCfg placeholders with DeformableObjectCfg
#       (or IsaacSim cloth prim) once cloth simulation is integrated.

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..shared.proj_base_scene_cfg import (
    ProjBaseSceneCfg,
    CONVEYOR_SURFACE_HEIGHT_M,
    ROBOT_MOUNT_HEIGHT_M,
)
from tensegrity_pick.robots import TENS_5DOF_GRIPPER_CFG

PLACE_MOUNT_HEIGHT_M = ROBOT_MOUNT_HEIGHT_M
SPAWN_HEIGHT_M = CONVEYOR_SURFACE_HEIGHT_M + 0.03

_INITIAL_JOINT_POS = {
    "base_y_joint": 0.0,
    "base_z_joint": -0.25,
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
class ShirtPlaceSceneCfg(ProjBaseSceneCfg):
    """Base scene + one shirt for the shirt place task.

    TODO: Add cloth-simulated shirt as a deformable object.
    """

    robot = TENS_5DOF_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.15, 0.0, PLACE_MOUNT_HEIGHT_M),
            joint_pos=_INITIAL_JOINT_POS,
        ),
    )

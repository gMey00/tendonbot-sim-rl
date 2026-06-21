"""UR10 (classic 6-DOF) + Robotiq 2F-140 robot configuration.

Wraps the pre-assembled ``res/UR10/UR10_Robotiq2F140.usd`` which references the
official Isaac UR10 arm (``UniversalRobots/ur10/ur10.usd``) and the Robotiq
2F-140 gripper (``Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd``).

PD gains
--------
Arm drive stiffness/damping/maxForce are transcribed from the joint
``PhysicsDriveAPI`` values actually authored in the referenced UR10 USD
(verified by loading the composed stage):

    shoulder_pan_joint  : stiff=153634.5  damp=5.121   maxForce=330
    shoulder_lift_joint : stiff=169797.3  damp=5.660   maxForce=330
    elbow_joint         : stiff=831589.3  damp=27.720  maxForce=150
    wrist_1_joint       : stiff=165646.5  damp=5.522   maxForce=56
    wrist_2_joint       : stiff=157650.2  damp=5.255   maxForce=56
    wrist_3_joint       : stiff=155870.9  damp=5.196   maxForce=56

Gripper actuator groups are imported from the shared centralized config.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .robotiq_2f140_gripper_cfg import get_gripper_actuators

PROJ_ASSETS_PATH = os.path.join(
    os.environ.get("PROJECT_PATH", "/home/robot/studentische-arbeiten"), "res"
)

TARGET_LINK_NAME = "robotiq_base_link"
CONTROLLED_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR10_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJ_ASSETS_PATH}/UR10/UR10_Robotiq2F140.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Arm joints — "ready" pose (arm up/forward, elbow bent).
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 1.5708,
            "wrist_1_joint": -1.5708,
            "wrist_2_joint": 1.5708,
            "wrist_3_joint": 0.0,
            # Gripper — open state (same Robotiq 2F-140 as tensegrity/UR10e).
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "left_inner_finger_pad_joint": 0.0,
            "right_inner_finger_pad_joint": 0.0,
        },
        joint_vel={},
    ),
    actuators={
        # Arm: per-joint gains transcribed from the UR10 USD PhysicsDriveAPI.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            ],
            effort_limit_sim={
                "shoulder_pan_joint": 330.0,
                "shoulder_lift_joint": 330.0,
                "elbow_joint": 150.0,
                "wrist_1_joint": 56.0,
                "wrist_2_joint": 56.0,
                "wrist_3_joint": 56.0,
            },
            stiffness={
                "shoulder_pan_joint": 153634.5,
                "shoulder_lift_joint": 169797.3,
                "elbow_joint": 831589.3,
                "wrist_1_joint": 165646.5,
                "wrist_2_joint": 157650.2,
                "wrist_3_joint": 155870.9,
            },
            damping={
                "shoulder_pan_joint": 5.121,
                "shoulder_lift_joint": 5.660,
                "elbow_joint": 27.720,
                "wrist_1_joint": 5.522,
                "wrist_2_joint": 5.255,
                "wrist_3_joint": 5.196,
            },
        ),
        # Gripper: imported from shared centralized config.
        **get_gripper_actuators("original"),
    },
)

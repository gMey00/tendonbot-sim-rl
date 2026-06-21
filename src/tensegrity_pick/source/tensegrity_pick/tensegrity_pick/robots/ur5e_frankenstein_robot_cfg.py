"""UR5e (6-DOF) + Tensegrity 2-DOF Wrist + Robotiq 2F-140 configuration.

"Frankenstein" variant: the official Isaac UR5e arm with the tensegrity
2-DOF wrist element (``tensegrity_twodof_wrist``) spliced between the arm
flange and the Robotiq 2F-140 gripper.  Pre-assembled in
``res/UR5E/UR5EFrankenstein_Robotiq2F140.usd``.

PD gains
--------
Arm drive gains are transcribed from the UR5e USD PhysicsDriveAPI (identical
to ``ur5e_robot_cfg.py``; see that file for the per-joint values).

The tensegrity wrist joints (``wrist_x_joint``, ``wrist_y_joint``) are shipped
**passive** in the USD (drive stiffness=0, damping=0, maxForce=0) with
revolute limits of ±50° (±0.873 rad).  Here they are given the same active
gains as the tensegrity manipulator's tendon wrist so the element is
controllable; FK workspace sampling teleports them within their USD limits.
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
    "wrist_x_joint",
    "wrist_y_joint",
]

UR5E_FRANKENSTEIN_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJ_ASSETS_PATH}/UR5E/UR5EFrankenstein_Robotiq2F140.usd",
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
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 1.5708,
            "wrist_1_joint": -1.5708,
            "wrist_2_joint": 1.5708,
            "wrist_3_joint": 0.0,
            # Tensegrity wrist — neutral.
            "wrist_x_joint": 0.0,
            "wrist_y_joint": 0.0,
            # Gripper — open state.
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
        # Arm: per-joint gains transcribed from the UR5e USD PhysicsDriveAPI.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            ],
            effort_limit_sim={
                "shoulder_pan_joint": 150.0,
                "shoulder_lift_joint": 150.0,
                "elbow_joint": 150.0,
                "wrist_1_joint": 28.0,
                "wrist_2_joint": 28.0,
                "wrist_3_joint": 28.0,
            },
            stiffness={
                "shoulder_pan_joint": 9400.5,
                "shoulder_lift_joint": 10020.9,
                "elbow_joint": 10230.2,
                "wrist_1_joint": 3940.6,
                "wrist_2_joint": 3940.6,
                "wrist_3_joint": 1000.1,
            },
            damping={
                "shoulder_pan_joint": 0.378,
                "shoulder_lift_joint": 0.412,
                "elbow_joint": 4.093,
                "wrist_1_joint": 1.579,
                "wrist_2_joint": 0.061,
                "wrist_3_joint": 0.004,
            },
        ),
        # Tensegrity wrist (USD ships passive 0/0/0) — active gains mirror the
        # tensegrity manipulator's tendon wrist [Klein 2023].
        "tendon_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_x_joint", "wrist_y_joint"],
            effort_limit_sim=3.5,
            velocity_limit_sim=9.0,
            stiffness=400.0,
            damping=20.0,
        ),
        # Gripper: imported from shared centralized config.
        **get_gripper_actuators("original"),
    },
)

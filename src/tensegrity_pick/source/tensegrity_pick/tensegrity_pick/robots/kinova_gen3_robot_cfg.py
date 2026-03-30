"""Kinova Gen3 (7-DOF) + Robotiq 2F-140 robot configuration.

Provides a Kinova Gen3 7-DOF arm with Robotiq 2F-140 gripper for
baseline comparison against the tensegrity manipulator and UR10e.

The combined USD is pre-assembled and lives at
``res/KinovaGen3/KinovaGen3_Robotiq2F140.usd``.

Actuator gains are based on community-validated configs from Isaac Lab
Discussion #4226 (Isaac Sim 5.1.0) and the louislelay/kinova_isaaclab_sim2real
repository (sim2real validated on real Kinova Gen3 hardware).

Joint specifications from the official Kinova URDF V12:
- Joints 1-4 (large actuators): 39 Nm effort, 1.3963 rad/s velocity
- Joints 5-7 (small actuators): 9 Nm effort, 1.2218 rad/s velocity

References:
- Kinova Gen3 URDF: github.com/Kinovarobotics/ros_kortex
- Isaac Lab Discussion #4226: Kinova Gen3 lift task config
- louislelay/kinova_isaaclab_sim2real: sim2real validated pipeline
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .robotiq_2f140_gripper_cfg import get_gripper_actuators

PROJ_ASSETS_PATH = "/home/robot/studentische-arbeiten/res"

TARGET_LINK_NAME = "end_effector_link"
CONTROLLED_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

KINOVA_GEN3_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJ_ASSETS_PATH}/KinovaGen3/KinovaGen3_Robotiq2F140.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_linear_velocity=10.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Arm joints — home pose (gripper pointing down for table mount).
            "joint_1": 0.0,
            "joint_2": 0.2618,
            "joint_3": 3.1416,
            "joint_4": -2.2689,
            "joint_5": 0.0,
            "joint_6": 0.9599,
            "joint_7": 3.1416,
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
        # ── Arm: split by actuator size ──
        # Large actuators (joints 1-2): shoulder
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_1", "joint_2"],
            effort_limit_sim=39.0,
            velocity_limit_sim=1.3963,
            stiffness=5000.0,
            damping=220.0,
        ),
        # Large actuators (joints 3-4): elbow
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint_3", "joint_4"],
            effort_limit_sim=39.0,
            velocity_limit_sim=1.3963,
            stiffness=2500.0,
            damping=160.0,
        ),
        # Small actuators (joints 5-7): wrist
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_5", "joint_6", "joint_7"],
            effort_limit_sim=9.0,
            velocity_limit_sim=1.2218,
            stiffness=2500.0,
            damping=160.0,
        ),
        # ── Gripper: imported from shared centralized config ──
        **get_gripper_actuators("original"),
    },
)

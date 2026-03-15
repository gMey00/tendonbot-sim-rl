"""Standard (position/velocity-controlled) tensegrity robot configurations.

Provides ready-to-use :class:`ArticulationCfg` instances for the 3-DOF
manipulator and 5-DOF manipulator-with-gripper.  All joints use
:class:`ImplicitActuatorCfg` (built-in PhysX PD drives).

Configs
-------
* ``TENS_3DOF_CFG`` — 3-DOF arm only (elbow + 2-DOF wrist)
* ``TENS_5DOF_GRIPPER_CFG`` — 2-DOF linear base + 3-DOF arm + Robotiq gripper
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


PROJ_ASSETS_PATH = "/home/robot/studentische-arbeiten/res"

TARGET_LINK_NAME_3DOF = "tool_link"
CONTROLLED_JOINT_NAMES_3DOF = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]
TARGET_LINK_NAME_5DOF = "tool_link_0"
CONTROLLED_JOINT_NAMES_5DOF = ["base_y_joint", "base_z_joint", "elbow_joint", "wrist_y_joint", "wrist_x_joint"]


TENS_3DOF_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJ_ASSETS_PATH}/Tensegrity/threedof_manipulator/threedof_manipulator.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "elbow_joint": 0.0,
            "wrist_y_joint": 0.0,
            "wrist_x_joint": 0.0,
        },
        joint_vel={},
    ),

    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            effort_limit_sim=40.0,
            velocity_limit_sim=2.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
)

TENS_5DOF_GRIPPER_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJ_ASSETS_PATH}/Tensegrity/fivedof_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base_y_joint": 0.0,
            "base_z_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_y_joint": 0.0,
            "wrist_x_joint": 0.0,
            # All gripper joints at 0 (open state).
            # USD drive targets are restored by a reset event.
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
        # Gains validated by tune_pd_gains.py on 2026-02-28.
        # See doc/pd_tuning_results.md for rationale and measurements.
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_y_joint", "base_z_joint"],
            effort_limit_sim=800.0,
            velocity_limit_sim=5.0,
            stiffness=8000.0,
            damping=800.0,
        ),
        # Arm: D=120→20 (ζ 6.4→1.1, settle 1.66s→0.30s).
        # Klein (2023) target: ζ≈1.0, settle 0.1–0.3s, overshoot<5%.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            effort_limit_sim=40.0,
            velocity_limit_sim=2.0,
            stiffness=400.0,
            damping=20.0,
        ),
        # Gripper: three-group config matching IsaacLab's reference
        # Robotiq 2F-140 (isaaclab_assets/robots/universal_robots.py).
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=11.25,
            damping=0.1,
            friction=0.0,
            armature=0.0,
        ),
        # Inner finger joints: strong spring keeps pads parallel.
        # K=10, D=0.05 from IsaacLab gear-assembly reference
        # (default K=0.2 is too weak to resist four-bar linkage forces).
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_joint",
                "right_inner_finger_joint",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=0.05,
            friction=0.0,
            armature=0.0,
        ),
        # Passive four-bar linkage: K=0, D=0 — follows drive mechanically.
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "right_outer_knuckle_joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)

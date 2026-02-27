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
            effort_limit=40.0,
            velocity_limit_sim=2.0,
            stiffness=400.0,
            damping=120.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[],
            stiffness=0.0,
            damping=0.0,
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
            "finger_joint": 0.0,
        },
        joint_vel={},
    ),

    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_y_joint", "base_z_joint"],
            effort_limit=200.0,
            velocity_limit_sim=5.0,
            stiffness=8000.0,
            damping=800.0,
        ),

        "arm": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            effort_limit=40.0,
            velocity_limit_sim=2.0,
            stiffness=400.0,
            damping=120.0,
        ),

        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit=50.0,
            velocity_limit_sim=5.0,
            stiffness=100.0,
            damping=20.0,
        ),

        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_outer_knuckle_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "left_inner_finger_joint",
                "right_inner_finger_joint",
                "left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint",
            ],
            effort_limit=200.0,
            velocity_limit_sim=5.0,
            stiffness=1000.0,
            damping=200.0,
        ),
    },
)

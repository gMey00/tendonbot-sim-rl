"""UR10e + Robotiq 2F-140 robot configuration.

Provides a UR10e robot with Robotiq gripper for baseline comparison
against the tensegrity manipulator (PG-6: Tendon Wrist Transfer).

The USD is loaded from the Nucleus server via the ``Robotiq_2f_140``
variant of the UR10e asset.  Arm actuator gains come from IsaacLab's
reference ``UR10e_CFG``; gripper actuator groups mirror the tensegrity
robot's Robotiq config (see ``tensegrity_robot_cfg.py``).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

TARGET_LINK_NAME = "robotiq_base_link"
CONTROLLED_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR10E_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        # Activate Robotiq 2F-140 gripper variant shipped with the USD.
        variants={"Gripper": "Robotiq_2f_140"},
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
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
            # Arm joints — straight-down home pose for ceiling mount.
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            # Gripper — open state (same joint names as tensegrity Robotiq).
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
        # Arm: per-group gains from IsaacLab's UR10e_CFG reference config.
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
        # Gripper: three-group config matching the tensegrity robot's Robotiq.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=11.25,
            damping=0.1,
            friction=0.0,
            armature=0.0,
        ),
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

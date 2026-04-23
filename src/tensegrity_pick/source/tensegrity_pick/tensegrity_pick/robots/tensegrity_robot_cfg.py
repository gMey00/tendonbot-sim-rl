"""Tensegrity robot configurations with approximated elbow (PD-controlled) .

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

from .robotiq_2f140_gripper_cfg import get_gripper_actuators


PROJ_ASSETS_PATH = "/home/robot/studentische-arbeiten/res"

TARGET_LINK_NAME_3DOF = "tool_link"
CONTROLLED_JOINT_NAMES_3DOF = ["elbow_joint", "wrist_x_joint", "wrist_y_joint"]
TARGET_LINK_NAME_5DOF = "tool_link_0"
CONTROLLED_JOINT_NAMES_5DOF = ["base_y_joint", "base_z_joint", "elbow_joint", "wrist_x_joint", "wrist_y_joint"]


""" 3DOF arm with PD-controlled elbow and wrist joints"""

TENS_3DOF_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJ_ASSETS_PATH}/Tensegrity/threedof_arm/tensegrity_threedof_arm_elbow_approx.usd",
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
        # Elbow: cable-derived max torque ≈ 34.8 N·m (160 N motor peak × 3:1 MA × 0.0725 m lever).
        # Klein (2023) §3.2.6 p.53, §3.5 p.65, spool radius 5 mm.
        # 35 N·m provides slight headroom.  Velocity matches physical spec
        # from Klein (2023) §4.2 p.77: 126°/s ≈ 2.2 rad/s (measured at 40° step).
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=2.5,
            stiffness=400.0,
            damping=20.0,
        ),
        # Wrist: cable-derived max torque ≈ 3.2 N·m (80 N motor continuous × 0.020 m
        # max lever × 2 contributing cables). Klein (2023) §3.2.2 p.42 (r = 20 mm),
        # §3.2.6 p.53. No pulley MA at wrist. 3.5 N·m provides headroom.
        # Velocity: 504°/s ≈ 8.8 rad/s measured (Klein 2023 §4.2 p.76).
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_x_joint", "wrist_y_joint"],
            effort_limit_sim=3.5,
            velocity_limit_sim=9.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
)

"3DOF arm with PD-controlled elbow and wrist joints (low resolution visualization)"

TENS_3DOF_LO_CFG = TENS_3DOF_CFG.replace(
    spawn=TENS_3DOF_CFG.spawn.replace(
        usd_path=f"{PROJ_ASSETS_PATH}/Tensegrity/threedof_arm/tensegrity_threedof_arm_elbow_approx_lo.usd"
    )
)

""" 5DOF arm + gripper with PD-controlled base and arm joints, plus Robotiq gripper"""

TENS_5DOF_GRIPPER_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJ_ASSETS_PATH}/Tensegrity/fivedof_manipulator/fivedof_linear_base_elbow_approx_robotiq2f140.usd",
        # R1 (cube_sort rework): contact sensors must be activated on the
        # SPAWNER, not on the ArticulationCfg root (Isaac Lab Issue #2985 —
        # silent no-op there). Required for the per-cube `is_holding`
        # predicate built in R2; see doc/reports/cube_sort_optimization_tracking.md.
        activate_contact_sensors=True,
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
        # Base per-joint effort limits from Klein (2023) joint spec.
        "base_y": ImplicitActuatorCfg(
            joint_names_expr=["base_y_joint"],
            effort_limit_sim=300.0,
            velocity_limit_sim=5.0,
            stiffness=8000.0,
            damping=800.0,
        ),
        "base_z": ImplicitActuatorCfg(
            joint_names_expr=["base_z_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=5.0,
            stiffness=8000.0,
            damping=800.0,
        ),
        # Arm: D=120→20 (ζ 6.4→1.1, settle 1.66s→0.30s).
        # Klein (2023) target: ζ≈1.0, settle 0.1–0.3s, overshoot<5%.
        # Elbow effort: cable-derived 34.8 N·m (160 N × 3:1 × 0.0725 m), use 35 for headroom.
        # Klein (2023) §3.2.6 p.53, spool radius 5 mm, 3:1 elbow pulley MA.
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=2.5,
            stiffness=400.0,
            damping=20.0,
        ),
        # Wrist effort: cable-derived max ≈ 3.2 N·m (80 N × 0.020 m × 2 cable
        # contributions). Klein (2023) §3.2.2 p.42 (r = 20 mm), no pulley MA.
        # 3.5 N·m provides headroom.
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_x_joint", "wrist_y_joint"],
            effort_limit_sim=3.5,
            velocity_limit_sim=9.0,
            stiffness=400.0,
            damping=20.0,
        ),
        # Gripper: imported from shared centralized config
        **get_gripper_actuators("original"),
    },
)

"5DOF arm + gripper with PD-controlled base and arm joints, plus Robotiq gripper (low resolution visualization)"

TENS_5DOF_GRIPPER_LO_CFG = TENS_5DOF_GRIPPER_CFG.replace(
    spawn=TENS_5DOF_GRIPPER_CFG.spawn.replace(
        usd_path=f"{PROJ_ASSETS_PATH}/Tensegrity/fivedof_manipulator/fivedof_linear_base_elbow_approx_robotiq2f140_lo.usd"
    )
)

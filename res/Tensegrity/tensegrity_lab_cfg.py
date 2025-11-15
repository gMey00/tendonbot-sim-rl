"""
- Controls arm joints: Joint1, Wrist_Joint_y, Wrist_Joint_x
- Currently no physical gripper DOF present; "gripper" actuator group is left empty (no-op)
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


PROJ_ASSETS_PATH = "/home/robot/studentische-arbeiten/res"

TENS_3DOF_CFG = ArticulationCfg(
    
    spawn_cfg=sim_utils.UsdFileCfg(
        usd_path="{PROJ_ASSETS_PATH}/Tensegrity/tensegrity_3dof.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Joint1": 0.0,
            "Wrist_Joint_y": 0.0,
            "Wrist_Joint_x": 0.0,
        },
        joint_vel={},
    ),
    
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Joint1", "Wrist_Joint_y", "Wrist_Joint_x"],
            effort_limit=87.0,   # adjust as needed
            stiffness=800.0,     # position-drive stiffness
            damping=40.0,        # position-drive damping
        ),
        # Gripper group: no DOFs exist in current URDF/USD (gripper mesh is fixed/commented)
        # Leaving this empty keeps task interfaces stable without driving non-existent joints.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[],  # no-op
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
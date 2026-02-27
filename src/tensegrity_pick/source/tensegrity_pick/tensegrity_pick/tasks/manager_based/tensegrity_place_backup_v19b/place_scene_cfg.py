# place_scene_cfg.py
#
# Scene for the simplified cube-placement task.
# Extends ProjBaseSceneCfg (ground, lights, robot, conveyor, drum) and adds
# exactly one green and one red cube as individual RigidObjects placed
# directly below the robot arm.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..tensegrity_pick.proj_base_scene_cfg import (
    ProjBaseSceneCfg,
    CONVEYOR_SURFACE_HEIGHT_M,
    ROBOT_MOUNT_HEIGHT_M,
    PROJ_ASSETS_PATH,
    DRUM_USD_DIAMETER_SCALE,
    DRUM_USD_HEIGHT_SCALE,
)
from ..tensegrity_pick.tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG

# Use the original project mount height
PLACE_MOUNT_HEIGHT_M = ROBOT_MOUNT_HEIGHT_M

# ---------------------------------------------------------------------------
# Cube parameters
# ---------------------------------------------------------------------------
CUBE_SIZE_M = 0.05
CUBE_MASS_KG = 0.05
# Cubes spawned directly below the robot (local-frame coords used in reset)
SPAWN_HEIGHT_M = CONVEYOR_SURFACE_HEIGHT_M + 0.03

ENV_NS = "/World/envs/env_.*/"


def _make_colored_cube(
    prim_name: str,
    color_rgb: tuple[float, float, float],
    spawn_pos: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{ENV_NS}{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color_rgb,
                metallic=0.05,
                roughness=0.75,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=spawn_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@configclass
class PlaceSceneCfg(ProjBaseSceneCfg):
    """Base scene + one green cube + one red cube for the place task."""

    # Override robot: stronger base actuator, stable gripper
    robot = TENS_5DOF_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.15, 0.0, PLACE_MOUNT_HEIGHT_M),
            joint_pos={
                "base_y_joint": 0.0,
                "base_z_joint": 0.0,
                "elbow_joint": 0.0,
                "wrist_y_joint": 0.0,
                "wrist_x_joint": 0.0,
                # Only drive joint needs explicit init; passive joints
                # settle to their physical equilibrium via the linkage.
                "finger_joint": 0.0,
            },
        ),
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=["base_y_joint", "base_z_joint"],
                effort_limit=800.0,
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
            # Gripper actuators: values from IsaacLab's reference
            # UR10e_ROBOTIQ_GRIPPER_CFG (universal_robots.py).
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit=10.0,
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
                effort_limit=1.0,
                velocity_limit_sim=1.0,
                stiffness=0.2,
                damping=0.001,
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
                effort_limit=1.0,
                velocity_limit_sim=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
            ),
        },
    )

    # Drum position inherited from ProjBaseSceneCfg (Y=0.85)

    green_cube: RigidObjectCfg = _make_colored_cube(
        prim_name="GreenCube",
        color_rgb=(0.0, 1.0, 0.0),
        # directly below robot mount (x=0.15), slightly offset in Y
        spawn_pos=(0.15, -0.04, SPAWN_HEIGHT_M),
    )

    red_cube: RigidObjectCfg = _make_colored_cube(
        prim_name="RedCube",
        color_rgb=(1.0, 0.0, 0.0),
        spawn_pos=(0.15, 0.04, SPAWN_HEIGHT_M),
    )

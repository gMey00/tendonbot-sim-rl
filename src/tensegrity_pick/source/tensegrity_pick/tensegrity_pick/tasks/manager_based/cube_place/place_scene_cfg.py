# place_scene_cfg.py
#
# Robot-independent scene for the cube-placement task.
# Extends ProjBaseSceneCfg (ground, lights, conveyor, drum) and adds
# exactly one green and one red cube as individual RigidObjects placed
# directly below the robot arm.
#
# The ``robot`` field is left as ``MISSING`` — each robot variant
# config fills it in via ``__post_init__``, following the same
# pattern as the reach task.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ..shared.proj_base_scene_cfg import (
    ProjBaseSceneCfg,
    CONVEYOR_SURFACE_HEIGHT_M,
)

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
    """Robot-independent base scene + one green cube + one red cube.

    The ``robot`` field is inherited as ``MISSING`` from
    ``ProjBaseSceneCfg``.  Each robot variant fills it in via
    ``__post_init__`` in its own ``joint_pos_env_cfg.py``.

    Drum position is inherited from ProjBaseSceneCfg (Y=0.85).
    """

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

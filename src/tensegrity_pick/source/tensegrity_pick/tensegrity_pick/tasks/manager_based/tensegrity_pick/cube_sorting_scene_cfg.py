# cube_sorting_scene_cfg_metric.py
#
# Cube Sorting Scene (metric, conveyor along +X)
# ------------------------------------------------
# Builds on top of ProjBaseSceneCfg and adds cube pools for sorting.
#
# Task design choices:
#   - **Reset option**: episodic. 
#       - No in-episode despawn/recycle.
#       - Episode ends once cubes have passed through (e.g., beyond conveyor end).
#   - **Randomization pattern**: separate pools/collections per "trash type" (green vs red).
#     - Later easily swap red/green pools out for different trash collections 
#       without changing the spawner logic.
#
# NOTE:
# - This file is *scene config only*. The actual randomization at reset (uniform pose sampling,
#   enabling/disabling subsets, and termination conditions) implemented via
#   environment's EventCfg/TerminationCfg (manager-based env).
# - Conveyor motion direction (+X).

from __future__ import annotations

from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.utils import configclass

from .proj_base_scene_cfg import (
    ProjBaseSceneCfg,
    CONVEYOR_LENGTH_M,
    CONVEYOR_SURFACE_HEIGHT_M,
    CONVEYOR_WIDTH_M,
    TOTAL_CONVEYOR_START_X,
    TOTAL_CONVEYOR_END_X,
)

# --------------------------------------------------------------------------------------
# Conveyor geometry constants (accessible at module level for env cfg)
# Both conveyors are end-to-end; physics & spawn ranges use the total extent.
# --------------------------------------------------------------------------------------
CONVEYOR_START_X = TOTAL_CONVEYOR_START_X   # upstream end of the upstream conveyor
CONVEYOR_END_X = TOTAL_CONVEYOR_END_X       # downstream end of the original conveyor
BELT_HEIGHT_M = CONVEYOR_SURFACE_HEIGHT_M
BELT_WIDTH_M = CONVEYOR_WIDTH_M
CONVEYOR_CENTER_Y = 0.0

# --------------------------------------------------------------------------------
# Cube pool parameters
# --------------------------------------------------------------------------------
CUBE_SIZE_M = 0.05
CUBE_MASS_KG = 0.05
NUM_CUBES_PER_COLOR = 8

SPAWN_X = CONVEYOR_START_X + 0.20  # near upstream end of the extended conveyor
SPAWN_Y = 0.0
SPAWN_Z = BELT_HEIGHT_M + 0.03

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

ENV_NS = "/World/envs/env_.*/"

def _make_colored_cube_spawn_cfg(
    cube_size_m: float,
    color_rgb: Tuple[float, float, float],
    mass_kg: float,
) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=(cube_size_m, cube_size_m, cube_size_m),
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
        mass_props=sim_utils.MassPropertiesCfg(mass=mass_kg),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )


def _make_cube_pool(
    pool_name: str,
    color_rgb: Tuple[float, float, float],
    num_cubes: int,
    spawn_center: Tuple[float, float, float],
    cube_size_m: float,
    cube_mass_kg: float,
) -> RigidObjectCollectionCfg:
    spawn_cfg = _make_colored_cube_spawn_cfg(cube_size_m, color_rgb, cube_mass_kg)

    cx, cy, cz = spawn_center
    # Spread a bit in Y so they don’t interpenetrate at reset
    dy = cube_size_m * 1.4

    rigid_objects: Dict[str, RigidObjectCfg] = {}
    for i in range(num_cubes):
        rigid_objects[f"{pool_name}_{i:02d}"] = RigidObjectCfg(
            # IMPORTANT: no intermediate folders, parent env prim already exists
            prim_path=f"{ENV_NS}{pool_name}_Cube_{i:02d}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cx, cy - 0.5 * (num_cubes - 1) * dy + i * dy, cz),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


# --------------------------------------------------------------------------------------
# Scene configuration
# --------------------------------------------------------------------------------------

@configclass
class CubeSortingSceneCfg(ProjBaseSceneCfg):
    """Cube sorting scene (conveyor along +X)."""

    green_cubes: RigidObjectCollectionCfg = _make_cube_pool(
        pool_name="green",
        color_rgb=(0.0, 1.0, 0.0),
        num_cubes=NUM_CUBES_PER_COLOR,
        spawn_center=(SPAWN_X, SPAWN_Y, SPAWN_Z),
        cube_size_m=CUBE_SIZE_M,
        cube_mass_kg=CUBE_MASS_KG,
    )

    red_cubes: RigidObjectCollectionCfg = _make_cube_pool(
        pool_name="red",
        color_rgb=(1.0, 0.0, 0.0),
        num_cubes=NUM_CUBES_PER_COLOR,
        spawn_center=(SPAWN_X, SPAWN_Y, SPAWN_Z),
        cube_size_m=CUBE_SIZE_M,
        cube_mass_kg=CUBE_MASS_KG,
    )

    # ----------------------------------------------------------------------------------
    # Notes for EnvCfg / TaskCfg (not scene):
    # ----------------------------------------------------------------------------------
    # Reset randomization (Option A, Pattern 1):
    #   - On reset: sample k_green and k_red (or fixed counts) and place those cubes in a spawn box:
    #       x in [y_start, y_start + 0.20] 
    #       y in [-0.35, +0.35] (within 0.8m belt width, leaving margin)
    #       z = belt_surface + small clearance
    #       yaw random
    #   - "Deactivate" the unused cubes by teleporting them to a parking location far away
    #     (e.g., (100, 100, 1)) and setting velocities to 0.
    #
    # Termination conditions:
    #   - End episode when all *active* cubes satisfy:
    #       * z < conveyor_surface_height - 0.05  (fallen off)
    #     plus a time-limit termination.

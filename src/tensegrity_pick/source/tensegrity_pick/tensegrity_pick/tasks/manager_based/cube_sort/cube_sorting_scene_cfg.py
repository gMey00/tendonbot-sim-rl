# cube_sorting_scene_cfg.py
#
# Cube Sorting Scene (metric, conveyor along +X)
# ------------------------------------------------
# Builds on top of ProjBaseSceneCfg and adds a unified labelled cube pool.
#
# Each cube carries an integer label (stored as a flat tensor in the env)
# that specifies its category (e.g. 0 = target/green, 1 = distractor/red).
# Functions filter cubes by label using vectorised boolean masks — no
# per-colour loops, peak GPU throughput.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.utils import configclass

from ..shared.proj_base_scene_cfg import (
    ProjBaseSceneCfg,
    CONVEYOR_SURFACE_HEIGHT_M,
    CONVEYOR_WIDTH_M,
    TOTAL_CONVEYOR_START_X,
    TOTAL_CONVEYOR_END_X,
)

# ---------------------------------------------------------------------------
# Conveyor geometry re-exports
# ---------------------------------------------------------------------------
CONVEYOR_START_X: Final = TOTAL_CONVEYOR_START_X
CONVEYOR_END_X: Final = TOTAL_CONVEYOR_END_X
BELT_HEIGHT_M: Final = CONVEYOR_SURFACE_HEIGHT_M
BELT_WIDTH_M: Final = CONVEYOR_WIDTH_M
CONVEYOR_CENTER_Y: Final = 0.0

# ---------------------------------------------------------------------------
# Cube parameters
# ---------------------------------------------------------------------------
CUBE_SIZE_M: Final = 0.05
CUBE_MASS_KG: Final = 0.05

SPAWN_X: Final = CONVEYOR_START_X + 0.20
SPAWN_Y: Final = 0.0
SPAWN_Z: Final = BELT_HEIGHT_M + 0.03

ENV_NS: Final = "/World/envs/env_.*/"


# ---------------------------------------------------------------------------
# Labelled-cube category system
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CubeCategory:
    """One category of cubes (e.g. *green / target*)."""

    label: int
    count: int
    color_rgb: Tuple[float, float, float]


TARGET_LABEL: Final = 0
DISTRACTOR_LABEL: Final = 1

CUBE_CATEGORIES: Final[Tuple[CubeCategory, ...]] = (
    CubeCategory(label=TARGET_LABEL, count=8, color_rgb=(0.0, 1.0, 0.0)),
    CubeCategory(label=DISTRACTOR_LABEL, count=8, color_rgb=(1.0, 0.0, 0.0)),
)

NUM_CUBES_TOTAL: Final = sum(c.count for c in CUBE_CATEGORIES)

CUBE_LABELS: Final[Tuple[int, ...]] = tuple(
    cat.label for cat in CUBE_CATEGORIES for _ in range(cat.count)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cube_spawn_cfg(
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


def _make_cube_collection(
    categories: Sequence[CubeCategory],
    spawn_center: Tuple[float, float, float],
    cube_size_m: float,
    cube_mass_kg: float,
) -> RigidObjectCollectionCfg:
    cx, cy, cz = spawn_center
    dy = cube_size_m * 1.4
    total = sum(cat.count for cat in categories)

    rigid_objects: dict[str, RigidObjectCfg] = {}
    global_idx = 0
    for cat in categories:
        spawn = _cube_spawn_cfg(cube_size_m, cat.color_rgb, cube_mass_kg)
        for _ in range(cat.count):
            y_offset = -0.5 * (total - 1) * dy + global_idx * dy
            rigid_objects[f"cube_{global_idx:02d}"] = RigidObjectCfg(
                prim_path=f"{ENV_NS}Cube_{global_idx:02d}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(cx, cy + y_offset, cz),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            global_idx += 1

    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@configclass
class CubeSortingSceneCfg(ProjBaseSceneCfg):
    """Cube sorting scene — single labelled collection on conveyor belt."""

    cubes: RigidObjectCollectionCfg = _make_cube_collection(
        categories=CUBE_CATEGORIES,
        spawn_center=(SPAWN_X, SPAWN_Y, SPAWN_Z),
        cube_size_m=CUBE_SIZE_M,
        cube_mass_kg=CUBE_MASS_KG,
    )

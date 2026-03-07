# shirt_sort_scene_cfg.py
#
# Scene for the shirt sorting task.
# Extends ProjBaseSceneCfg (ground, lights, robot, conveyor, drum) and
# replaces cube pools with cloth-simulated T-shirts on a moving conveyor.
#
# TODO: Replace RigidObjectCollectionCfg placeholders with deformable
#       object pools once cloth simulation is integrated.

from __future__ import annotations

from isaaclab.utils import configclass

from ..shared.proj_base_scene_cfg import (
    ProjBaseSceneCfg,
    CONVEYOR_SURFACE_HEIGHT_M,
    CONVEYOR_WIDTH_M,
    TOTAL_CONVEYOR_START_X,
    TOTAL_CONVEYOR_END_X,
)

CONVEYOR_START_X = TOTAL_CONVEYOR_START_X
CONVEYOR_END_X = TOTAL_CONVEYOR_END_X
BELT_HEIGHT_M = CONVEYOR_SURFACE_HEIGHT_M
BELT_WIDTH_M = CONVEYOR_WIDTH_M
CONVEYOR_CENTER_Y = 0.0


@configclass
class ShirtSortingSceneCfg(ProjBaseSceneCfg):
    """Shirt sorting scene (conveyor along +X).

    TODO: Add pools of cloth-simulated shirts (clean vs dirty).
    """

    pass

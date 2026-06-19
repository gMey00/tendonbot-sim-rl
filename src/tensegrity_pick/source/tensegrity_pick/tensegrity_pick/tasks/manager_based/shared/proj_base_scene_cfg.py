# proj_base_scene_cfg.py
#
# Scene configuration for a manager-based Isaac Lab task.
# Spawns: ground, lighting, robot, conveyor, plastic drums.

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path

from tensegrity_pick.robots.tensegrity_robot_cfg import PROJ_ASSETS_PATH, TENS_5DOF_GRIPPER_CFG

ENV_NS = "{ENV_REGEX_NS}"

# --------------------------------------------------------------------------------------
# Metric constants from project specifications (CAD + USD analysis)
# --------------------------------------------------------------------------------------
CONVEYOR_WIDTH_M = 0.800
CONVEYOR_SURFACE_HEIGHT_M = 0.8 #0.793

DRUM_DIAMETER_M = 0.547
DRUM_HEIGHT_M = 0.880

DRUM_USD_DIAMETER_SCALE = 0.02186 * DRUM_DIAMETER_M
DRUM_USD_HEIGHT_SCALE = 0.01115 * DRUM_HEIGHT_M

DRUM_CENTER_TO_CONVEYOR_EDGE_M = 0.450
ROBOT_TO_CONVEYOR_GAP_M = 0.040

# Per-robot optimal mount heights (determined by workspace coverage sweep).
TENSEGRITY_MOUNT_HEIGHT_M = 2.30
UR10E_MOUNT_HEIGHT_M = 1.40
KINOVA_MOUNT_HEIGHT_M = 1.40

# Default robot mount height (can be overridden per task)
ROBOT_MOUNT_HEIGHT_M = TENSEGRITY_MOUNT_HEIGHT_M

CONVEYOR_LENGTH_M = 2.0
CONVEYOR_USD_ORIGIN_TO_BELT_SURFACE_M = 1.78056  # from USD geometry analysis

# Two identical conveyors placed end-to-end along +X.
# The upstream conveyor is placed before the original one.
TOTAL_CONVEYOR_LENGTH_M = 2 * CONVEYOR_LENGTH_M  # 4.0 m total
TOTAL_CONVEYOR_START_X = -CONVEYOR_LENGTH_M - CONVEYOR_LENGTH_M / 2  # = -3.0
TOTAL_CONVEYOR_END_X = CONVEYOR_LENGTH_M / 2  # = +1.0 (same downstream end)

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Scene configuration
# --------------------------------------------------------------------------------------
@configclass
class ProjBaseSceneCfg(InteractiveSceneCfg):
    """Scene configuration: ground + robot + conveyor + plastic drum."""

    # ---- terrain / ground ----
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )
    
    # ---- lighting ----
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )

    # ---- robot (from separate robot cfg file) ----
    robot: ArticulationCfg = MISSING  # to be specified by individual task env configs; different tasks may use different robot assets and mounts

    # ---- conveyor belts (two identical conveyors end-to-end along +X) ----
    # Note: Using AssetBaseCfg instead of RigidObjectCfg since the conveyor USD has its own physics prim

    # Original (downstream) conveyor
    conveyor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Conveyor",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A06.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # Position: origin offset to center conveyor; USD origin is not at belt surface/center
            pos=(-1.0, 0.0, CONVEYOR_SURFACE_HEIGHT_M - CONVEYOR_USD_ORIGIN_TO_BELT_SURFACE_M),
        ),
    )

    # Upstream conveyor (placed one conveyor-length before the original)
    conveyor_upstream = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorUpstream",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A06.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.0 - CONVEYOR_LENGTH_M, 0.0, CONVEYOR_SURFACE_HEIGHT_M - CONVEYOR_USD_ORIGIN_TO_BELT_SURFACE_M),
        ),
    )

    # ---- plastic drums (rigid object from custom USD) ----
    drum_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DrumTarget",
        spawn=sim_utils.UsdFileCfg(
            usd_path= f"{PROJ_ASSETS_PATH}/Props/PlasticDrum_B03_PR_NVD_01.usd",
            scale=(DRUM_USD_DIAMETER_SCALE, DRUM_USD_DIAMETER_SCALE, DRUM_USD_HEIGHT_SCALE),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                0.15,  # slightly downstream from center
                CONVEYOR_WIDTH_M * 0.5 + DRUM_CENTER_TO_CONVEYOR_EDGE_M,  
                0.0,
            )
        ),
    )
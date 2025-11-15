# proj_base_scene_cfg.py
#
# Scene configuration for a manager-based Isaac Lab task.
# Spawns: ground, lighting, robot, conveyor, plastic drum.

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass

from .tensegrity_3dof import TENS_3DOF_CFG  # or MyRobotCfg, etc.

PROJ_ASSETS_PATH = "/home/robot/studentische-arbeiten/res"

@configclass
class ProjBaseSceneCfg(InteractiveSceneCfg):
    """Scene configuration: ground + robot + conveyor + plastic drum."""

    # ---- required base fields from InteractiveSceneCfg ----
    # You can override these from the env_cfg via Hydra if you want.
    num_envs: int = 1024
    env_spacing: float = 2.0
    lazy_sensor_update: bool = True
    replicate_physics: bool = True

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
    # We only change the prim_path to use Isaac Lab's per-env namespace.
    robot = TENS_3DOF_CFG.replace(
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0)
        )
    )

    # ---- conveyor belt (SimReady USD without stands) ----
    conveyor = RigidObjectCfg(
        prim_path="/World/Conveyor",
        spawn=sim_utils.UsdFileCfg(
            usd_path="{PROJ_ASSETS_PATH}/Conveyor/ConveyorBelt_A06_PR_NVD_01.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # position & orientation of the conveyor root in world/env frame
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # ---- plastic drum (rigid object from USD) ----
    drum = RigidObjectCfg(
        prim_path="/World/Drum",
        spawn=sim_utils.UsdFileCfg(
            usd_path="{PROJ_ASSETS_PATH}/PlasticDrum_B03_PR_NVD_01.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-80.0, 0.0, 0.0),
        ),
    )
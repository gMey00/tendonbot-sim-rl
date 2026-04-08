# shirt_place_scene_cfg.py
#
# Scene for the shirt placement task.
# Extends ProjBaseSceneCfg (ground, lights, robot, conveyor, drum) and
# adds a PBD particle cloth T-shirt as the manipulation target.
#
# The cloth mesh is spawned by a startup event (apply_cloth_startup_event)
# which runs between scene creation and sim.reset().  After sim.reset()
# the env acquires a ParticleClothView for GPU tensor access.
#
# A kinematic rigid-body proxy (``shirt_proxy``) is kept for reward /
# observation tracking: its position is synced to the cloth centroid
# each step so all reward functions work unchanged.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from ..shared.proj_base_scene_cfg import (
    CONVEYOR_SURFACE_HEIGHT_M,
    PROJ_ASSETS_PATH,
    ProjBaseSceneCfg,
    ROBOT_MOUNT_HEIGHT_M,
)
from ..shared.cloth_object import ClothObjectCfg, PBDClothParams
from tensegrity_pick.robots import TENS_5DOF_GRIPPER_CFG

PLACE_MOUNT_HEIGHT_M = ROBOT_MOUNT_HEIGHT_M
SPAWN_HEIGHT_M = CONVEYOR_SURFACE_HEIGHT_M + 0.03

# T-shirt USD asset paths
TSHIRT_USD_PATH = f"{PROJ_ASSETS_PATH}/Props/Cloth/tshirt_mod.usdc"
TSHIRT_MESH_PRIM = "/root/World/mesh/Mesh"

# Proxy cube: kinematic rigid-body synced to cloth centroid
PROXY_SIZE_M = 0.01  # tiny — only used as position tracker
PROXY_MASS_KG = 0.01

_INITIAL_JOINT_POS = {
    "base_y_joint": 0.0,
    "base_z_joint": -0.25,
    "elbow_joint": 0.0,
    "wrist_y_joint": 0.0,
    "wrist_x_joint": 0.0,
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_outer_finger_joint": 0.0,
    "right_outer_finger_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "left_inner_finger_pad_joint": 0.0,
    "right_inner_finger_pad_joint": 0.0,
}


# ---------------------------------------------------------------------------
# Cloth configuration
# ---------------------------------------------------------------------------

SHIRT_CLOTH_CFG = ClothObjectCfg(
    prim_path="{ENV_REGEX_NS}/Shirt",
    usd_path=TSHIRT_USD_PATH,
    mesh_prim_path=TSHIRT_MESH_PRIM,
    pbd_params=PBDClothParams(),
    init_pos=(0.15, -0.04, SPAWN_HEIGHT_M),
)


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@configclass
class ShirtPlaceSceneCfg(ProjBaseSceneCfg):
    """Base scene for the shirt-placement task.

    ``replicate_physics`` is disabled because PBD particle cloth
    requires per-env physics (no GPU replication).

    The ``shirt_proxy`` is a kinematic rigid body whose position is
    synced to the cloth centroid each step.  All reward / observation
    functions read from the proxy so they work unchanged.
    """

    # Cloth requires per-env physics — no GPU replication
    replicate_physics: bool = False

    robot = TENS_5DOF_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.15, 0.0, PLACE_MOUNT_HEIGHT_M),
            joint_pos=_INITIAL_JOINT_POS,
        ),
    )

    # Kinematic proxy — collision-free, synced to cloth centroid
    shirt_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ShirtProxy",
        spawn=sim_utils.CuboidCfg(
            size=(PROXY_SIZE_M, PROXY_SIZE_M, PROXY_SIZE_M),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=PROXY_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.15, -0.04, SPAWN_HEIGHT_M),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


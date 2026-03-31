# shirt_place_scene_cfg.py
#
# Scene for the shirt placement task.
# Extends ProjBaseSceneCfg (ground, lights, robot, conveyor, drum) and
# adds a cloth-simulated T-shirt as the manipulation target.
#
# The shirt is declared via ClothObjectCfg — a stub wrapper that will
# be materialized as PBD particle cloth or XPBD surface deformable at
# Isaac Sim runtime.  Currently the cloth object is NOT spawned by the
# scene infrastructure (Isaac Lab's InteractiveSceneCfg does not support
# cloth); it is created programmatically in the environment class.
#
# For now we also keep a RigidObject "proxy cube" that stands in for
# the cloth centroid in reward/observation computations until the full
# ClothObject tensor API is wired up.

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
from ..shared.cloth_object import ClothBackend, ClothObjectCfg, PBDClothParams, XPBDClothParams
from tensegrity_pick.robots import TENS_5DOF_GRIPPER_CFG

PLACE_MOUNT_HEIGHT_M = ROBOT_MOUNT_HEIGHT_M
SPAWN_HEIGHT_M = CONVEYOR_SURFACE_HEIGHT_M + 0.03

# T-shirt USD asset paths
TSHIRT_USD_PATH = f"{PROJ_ASSETS_PATH}/Props/Cloth/tshirt_mod.usdc"
TSHIRT_MESH_PRIM = "/World/tshirt_mod/mesh"

# Proxy cube: temporary rigid-body stand-in for cloth centroid
PROXY_SIZE_M = 0.05
PROXY_MASS_KG = 0.20  # cotton T-shirt mass

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

ENV_NS = "/World/envs/env_.*/"


# ---------------------------------------------------------------------------
# Cloth configuration
# ---------------------------------------------------------------------------

SHIRT_CLOTH_CFG = ClothObjectCfg(
    prim_path="{ENV_REGEX_NS}/Shirt",
    usd_path=TSHIRT_USD_PATH,
    mesh_prim_path=TSHIRT_MESH_PRIM,
    backend=ClothBackend.PBD,
    pbd_params=PBDClothParams(),
    xpbd_params=XPBDClothParams(),
    init_pos=(0.15, -0.04, SPAWN_HEIGHT_M),
    num_keypoints=8,
)


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@configclass
class ShirtPlaceSceneCfg(ProjBaseSceneCfg):
    """Base scene + one shirt (proxy cube) for the shirt place task.

    The ``robot`` field is inherited as ``MISSING`` from ProjBaseSceneCfg.
    Each robot variant fills it in via ``__post_init__``.

    The ``shirt_proxy`` RigidObject is a temporary stand-in for the cloth
    centroid.  It will be replaced by actual cloth state tensors once
    ClothObject.initialize() is implemented.
    """

    robot = TENS_5DOF_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.15, 0.0, PLACE_MOUNT_HEIGHT_M),
            joint_pos=_INITIAL_JOINT_POS,
        ),
    )

    # Proxy cube standing in for the cloth centroid
    shirt_proxy: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"{ENV_NS}ShirtProxy",
        spawn=sim_utils.CuboidCfg(
            size=(PROXY_SIZE_M, PROXY_SIZE_M, PROXY_SIZE_M),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.5, 0.9),
                metallic=0.0,
                roughness=0.9,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=PROXY_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.15, -0.04, SPAWN_HEIGHT_M),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


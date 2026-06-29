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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

# Box collider that replaces the conveyor's triangle-mesh / convex-hull
# colliders for PBD cloth interaction.  Spans BOTH conveyors end-to-end
# (downstream X∈[-1,1] + upstream X∈[-3,-1] → X∈[-3,1]) at the belt width
# (0.9 m), with its top face exactly at the belt surface height (0.8 m).  The
# upstream half is the "free side" used to settle the shirt before teleporting
# it to the work area.
# A thick slab (0.40 m): the top face stays exactly at the belt surface, but the
# extra depth stops a fast-falling flat cloth from tunnelling through a thin box
# (CCD is off for PBD cloth).  The box is invisible, so depth has no visual cost.
CONVEYOR_BOX_SIZE = (4.0, 0.90, 0.40)
CONVEYOR_BOX_POS = (-1.0, 0.0, CONVEYOR_SURFACE_HEIGHT_M - CONVEYOR_BOX_SIZE[2] / 2)

# Centre of the upstream (free) conveyor — used to settle the shirt cleanly,
# away from the robot, before teleporting it to the work area.
UPSTREAM_SETTLE_POS = (-2.0, 0.0)  # (x, y) in env-local coords

# T-shirt USD asset.  A flat-laid, metric, single-layer ClothesNet shirt
# (``Ts1_0``), built fresh for cloth sim — NOT the old inflated ``tshirt_mod`` /
# short-sleeve shells.  Its default prim ``/World`` collapses onto the reference,
# so the mesh ends up at ``{ENV}/Shirt/mesh``.
TSHIRT_USD_PATH = f"{PROJ_ASSETS_PATH}/Props/Cloth/tshirt_clothesnet.usd"
# Same garment with the XPBD surface-deformable schema + material pre-authored
# (no physics scene) — referenced when the cloth backend is XPBD.
TSHIRT_XPBD_USD_PATH = f"{PROJ_ASSETS_PATH}/Props/Cloth/tshirt_clothesnet_xpbd.usd"
TSHIRT_MESH_PRIM = "/mesh"

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

# Cloth backend — switch between PBD particle cloth and XPBD surface deformable
# by changing this one value (or override SHIRT_CLOTH_CFG.backend per task).
SHIRT_CLOTH_BACKEND = ClothBackend.PBD

SHIRT_CLOTH_CFG = ClothObjectCfg(
    prim_path="{ENV_REGEX_NS}/Shirt",
    usd_path=TSHIRT_USD_PATH,
    xpbd_usd_path=TSHIRT_XPBD_USD_PATH,
    mesh_prim_path=TSHIRT_MESH_PRIM,
    backend=SHIRT_CLOTH_BACKEND,
    pbd_params=PBDClothParams(),
    xpbd_params=XPBDClothParams(),
    init_pos=(0.15, -0.04, SPAWN_HEIGHT_M),
)

# ── Fixed cloth placement on the belt (first iteration) ────────────────────
# The shirt is laid flat and centred under the gripper's default grasp position
# at every reset — no fold, no crumble, no XY/yaw randomization.  The X is set to
# the gripper's grasp-centre X at the default pose so the shirt's highest point
# sits directly under the open gripper.
SHIRT_REST_XY = (0.18, 0.0)    # (x, y) local to env origin, under the gripper
SHIRT_BELT_CLEARANCE_M = 0.02  # drop height above the belt before settling


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

    # Box collider for the belt surface.  PBD particle cloth tunnels through
    # the conveyor USD's triangle-mesh / convex-hull colliders, so those are
    # disabled by ``disable_complex_colliders_event`` and the cloth rests on
    # this simple static box instead (top face at the belt height).
    conveyor_collider: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorCollider",
        spawn=sim_utils.CuboidCfg(
            size=CONVEYOR_BOX_SIZE,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=CONVEYOR_BOX_POS),
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


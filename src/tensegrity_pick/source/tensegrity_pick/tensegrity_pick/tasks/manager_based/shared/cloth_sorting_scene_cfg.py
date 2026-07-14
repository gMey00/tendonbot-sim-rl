# cloth_sorting_scene_cfg.py
#
# Central scene for the three master-thesis cloth-sorting pipeline tasks:
#
#   1. shirt_pick       — retrieving robot picks a shirt from the conveyor and
#                         holds it in front of the inspection camera
#   2. shirt_present    — second robot grasps a second point and stretches the
#                         hanging shirt for front/back camera assessment
#   3. shirt_distribute — second robot throws/places the shirt into the bin
#                         matching its condition label (reusable/recyclable/trash)
#
# All three tasks import THIS scene so layout changes propagate everywhere.
# The layout rebuilds the hand-authored showcase stage
# ``res/Scenes/cloth_sorting_showcase.usd`` (positions extracted from its USD
# xformOps) on top of :class:`ProjBaseSceneCfg`:
#
#   * two conveyors end-to-end along +X (inherited from the base scene)
#   * retrieving robot hanging over the belt at ``RETRIEVE_MOUNT_*``
#     (tensegrity showcase pose: ``(0.15, 0.0, 2.30)``)
#   * second robot standing on a pedestal at ``SECOND_ROBOT_MOUNT_POS``
#     ``(0.75, 1.0, 0.75)`` — the same workspace-analysis pose used by the
#     F140 reach variants (``reach/config/f140_reach_common.py``)
#   * three sorting drums replacing the base scene's single ``drum_target``
#   * inspection-camera frame constants (no camera sensor is spawned — the
#     observation design stays camera-realistic without camera simulation)
#   * the PBD cloth T-shirt + belt box collider + kinematic centroid proxy,
#     reusing the machinery validated in the shirt_place task
#     (``shared/cloth_object.py``).

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from .cloth_object import ClothBackend, ClothObjectCfg, GraspMode, PBDClothParams, XPBDClothParams
from .proj_base_scene_cfg import (
    CONVEYOR_SURFACE_HEIGHT_M,
    DRUM_USD_DIAMETER_SCALE,
    DRUM_USD_HEIGHT_SCALE,
    PROJ_ASSETS_PATH,
    ProjBaseSceneCfg,
    TENSEGRITY_MOUNT_HEIGHT_M,
)

# ──────────────────────────────────────────────────────────────────────────
# Layout constants (extracted from res/Scenes/cloth_sorting_showcase.usd)
# ──────────────────────────────────────────────────────────────────────────

# Retrieving robot (task 1 agent): hanging mount above the belt.
# The tensegrity showcase pose; UR/Kinova retrieval variants override the
# height via their per-robot mount constants in proj_base_scene_cfg.
RETRIEVE_MOUNT_XY = (0.15, 0.0)
RETRIEVE_MOUNT_HEIGHT_M = TENSEGRITY_MOUNT_HEIGHT_M  # 2.30

# Second robot (task 2 + 3 agent): standing on a pedestal beyond the belt.
# Same pose as the F140 reach comparison variants (workspace-analysis pose).
SECOND_ROBOT_MOUNT_POS = (0.75, 1.0, 0.75)
SECOND_ROBOT_MOUNT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity (Kinova showcase pose)
# The showcase mounts the UR5e yaw-rotated by +90° so its home pose faces the belt.
SECOND_ROBOT_MOUNT_ROT_UR = (0.7071068, 0.0, 0.0, 0.7071068)

# Pedestal under the second robot.  The showcase uses the Isaac
# ``Props/Mounts/Stand/stand_instanceable.usd`` (remote asset, scale 1.5); we
# spawn a simple visual column instead to avoid the remote dependency.
PEDESTAL_SIZE = (0.30, 0.30, SECOND_ROBOT_MOUNT_POS[2])
PEDESTAL_POS = (SECOND_ROBOT_MOUNT_POS[0], SECOND_ROBOT_MOUNT_POS[1], SECOND_ROBOT_MOUNT_POS[2] / 2)

# Three sorting drums (showcase positions).  The label → drum assignment is a
# *task* decision: shirt_distribute randomizes it per episode so "nearest bin"
# and "correct bin" diverge (goal-conditioned formulation, see
# doc/reports/tmp/RESEARCH_cloth_sorting_pipeline.md §4).
DRUM_REUSABLE_POS = (0.15, 1.0, 0.0)
DRUM_RECYCLABLE_POS = (1.35, 1.0, 0.0)
DRUM_TRASH_POS = (0.75, 1.6, 0.0)
DRUM_NAMES = ("drum_reusable", "drum_recyclable", "drum_trash")
DRUM_POSITIONS = (DRUM_REUSABLE_POS, DRUM_RECYCLABLE_POS, DRUM_TRASH_POS)

# Inspection camera frame (NO camera sensor is spawned — observations must stay
# camera-derivable, see the research report §6).  Showcase pose: at y=+2.0
# looking along −Y toward the presentation point, 1.25 m above ground.
INSPECTION_CAMERA_POS = (0.15, 2.0, 1.25)
INSPECTION_CAMERA_VIEW_DIR = (0.0, -1.0, 0.0)

# Where the retrieving robot presents the shirt to the camera (task 1 goal /
# task 2 start).  Set per the Project-Thesis workspace analysis (user-checked
# in the GUI, 2026-07-03) and verified empirically by
# scripts/model_validation/probe_present_pose.py: reached in 8/8 envs with
# the shirt held (hold distance 0.000 m), NO joint at a position limit
# (elbow +1.19 of ±1.5, base_y +0.19 of ±0.5) and NO torque saturation
# (peaks: wrist_x 0.78, elbow 0.22 of the Klein effort limits) — so neither
# action clips nor effort limits needed changes.  The hanging shirt
# (~0.35 m below the grasp) clears the belt footprint (collider y ≤ 0.45)
# and sits near the inspection camera's −Y optical axis (camera at
# (0.15, 2.0, 1.25)).
# History: two earlier, lower poses came from over-conservative reach
# estimates produced by crude scripted probes (stale-Jacobian servo stalls,
# misread as envelope limits) — see shirt_pick_optimization_tracking.md.
PRESENTATION_POS = (0.15, 0.90, 1.60)

# ──────────────────────────────────────────────────────────────────────────
# Cloth configuration (same garment + backend as the validated shirt_place)
# ──────────────────────────────────────────────────────────────────────────

# Flat-laid welded ClothesNet shirt — the asset validated for both backends in
# shirt_place (see res/Props/Cloth/README.md).
TSHIRT_USD_PATH = f"{PROJ_ASSETS_PATH}/Props/Cloth/tshirt_clothesnet.usd"
TSHIRT_XPBD_USD_PATH = f"{PROJ_ASSETS_PATH}/Props/Cloth/tshirt_clothesnet_xpbd.usd"
TSHIRT_MESH_PRIM = "/mesh"

SPAWN_HEIGHT_M = CONVEYOR_SURFACE_HEIGHT_M + 0.03

CLOTH_SORTING_SHIRT_CFG = ClothObjectCfg(
    prim_path="{ENV_REGEX_NS}/Shirt",
    usd_path=TSHIRT_USD_PATH,
    xpbd_usd_path=TSHIRT_XPBD_USD_PATH,
    mesh_prim_path=TSHIRT_MESH_PRIM,
    backend=ClothBackend.PBD,
    grasp_mode=GraspMode.ANCHOR,
    pbd_params=PBDClothParams(),
    xpbd_params=XPBDClothParams(),
    init_pos=(RETRIEVE_MOUNT_XY[0], -0.04, SPAWN_HEIGHT_M),
)

# Default flat-lay pose on the belt (task envs override per task).
SHIRT_REST_XY = (0.18, 0.0)

# Cached crumpled-state bank (drop-and-settle protocol) — generated by
# ``scripts/generate_crumpled_bank.py``; consumed by ``ClothSortingEnvBase``
# when a task env sets ``crumpled_bank_path`` (shirt_pick does).
CRUMPLED_BANK_PATH = f"{PROJ_ASSETS_PATH}/Props/Cloth/banks/tshirt_crumpled_bank.pt"
# Cached hanging-state bank (shirt hanging relaxed from ONE random particle) —
# generated by ``scripts/generate_hanging_bank.py``; states are stored
# anchor-centred so they can be restored at ANY world anchor point (static
# holder anchor in shirt_present, the learning robot's own finger tip in
# shirt_distribute).  Consumed via ``hanging_bank_path`` +
# ``ClothSortingEnvBase._reset_cloth_hanging_from_bank``.
HANGING_BANK_PATH = f"{PROJ_ASSETS_PATH}/Props/Cloth/banks/tshirt_hanging_bank.pt"
SHIRT_BELT_CLEARANCE_M = 0.02
# Centre of the upstream (free) conveyor — clean settle spot for the one-off
# cloth pre-settle, away from the robots (same trick as shirt_place).
UPSTREAM_SETTLE_POS = (-2.0, 0.0)

# Box collider replacing the conveyors' triangle-mesh colliders (PBD cloth
# tunnels through those).  Identical to the validated shirt_place setup.
CONVEYOR_BOX_SIZE = (4.0, 0.90, 0.40)
CONVEYOR_BOX_POS = (-1.0, 0.0, CONVEYOR_SURFACE_HEIGHT_M - CONVEYOR_BOX_SIZE[2] / 2)

# Kinematic proxy synced to the cloth centroid (rewards/obs read this).
PROXY_SIZE_M = 0.01
PROXY_MASS_KG = 0.01


# ──────────────────────────────────────────────────────────────────────────
# Scene configuration
# ──────────────────────────────────────────────────────────────────────────

@configclass
class ClothSortingSceneCfg(ProjBaseSceneCfg):
    """Shared scene for shirt_pick / shirt_present / shirt_distribute.

    ``robot`` is the *learning* agent of the task at hand and stays MISSING —
    each robot variant fills it (retriever for shirt_pick, second arm for
    shirt_present / shirt_distribute).  ``holder_robot`` is the optional
    passive partner arm (e.g. the retriever statically holding the shirt
    during shirt_present); tasks that do not need it leave it ``None`` and the
    scene skips it.

    ``replicate_physics`` is disabled because PBD particle cloth requires
    per-env physics (no GPU replication) — same as shirt_place.
    """

    replicate_physics: bool = False

    # Learning agent — filled by each task's robot variant config.
    robot: ArticulationCfg = MISSING

    # Optional passive partner robot (None = not spawned).
    holder_robot: ArticulationCfg | None = None

    # The base scene's single target drum is replaced by the three sorting bins.
    drum_target = None

    drum_reusable = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DrumReusable",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PROJ_ASSETS_PATH}/Props/PlasticDrum_B03_PR_NVD_01.usd",
            scale=(DRUM_USD_DIAMETER_SCALE, DRUM_USD_DIAMETER_SCALE, DRUM_USD_HEIGHT_SCALE),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=DRUM_REUSABLE_POS),
    )
    drum_recyclable = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DrumRecyclable",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PROJ_ASSETS_PATH}/Props/PlasticDrum_B03_PR_NVD_01.usd",
            scale=(DRUM_USD_DIAMETER_SCALE, DRUM_USD_DIAMETER_SCALE, DRUM_USD_HEIGHT_SCALE),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=DRUM_RECYCLABLE_POS),
    )
    drum_trash = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DrumTrash",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{PROJ_ASSETS_PATH}/Props/PlasticDrum_B03_PR_NVD_01.usd",
            scale=(DRUM_USD_DIAMETER_SCALE, DRUM_USD_DIAMETER_SCALE, DRUM_USD_HEIGHT_SCALE),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=DRUM_TRASH_POS),
    )

    # Visual pedestal under the second robot (stands in for the showcase's
    # remote ``stand_instanceable`` asset; collision-free so it never
    # interferes with training).
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=PEDESTAL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PEDESTAL_POS),
    )

    # Box collider for the belt surface (PBD cloth support, see above).
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

    # Kinematic proxy — collision-free, synced to the cloth centroid each step.
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
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CLOTH_SORTING_SHIRT_CFG.init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def configure_cloth_sim(env_cfg) -> None:
    """Apply the cloth-validated sim profile to a ``ManagerBasedRLEnvCfg``.

    Copies the profile tuned/validated in shirt_place
    (``shirt_place_env_cfg.ShirtPlaceEnvCfg.__post_init__``): 60 Hz physics at
    decimation 1 (the ~11 k-particle cloth is the sim bottleneck), 24 PBD
    position iterations with global self-collision, and the enlarged GPU
    buffers PBD particle contacts require.  Call from each task cfg's
    ``__post_init__``.
    """
    env_cfg.decimation = 1
    env_cfg.sim.dt = 1.0 / 60.0
    env_cfg.sim.render_interval = env_cfg.decimation

    cp = CLOTH_SORTING_SHIRT_CFG.pbd_params
    cp.solver_position_iterations = 24
    cp.enable_ccd = False
    cp.global_self_collision = True

    env_cfg.sim.physx.solver_type = 1
    env_cfg.sim.physx.bounce_threshold_velocity = 0.2
    env_cfg.sim.physx.enable_stabilization = True
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**21
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**19
    env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
    env_cfg.sim.physx.gpu_max_particle_contacts = 2**22
    env_cfg.sim.physx.friction_correlation_distance = 0.00625
    # Signed-32-bit max: 2**31 overflows negative → PhysX drops contacts
    # silently (see shirt_place lesson).
    env_cfg.sim.physx.gpu_collision_stack_size = 2**31 - 1

"""Cloth integration for Isaac Lab environments — two switchable backends.

``ClothBackend.PBD``  : PhysX particle cloth (PBD mass-spring) authored at runtime
                        with ``omni.physx.scripts.particleUtils`` and driven via a
                        ``ParticleClothView``.  Offsets are keyed to the mesh edge
                        length (the decisive stability fix), self-collision on,
                        realistic mass — see ``PBDClothParams``.
``ClothBackend.XPBD`` : PhysX surface deformable (XPBD-FEM) whose schema + material
                        are *pre-authored* into a USD asset and referenced per env;
                        driven via a ``surface_deformable_body_view``.  The flat
                        ``restShapePoints`` give a genuinely flat lay — see
                        ``XPBDClothParams``.

Lifecycle (both backends):
  1. ``apply_cloth_startup_event()`` (prestartup, before ``sim.reset()``) spawns the
     garment per-env and authors / references the chosen backend.
  2. After ``sim.reset()`` the env creates ``ClothObject()``, which acquires the
     backend-appropriate tensor view and stores default state.
  3. Each step ``update()`` reads positions/velocities; each reset writes them.

The RL task keeps a kinematic rigid-body proxy (``shirt_proxy``) synced to the cloth
centroid, so reward/observation functions work unchanged for either backend.

See ``doc/reports/cloth_sim_research/RESEARCH_isaaclab_cloth.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

logger = logging.getLogger(__name__)

# When True the cloth's UsdPreviewSurface visual material is bound
# *stronger than descendants*, so it overrides the per-face GeomSubset
# material bindings inherited from the source garment USD.  Those subset
# materials reference textures by a relative path that does not resolve once
# the mesh is referenced under ``{ENV}/Shirt``, which left the shirt rendering
# as an unlit/!invisible surface.  Set to False only to reproduce that bug.
_VIS_BINDING_STRONG = True


# ---------------------------------------------------------------------------
# Cloth backend enum
# ---------------------------------------------------------------------------

class ClothBackend(Enum):
    """Supported cloth simulation backends in Isaac Sim 5.1.

    - ``PBD``  : PhysX particle cloth (PBD mass-spring), the path both reference
      repos (GarmentLab / DexGarmentLab) use.
    - ``XPBD`` : PhysX surface deformables (XPBD-FEM corotational), the sanctioned
      5.1 replacement for particle cloth, with an explicit flat rest shape.
    """

    PBD = "pbd"
    XPBD = "xpbd"


# ---------------------------------------------------------------------------
# Physics parameter data classes
# ---------------------------------------------------------------------------

@dataclass
class PBDClothParams:
    """PBD particle-cloth parameters, per the Isaac Sim 5.1 cloth research
    (``doc/reports/cloth_sim_research/RESEARCH_isaaclab_cloth.md``).

    The decisive correction over every earlier attempt: **the collision offsets
    must track the mesh particle spacing.**  NVIDIA's particle-cloth docs:
    ``particle_contact_offset ≈ mean edge length`` and
    ``solid_rest_offset ≈ ½ × mean edge length``.  Offsets larger than ½ edge
    length (our old reference-copied 0.02 / 0.0075) make spring-connected
    neighbours start *inside* each other's contact radius, so PBD injects
    depenetration energy every step → the gradual divergence (F3) and the
    self-collision explosion (F4).  With offsets matched to spacing:
      * high stretch stiffness is stable (PBD *projects* inextensible
        constraints; it does not integrate a stiff spring force), so 1e5 is fine;
      * self-collision can stay ON (graspable two-sided cloth);
      * a realistic 0.15 kg garment settles without sinking — no 1 kg hack.

    Validated in a standalone DexGarmentLab-style settle test (isaacsim.core
    ParticleSystem + ClothPrim, GPU dynamics, a ClothesNet shirt): finite,
    self-colliding, drapes onto the ground and stays bounded.
    """

    # Spring stiffness.  PBD projects stiff constraints stably, so stretch is high
    # (inextensible cloth); bend/shear moderate for cotton drape (research 1e5/100/100).
    stretch_stiffness: float = 1.0e5
    bend_stiffness: float = 100.0
    shear_stiffness: float = 100.0
    spring_damping: float = 0.2

    # PBD material (cotton-ish).
    # Aerodynamic drag / lift are OFF: with them on (0.02) the light cloth floats
    # like plastic foil — slowly settling and carried sideways by "air".  A real
    # garment falls under gravity, so we zero the aero terms and add material
    # damping so motion dissipates quickly instead of drifting.
    drag: float = 0.0
    lift: float = 0.0
    friction: float = 0.3
    damping: float = 0.5

    # Adhesion (for grasping).  adhesion_offset_scale MUST be > 0 or the adhesion
    # force has no range and is silently inert.
    adhesion: float = 0.2
    adhesion_offset_scale: float = 2.0
    particle_adhesion_scale: float = 1.0
    particle_friction_scale: float = 1.0

    # Total garment mass (kg).  Per-particle mass is mass / n_particles; PhysX
    # divides it evenly.  Raised from a (too-light, floaty) 0.15 kg so the shirt
    # has enough weight to fall and hang naturally under gravity.
    mass_kg: float = 0.30

    # Collision offsets.  ``particle_contact_offset`` defaults to the mesh mean
    # edge length (set a float to override); ``solid_rest_offset`` is
    # ``solid_rest_offset_scale × edge`` (½ edge per the cloth docs).  The rest of
    # the hierarchy auto-derives (rest = 0.99·pco, contact = pco, fluid = 0.6·solid).
    particle_contact_offset: float | None = None
    solid_rest_offset_scale: float = 0.5

    # Drop height (m) for the one-off pre-settle (relaxes the spawned mesh onto
    # the belt and adopts the draped shape as the per-reset rest pose).
    presettle_drop_height_m: float = 0.03

    # Solver
    solver_position_iterations: int = 16
    max_neighborhood: int = 96
    # Self-collision ON — with offsets matched to spacing the in-plane springs are
    # filtered (self_collision_filter) and only genuine folds collide, so the
    # garment is graspable/foldable and does not explode.
    global_self_collision: bool = True
    enable_ccd: bool = True

    # Cloth runs on the GPU deformable-surface / particle contact path in 5.1;
    # these buffers (not the rigid one) govern cloth contact overflow.
    gpu_max_deformable_surface_contacts: int = 2 ** 22


# ---------------------------------------------------------------------------
# Cloth object configuration
# ---------------------------------------------------------------------------

@dataclass
class XPBDClothParams:
    """XPBD surface-deformable (FEM) parameters — the sanctioned 5.1 cloth path.

    Surface deformables use an XPBD corotational-FEM model with an explicit flat
    ``restShapePoints`` attribute, so the flat rest shape is first-class (unlike
    particle cloth, whose rest shape is whatever mesh you apply it to).  Material
    is governed by Young's modulus / Poisson ratio / thickness rather than spring
    stiffnesses.  Requires ``/physics/enableDeformableBeta``.
    """

    # Young's modulus: 5e4 diverged (NaN on step 1); 5000 is the validated
    # cotton-soft value (the FEM default solver is ~50 MPa, so this is much
    # softer).  Lower = floppier.
    youngs_modulus: float = 5000.0
    poissons_ratio: float = 0.3
    thickness: float = 1.0e-3          # shell thickness (m)
    dynamic_friction: float = 0.3
    density: float = 350.0             # cotton ~300-400 kg/m^3
    presettle_drop_height_m: float = 0.03


@dataclass
class ClothObjectCfg:
    """Declarative configuration for a cloth asset (PBD or XPBD backend)."""

    prim_path: str = ""
    """Isaac Lab prim path pattern, e.g. ``{ENV_REGEX_NS}/Shirt``."""

    usd_path: str = ""
    """Absolute path to the PBD cloth mesh USD (a flat-laid single-mesh garment)."""

    xpbd_usd_path: str = ""
    """Absolute path to the XPBD variant — the same garment with the surface
    deformable schema + material pre-authored (no physics scene).  Referenced
    per-env when ``backend is ClothBackend.XPBD``."""

    mesh_prim_path: str = ""
    """Mesh prim path *inside* the referenced USD, relative form after the
    reference collapses the default prim — e.g. ``/mesh`` for an asset whose
    default prim ``/World`` has a child ``mesh``."""

    backend: ClothBackend = ClothBackend.PBD
    """Which cloth simulation backend to author / drive."""

    pbd_params: PBDClothParams = field(default_factory=PBDClothParams)
    xpbd_params: XPBDClothParams = field(default_factory=XPBDClothParams)

    init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial position (local to env origin)."""


# ---------------------------------------------------------------------------
# Startup event: disable complex colliders (so PBD cloth uses a box instead)
# ---------------------------------------------------------------------------

def disable_complex_colliders_event(
    env: object,
    env_ids: torch.Tensor | None,
    asset_keys: Tuple[str, ...],
) -> None:
    """Disable every PhysX collider under the named scene assets.

    PBD particle cloth collides robustly only with simple/analytic shapes.
    The conveyor USD ships triangle-mesh and convex-hull colliders that the
    cloth tunnels through, so we turn them off here and rely on a separate
    box collider (see ``ShirtPlaceSceneCfg.conveyor_collider``) for the belt
    surface.  Runs ``prestartup`` (after scene creation, before sim.reset()).
    """
    stage = env.sim.stage
    n_disabled = 0
    for ep in env.scene.env_prim_paths:
        for key in asset_keys:
            root = stage.GetPrimAtPath(Sdf.Path(f"{ep}/{key}"))
            if not root or not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                    n_disabled += 1
    logger.info("Disabled %d complex colliders under %s", n_disabled, asset_keys)


# ---------------------------------------------------------------------------
# Startup event: spawn cloth mesh and apply PBD physics
# ---------------------------------------------------------------------------

def apply_cloth_startup_event(
    env: object,
    env_ids: torch.Tensor | None,
    cloth_cfg: ClothObjectCfg,
) -> None:
    """Startup event — spawn the garment mesh and author the chosen cloth backend.

    Runs between scene creation and ``sim.reset()`` so PhysX recognises the
    schemas at GPU initialisation time.  Dispatches on ``cloth_cfg.backend``
    (PBD particle cloth or XPBD surface deformable).
    """
    if cloth_cfg.backend is ClothBackend.XPBD:
        _apply_xpbd_cloth(env, cloth_cfg)
    else:
        _apply_pbd_cloth(env, cloth_cfg)


def _apply_pbd_cloth(env: object, cloth_cfg: ClothObjectCfg) -> None:
    """Author PhysX PBD particle cloth: one shared particle system, per-env set."""
    from omni.physx.scripts import particleUtils, physicsUtils

    stage = env.sim.stage
    scene_path = env.scene.physics_scene_path
    env_prim_paths = env.scene.env_prim_paths
    params = cloth_cfg.pbd_params

    # Particle cloth requires GPU dynamics; raise the deformable-surface /
    # particle contact buffers (these, not the rigid buffer, govern cloth overflow).
    n_envs = max(1, len(env_prim_paths))
    scene_prim = stage.GetPrimAtPath(Sdf.Path(scene_path))
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    scene_api.CreateEnableGPUDynamicsAttr(True)
    scene_api.CreateBroadphaseTypeAttr("GPU")
    max_surf_contacts = min(2 ** 26, n_envs * params.gpu_max_deformable_surface_contacts)
    scene_api.CreateGpuMaxDeformableSurfaceContactsAttr(int(max_surf_contacts))

    prim_name = cloth_cfg.prim_path.split("/")[-1]  # "Shirt"
    mesh_rel = cloth_cfg.mesh_prim_path.removeprefix("/root")  # "/World/mesh/Mesh"

    # ── Spawn USD reference in each env and collect mesh paths ────────
    mesh_paths: list[str] = []
    for ep in env_prim_paths:
        ref_path = f"{ep}/{prim_name}"
        prim = stage.DefinePrim(ref_path, "Xform")
        prim.GetReferences().AddReference(cloth_cfg.usd_path)
        mesh_paths.append(f"{ref_path}{mesh_rel}")

    # ── Bake transforms into the mesh points; clear all xforms ────────
    # CRITICAL for particle cloth: the render mesh is driven directly by the
    # particle positions, so any non-identity transform on the mesh (or its
    # parents) makes the *rendered* shirt drift away from the *simulated* one
    # (it appeared ~1.5 m below the belt — "underneath the conveyor").  The
    # source garment USD carries internal translate/orient/scale on its mesh
    # hierarchy, so we bake the mesh→env-root transform into the points.
    tc = Usd.TimeCode.Default()
    mesh0 = UsdGeom.Mesh(stage.GetPrimAtPath(Sdf.Path(mesh_paths[0])))
    env0 = stage.GetPrimAtPath(Sdf.Path(env_prim_paths[0]))
    rel = (
        UsdGeom.Xformable(mesh0).ComputeLocalToWorldTransform(tc)
        * UsdGeom.Xformable(env0).ComputeLocalToWorldTransform(tc).GetInverse()
    )
    rel_np = np.array(rel, dtype=np.float64).reshape(4, 4)
    local_pts = np.asarray(mesh0.GetPointsAttr().Get(), dtype=np.float64)
    baked = (np.hstack([local_pts, np.ones((len(local_pts), 1))]) @ rel_np)[:, :3]

    # ── Place the (already flat-laid, centred) garment on the belt ────
    # The asset is a flat single-layer ClothesNet shirt (thin axis = Z, centred at
    # origin, metric).  NO geometric flatten/compression is applied — that hack
    # caused sliver triangles and the self-collision explosions.  We only
    # translate so the centroid sits at init_pos in XY and the lowest particle
    # rests at init_pos.z, so the shirt spawns lying flat ON the belt.
    init_pos = np.asarray(cloth_cfg.init_pos, dtype=np.float64)
    baked[:, 0] += init_pos[0] - baked[:, 0].mean()
    baked[:, 1] += init_pos[1] - baked[:, 1].mean()
    baked[:, 2] += init_pos[2] - baked[:, 2].min()
    logger.info(
        "Cloth placed: ext %s, thickness %.4f m, resting at z=%.3f",
        np.round(baked.max(0) - baked.min(0), 3),
        float(baked[:, 2].max() - baked[:, 2].min()), float(baked[:, 2].min()),
    )
    baked_vt = Vt.Vec3fArray.FromNumpy(baked.astype(np.float32))

    # Clear any xformOps on the reference prim, the (collapsed) /World, and the
    # mesh so the render mesh is driven purely by the particle positions.
    chain = ["", "/World", mesh_rel]
    for ep in env_prim_paths:
        ref_path = f"{ep}/{prim_name}"
        for suf in chain:
            xp = UsdGeom.Xformable(stage.GetPrimAtPath(Sdf.Path(ref_path + suf)))
            if xp:
                xp.ClearXformOpOrder()
        UsdGeom.Mesh(
            stage.GetPrimAtPath(Sdf.Path(f"{ref_path}{mesh_rel}"))
        ).GetPointsAttr().Set(baked_vt)

    # ── Collision offsets keyed to particle spacing (THE research fix) ──
    # NVIDIA particle-cloth docs: particle_contact_offset ≈ mean edge length;
    # solid_rest_offset ≈ ½ edge length.  Offsets larger than ½ edge make
    # spring-connected neighbours start inside each other's contact radius, so the
    # solver injects depenetration energy every step → the gradual divergence and
    # self-collision explosions we used to see.  The rest auto-derive.
    idx = np.asarray(mesh0.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    tris = idx.reshape(-1, 3)
    a, b, c = baked[tris[:, 0]], baked[tris[:, 1]], baked[tris[:, 2]]
    edges = np.concatenate([
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(a - c, axis=1),
    ])
    avg_edge = float(np.mean(edges))
    pco = (
        params.particle_contact_offset
        if params.particle_contact_offset is not None
        else avg_edge
    )
    particle_contact_offset = pco
    contact_offset = pco
    rest_offset = 0.99 * pco
    solid_rest_offset = params.solid_rest_offset_scale * avg_edge
    fluid_rest_offset = 0.6 * solid_rest_offset

    logger.info(
        "Cloth mesh: %d verts, avg edge %.4f m -> pco %.4f contact %.4f rest %.4f solid_rest %.4f",
        len(baked), avg_edge, pco, contact_offset, rest_offset, solid_rest_offset,
    )

    # ── Shared particle system ────────────────────────────────────────
    ps_path = "/World/clothParticleSystem"
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=ps_path,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
        particle_contact_offset=particle_contact_offset,
        solid_rest_offset=solid_rest_offset,
        fluid_rest_offset=fluid_rest_offset,
        solver_position_iterations=params.solver_position_iterations,
        simulation_owner=scene_path,
    )
    ps_prim = stage.GetPrimAtPath(Sdf.Path(ps_path))
    ps_api = PhysxSchema.PhysxParticleSystem(ps_prim)
    ps_api.CreateMaxNeighborhoodAttr(params.max_neighborhood)
    # Self-collision enabled at the *system* level too (not only the per-cloth
    # API); with offsets matched to spacing the in-plane springs are filtered.
    ps_api.CreateGlobalSelfCollisionEnabledAttr(params.global_self_collision)
    ps_api.CreateNonParticleCollisionEnabledAttr(True)
    if params.enable_ccd:
        ps_api.CreateEnableCCDAttr(True)

    # ── PBD material ──────────────────────────────────────────────────
    mat_path = "/World/clothMaterial"
    particleUtils.add_pbd_particle_material(
        stage, mat_path,
        drag=params.drag, lift=params.lift,
        friction=params.friction, damping=params.damping,
    )
    mat_prim = stage.GetPrimAtPath(Sdf.Path(mat_path))
    pbd_api = PhysxSchema.PhysxPBDMaterialAPI(mat_prim)
    pbd_api.CreateAdhesionAttr(params.adhesion)
    # Adhesion is inert unless adhesionOffsetScale > 0 — without it the
    # configured grasp adhesion does nothing.  This was the missing piece that
    # made PBD cloth grasping unreliable.
    pbd_api.CreateAdhesionOffsetScaleAttr(params.adhesion_offset_scale)
    pbd_api.CreateParticleAdhesionScaleAttr(params.particle_adhesion_scale)
    pbd_api.CreateParticleFrictionScaleAttr(params.particle_friction_scale)
    physicsUtils.add_physics_material_to_prim(
        stage, ps_prim, Sdf.Path(mat_path),
    )

    # ── Apply cloth to each env's mesh ────────────────────────────────
    for mp in mesh_paths:
        mesh_prim = stage.GetPrimAtPath(Sdf.Path(mp))
        mesh_geom = UsdGeom.Mesh(mesh_prim)
        mesh_geom.GetSubdivisionSchemeAttr().Set("none")
        # Cloth must render double-sided: once the closed shell collapses/folds,
        # the visible side faces away from the camera and a single-sided mesh is
        # back-face culled (the shirt looked invisible from above).
        mesh_geom.CreateDoubleSidedAttr(True)
        particleUtils.add_physx_particle_cloth(
            stage=stage,
            path=mp,
            dynamic_mesh_path=None,
            particle_system_path=ps_path,
            spring_stretch_stiffness=params.stretch_stiffness,
            spring_bend_stiffness=params.bend_stiffness,
            spring_shear_stiffness=params.shear_stiffness,
            spring_damping=params.spring_damping,
            self_collision=params.global_self_collision,
            self_collision_filter=params.global_self_collision,
        )
        UsdPhysics.MassAPI.Apply(mesh_prim).GetMassAttr().Set(params.mass_kg)

    # ── Apply a visible UsdPreviewSurface material to the cloth ───────
    # The source garment binds its faces through GeomSubsets to materials
    # whose diffuse textures use a relative path that does NOT resolve once
    # the mesh is referenced under ``{ENV}/Shirt`` — which left the shirt
    # rendering invisible.  We author one shared UsdPreviewSurface material and
    # bind it *stronger than descendants* (and directly onto each GeomSubset)
    # so it reliably overrides the broken subset bindings on every env.
    vis_mat_path = "/World/clothVisualMaterial"
    vis_mat = UsdShade.Material.Define(stage, vis_mat_path)
    shader = UsdShade.Shader.Define(stage, f"{vis_mat_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    # A vivid, slightly emissive red so the (thin, flat) shirt reads clearly as
    # a garment against the blue/grey belt and drum under the bright dome light,
    # and is easy to follow during grasp/transport.
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.85, 0.06, 0.06),
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.25, 0.0, 0.0),
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    strength = (
        UsdShade.Tokens.strongerThanDescendants
        if _VIS_BINDING_STRONG
        else UsdShade.Tokens.weakerThanDescendants
    )
    for mp in mesh_paths:
        mesh_prim = stage.GetPrimAtPath(Sdf.Path(mp))
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(vis_mat, strength)
        # Also override the per-face subset bindings directly.
        for child in mesh_prim.GetChildren():
            if child.IsA(UsdGeom.Subset):
                UsdShade.MaterialBindingAPI.Apply(child).Bind(vis_mat, strength)

    logger.info(
        "PBD cloth applied to %d envs (particle system: %s)", len(mesh_paths), ps_path,
    )


def _apply_xpbd_cloth(env: object, cloth_cfg: ClothObjectCfg) -> None:
    """Spawn the XPBD surface-deformable garment, one per env, placed on the belt.

    The surface-deformable schema + material are *pre-authored* into
    ``cloth_cfg.xpbd_usd_path`` (offline, no physics scene) because PhysX cooks a
    runtime-applied deformable on a *referenced* mesh unreliably, whereas a
    pre-authored-then-referenced one cooks correctly.  The flat ``restShapePoints``
    baked into that asset make the garment settle flat.  Here we only: enable the
    deformable-beta flag, set the scene's GPU/deformable buffers, reference the
    asset per env, and translate it onto the belt (a rigid translation induces no
    FEM strain, so the cloth stays put and then falls/drapes).
    """
    import carb

    stage = env.sim.stage
    scene_path = env.scene.physics_scene_path
    env_prim_paths = env.scene.env_prim_paths
    params = cloth_cfg.xpbd_params
    assert cloth_cfg.xpbd_usd_path, "ClothObjectCfg.xpbd_usd_path must be set for the XPBD backend"

    # Surface deformables are a 5.1 beta feature gated behind this carb flag.
    carb.settings.get_settings().set_bool("/physics/enableDeformableBeta", True)

    # Scene: GPU dynamics + raised deformable-surface contact / collision buffers.
    n_envs = max(1, len(env_prim_paths))
    scene_prim = stage.GetPrimAtPath(Sdf.Path(scene_path))
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    scene_api.CreateEnableGPUDynamicsAttr(True)
    scene_api.CreateBroadphaseTypeAttr("GPU")
    scene_api.CreateGpuCollisionStackSizeAttr(2 ** 31 - 1)
    max_surf = min(2 ** 26, n_envs * 4_000_000)
    scene_api.CreateGpuMaxDeformableSurfaceContactsAttr(int(max_surf))

    prim_name = cloth_cfg.prim_path.split("/")[-1]
    mesh_rel = cloth_cfg.mesh_prim_path.removeprefix("/root")

    mesh_paths: list[str] = []
    for ep in env_prim_paths:
        ref_path = f"{ep}/{prim_name}"
        prim = stage.DefinePrim(ref_path, "Xform")
        prim.GetReferences().AddReference(cloth_cfg.xpbd_usd_path)
        mesh_paths.append(f"{ref_path}{mesh_rel}")

    # Bake the env transform and translate onto the belt (centroid XY -> init_pos
    # XY, lowest node at init_pos.z).  restShapePoints stay as authored (flat at
    # origin); a rigid translation is strain-free for the corotational FEM.
    tc = Usd.TimeCode.Default()
    mesh0 = UsdGeom.Mesh(stage.GetPrimAtPath(Sdf.Path(mesh_paths[0])))
    env0 = stage.GetPrimAtPath(Sdf.Path(env_prim_paths[0]))
    rel = (
        UsdGeom.Xformable(mesh0).ComputeLocalToWorldTransform(tc)
        * UsdGeom.Xformable(env0).ComputeLocalToWorldTransform(tc).GetInverse()
    )
    rel_np = np.array(rel, dtype=np.float64).reshape(4, 4)
    local_pts = np.asarray(mesh0.GetPointsAttr().Get(), dtype=np.float64)
    baked = (np.hstack([local_pts, np.ones((len(local_pts), 1))]) @ rel_np)[:, :3]
    init_pos = np.asarray(cloth_cfg.init_pos, dtype=np.float64)
    baked[:, 0] += init_pos[0] - baked[:, 0].mean()
    baked[:, 1] += init_pos[1] - baked[:, 1].mean()
    baked[:, 2] += init_pos[2] - baked[:, 2].min()
    baked_vt = Vt.Vec3fArray.FromNumpy(baked.astype(np.float32))

    chain = ["", "/World", mesh_rel]
    for ep in env_prim_paths:
        ref_path = f"{ep}/{prim_name}"
        for suf in chain:
            xp = UsdGeom.Xformable(stage.GetPrimAtPath(Sdf.Path(ref_path + suf)))
            if xp:
                xp.ClearXformOpOrder()
        UsdGeom.Mesh(
            stage.GetPrimAtPath(Sdf.Path(f"{ref_path}{mesh_rel}"))
        ).GetPointsAttr().Set(baked_vt)

    # Override per-env material params from the cfg (so tuning the dataclass takes
    # effect without re-authoring the asset).
    for ep in env_prim_paths:
        matp = stage.GetPrimAtPath(Sdf.Path(f"{ep}/{prim_name}/deformableMaterial"))
        if matp and matp.IsValid():
            for attr, val in (
                ("omniphysics:youngsModulus", params.youngs_modulus),
                ("omniphysics:poissonsRatio", params.poissons_ratio),
                ("omniphysics:surfaceThickness", params.thickness),
                ("omniphysics:density", params.density),
                ("omniphysics:dynamicFriction", params.dynamic_friction),
            ):
                a = matp.GetAttribute(attr)
                if a:
                    a.Set(val)

    # Visible red material (bound for the *default* render purpose so it does not
    # clash with the physics-purpose deformable material binding).
    _bind_red_visual(stage, mesh_paths)
    logger.info("XPBD surface-deformable cloth applied to %d envs", len(mesh_paths))


def _bind_red_visual(stage, mesh_paths: list[str]) -> None:
    """Bind a vivid red UsdPreviewSurface to each cloth mesh (default purpose)."""
    vis_mat_path = "/World/clothVisualMaterial"
    if not stage.GetPrimAtPath(Sdf.Path(vis_mat_path)).IsValid():
        vis_mat = UsdShade.Material.Define(stage, vis_mat_path)
        shader = UsdShade.Shader.Define(stage, f"{vis_mat_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.06, 0.06))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    vis_mat = UsdShade.Material(stage.GetPrimAtPath(Sdf.Path(vis_mat_path)))
    for mp in mesh_paths:
        mesh_prim = stage.GetPrimAtPath(Sdf.Path(mp))
        UsdGeom.Mesh(mesh_prim).CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
            vis_mat, UsdShade.Tokens.strongerThanDescendants)


# ---------------------------------------------------------------------------
# ClothObject — GPU tensor-backed particle cloth wrapper
# ---------------------------------------------------------------------------

class ClothObject:
    """PBD particle cloth wrapper with GPU tensor access for parallel envs.

    Create *after* ``sim.reset()`` so the GPU particle buffers exist.
    """

    def __init__(
        self,
        cfg: ClothObjectCfg,
        num_envs: int,
        device: str,
    ) -> None:
        import omni.physics.tensors.impl.api as physx
        from isaaclab.sim.utils.stage import get_current_stage_id

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._backend = cfg.backend

        # Build glob pattern for the cloth view
        prim_name = cfg.prim_path.split("/")[-1]
        mesh_rel = cfg.mesh_prim_path.removeprefix("/root")
        self._cloth_pattern = f"/World/envs/env_*/{prim_name}{mesh_rel}"

        # Create simulation view + the backend-specific body view.
        stage_id = get_current_stage_id()
        self._sim_view = physx.create_simulation_view("torch", stage_id)
        if self._backend is ClothBackend.XPBD:
            self._cloth_view = self._sim_view.create_surface_deformable_body_view(
                self._cloth_pattern
            )
        else:
            self._cloth_view = self._sim_view.create_particle_cloth_view(
                self._cloth_pattern
            )

        try:
            count = self._cloth_view.count
        except Exception:
            count = 0
        if self._cloth_view is None or count != num_envs:
            raise RuntimeError(
                f"Failed to create a {self._backend.value} cloth view for pattern "
                f"'{self._cloth_pattern}' (got count={count}, expected {num_envs}). "
                f"Ensure the cloth schemas were applied before sim.reset() and that "
                f"GPU dynamics / enableDeformableBeta are set."
            )

        if self._backend is ClothBackend.XPBD:
            self._max_particles = self._cloth_view.max_simulation_nodes_per_body
        else:
            self._max_particles = self._cloth_view.max_particles_per_cloth

        # Store default state (for reset) — always kept as flat [num_envs, P*3].
        self._default_pos = self._get_positions().clone()
        self._default_vel = self._get_velocities().clone()

        # Working buffers — updated by update()
        self._pos_flat = self._default_pos.clone()
        self._vel_flat = self._default_vel.clone()

        self._ALL_INDICES = torch.arange(
            num_envs, device=device, dtype=torch.int32,
        )

        # ── Deterministic attachment-grasp state (per env) ────────────────
        # Mirrors the GarmentLab / DexGarmentLab AttachmentBlock: particles
        # within a radius of the gripper tip are welded to it and tracked as the
        # arm moves.  ``attach`` freezes the per-particle offset, ``hold`` re-pins
        # each control step, ``detach`` releases.
        self._attach_topk = 50  # particles averaged for ``highest_point_w``
        self._attach_mask = torch.zeros(
            num_envs, self._max_particles, dtype=torch.bool, device=device,
        )
        self._attach_offset = torch.zeros(
            num_envs, self._max_particles, 3, device=device,
        )
        self._attached = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Canonical rest shape, laid *flat*.  The cloth was flattened at apply
        # time along its thinnest axis, so here we orient that thin (sheet
        # normal) axis to +Z and the two large axes into the belt (XY) plane.
        # Detecting the thin axis (instead of hardcoding the source orientation)
        # keeps this correct regardless of how the garment USD is authored.
        dp0 = self._default_pos.view(num_envs, self._max_particles, 3)[0]
        rest_local = dp0 - dp0.mean(dim=0, keepdim=True)
        ext = rest_local.max(dim=0).values - rest_local.min(dim=0).values
        thin = int(torch.argmin(ext).item())
        plane = [a for a in (0, 1, 2) if a != thin]
        self._flat_rest = torch.stack(
            [rest_local[:, plane[0]], rest_local[:, plane[1]], rest_local[:, thin]],
            dim=-1,
        )  # [P, 3], centroid ≈ origin, thin axis → Z
        # Bottom of the laid-flat shirt relative to its centroid (negative).
        self._flat_rest_min_z = float(self._flat_rest[:, 2].min())
        # Long-axis (Y) extent, used for folding.
        self._flat_y_min = float(self._flat_rest[:, 1].min())
        self._flat_y_max = float(self._flat_rest[:, 1].max())
        self._flat_thickness = float(
            self._flat_rest[:, 2].max() - self._flat_rest[:, 2].min()
        )

        logger.info(
            "ClothObject: %d envs, %d particles/cloth, pattern='%s'",
            num_envs, self._max_particles, self._cloth_pattern,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_particles(self) -> int:
        """Number of particles (vertices) per cloth."""
        return self._max_particles

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        """Particle world positions ``[num_envs, num_particles, 3]``."""
        return self._pos_flat.view(self.num_envs, self._max_particles, 3)

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        """Particle world velocities ``[num_envs, num_particles, 3]``."""
        return self._vel_flat.view(self.num_envs, self._max_particles, 3)

    @property
    def centroid_pos_w(self) -> torch.Tensor:
        """Cloth centroid world position ``[num_envs, 3]``."""
        return self.nodal_pos_w.mean(dim=1)

    @property
    def centroid_vel_w(self) -> torch.Tensor:
        """Cloth centroid world velocity ``[num_envs, 3]``."""
        return self.nodal_vel_w.mean(dim=1)

    # ── Backend-specific view I/O (kept flat [num_envs, P*3] internally) ──

    def _get_positions(self) -> torch.Tensor:
        if self._backend is ClothBackend.XPBD:
            return self._cloth_view.get_simulation_nodal_positions().reshape(
                self.num_envs, self._max_particles * 3)
        return self._cloth_view.get_positions()

    def _get_velocities(self) -> torch.Tensor:
        if self._backend is ClothBackend.XPBD:
            return self._cloth_view.get_simulation_nodal_velocities().reshape(
                self.num_envs, self._max_particles * 3)
        return self._cloth_view.get_velocities()

    def _set_positions(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        if self._backend is ClothBackend.XPBD:
            self._cloth_view.set_simulation_nodal_positions(
                flat.view(self.num_envs, self._max_particles, 3), indices)
        else:
            self._cloth_view.set_positions(flat, indices)

    def _set_velocities(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        if self._backend is ClothBackend.XPBD:
            self._cloth_view.set_simulation_nodal_velocities(
                flat.view(self.num_envs, self._max_particles, 3), indices)
        else:
            self._cloth_view.set_velocities(flat, indices)

    # ── Runtime methods ───────────────────────────────────────────────

    def update(self) -> None:
        """Read positions and velocities from GPU PhysX buffers."""
        self._pos_flat = self._get_positions()
        self._vel_flat = self._get_velocities()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset cloth to its spawned default state for *env_ids*."""
        if len(env_ids) == 0:
            return

        indices = env_ids.to(dtype=torch.int32, device=self.device)

        # Write default positions
        self._pos_flat[env_ids] = self._default_pos[env_ids]
        self._set_positions(self._pos_flat, indices)

        # Write zero velocities
        self._vel_flat[env_ids] = self._default_vel[env_ids]
        self._set_velocities(self._vel_flat, indices)

    # ── Properties for randomized placement ───────────────────────────

    @property
    def flat_rest_min_z(self) -> float:
        """Lowest point of the laid-flat shirt relative to its centroid (≤ 0)."""
        return self._flat_rest_min_z

    def recompute_flat_rest_from_current(self) -> None:
        """Adopt the current (settled) shape of env 0 as the canonical rest.

        Used after a one-off pre-settle so subsequent resets place an already
        relaxed, collapsed sheet — avoiding the first-frame self-collision pop
        of the raw, laid-flat closed garment mesh.
        """
        self.update()
        pts = self.nodal_pos_w[0]
        self._flat_rest = (pts - pts.mean(dim=0, keepdim=True)).clone()
        self._flat_rest_min_z = float(self._flat_rest[:, 2].min())
        self._flat_y_min = float(self._flat_rest[:, 1].min())
        self._flat_y_max = float(self._flat_rest[:, 1].max())
        self._flat_thickness = float(
            self._flat_rest[:, 2].max() - self._flat_rest[:, 2].min()
        )

    def _apply_fold(
        self, base: torch.Tensor, fold_fracs: torch.Tensor,
    ) -> torch.Tensor:
        """Fold the laid-flat shirt about a hinge across its long (Y) axis.

        For each env with ``fold_fracs > 0`` the portion beyond the hinge line
        (``y > y_hinge``) is reflected over the hinge and lifted by the cloth
        thickness, producing a half-folded initial state.  ``fold_fracs <= 0``
        leaves the shirt unfolded.  Vectorised over envs and particles.

        Args:
            base:       ``[K, P, 3]`` laid-flat positions (modified copy returned).
            fold_fracs: ``[K]`` hinge position as a fraction of the Y extent,
                        or ``<= 0`` to skip folding for that env.
        """
        y_min, y_max = self._flat_y_min, self._flat_y_max
        y_hinge = y_min + fold_fracs.clamp(min=0.0, max=1.0) * (y_max - y_min)  # [K]
        do_fold = (fold_fracs > 0.0).view(-1, 1)                                # [K,1]
        y = base[..., 1]                                                        # [K,P]
        beyond = (y > y_hinge.view(-1, 1)) & do_fold                            # [K,P]
        # Reflect Y about the hinge and stack on top by ~1.2× the thickness.
        y_folded = 2.0 * y_hinge.view(-1, 1) - y
        z_folded = base[..., 2] + 1.2 * self._flat_thickness
        base[..., 1] = torch.where(beyond, y_folded, base[..., 1])
        base[..., 2] = torch.where(beyond, z_folded, base[..., 2])
        return base

    def reset_randomized(
        self,
        env_ids: torch.Tensor,
        centroids_w: torch.Tensor,
        yaws: torch.Tensor,
        fold_fracs: torch.Tensor | None = None,
    ) -> None:
        """Reset the cloth laid flat at a randomized pose (+ optional fold).

        Args:
            env_ids:     ``[K]`` long env indices to reset.
            centroids_w: ``[K, 3]`` target world centroid for each shirt.
            yaws:        ``[K]`` yaw angle (rad) about the vertical axis.
            fold_fracs:  optional ``[K]`` fold hinge fraction (≤ 0 = no fold).
        """
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        k = env_ids.numel()
        if k == 0:
            return
        p = self._max_particles

        base = self._flat_rest.unsqueeze(0).expand(k, p, 3).clone()  # [K,P,3]
        if fold_fracs is not None:
            base = self._apply_fold(base, fold_fracs.to(self.device))

        cy, sy = torch.cos(yaws), torch.sin(yaws)
        rot = torch.zeros(k, 3, 3, device=self.device, dtype=base.dtype)
        rot[:, 0, 0], rot[:, 0, 1] = cy, -sy
        rot[:, 1, 0], rot[:, 1, 1] = sy, cy
        rot[:, 2, 2] = 1.0
        new_world = torch.einsum("kij,kpj->kpi", rot, base) + centroids_w.unsqueeze(1)

        indices = env_ids.to(dtype=torch.int32)
        self._pos_flat[env_ids] = new_world.reshape(k, p * 3)
        self._set_positions(self._pos_flat, indices)
        self._vel_flat[env_ids] = 0.0
        self._set_velocities(self._vel_flat, indices)

    def write_nodal_pos_to_sim(
        self,
        positions: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Write arbitrary particle positions into the simulation.

        Args:
            positions: Flat tensor ``[len(env_ids), max_particles * 3]``.
            env_ids: Environment indices to write.
        """
        indices = env_ids.to(dtype=torch.int32, device=self.device)
        self._pos_flat[env_ids] = positions
        self._set_positions(self._pos_flat, indices)

    # ── Deterministic attachment grasp ────────────────────────────────────
    # Batched generalisation of the validated single-env weld in
    # ``run_shirt_validation.py`` (GarmentLab / DexGarmentLab AttachmentBlock
    # logic): instead of relying on PBD friction — which does not lift this
    # cloth — the particles around the gripper tip are welded to it and tracked.

    @property
    def highest_point_w(self) -> torch.Tensor:
        """Mean position of the top-N highest particles per env ``[num_envs, 3]``.

        A *stable* highest region of the cloth (a single argmax jitters across
        the near-coplanar top layer of a flat-laid sheet).  Used as the
        deterministic grasp target; once the cloth is lifted by its welded patch,
        that patch is the highest region, so this tracks the gripper tip.
        """
        pts = self.nodal_pos_w                                   # [N, P, 3]
        k = min(self._attach_topk, self._max_particles)
        idx = torch.topk(pts[:, :, 2], k, dim=1).indices         # [N, k]
        top = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        return top.mean(dim=1)

    @property
    def is_attached(self) -> torch.Tensor:
        """Per-env bool ``[num_envs]``: True while a welded grasp is held."""
        return self._attached

    def attach(
        self,
        env_ids: torch.Tensor,
        grasp_centers: torch.Tensor,
        radius: float,
    ) -> None:
        """Weld every particle within ``radius`` of the grasp centre to the tip.

        Args:
            env_ids:       ``[k]`` env indices to (try to) attach.
            grasp_centers: ``[num_envs, 3]`` gripper-tip world positions.
            radius:        attachment overlap radius (m).
        """
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        gc = grasp_centers[env_ids]                              # [k, 3]
        pts = self.nodal_pos_w[env_ids]                          # [k, P, 3]
        dist = torch.norm(pts - gc.unsqueeze(1), dim=2)          # [k, P]
        mask = dist < radius                                     # [k, P]
        caught = mask.any(dim=1)                                 # [k]
        # Only weld envs that actually caught particles.
        mask = mask & caught.unsqueeze(1)
        self._attach_mask[env_ids] = mask
        self._attach_offset[env_ids] = pts - gc.unsqueeze(1)     # masked use only
        self._attached[env_ids] = caught

    def hold(self, grasp_centers: torch.Tensor) -> None:
        """Re-pin welded particles to ``grasp_centre + frozen_offset``.

        Call once per control step for all attached envs.  Non-welded particles
        keep their simulated positions, so the rest of the shirt hangs from the
        grasped patch via the PBD constraints.
        """
        attached_ids = torch.nonzero(self._attached, as_tuple=False).squeeze(-1)
        if attached_ids.numel() == 0:
            return
        gc = grasp_centers[attached_ids]                         # [m, 3]
        mask = self._attach_mask[attached_ids]                   # [m, P]
        target = gc.unsqueeze(1) + self._attach_offset[attached_ids]  # [m, P, 3]
        pos = self.nodal_pos_w[attached_ids].clone()             # [m, P, 3]
        pos = torch.where(mask.unsqueeze(-1), target, pos)
        m = attached_ids.numel()
        self.write_nodal_pos_to_sim(pos.reshape(m, -1), attached_ids)
        # Zero the welded particles' velocity so the solver does not fling them.
        vel = self.nodal_vel_w[attached_ids].clone()
        vel = torch.where(mask.unsqueeze(-1), torch.zeros_like(vel), vel)
        self._vel_flat[attached_ids] = vel.reshape(m, -1)
        self._set_velocities(self._vel_flat, attached_ids.to(torch.int32))

    def detach(self, env_ids: torch.Tensor) -> None:
        """Release the welded grasp for ``env_ids``."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self._attached[env_ids] = False
        self._attach_mask[env_ids] = False

    def reset_attachment(self, env_ids: torch.Tensor) -> None:
        """Clear attachment state on env reset (alias of :meth:`detach`)."""
        self.detach(env_ids)

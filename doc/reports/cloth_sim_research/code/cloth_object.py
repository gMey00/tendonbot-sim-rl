"""PBD particle cloth integration for Isaac Lab environments.

Uses ``omni.physx.scripts.particleUtils`` for USD schema application and
``omni.physics.tensors.impl.api.ParticleClothView`` for GPU tensor access.

Typical lifecycle:
  1. Startup event calls ``apply_cloth_startup_event()`` — spawns the
     T-shirt mesh per-env and applies PBD cloth schemas.  Runs between
     scene creation and ``sim.reset()``.
  2. After ``sim.reset()``, the custom env creates ``ClothObject()`` which
     acquires the ``ParticleClothView`` and stores default state.
  3. Each env step: ``update()`` reads particle positions/velocities.
  4. Each env reset: ``reset(env_ids)`` writes default positions/velocities.

The RL task keeps a kinematic rigid-body proxy (``shirt_proxy``) whose
position is synced to the cloth centroid each step, so all existing reward
and observation functions work unchanged.
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
    """Supported cloth simulation backends in Isaac Sim 5.1."""

    PBD = "pbd"


# ---------------------------------------------------------------------------
# Physics parameter data classes
# ---------------------------------------------------------------------------

@dataclass
class PBDClothParams:
    """PBD particle cloth material and solver parameters.

    Defaults tuned for a cotton T-shirt (~0.2 kg, ~6700 vertices).

    The stiffness regime (high stretch, moderate bend/shear, self-collision ON)
    follows the two reference Isaac Sim cloth pipelines:
      * GarmentLab — ``Env/Config/GarmentConfig.py``
        (stretch 1e4, bend 100, shear 100, global_self_collision=True)
      * DexGarmentLab — ``Env_Config/Garment/Particle_Garment.py``
        (stretch 1e6–1e12, bend 100, shear 100, contact_offset 0.010,
        rest_offset 0.0075, global_self_collision_enabled=True)
    Real cloth is essentially *inextensible* (very high stretch stiffness) and
    soft only in *bending*; the earlier low-stretch (300) / near-zero-bend (0.8)
    setup behaved like a stretchy rubber sheet, not cotton.
    """

    # Spring stiffness.  Stretch is very high so the panels keep their area
    # (cloth does not stretch — cf. GarmentLab stretch=1e4, DexGarmentLab 1e6+).
    # Bend/shear are moderate-low for a soft cotton hand that still folds and
    # drapes (the references use 100; we run softer for a limper tee).  The shirt
    # is genuinely flattened at apply time (see ``flatten_scale``), so we no
    # longer rely on near-zero bend to fight the source mesh's 3-D shoulder
    # curvature.
    stretch_stiffness: float = 800.0
    bend_stiffness: float = 20.0
    shear_stiffness: float = 20.0
    spring_damping: float = 0.5

    # PBD material
    drag: float = 0.02
    lift: float = 0.02
    friction: float = 0.8
    damping: float = 0.3

    # Adhesion (for grasping).  adhesion_offset_scale MUST be > 0 or the
    # adhesion force has no range and is silently inert.
    adhesion: float = 0.2
    adhesion_offset_scale: float = 2.0
    particle_adhesion_scale: float = 1.0
    particle_friction_scale: float = 1.0

    # Mass of the whole garment (kg).  Heavier than a real tee (~0.15 kg): with
    # the very light per-particle mass of a 6.7k-vertex mesh, contact forces from
    # the belt produce large accelerations and a resting cloth jitters into
    # divergence.  A heavier garment (gravity-dominated) settles stably and still
    # drapes well.  Paired with the reference-style contact offsets below.
    # (At 0.5 kg the cloth slowly penetrated the belt collider over an episode;
    # 1.0 kg gives stable contact and rests on the belt without sinking.)
    mass_kg: float = 1.0

    # Geometric flattening applied to the source mesh at spawn time.  The source
    # ``tshirt_mod.usdc`` is modelled as a *worn / inflated* 3-D shirt: its front
    # and back panels are separated by ~0.25 m along the body-depth axis.  We
    # reorient that thin axis to vertical and *scale* it by this factor so the
    # garment becomes a flat double layer (~0.25 m × 0.06 ≈ 1.5 cm thick).
    # Because PhysX bakes the spring rest-lengths from the mesh AT APPLY TIME,
    # this flat sheet becomes the zero-energy rest shape — the shirt stays flat
    # instead of re-inflating to the puffy shell.  We *scale* (not project to a
    # plane): projecting collapses the seam triangles to zero area and explodes
    # the solver; scaling keeps every triangle non-degenerate.
    #
    # ~0.277 m × flatten_scale ≈ 1.4 cm: a realistic flat-folded-tee thickness.
    # (With self-collision OFF there is no inter-layer repulsion, so the gap can
    # be small; the flattened sheet is simply the rest shape and stays flat.)
    flatten_scale: float = 0.05

    # Drop height (m) used by the one-off pre-settle: the flattened sheet is
    # placed this far above the belt and allowed to relax onto it.
    presettle_drop_height_m: float = 0.02

    # Clamp on per-particle speed.  Without it, occasional PBD spring/contact
    # overshoot launches individual particles, which render as flickering spikes
    # around the shirt.  A modest cap removes the spikes without affecting the
    # bulk drape dynamics.
    max_velocity: float = 5.0

    # Collision offsets (m).  If None they are derived from the mean edge length
    # (rest = 0.5·edge, contact = 1.5·rest).  The references use *larger* fixed
    # offsets — GarmentLab ``particle_contact_offset`` 0.02, DexGarmentLab
    # contact 0.010 / rest 0.0075 — which give a thicker, less jittery contact
    # band.  A too-thin band lets a resting cloth vibrate against the collider and
    # slowly pump energy until the solver diverges, so we expose these to tune.
    rest_offset: float | None = 0.0075
    contact_offset: float | None = 0.01
    particle_contact_offset: float | None = 0.02

    # Solver
    solver_position_iterations: int = 20
    max_neighborhood: int = 96
    # Self-collision OFF.  GarmentLab/DexGarmentLab enable it because they load
    # *natively flat* garment patterns; our source mesh is a worn/inflated shell
    # that we *geometrically compress* to a flat sheet, which spawns the two
    # layers ~1.4 cm apart — inside the self-collision contact radius — so turning
    # self-collision on makes the layers explosively repel and the solver blows
    # up (positions → 1e11).  With the flat sheet already the rest shape, the
    # shirt stays flat and drapes softly without it; self-collision only adds
    # fold-on-fold realism, which is not worth the instability for this asset.
    global_self_collision: bool = False
    enable_ccd: bool = False

    # In Isaac Sim 5.1 PBD particle cloth is processed by the FEM cloth solver,
    # which uses the *deformable-surface* contact buffer.  This caps that buffer
    # (scaled by env count at apply time, capped for VRAM).  With self-collision
    # OFF the contact count is modest and no overflow occurs in practice; the cap
    # mainly matters if self-collision is re-enabled for a natively flat garment.
    gpu_max_deformable_surface_contacts: int = 2 ** 21


# ---------------------------------------------------------------------------
# Cloth object configuration
# ---------------------------------------------------------------------------

@dataclass
class ClothObjectCfg:
    """Declarative configuration for a PBD particle cloth asset."""

    prim_path: str = ""
    """Isaac Lab prim path pattern, e.g. ``{ENV_REGEX_NS}/Shirt``."""

    usd_path: str = ""
    """Absolute path to the cloth mesh USD file."""

    mesh_prim_path: str = ""
    """Mesh prim path *inside* the USD file, e.g. ``/root/World/mesh/Mesh``."""

    pbd_params: PBDClothParams = field(default_factory=PBDClothParams)

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
    """Startup event — spawn T-shirt mesh and apply PBD particle cloth.

    Runs between scene creation and ``sim.reset()`` so that PhysX
    recognises the cloth schemas at GPU initialisation time.
    """
    from omni.physx.scripts import particleUtils, physicsUtils

    stage = env.sim.stage
    scene_path = env.scene.physics_scene_path
    env_prim_paths = env.scene.env_prim_paths
    params = cloth_cfg.pbd_params

    # PBD cloth uses the FEM deformable-surface contact buffer in Isaac Sim 5.1;
    # raise its cap (scaled by env count, capped for VRAM) so self-collision
    # contacts are not dropped.
    n_envs = max(1, len(env_prim_paths))
    scene_prim = stage.GetPrimAtPath(Sdf.Path(scene_path))
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    max_surf_contacts = min(2 ** 25, n_envs * params.gpu_max_deformable_surface_contacts)
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

    # ── Flatten the worn/inflated shell into a flat laid-out sheet ────
    # The source mesh is a 3-D shirt: its front and back panels are separated by
    # ~0.25 m along its *thinnest* axis (the body-depth direction).  We:
    #   1. reorient that thin axis to vertical (+Z), laying the two large axes
    #      (shoulder width, neck→waist length) into the horizontal belt plane;
    #   2. *scale* the thin axis by ``flatten_scale`` so the panels collapse into
    #      a ~1.5 cm two-layer sheet (a scale, NOT a projection — projecting to a
    #      plane collapses the seam triangles to zero area and blows up PhysX);
    #   3. place the sheet so its lowest point rests at ``init_pos`` (just above
    #      the belt), centred in XY on ``init_pos`` — so the very first frame
    #      shows the shirt lying flat ON the belt, never standing through it.
    # PhysX bakes spring rest-lengths from these points, so the flat sheet is the
    # zero-energy rest shape and the shirt stays flat instead of re-inflating.
    baked = baked - baked.mean(axis=0)
    ext = baked.max(axis=0) - baked.min(axis=0)
    thin = int(np.argmin(ext))
    plane = [a for a in (0, 1, 2) if a != thin]
    flat = np.stack([baked[:, plane[0]], baked[:, plane[1]], baked[:, thin]], axis=1)
    flat[:, 2] *= params.flatten_scale
    init_pos = np.asarray(cloth_cfg.init_pos, dtype=np.float64)
    flat[:, 0] += init_pos[0]
    flat[:, 1] += init_pos[1]
    flat[:, 2] += init_pos[2] - flat[:, 2].min()  # lowest particle at init_pos.z
    baked = flat
    logger.info(
        "Cloth flattened: source ext %s (thin axis %d) -> flat sheet %s, "
        "thickness %.4f m, resting at z=%.3f",
        np.round(ext, 3), thin, np.round(baked.max(0) - baked.min(0), 3),
        float(baked[:, 2].max() - baked[:, 2].min()), float(baked[:, 2].min()),
    )
    baked_vt = Vt.Vec3fArray.FromNumpy(baked.astype(np.float32))

    chain = ["", "/World", "/World/mesh", "/World/mesh/Mesh"]
    for ep in env_prim_paths:
        ref_path = f"{ep}/{prim_name}"
        for suf in chain:
            xp = UsdGeom.Xformable(stage.GetPrimAtPath(Sdf.Path(ref_path + suf)))
            if xp:
                xp.ClearXformOpOrder()
        UsdGeom.Mesh(
            stage.GetPrimAtPath(Sdf.Path(f"{ref_path}{mesh_rel}"))
        ).GetPointsAttr().Set(baked_vt)

    # ── Offsets from the baked (world-scale) geometry ────────────────
    idx = np.asarray(mesh0.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    tris = idx.reshape(-1, 3)
    a, b, c = baked[tris[:, 0]], baked[tris[:, 1]], baked[tris[:, 2]]
    edges = np.concatenate([
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(a - c, axis=1),
    ])
    avg_edge = float(np.mean(edges))
    rest_offset = params.rest_offset if params.rest_offset is not None else 0.5 * avg_edge
    contact_offset = (
        params.contact_offset if params.contact_offset is not None else rest_offset * 1.5
    )
    particle_contact_offset = (
        params.particle_contact_offset
        if params.particle_contact_offset is not None
        else contact_offset
    )

    logger.info(
        "Cloth mesh: %d verts, avg edge %.4f m, rest_offset %.4f, contact_offset %.4f, "
        "particle_contact_offset %.4f",
        len(baked), avg_edge, rest_offset, contact_offset, particle_contact_offset,
    )

    # ── Shared particle system ────────────────────────────────────────
    ps_path = "/World/clothParticleSystem"
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=ps_path,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
        particle_contact_offset=particle_contact_offset,
        solid_rest_offset=rest_offset,
        fluid_rest_offset=0.0,
        solver_position_iterations=params.solver_position_iterations,
        simulation_owner=scene_path,
    )
    ps_prim = stage.GetPrimAtPath(Sdf.Path(ps_path))
    ps_api = PhysxSchema.PhysxParticleSystem(ps_prim)
    ps_api.CreateMaxNeighborhoodAttr(params.max_neighborhood)
    # Self-collision must be enabled at the *system* level too, not only on the
    # per-cloth PhysxParticleAPI, or folded layers pass through each other.
    ps_api.CreateGlobalSelfCollisionEnabledAttr(params.global_self_collision)
    ps_api.CreateNonParticleCollisionEnabledAttr(True)
    # Clamp particle speed to suppress the flickering "spike" artifacts caused by
    # occasional spring/contact overshoot launching individual particles.
    ps_api.CreateMaxVelocityAttr(params.max_velocity)
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

        # Build glob pattern for the cloth view
        prim_name = cfg.prim_path.split("/")[-1]
        mesh_rel = cfg.mesh_prim_path.removeprefix("/root")
        self._cloth_pattern = f"/World/envs/env_*/{prim_name}{mesh_rel}"

        # Create simulation view + cloth view
        stage_id = get_current_stage_id()
        self._sim_view = physx.create_simulation_view("torch", stage_id)
        self._cloth_view = self._sim_view.create_particle_cloth_view(
            self._cloth_pattern
        )

        if self._cloth_view is None or self._cloth_view._backend is None:
            raise RuntimeError(
                f"Failed to create ParticleClothView for pattern "
                f"'{self._cloth_pattern}'. Ensure PBD cloth schemas were "
                f"applied before sim.reset()."
            )

        assert self._cloth_view.count == num_envs, (
            f"Expected {num_envs} cloth instances, got {self._cloth_view.count}"
        )

        self._max_particles = self._cloth_view.max_particles_per_cloth

        # Store default state (for reset)
        self._default_pos = self._cloth_view.get_positions().clone()
        self._default_vel = self._cloth_view.get_velocities().clone()

        # Working buffers — updated by update()
        self._pos_flat = self._default_pos.clone()
        self._vel_flat = self._default_vel.clone()

        self._ALL_INDICES = torch.arange(
            num_envs, device=device, dtype=torch.int32,
        )

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

    # ── Runtime methods ───────────────────────────────────────────────

    def update(self) -> None:
        """Read positions and velocities from GPU PhysX buffers."""
        self._pos_flat = self._cloth_view.get_positions()
        self._vel_flat = self._cloth_view.get_velocities()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset cloth to its spawned default state for *env_ids*."""
        if len(env_ids) == 0:
            return

        indices = env_ids.to(dtype=torch.int32, device=self.device)

        # Write default positions
        self._pos_flat[env_ids] = self._default_pos[env_ids]
        self._cloth_view.set_positions(self._pos_flat, indices)

        # Write zero velocities
        self._vel_flat[env_ids] = self._default_vel[env_ids]
        self._cloth_view.set_velocities(self._vel_flat, indices)

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
        self._cloth_view.set_positions(self._pos_flat, indices)
        self._vel_flat[env_ids] = 0.0
        self._cloth_view.set_velocities(self._vel_flat, indices)

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
        self._cloth_view.set_positions(self._pos_flat, indices)

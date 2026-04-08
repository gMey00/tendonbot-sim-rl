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
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

logger = logging.getLogger(__name__)


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
    """

    # Spring stiffness
    stretch_stiffness: float = 2000.0
    bend_stiffness: float = 200.0
    shear_stiffness: float = 100.0
    spring_damping: float = 0.3

    # PBD material
    drag: float = 0.02
    lift: float = 0.02
    friction: float = 0.8
    damping: float = 0.3

    # Adhesion (for grasping)
    adhesion: float = 0.2
    particle_adhesion_scale: float = 1.0
    particle_friction_scale: float = 1.0

    # Mass
    mass_kg: float = 0.2

    # Solver
    solver_position_iterations: int = 16
    max_neighborhood: int = 96
    global_self_collision: bool = True
    enable_ccd: bool = False


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

    prim_name = cloth_cfg.prim_path.split("/")[-1]  # "Shirt"
    mesh_rel = cloth_cfg.mesh_prim_path.removeprefix("/root")  # "/World/mesh/Mesh"

    # ── Spawn USD reference in each env and collect mesh paths ────────
    mesh_paths: list[str] = []
    for ep in env_prim_paths:
        ref_path = f"{ep}/{prim_name}"
        prim = stage.DefinePrim(ref_path, "Xform")
        prim.GetReferences().AddReference(cloth_cfg.usd_path)
        UsdGeom.Xform(prim).AddTranslateOp().Set(
            Gf.Vec3d(*cloth_cfg.init_pos),
        )
        mesh_paths.append(f"{ref_path}{mesh_rel}")

    # ── Compute offsets from first mesh ───────────────────────────────
    first_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(Sdf.Path(mesh_paths[0])))
    pts = np.asarray(first_mesh.GetPointsAttr().Get(), dtype=np.float64)

    # The USD mesh vertices may not be centered at the origin (e.g. the
    # T-shirt has centroid Z ≈ 1.34 m).  Compute the vertex centroid
    # and adjust the translate op so that ``init_pos`` specifies where
    # the cloth centroid ends up in env-local coordinates.
    mesh_centroid = pts.mean(axis=0)
    adjusted_pos = Gf.Vec3d(
        cloth_cfg.init_pos[0] - mesh_centroid[0],
        cloth_cfg.init_pos[1] - mesh_centroid[1],
        cloth_cfg.init_pos[2] - mesh_centroid[2],
    )
    for ep in env_prim_paths:
        ref_path = f"{ep}/{prim_name}"
        xform = UsdGeom.Xform(stage.GetPrimAtPath(Sdf.Path(ref_path)))
        xform.GetOrderedXformOps()[0].Set(adjusted_pos)

    idx = np.asarray(first_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    tris = idx.reshape(-1, 3)
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    edges = np.concatenate([
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(a - c, axis=1),
    ])
    avg_edge = float(np.mean(edges))
    rest_offset = 0.5 * avg_edge
    contact_offset = rest_offset * 1.5

    logger.info(
        "Cloth mesh: %d verts, avg edge %.4f m, rest_offset %.4f, contact_offset %.4f, "
        "centroid offset (%.3f, %.3f, %.3f)",
        len(pts), avg_edge, rest_offset, contact_offset,
        mesh_centroid[0], mesh_centroid[1], mesh_centroid[2],
    )

    # ── Shared particle system ────────────────────────────────────────
    ps_path = "/World/clothParticleSystem"
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=ps_path,
        contact_offset=contact_offset,
        rest_offset=rest_offset,
        particle_contact_offset=contact_offset,
        solid_rest_offset=rest_offset,
        fluid_rest_offset=0.0,
        solver_position_iterations=params.solver_position_iterations,
        simulation_owner=scene_path,
    )
    ps_prim = stage.GetPrimAtPath(Sdf.Path(ps_path))
    PhysxSchema.PhysxParticleSystem(ps_prim).CreateMaxNeighborhoodAttr(
        params.max_neighborhood
    )
    if params.enable_ccd:
        PhysxSchema.PhysxParticleSystem(ps_prim).CreateEnableCCDAttr(True)

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
    pbd_api.CreateParticleAdhesionScaleAttr(params.particle_adhesion_scale)
    pbd_api.CreateParticleFrictionScaleAttr(params.particle_friction_scale)
    physicsUtils.add_physics_material_to_prim(
        stage, ps_prim, Sdf.Path(mat_path),
    )

    # ── Apply cloth to each env's mesh ────────────────────────────────
    for mp in mesh_paths:
        mesh_prim = stage.GetPrimAtPath(Sdf.Path(mp))
        UsdGeom.Mesh(mesh_prim).GetSubdivisionSchemeAttr().Set("none")
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
    # The RTX renderer ignores displayColor primvars; it requires a
    # proper UsdPreviewSurface material binding.  GeomSubset bindings
    # from the source USD often don't survive the PBD particle cloth
    # conversion, so we create and bind an explicit material.
    vis_mat_path = f"{env_prim_paths[0]}/{prim_name}/ClothVisualMaterial"
    vis_mat = UsdShade.Material.Define(stage, vis_mat_path)
    shader_path = f"{vis_mat_path}/PreviewSurface"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.75, 0.75, 0.85),
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    for mp in mesh_paths:
        mesh_prim = stage.GetPrimAtPath(Sdf.Path(mp))
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(vis_mat)

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

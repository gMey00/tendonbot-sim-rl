"""Create a PBD particle-cloth USD from an existing triangle mesh.

Authors the full PhysX PBD particle-cloth schema stack directly with OpenUSD
(``pxr.UsdPhysics`` + ``pxr.PhysxSchema``).  Because it only *authors* USD it
needs neither a running Kit application nor ``omni.physx`` — it runs in plain
Python as long as the OpenUSD + PhysxSchema libraries are importable
(see ``Scripts/_usdenv.sh``).  It also still works inside the Isaac Sim Script
Editor.

Pipeline:

  1. Physics Scene (TGS solver, GPU dynamics + GPU broadphase)
  2. Particle System with offsets derived from the mesh edge length
  3. PBD Material (cotton T-shirt defaults)
  4. Particle Cloth (PhysxParticleClothAPI + PhysxAutoParticleClothAPI) on the mesh
  5. Mass / self-collision properties

Usage (standalone)::

    source Scripts/_usdenv.sh
    "$KPY" Scripts/make_pbd_cloth.py \
        --input  tshirt_mod.usdc \
        --output tshirt_pbd.usd \
        --prim   /root/World/mesh/Mesh

Reference: cloth_simulation_isaacsim_research.md §3 (PBD setup + cotton parameters)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PhysxSchema is required to author particle cloth.  Source "
        "Scripts/_usdenv.sh (or run inside Isaac Sim) so the PhysX USD "
        f"schemas are importable.  Import error: {exc}"
    )


# ---------------------------------------------------------------------------
# Cotton T-shirt PBD defaults (from research §3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PBDClothParameters:
    # Spring stiffness (N/m) — generated from triangle topology at cook time
    stretch_stiffness: float = 2000.0
    bend_stiffness: float = 200.0
    shear_stiffness: float = 100.0
    spring_damping: float = 0.3

    # PBD material
    drag: float = 0.02
    lift: float = 0.02
    friction: float = 0.8
    damping: float = 0.3
    density: float = 350.0  # cotton ~300–400 kg/m3

    # Adhesion (DexGarmentLab-style grasping).  Disabled by default so a
    # free-draping shirt does not stick to the floor/colliders.  When raised,
    # ``adhesion_offset_scale`` MUST be > 0 or the adhesion force has no range
    # and is silently inert.
    adhesion: float = 0.0
    adhesion_offset_scale: float = 2.0
    particle_adhesion_scale: float = 1.0
    particle_friction_scale: float = 1.0

    # Mass (kg) — per-particle mass = mass_kg / num_particles
    mass_kg: float = 0.2

    # Particle system
    solver_position_iterations: int = 16
    max_neighborhood: int = 96
    global_self_collision: bool = True
    enable_ccd: bool = False

    # Offsets (metres).  <= 0 means "derive from mesh edge length".
    contact_offset: float = 0.0
    rest_offset: float = 0.0


COTTON_TSHIRT = PBDClothParameters()


# ---------------------------------------------------------------------------
# Stage-aware gravity
# ---------------------------------------------------------------------------

def stage_gravity(stage: Usd.Stage) -> tuple[Gf.Vec3f, float]:
    """Gravity (direction, magnitude) honouring the stage up-axis and units.

    PhysX works in stage units, so 9.81 m/s^2 must be converted with
    metersPerUnit (e.g. on a centimetre stage gravity is 981 units/s^2), and the
    direction must point along -up (Y-up or Z-up)."""
    up = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    direction = Gf.Vec3f(0.0, -1.0, 0.0) if up == UsdGeom.Tokens.y else Gf.Vec3f(0.0, 0.0, -1.0)
    return direction, 9.81 / mpu


# ---------------------------------------------------------------------------
# Mesh analysis (offset computation)
# ---------------------------------------------------------------------------

def compute_offsets_from_mesh(mesh: UsdGeom.Mesh) -> Tuple[float, float]:
    """Derive (solid_rest_offset, particle_contact_offset) from edge length.

    ``solidRestOffset ~= 0.5 * average_edge_length`` (research §3) keeps
    same-cloth particles from overlapping at rest; the particle contact offset
    is 1.5x that, which stays below the edge length so spring-connected
    neighbours are not flagged as colliding.
    """
    import numpy as np

    points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)

    if points.size == 0 or indices.size < 3:
        return 0.005, 0.0075

    tris = indices.reshape(-1, 3)
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    lengths = np.concatenate([
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(a - c, axis=1),
    ])
    average_edge = float(np.mean(lengths))

    rest_offset = 0.5 * average_edge
    contact_offset = rest_offset * 1.5
    return rest_offset, contact_offset


# ---------------------------------------------------------------------------
# PBD cloth creation
# ---------------------------------------------------------------------------

def create_pbd_cloth(
    stage: Usd.Stage,
    mesh_path: str,
    params: PBDClothParameters = COTTON_TSHIRT,
    scene_path: Optional[str] = None,
    particle_system_path: Optional[str] = None,
    material_path: Optional[str] = None,
) -> None:
    """Author the full PBD particle-cloth schema stack on an existing mesh."""
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        default_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(default_prim)
    root = default_prim.GetPath()

    mesh_prim = stage.GetPrimAtPath(Sdf.Path(mesh_path))
    if not mesh_prim or not mesh_prim.IsValid():
        raise ValueError(f"No prim at path: {mesh_path}")
    mesh = UsdGeom.Mesh(mesh_prim)
    if not mesh:
        raise ValueError(f"Prim at {mesh_path} is not a UsdGeom.Mesh")

    # A rigid body and a particle cloth on the same prim is contradictory.
    if mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(
            f"{mesh_path} has UsdPhysics.RigidBodyAPI; remove it before "
            "applying particle cloth."
        )

    # Cloth springs come from triangle topology, so subdivision must be off.
    scheme = mesh.GetSubdivisionSchemeAttr()
    if scheme.Get() not in (None, "", "none"):
        scheme.Set("none")
        print("  Set subdivisionScheme to 'none'")

    # Offsets
    rest_offset, contact_offset = params.rest_offset, params.contact_offset
    if rest_offset <= 0 or contact_offset <= 0:
        rest_offset, contact_offset = compute_offsets_from_mesh(mesh)
        print(f"  Auto offsets: solidRest={rest_offset:.6f} particleContact={contact_offset:.6f}")

    # 1. Physics Scene -----------------------------------------------------
    if scene_path is None:
        scene_path = root.AppendChild("physicsScene").pathString
    if not stage.GetPrimAtPath(Sdf.Path(scene_path)).IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        grav_dir, grav_mag = stage_gravity(stage)
        scene.CreateGravityDirectionAttr(grav_dir)
        scene.CreateGravityMagnitudeAttr(grav_mag)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene.CreateSolverTypeAttr("TGS")
        physx_scene.CreateEnableGPUDynamicsAttr(True)
        physx_scene.CreateBroadphaseTypeAttr("GPU")
        # A flat-laid garment has its front/back layers within a contact offset
        # of each other, so self-collision generates a very large contact count.
        # The default GPU buffers overflow ("Contacts have been dropped"); raise
        # them so the cloth simulates stably.
        physx_scene.CreateGpuCollisionStackSizeAttr(256 * 1024 * 1024)
        physx_scene.CreateGpuMaxParticleContactsAttr(4_000_000)
        physx_scene.CreateGpuFoundLostPairsCapacityAttr(1_000_000)
        print(f"  Created physics scene: {scene_path}")

    # 2. Particle System ---------------------------------------------------
    if particle_system_path is None:
        particle_system_path = root.AppendChild("particleSystem").pathString
    if not stage.GetPrimAtPath(Sdf.Path(particle_system_path)).IsValid():
        ps = PhysxSchema.PhysxParticleSystem.Define(stage, Sdf.Path(particle_system_path))
        ps.CreateSimulationOwnerRel().SetTargets([Sdf.Path(scene_path)])
        # particleContactOffset must be the largest offset; rest/contact for
        # particle–rigid collisions are left to auto (0.99 x particleContact).
        ps.CreateParticleContactOffsetAttr().Set(contact_offset)
        ps.CreateSolidRestOffsetAttr().Set(rest_offset)
        ps.CreateFluidRestOffsetAttr().Set(0.0)
        ps.CreateSolverPositionIterationCountAttr(params.solver_position_iterations)
        ps.CreateMaxNeighborhoodAttr(params.max_neighborhood)
        ps.CreateGlobalSelfCollisionEnabledAttr(params.global_self_collision)
        ps.CreateNonParticleCollisionEnabledAttr(True)
        ps.CreateEnableCCDAttr(params.enable_ccd)
        print(f"  Created particle system: {particle_system_path}")

    # 3. PBD Material ------------------------------------------------------
    if material_path is None:
        material_path = root.AppendChild("particleMaterial").pathString
    if not stage.GetPrimAtPath(Sdf.Path(material_path)).IsValid():
        material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
        pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
        pbd.CreateFrictionAttr().Set(params.friction)
        pbd.CreateDampingAttr().Set(params.damping)
        pbd.CreateDragAttr().Set(params.drag)
        pbd.CreateLiftAttr().Set(params.lift)
        pbd.CreateDensityAttr().Set(params.density)
        pbd.CreateParticleFrictionScaleAttr().Set(params.particle_friction_scale)
        if params.adhesion > 0.0:
            pbd.CreateAdhesionAttr().Set(params.adhesion)
            pbd.CreateAdhesionOffsetScaleAttr().Set(params.adhesion_offset_scale)
            pbd.CreateParticleAdhesionScaleAttr().Set(params.particle_adhesion_scale)
        # Bind the material to the particle system with the 'physics' purpose.
        ps_prim = stage.GetPrimAtPath(Sdf.Path(particle_system_path))
        UsdShade.MaterialBindingAPI.Apply(ps_prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        print(f"  Created + bound PBD material: {material_path}")

    # 4. Particle Cloth on the mesh ---------------------------------------
    cloth_api = PhysxSchema.PhysxParticleClothAPI.Apply(mesh_prim)
    auto_api = PhysxSchema.PhysxAutoParticleClothAPI.Apply(mesh_prim)
    particle_api = PhysxSchema.PhysxParticleAPI(cloth_api)

    particle_api.CreateParticleSystemRel().SetTargets([Sdf.Path(particle_system_path)])
    particle_api.CreateSelfCollisionAttr().Set(params.global_self_collision)
    cloth_api.CreateSelfCollisionFilterAttr().Set(True)

    auto_api.CreateSpringStretchStiffnessAttr().Set(params.stretch_stiffness)
    auto_api.CreateSpringBendStiffnessAttr().Set(params.bend_stiffness)
    auto_api.CreateSpringShearStiffnessAttr().Set(params.shear_stiffness)
    auto_api.CreateSpringDampingAttr().Set(params.spring_damping)
    print(f"  Applied particle cloth to: {mesh_path}")

    # 5. Mass --------------------------------------------------------------
    mass_api = UsdPhysics.MassAPI.Apply(mesh_prim)
    mass_api.CreateMassAttr().Set(params.mass_kg)
    print(f"  Set mass: {params.mass_kg} kg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Author PBD particle cloth on a triangle mesh in a USD file."
    )
    parser.add_argument("--input", required=True, help="Input USD file with triangle mesh.")
    parser.add_argument("--output", required=True, help="Output USD file path.")
    parser.add_argument("--prim", default=None, help="Mesh prim path (auto-discover if omitted).")

    parser.add_argument("--stretch", type=float, default=COTTON_TSHIRT.stretch_stiffness)
    parser.add_argument("--bend", type=float, default=COTTON_TSHIRT.bend_stiffness)
    parser.add_argument("--shear", type=float, default=COTTON_TSHIRT.shear_stiffness)
    parser.add_argument("--mass", type=float, default=COTTON_TSHIRT.mass_kg)
    parser.add_argument("--friction", type=float, default=COTTON_TSHIRT.friction)
    parser.add_argument("--adhesion", type=float, default=COTTON_TSHIRT.adhesion,
                        help="Particle-rigid adhesion (0 = off). Needs adhesion-offset-scale > 0.")
    parser.add_argument("--adhesion-offset-scale", type=float,
                        default=COTTON_TSHIRT.adhesion_offset_scale)

    args = parser.parse_args(argv)

    stage = Usd.Stage.Open(args.input)
    if stage is None:
        print(f"ERROR: Cannot open {args.input}")
        return 2

    prim_path = args.prim
    if prim_path is None:
        from check_cloth_readiness import find_first_mesh
        prim_path = find_first_mesh(stage)
        if prim_path is None:
            print("ERROR: No UsdGeom.Mesh found in stage.")
            return 2
        print(f"Auto-discovered mesh: {prim_path}")

    params = PBDClothParameters(
        stretch_stiffness=args.stretch,
        bend_stiffness=args.bend,
        shear_stiffness=args.shear,
        mass_kg=args.mass,
        friction=args.friction,
        adhesion=args.adhesion,
        adhesion_offset_scale=args.adhesion_offset_scale,
    )

    create_pbd_cloth(stage, prim_path, params)

    stage.GetRootLayer().Export(args.output)
    print(f"\nSaved PBD cloth USD to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

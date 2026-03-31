"""Create a PBD particle cloth USD from an existing triangle mesh.

Runs inside Isaac Sim (Script Editor or standalone launcher).  Reads a
source USD mesh and applies the full PhysX PBD particle cloth pipeline:

  1. Physics Scene (TGS solver, GPU broadphase)
  2. Particle System with tuned offsets
  3. PBD Material (cotton T-shirt defaults)
  4. Particle Cloth on the mesh
  5. Mass / self-collision / adhesion properties

Usage (Isaac Sim Script Editor)::

    Set constants below and run (Ctrl+Enter).

Usage (standalone)::

    ~/.local/share/ov/pkg/isaac-sim-*/python.sh make_pbd_cloth.py \\
        --input  /path/to/tshirt_mesh.usd \\
        --output /path/to/tshirt_pbd.usd  \\
        --prim   /World/tshirt_mod/mesh

Reference: cloth_simulation_isaacsim_research.md §3 (PBD setup + recommended cotton parameters)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

try:
    from pxr import PhysxSchema  # type: ignore[attr-defined]
except ImportError:
    PhysxSchema = None

try:
    import omni.usd
    from omni.physx.scripts import particleUtils, physicsUtils

    HAS_OMNI = True
except ImportError:
    HAS_OMNI = False


# ---------------------------------------------------------------------------
# Cotton T-shirt PBD defaults (from research §3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PBDClothParameters:
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

    # Adhesion (DexGarmentLab style grasping)
    adhesion: float = 0.2
    particle_adhesion_scale: float = 1.0
    particle_friction_scale: float = 1.0

    # Mass
    mass_kg: float = 0.2

    # Particle system
    solver_position_iterations: int = 16
    max_neighborhood: int = 96
    global_self_collision: bool = True
    enable_ccd: bool = False

    # Auto-computed from mesh (set to 0 for auto)
    contact_offset: float = 0.0
    rest_offset: float = 0.0


COTTON_TSHIRT = PBDClothParameters()


# ---------------------------------------------------------------------------
# Mesh analysis (offset computation)
# ---------------------------------------------------------------------------

def compute_offsets_from_mesh(
    mesh: UsdGeom.Mesh,
) -> tuple[float, float]:
    """Derive solidRestOffset and contactOffset from average edge length.

    Returns (rest_offset, contact_offset).
    """
    import numpy as np

    points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)

    if points.size == 0 or indices.size < 3:
        return 0.005, 0.0075

    tris = indices.reshape(-1, 3)
    a = points[tris[:, 0]]
    b = points[tris[:, 1]]
    c = points[tris[:, 2]]
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
    """Apply full PBD particle cloth setup to an existing triangle mesh.

    Requires Isaac Sim runtime (omni.physx.scripts).
    """
    if not HAS_OMNI:
        raise RuntimeError(
            "omni.physx is required.  Run this script inside Isaac Sim "
            "(standalone launcher or Script Editor)."
        )

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

    # Ensure triangulated + no subdivision
    scheme = mesh.GetSubdivisionSchemeAttr()
    if scheme.Get() not in (None, "", "none"):
        scheme.Set("none")
        print(f"  Set subdivisionScheme to 'none'")

    # Compute offsets
    rest_offset = params.rest_offset
    contact_offset = params.contact_offset
    if rest_offset <= 0 or contact_offset <= 0:
        rest_offset, contact_offset = compute_offsets_from_mesh(mesh)
        print(f"  Auto-computed offsets: rest={rest_offset:.6f}, contact={contact_offset:.6f}")

    # 1. Physics Scene
    if scene_path is None:
        scene_path = root.AppendChild("physicsScene").pathString
    if not stage.GetPrimAtPath(Sdf.Path(scene_path)).IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene.CreateSolverTypeAttr("TGS")
        physx_scene.CreateEnableGPUDynamicsAttr(True)
        physx_scene.CreateBroadphaseTypeAttr("GPU")
        print(f"  Created physics scene: {scene_path}")

    # 2. Particle System
    if particle_system_path is None:
        particle_system_path = root.AppendChild("particleSystem").pathString
    if not stage.GetPrimAtPath(Sdf.Path(particle_system_path)).IsValid():
        particleUtils.add_physx_particle_system(
            stage=stage,
            particle_system_path=particle_system_path,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
            particle_contact_offset=contact_offset,
            solid_rest_offset=rest_offset,
            fluid_rest_offset=0.0,
            solver_position_iterations=params.solver_position_iterations,
            simulation_owner=scene_path,
        )
        # Extra properties
        ps_prim = stage.GetPrimAtPath(Sdf.Path(particle_system_path))
        ps_api = PhysxSchema.PhysxParticleSystem(ps_prim)
        if ps_api:
            ps_api.CreateMaxNeighborhoodAttr(params.max_neighborhood)
            ps_api.CreateGlobalSelfCollisionEnabledAttr(params.global_self_collision)
            ps_api.CreateEnableCCDAttr(params.enable_ccd)
        print(f"  Created particle system: {particle_system_path}")

    # 3. PBD Material
    if material_path is None:
        material_path = root.AppendChild("particleMaterial").pathString
    if not stage.GetPrimAtPath(Sdf.Path(material_path)).IsValid():
        particleUtils.add_pbd_particle_material(
            stage,
            material_path,
            drag=params.drag,
            lift=params.lift,
            friction=params.friction,
            damping=params.damping,
        )
        # Adhesion properties (DexGarmentLab approach)
        mat_prim = stage.GetPrimAtPath(Sdf.Path(material_path))
        pbd_mat = PhysxSchema.PhysxPBDMaterialAPI(mat_prim)
        if pbd_mat:
            pbd_mat.CreateAdhesionAttr(params.adhesion)
            pbd_mat.CreateParticleAdhesionScaleAttr(params.particle_adhesion_scale)
            pbd_mat.CreateParticleFrictionScaleAttr(params.particle_friction_scale)

        physicsUtils.add_physics_material_to_prim(
            stage,
            stage.GetPrimAtPath(Sdf.Path(particle_system_path)),
            Sdf.Path(material_path),
        )
        print(f"  Created PBD material: {material_path}")

    # 4. Particle Cloth on mesh
    particleUtils.add_physx_particle_cloth(
        stage=stage,
        path=mesh_path,
        dynamic_mesh_path=None,
        particle_system_path=particle_system_path,
        spring_stretch_stiffness=params.stretch_stiffness,
        spring_bend_stiffness=params.bend_stiffness,
        spring_shear_stiffness=params.shear_stiffness,
        spring_damping=params.spring_damping,
        self_collision=params.global_self_collision,
        self_collision_filter=True,
    )
    print(f"  Applied particle cloth to: {mesh_path}")

    # 5. Mass
    mass_api = UsdPhysics.MassAPI.Apply(mesh_prim)
    mass_api.GetMassAttr().Set(params.mass_kg)
    print(f"  Set mass: {params.mass_kg} kg")


# ---------------------------------------------------------------------------
# CLI (standalone Isaac Sim launcher)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create PBD particle cloth USD from a triangle mesh."
    )
    parser.add_argument("--input", required=True, help="Input USD file with triangle mesh.")
    parser.add_argument("--output", required=True, help="Output USD file path.")
    parser.add_argument("--prim", default=None, help="Mesh prim path (auto-discover if omitted).")

    # Optional parameter overrides
    parser.add_argument("--stretch", type=float, default=COTTON_TSHIRT.stretch_stiffness)
    parser.add_argument("--bend", type=float, default=COTTON_TSHIRT.bend_stiffness)
    parser.add_argument("--shear", type=float, default=COTTON_TSHIRT.shear_stiffness)
    parser.add_argument("--mass", type=float, default=COTTON_TSHIRT.mass_kg)
    parser.add_argument("--friction", type=float, default=COTTON_TSHIRT.friction)
    parser.add_argument("--adhesion", type=float, default=COTTON_TSHIRT.adhesion)

    args = parser.parse_args(argv)

    stage = Usd.Stage.Open(args.input)
    if stage is None:
        print(f"ERROR: Cannot open {args.input}")
        return 2

    # Find mesh prim
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
    )

    create_pbd_cloth(stage, prim_path, params)

    stage.GetRootLayer().Export(args.output)
    print(f"\nSaved PBD cloth USD to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

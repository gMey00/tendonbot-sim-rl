"""Create an XPBD surface deformable cloth USD from an existing triangle mesh.

Runs inside Isaac Sim (Script Editor or standalone launcher).  Reads a
source USD mesh and applies the XPBD surface deformable pipeline:

  1. Physics Scene (TGS solver, GPU broadphase, deformable beta enabled)
  2. Surface Deformable Body on the mesh
  3. Deformable Material (cotton T-shirt defaults)

Usage (Isaac Sim Script Editor)::

    Set constants below and run (Ctrl+Enter).

Usage (standalone)::

    ~/.local/share/ov/pkg/isaac-sim-*/python.sh make_xpbd_cloth.py \\
        --input  /path/to/tshirt_mesh.usd \\
        --output /path/to/tshirt_xpbd.usd  \\
        --prim   /World/tshirt_mod/mesh

Reference: cloth_simulation_isaacsim_research.md §3 (XPBD setup + parameter mapping)

Note: Requires ``physics.enableDeformableBeta = true`` in Isaac Sim settings.
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
    from omni.physx.scripts import deformableUtils

    HAS_OMNI = True
except ImportError:
    HAS_OMNI = False


# ---------------------------------------------------------------------------
# Cotton T-shirt XPBD defaults (from research §3, parameter mapping)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class XPBDClothParameters:
    # Material
    youngs_modulus: float = 5000.0
    poissons_ratio: float = 0.3
    elasticity_damping: float = 0.01
    dynamic_friction: float = 0.8
    density: float = 350.0

    # Solver
    solver_position_iterations: int = 16
    vertex_velocity_damping: float = 0.005

    # Collision
    self_collision: bool = True
    self_collision_filter_distance: float = 0.0  # 0 = auto
    collision_simplification: bool = False

    # Remeshing (for meshes exceeding vertex budget)
    remeshing_enabled: bool = False
    target_triangle_count: int = 3000
    remeshing_resolution: int = 0  # 0 = auto


COTTON_TSHIRT = XPBDClothParameters()


# ---------------------------------------------------------------------------
# XPBD surface deformable creation
# ---------------------------------------------------------------------------

def create_xpbd_cloth(
    stage: Usd.Stage,
    mesh_path: str,
    params: XPBDClothParameters = COTTON_TSHIRT,
    scene_path: Optional[str] = None,
    material_path: Optional[str] = None,
) -> None:
    """Apply XPBD surface deformable setup to an existing triangle mesh.

    Requires Isaac Sim runtime (omni.physx.scripts) and
    ``physics.enableDeformableBeta = true``.
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

    # 2. Surface Deformable Body
    deformableUtils.set_physics_surface_deformable_body(
        stage=stage,
        prim_path=mesh_path,
    )
    print(f"  Applied surface deformable to: {mesh_path}")

    # Configure deformable body properties
    deformable_api = PhysxSchema.PhysxDeformableBodyAPI(mesh_prim)
    if deformable_api:
        deformable_api.CreateSolverPositionIterationCountAttr(
            params.solver_position_iterations
        )
        deformable_api.CreateSelfCollisionAttr(params.self_collision)
        deformable_api.CreateVertexVelocityDampingAttr(
            params.vertex_velocity_damping
        )
        if params.self_collision_filter_distance > 0:
            deformable_api.CreateSelfCollisionFilterDistanceAttr(
                params.self_collision_filter_distance
            )

    # Remeshing / simplification (for meshes exceeding vertex budget)
    if params.remeshing_enabled:
        simplification_api = PhysxSchema.PhysxAutoDeformableMeshSimplificationAPI.Apply(
            mesh_prim
        )
        simplification_api.CreateRemeshingEnabledAttr(True)
        simplification_api.CreateTargetTriangleCountAttr(
            params.target_triangle_count
        )
        if params.remeshing_resolution > 0:
            simplification_api.CreateRemeshingResolutionAttr(
                params.remeshing_resolution
            )
        print(f"  Enabled remeshing: target {params.target_triangle_count} triangles")

    # 3. Deformable Material
    if material_path is None:
        material_path = root.AppendChild("deformableMaterial").pathString
    if not stage.GetPrimAtPath(Sdf.Path(material_path)).IsValid():
        deformableUtils.add_deformable_material(
            stage=stage,
            path=material_path,
            youngs_modulus=params.youngs_modulus,
            poissons_ratio=params.poissons_ratio,
            elasticity_damping=params.elasticity_damping,
            dynamic_friction=params.dynamic_friction,
            density=params.density,
        )
        print(f"  Created deformable material: {material_path}")

    # Bind material to mesh
    UsdPhysics.MaterialBindingAPI.Apply(mesh_prim).Bind(
        stage.GetPrimAtPath(Sdf.Path(material_path))
    )
    print(f"  Bound material to mesh")


# ---------------------------------------------------------------------------
# CLI (standalone Isaac Sim launcher)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create XPBD surface deformable cloth USD from a triangle mesh."
    )
    parser.add_argument("--input", required=True, help="Input USD file with triangle mesh.")
    parser.add_argument("--output", required=True, help="Output USD file path.")
    parser.add_argument("--prim", default=None, help="Mesh prim path (auto-discover if omitted).")

    # Optional parameter overrides
    parser.add_argument("--youngs", type=float, default=COTTON_TSHIRT.youngs_modulus)
    parser.add_argument("--poissons", type=float, default=COTTON_TSHIRT.poissons_ratio)
    parser.add_argument("--friction", type=float, default=COTTON_TSHIRT.dynamic_friction)
    parser.add_argument("--density", type=float, default=COTTON_TSHIRT.density)
    parser.add_argument("--remesh", action="store_true", help="Enable remeshing.")
    parser.add_argument("--target-tris", type=int, default=COTTON_TSHIRT.target_triangle_count)

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

    params = XPBDClothParameters(
        youngs_modulus=args.youngs,
        poissons_ratio=args.poissons,
        dynamic_friction=args.friction,
        density=args.density,
        remeshing_enabled=args.remesh,
        target_triangle_count=args.target_tris,
    )

    create_xpbd_cloth(stage, prim_path, params)

    stage.GetRootLayer().Export(args.output)
    print(f"\nSaved XPBD cloth USD to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

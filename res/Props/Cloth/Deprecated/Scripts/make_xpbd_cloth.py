"""Create an XPBD/FEM surface-deformable cloth USD from a triangle mesh.

Authors the *new* PhysX surface-deformable beta (XPBD corotational FEM on
triangles) schema stack directly with OpenUSD.  It only authors USD, so it runs
in plain Python with the OpenUSD + PhysxSchema libraries on the path
(see ``Scripts/_usdenv.sh``) and also works inside the Isaac Sim Script Editor.

Pipeline:

  1. Physics Scene (TGS solver, GPU dynamics + GPU broadphase)
  2. Surface deformable body on the mesh, mirroring
     ``omni.physx.scripts.deformableUtils.set_physics_surface_deformable_body``:
        - OmniPhysicsDeformableBodyAPI
        - OmniPhysicsSurfaceDeformableSimAPI (rest points / triangle indices)
        - UsdPhysics.CollisionAPI
  3. Surface deformable Material (OmniPhysicsSurfaceDeformableMaterialAPI):
     Young's modulus / Poisson ratio / thickness / friction / density

Usage (standalone)::

    source Scripts/_usdenv.sh
    "$KPY" Scripts/make_xpbd_cloth.py \
        --input  tshirt_mod.usdc \
        --output tshirt_xpbd.usd \
        --prim   /root/World/mesh/Mesh

Requires ``/physics/enableDeformableBeta = true`` at *simulation* time (set it in
Isaac Sim preferences, or via ``carb`` in a standalone app — see
``Scripts/simulate_cloth_check.py``).

Reference: cloth_simulation_isaacsim_research.md §3 (surface deformable + mapping)

Why the new OmniPhysics beta and not ``PhysxDeformableSurfaceAPI``?
------------------------------------------------------------------
The deprecated ``PhysxSchema.PhysxDeformableSurfaceAPI`` is no longer cooked by
the PhysX solver in Isaac Sim 5.1 — authoring it and pressing Play crashes the
``omni.physx`` plugin.  The supported path is the OmniPhysics deformable beta.
Its body schema deliberately exposes *no* solver-iteration / self-collision /
velocity-damping knobs (those are global/scene concerns in the beta); elasticity
is governed by Young's modulus, Poisson ratio and surface thickness, with
bending controlled by ``surfaceBendStiffness``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema  # noqa: F401  (ensures PhysX USD schema plugins load)
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PhysX USD schemas are required to author surface deformables.  Source "
        "Scripts/_usdenv.sh (or run inside Isaac Sim).  Import error: " + str(exc)
    )

# OmniPhysics deformable schemas are codeless (no Python class); applied by name.
_BODY_API = "OmniPhysicsDeformableBodyAPI"
_SIM_API = "OmniPhysicsSurfaceDeformableSimAPI"
_MAT_API = "OmniPhysicsDeformableMaterialAPI"
_SURF_MAT_API = "OmniPhysicsSurfaceDeformableMaterialAPI"


@dataclass(frozen=True)
class XPBDClothParameters:
    # Material — elasticity is FEM, so stretch/shear come from these
    youngs_modulus: float = 5000.0     # soft, cotton-like (default solver is ~50 MPa)
    poissons_ratio: float = 0.3
    dynamic_friction: float = 0.8
    static_friction: float = 0.8
    density: float = 350.0             # cotton ~300-400 kg/m3
    surface_thickness: float = 0.001   # 1 mm shell thickness
    surface_bend_stiffness: float = 0.0    # extra bending (0 = FEM only)
    surface_shear_stiffness: float = 0.0   # 0 -> governed by youngs/poisson
    surface_stretch_stiffness: float = 0.0 # 0 -> governed by youngs/poisson


COTTON_TSHIRT = XPBDClothParameters()


def stage_gravity(stage: Usd.Stage) -> tuple[Gf.Vec3f, float]:
    """Gravity (direction, magnitude) honouring the stage up-axis and units."""
    up = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    direction = Gf.Vec3f(0.0, -1.0, 0.0) if up == UsdGeom.Tokens.y else Gf.Vec3f(0.0, 0.0, -1.0)
    return direction, 9.81 / mpu


def create_xpbd_cloth(
    stage: Usd.Stage,
    mesh_path: str,
    params: XPBDClothParameters = COTTON_TSHIRT,
    scene_path: Optional[str] = None,
    material_path: Optional[str] = None,
) -> None:
    """Author the OmniPhysics surface-deformable schema stack on an existing mesh."""
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

    if mesh_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(
            f"{mesh_path} has UsdPhysics.RigidBodyAPI; a surface deformable "
            "cannot also be a rigid body."
        )

    # Surface deformables are limited to triangle faces and no subdivision.
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    if any(c != 3 for c in counts):
        raise ValueError(f"{mesh_path} has non-triangle faces; triangulate first.")
    scheme = mesh.GetSubdivisionSchemeAttr()
    if scheme.Get() not in (None, "", "none"):
        scheme.Set("none")
        print("  Set subdivisionScheme to 'none'")

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
        physx_scene.CreateGpuMaxDeformableSurfaceContactsAttr(4_000_000)
        physx_scene.CreateGpuFoundLostPairsCapacityAttr(1_000_000)
        print(f"  Created physics scene: {scene_path}")

    # 2. Surface deformable body ------------------------------------------
    if not mesh_prim.ApplyAPI(_BODY_API):
        raise RuntimeError(f"Failed to apply {_BODY_API} to {mesh_path}")
    if not mesh_prim.ApplyAPI(_SIM_API):
        raise RuntimeError(f"Failed to apply {_SIM_API} to {mesh_path}")
    UsdPhysics.CollisionAPI.Apply(mesh_prim)

    # The sim API needs the rest shape (points + triangle vertex indices).
    points = mesh.GetPointsAttr().Get()
    tri = [Gf.Vec3i(int(indices[3 * i]), int(indices[3 * i + 1]), int(indices[3 * i + 2]))
           for i in range(len(counts))]
    mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(points)
    mesh_prim.GetAttribute("omniphysics:restTriVtxIndices").Set(tri)
    mesh_prim.GetAttribute("omniphysics:deformableBodyEnabled").Set(True)
    print(f"  Applied surface deformable to: {mesh_path}")

    # 3. Surface deformable Material --------------------------------------
    if material_path is None:
        material_path = root.AppendChild("deformableMaterial").pathString
    if not stage.GetPrimAtPath(Sdf.Path(material_path)).IsValid():
        material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
        mp = material.GetPrim()
        mp.ApplyAPI(_MAT_API)
        mp.ApplyAPI(_SURF_MAT_API)
        mp.GetAttribute("omniphysics:youngsModulus").Set(params.youngs_modulus)
        mp.GetAttribute("omniphysics:poissonsRatio").Set(params.poissons_ratio)
        mp.GetAttribute("omniphysics:dynamicFriction").Set(params.dynamic_friction)
        mp.GetAttribute("omniphysics:staticFriction").Set(params.static_friction)
        mp.GetAttribute("omniphysics:density").Set(params.density)
        mp.GetAttribute("omniphysics:surfaceThickness").Set(params.surface_thickness)
        mp.GetAttribute("omniphysics:surfaceBendStiffness").Set(params.surface_bend_stiffness)
        mp.GetAttribute("omniphysics:surfaceShearStiffness").Set(params.surface_shear_stiffness)
        mp.GetAttribute("omniphysics:surfaceStretchStiffness").Set(params.surface_stretch_stiffness)
        print(f"  Created surface deformable material: {material_path}")

    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(Sdf.Path(material_path))),
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )
    print("  Bound material to mesh")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Author an XPBD/FEM surface deformable on a triangle mesh."
    )
    parser.add_argument("--input", required=True, help="Input USD file with triangle mesh.")
    parser.add_argument("--output", required=True, help="Output USD file path.")
    parser.add_argument("--prim", default=None, help="Mesh prim path (auto-discover if omitted).")

    parser.add_argument("--youngs", type=float, default=COTTON_TSHIRT.youngs_modulus)
    parser.add_argument("--poissons", type=float, default=COTTON_TSHIRT.poissons_ratio)
    parser.add_argument("--friction", type=float, default=COTTON_TSHIRT.dynamic_friction)
    parser.add_argument("--density", type=float, default=COTTON_TSHIRT.density)
    parser.add_argument("--thickness", type=float, default=COTTON_TSHIRT.surface_thickness)

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

    params = XPBDClothParameters(
        youngs_modulus=args.youngs,
        poissons_ratio=args.poissons,
        dynamic_friction=args.friction,
        static_friction=args.friction,
        density=args.density,
        surface_thickness=args.thickness,
    )

    create_xpbd_cloth(stage, prim_path, params)

    stage.GetRootLayer().Export(args.output)
    print(f"\nSaved surface deformable cloth USD to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
make_cloth.py — Isaac Sim 5.1.x UI helper

Runs a battery of mesh validity checks that frequently cause
    [omni.physx.plugin] PxValidateTriangleMesh for PxDeformableSurface failed
and only then applies the Surface Deformable setup.

Usage:
- Open Isaac Sim → Window → Script Editor
- Set TSHIRT_MESH_PATH below to the *UsdGeom.Mesh* prim you want to simulate
- Run (Ctrl+Enter)

Notes:
- This script is intentionally "defensive": it aborts before authoring schemas
  when it detects issues that almost always break PhysX deformable cooking.
- Geometry checks delegate to check_cloth_readiness.py (shared standalone module).
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

# Optional schemas: not always present, so import defensively.
try:
    from pxr import PhysxSchema  # type: ignore
except Exception:  # pragma: no cover
    PhysxSchema = None  # type: ignore

from omni.physx.scripts import deformableUtils

# Make sibling module importable (for Script Editor usage)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from check_cloth_readiness import (
    CheckResult,
    check_duplicate_triangles,
    check_index_ranges,
    check_kinematic_rigid_body,
    check_mesh_attributes,
    check_nonmanifold_edges,
    check_point_finiteness,
    check_triangle_quality,
    load_mesh,
    merge_results,
)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

TSHIRT_MESH_PATH: str = "/World/tshirt_mod/mesh"  # <- set to *Mesh* prim path
ABORT_ON_WARNINGS: bool = False  # if True: abort also on warnings (not just errors)


# -----------------------------------------------------------------------------
# HELPERS (Isaac Sim-specific — not in the standalone checker)
# -----------------------------------------------------------------------------

def _as_tuple(xs: Iterable[str]) -> Tuple[str, ...]:
    return tuple(xs)


def _get_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_block(label: str, lines: Iterable[str]) -> None:
    lines = list(lines)
    if not lines:
        return
    print(f"\n{label}:")
    for ln in lines:
        print(f"  - {ln}")


def _check_editability(stage: Usd.Stage, prim: Usd.Prim) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        if prim.IsInstance() or prim.IsInstanceable():
            warnings.append("Prim is instance/instanceable; authoring may not apply as expected. Consider uninstancing/duplicating.")
    except Exception:
        pass

    try:
        if prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
            warnings.append("Prim has authored references/payloads; ensure your Edit Target is a writable layer.")
    except Exception:
        pass

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _check_collision_api_count(stage: Usd.Stage, mesh_prim: Usd.Prim) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    # Omni PhysX surface deformables currently have strict collider limitations.
    # Heuristic: warn if there are multiple CollisionAPI prims under the shirt hierarchy.
    root = mesh_prim
    # climb to the nearest non-Mesh parent that represents the garment container (usually the referenced asset root)
    while root and root.GetParent() and root.GetParent().IsValid() and root.GetParent().GetPath() != Sdf.Path.absoluteRootPath:
        # stop at first parent that is not a Mesh
        if not root.GetParent().IsA(UsdGeom.Mesh):
            root = root.GetParent()
            break
        root = root.GetParent()

    colliders: List[str] = []
    for p in Usd.PrimRange(root):
        if UsdPhysics.CollisionAPI(p):
            colliders.append(p.GetPath().pathString)

    if len(colliders) == 0:
        warnings.append("No UsdPhysics.CollisionAPI found under garment. Surface deformables require collision enabled on the sim mesh; helper may add it.")
    elif len(colliders) > 1:
        warnings.append(f"Multiple CollisionAPI prims under garment ({len(colliders)}). Surface deformables often require exactly one collider (the sim mesh).")
        warnings.append("CollisionAPI prims: " + ", ".join(colliders))
    else:
        warnings.append(f"CollisionAPI prim: {colliders[0]}")

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _force_subdivision_none(mesh: UsdGeom.Mesh) -> None:
    try:
        mesh.GetSubdivisionSchemeAttr().Set("none")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    _print_header("Surface deformable preflight checks")

    stage = _get_stage()
    mesh, prim = load_mesh(stage, TSHIRT_MESH_PATH)

    if mesh is None:
        if prim is None:
            print(f"ERROR: Prim does not exist at path: {TSHIRT_MESH_PATH}")
        else:
            print(f"ERROR: Prim at path is not a UsdGeom.Mesh: {TSHIRT_MESH_PATH} (type={prim.GetTypeName()})")
            print("Tip: select the Mesh prim (not the Xform or GeomSubset) and copy its prim path.")
        return

    # Basic attribute checks (triangulation, counts/indices consistency, subdivision)
    r_attr = check_mesh_attributes(mesh)
    _print_block("Warnings", r_attr.warnings)
    _print_block("Errors", r_attr.errors)
    if not r_attr.ok and not ABORT_ON_WARNINGS:
        print("\nABORT: Fix the errors above (usually: triangulate / indices mismatch) and re-run.")
        return

    # Force subdivision scheme
    _force_subdivision_none(mesh)

    # Load arrays for geometry checks
    points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)

    results = [
        _check_editability(stage, prim),
        check_point_finiteness(points),
        check_index_ranges(indices, num_points=int(points.shape[0]) if points.ndim == 2 else 0),
        check_triangle_quality(points, indices),
        check_duplicate_triangles(indices),
        check_nonmanifold_edges(indices),
        _check_collision_api_count(stage, prim),
        check_kinematic_rigid_body(prim),
    ]
    merged = merge_results(results)

    _print_block("Warnings", merged.warnings)
    _print_block("Errors", merged.errors)

    if not merged.ok or (ABORT_ON_WARNINGS and len(merged.warnings) > 0):
        print("\nABORT: Preflight checks indicate the mesh/setup is likely to fail PhysX deformable cooking.")
        print("Recommended fixes (in order):")
        print("  1) In Blender: Apply transforms, Merge by Distance, Delete Loose, remove double-wall thickness, Triangulate, Recalc normals, re-export USD.")
        print("  2) Alternatively: use PhysX Auto-Deformable remeshing (targetTriangleCount) to generate a valid sim mesh.")
        return

    _print_header("Applying Surface Deformable setup")

    ok = deformableUtils.set_physics_surface_deformable_body(stage, TSHIRT_MESH_PATH)
    print("surface deformable setup:", bool(ok))

    # Re-print final subdivision state
    scheme = mesh.GetSubdivisionSchemeAttr().Get()
    print("subdivisionScheme:", scheme)

    print("\nDone. Press Play to simulate. If you still get PxValidateTriangleMesh failures, the mesh likely has")
    print("self-intersections / non-manifold topology not caught by these checks; do the Blender cleanup or Auto-Deformable remesh.")


if __name__ == "__main__":
    main()

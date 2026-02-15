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
"""

from __future__ import annotations

from dataclasses import dataclass
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

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

TSHIRT_MESH_PATH: str = "/World/tshirt_mod/mesh"  # <- set to *Mesh* prim path
ABORT_ON_WARNINGS: bool = False  # if True: abort also on warnings (not just errors)

# Geometry thresholds (tune if needed)
DEGENERATE_AREA_EPS: float = 1e-12
NEAR_DEGENERATE_AREA_EPS: float = 1e-10
MAX_WARN_NONMANIFOLD_EDGES: int = 0
MAX_WARN_DUP_TRI_GROUPS: int = 0


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    ok: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]


def _as_tuple(xs: Iterable[str]) -> Tuple[str, ...]:
    return tuple(xs)


def _get_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


def _get_mesh(stage: Usd.Stage, mesh_path: str) -> Tuple[Optional[UsdGeom.Mesh], Optional[Usd.Prim]]:
    prim = stage.GetPrimAtPath(Sdf.Path(mesh_path))
    if not prim or not prim.IsValid():
        return None, None
    if not prim.IsA(UsdGeom.Mesh):
        return None, prim
    return UsdGeom.Mesh(prim), prim


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

    # Instanceable prims are a common reason why changes "don't stick"
    try:
        if prim.IsInstance() or prim.IsInstanceable():
            warnings.append("Prim is instance/instanceable; authoring may not apply as expected. Consider uninstancing/duplicating.")
    except Exception:
        pass

    # Referenced payloads can still be editable via layers, but warn if not in local layer stack.
    # (We cannot reliably detect edit target in all UI contexts; this is a heuristic.)
    try:
        if prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
            warnings.append("Prim has authored references/payloads; ensure your Edit Target is a writable layer.")
    except Exception:
        pass

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _check_mesh_attributes(mesh: UsdGeom.Mesh) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    points = mesh.GetPointsAttr().Get() or []

    if len(points) == 0:
        errors.append("Mesh has zero points.")
    if len(counts) == 0:
        errors.append("Mesh has zero faces (faceVertexCounts empty).")
    if len(indices) == 0:
        errors.append("Mesh has zero indices (faceVertexIndices empty).")

    if counts and indices:
        expected = int(np.sum(np.asarray(counts, dtype=np.int64)))
        if expected != len(indices):
            errors.append(f"faceVertexIndices length mismatch: expected sum(counts)={expected}, got {len(indices)}.")

    # Subdivision scheme must not be CatmullClark for deformable cooking; set to 'none'.
    scheme = mesh.GetSubdivisionSchemeAttr().Get()
    if scheme not in (None, "", "none"):
        warnings.append(f"subdivisionScheme is '{scheme}' (recommended: 'none'). Script will force it to 'none'.")

    # Surface deformables need triangles; non-tri faces typically fail cooking.
    non_tri = [c for c in counts if c != 3]
    if len(non_tri) > 0:
        errors.append(f"Found {len(non_tri)} non-triangle faces. Surface deformables require triangles only.")

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _check_point_finiteness(points_np: np.ndarray) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if points_np.ndim != 2 or points_np.shape[1] != 3:
        errors.append(f"Points array has unexpected shape {points_np.shape}; expected (N, 3).")
        return CheckResult(ok=False, errors=_as_tuple(errors), warnings=_as_tuple(warnings))

    bad = np.logical_not(np.isfinite(points_np)).any(axis=1)
    n_bad = int(bad.sum())
    if n_bad > 0:
        errors.append(f"Found {n_bad} points with NaN/Inf coordinates.")

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _check_index_ranges(indices_np: np.ndarray, n_points: int) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if indices_np.size == 0 or n_points <= 0:
        return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))

    min_i = int(indices_np.min())
    max_i = int(indices_np.max())
    if min_i < 0:
        errors.append(f"Negative vertex index detected: min index = {min_i}.")
    if max_i >= n_points:
        errors.append(f"Out-of-range vertex index detected: max index = {max_i} >= num_points = {n_points}.")

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _triangle_areas(points_np: np.ndarray, tris_np: np.ndarray) -> np.ndarray:
    a = points_np[tris_np[:, 0]]
    b = points_np[tris_np[:, 1]]
    c = points_np[tris_np[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _check_triangle_quality(points_np: np.ndarray, indices_np: np.ndarray) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if indices_np.size == 0:
        return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))

    if indices_np.size % 3 != 0:
        errors.append(f"Indices length {indices_np.size} is not divisible by 3; cannot form triangles safely.")
        return CheckResult(ok=False, errors=_as_tuple(errors), warnings=_as_tuple(warnings))

    tris = indices_np.reshape(-1, 3)
    areas = _triangle_areas(points_np, tris)
    min_area = float(np.min(areas)) if areas.size else 0.0
    n_deg = int(np.sum(areas < DEGENERATE_AREA_EPS))
    n_near = int(np.sum((areas >= DEGENERATE_AREA_EPS) & (areas < NEAR_DEGENERATE_AREA_EPS)))

    if n_deg > 0:
        errors.append(f"Found {n_deg} degenerate triangles (area < {DEGENERATE_AREA_EPS:g}).")
    if n_near > 0:
        warnings.append(f"Found {n_near} near-degenerate triangles (area < {NEAR_DEGENERATE_AREA_EPS:g}); may cause instability/cooking failures.")
    warnings.append(f"Triangle area stats: min_area={min_area:g}, mean_area={float(np.mean(areas)):g}")

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _check_duplicate_triangles(indices_np: np.ndarray) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if indices_np.size == 0 or indices_np.size % 3 != 0:
        return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))

    tris = indices_np.reshape(-1, 3)
    sorted_tris = np.sort(tris, axis=1)
    _, counts = np.unique(sorted_tris, axis=0, return_counts=True)
    dup_groups = int(np.sum(counts > 1))

    if dup_groups > 0:
        msg = f"Found {dup_groups} duplicate triangle groups (same 3 vertex indices). This frequently breaks PhysX mesh cooking."
        if dup_groups > MAX_WARN_DUP_TRI_GROUPS:
            errors.append(msg)
        else:
            warnings.append(msg)

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _check_nonmanifold_edges(indices_np: np.ndarray) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if indices_np.size == 0 or indices_np.size % 3 != 0:
        return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))

    tris = indices_np.reshape(-1, 3)
    e01 = np.sort(tris[:, [0, 1]], axis=1)
    e12 = np.sort(tris[:, [1, 2]], axis=1)
    e20 = np.sort(tris[:, [2, 0]], axis=1)
    edges = np.vstack([e01, e12, e20])

    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    nonmanifold = int(np.sum(edge_counts > 2))
    boundary = int(np.sum(edge_counts == 1))

    # Boundary edges are expected for a shirt (open boundaries at sleeves/neck/hem).
    warnings.append(f"Edge stats: boundary_edges={boundary}, nonmanifold_edges(>2 faces)={nonmanifold}")

    if nonmanifold > 0:
        msg = f"Non-manifold edges detected (used by >2 triangles): {nonmanifold}. This commonly causes PxValidateTriangleMesh to fail."
        if nonmanifold > MAX_WARN_NONMANIFOLD_EDGES:
            errors.append(msg)
        else:
            warnings.append(msg)

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


def _check_kinematic_rigid_body(mesh_prim: Usd.Prim) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    # Kinematic deformable bodies are not supported; warn if rigid body APIs indicate kinematic usage.
    if UsdPhysics.RigidBodyAPI(mesh_prim):
        warnings.append("UsdPhysics.RigidBodyAPI is applied on the mesh prim. Deformables should generally NOT also be rigid bodies.")

    if PhysxSchema is not None:
        try:
            rb = PhysxSchema.PhysxRigidBodyAPI(mesh_prim)
            if rb:
                kin_attr = rb.GetKinematicEnabledAttr()
                kin = bool(kin_attr.Get()) if kin_attr else False
                if kin:
                    errors.append("PhysxRigidBodyAPI.kinematicEnabled == True on the garment. Kinematic deformables are not supported.")
        except Exception:
            pass

    # Also warn if parent transform is animated (common cause of implicit kinematic behavior).
    # We can't robustly detect animation without scanning time samples, so add a hint.
    warnings.append("If you still see 'kinematic deformable bodies not supported', ensure the garment prim (and parents) are not animated/teleported each frame.")

    return CheckResult(ok=(len(errors) == 0), errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def _merge_results(results: Iterable[CheckResult]) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []
    ok = True
    for r in results:
        ok = ok and r.ok
        errors.extend(r.errors)
        warnings.extend(r.warnings)
    return CheckResult(ok=ok, errors=_as_tuple(errors), warnings=_as_tuple(warnings))


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
    mesh, prim = _get_mesh(stage, TSHIRT_MESH_PATH)

    if mesh is None:
        if prim is None:
            print(f"ERROR: Prim does not exist at path: {TSHIRT_MESH_PATH}")
        else:
            print(f"ERROR: Prim at path is not a UsdGeom.Mesh: {TSHIRT_MESH_PATH} (type={prim.GetTypeName()})")
            print("Tip: select the Mesh prim (not the Xform or GeomSubset) and copy its prim path.")
        return

    # Basic attribute checks (triangulation, counts/indices consistency, subdivision)
    r_attr = _check_mesh_attributes(mesh)
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
        _check_point_finiteness(points),
        _check_index_ranges(indices, n_points=int(points.shape[0]) if points.ndim == 2 else 0),
        _check_triangle_quality(points, indices),
        _check_duplicate_triangles(indices),
        _check_nonmanifold_edges(indices),
        _check_collision_api_count(stage, prim),
        _check_kinematic_rigid_body(prim),
    ]
    merged = _merge_results(results)

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

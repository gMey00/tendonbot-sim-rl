#!/usr/bin/env python3
"""Standalone cloth-readiness validator for USD meshes.

Runs entirely outside Isaac Sim — depends only on ``pxr`` (OpenUSD) and
``numpy``.  Checks whether a USD mesh prim is suitable for PhysX cloth
simulation (both PBD particle cloth and XPBD surface deformables).

Usage::

    python check_cloth_readiness.py path/to/mesh.usd [--prim /World/mesh]
    python check_cloth_readiness.py path/to/mesh.usdc --prim /World/tshirt_mod/mesh

If ``--prim`` is omitted the script auto-discovers the first
``UsdGeom.Mesh`` prim in the stage.

Exit codes: 0 = pass, 1 = fail (errors found), 2 = usage error.

Reference: cloth_simulation_isaacsim_research.md §2 (mesh requirements)
  - Particle cloth does NOT require a closed manifold
  - Must be triangulated (PhysxAutoParticleClothAPI generates springs
    from triangle topology; surface deformables limited to triangle faces)
  - Recommended ≤2000–3000 vertices per garment for 4096 environments
  - Each particle ≈64 bytes (position Vec4, velocity Vec4, phase, indices)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

# Try importing PhysxSchema — not always available outside Isaac Sim.
try:
    from pxr import PhysxSchema  # type: ignore[attr-defined]
except Exception:
    PhysxSchema = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

DEGENERATE_AREA_EPS: float = 1e-12
NEAR_DEGENERATE_AREA_EPS: float = 1e-10
MAX_WARN_NONMANIFOLD_EDGES: int = 0
MAX_WARN_DUP_TRI_GROUPS: int = 0

# Memory estimation constants (from research §2 / §6)
BYTES_PER_PARTICLE: int = 64
SOLVER_OVERHEAD_FACTOR: float = 3.0  # solver buffers ≈ 3× particle state
ROBOT_ARTICULATION_VRAM_GB: float = 3.0
RL_TRAINING_VRAM_GB: float = 5.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    ok: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]


def _as_tuple(items: Sequence[str]) -> Tuple[str, ...]:
    return tuple(items)


# ---------------------------------------------------------------------------
# Mesh discovery
# ---------------------------------------------------------------------------

def find_first_mesh(stage: Usd.Stage) -> Optional[str]:
    """Return the prim path of the first UsdGeom.Mesh in the stage."""
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            return prim.GetPath().pathString
    return None


def load_mesh(
    stage: Usd.Stage, prim_path: str
) -> Tuple[Optional[UsdGeom.Mesh], Optional[Usd.Prim]]:
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim or not prim.IsValid():
        return None, None
    if not prim.IsA(UsdGeom.Mesh):
        return None, prim
    return UsdGeom.Mesh(prim), prim


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def triangle_areas(
    points: NDArray[np.float64], triangles: NDArray[np.int64]
) -> NDArray[np.float64]:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def edge_lengths(
    points: NDArray[np.float64], triangles: NDArray[np.int64]
) -> NDArray[np.float64]:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return np.concatenate([
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(a - c, axis=1),
    ])


# ---------------------------------------------------------------------------
# Individual checks (pure functions — no omni.* dependency)
# ---------------------------------------------------------------------------

def check_mesh_attributes(mesh: UsdGeom.Mesh) -> CheckResult:
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
            errors.append(
                f"faceVertexIndices length mismatch: "
                f"expected sum(counts)={expected}, got {len(indices)}."
            )

    scheme = mesh.GetSubdivisionSchemeAttr().Get()
    if scheme not in (None, "", "none"):
        warnings.append(
            f"subdivisionScheme is '{scheme}' (must be 'none' for cloth). "
            f"Set mesh.GetSubdivisionSchemeAttr().Set('none')."
        )

    non_tri = [c for c in counts if c != 3]
    if non_tri:
        errors.append(
            f"Found {len(non_tri)} non-triangle faces. "
            f"Cloth simulation requires triangles only."
        )

    return CheckResult(ok=len(errors) == 0, errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def check_point_finiteness(points: NDArray[np.float64]) -> CheckResult:
    errors: List[str] = []

    if points.ndim != 2 or points.shape[1] != 3:
        errors.append(f"Points array has unexpected shape {points.shape}; expected (N, 3).")
        return CheckResult(ok=False, errors=_as_tuple(errors), warnings=())

    bad_count = int(np.logical_not(np.isfinite(points)).any(axis=1).sum())
    if bad_count > 0:
        errors.append(f"Found {bad_count} points with NaN/Inf coordinates.")

    return CheckResult(ok=len(errors) == 0, errors=_as_tuple(errors), warnings=())


def check_index_ranges(indices: NDArray[np.int64], num_points: int) -> CheckResult:
    errors: List[str] = []

    if indices.size == 0 or num_points <= 0:
        return CheckResult(ok=True, errors=(), warnings=())

    min_index = int(indices.min())
    max_index = int(indices.max())
    if min_index < 0:
        errors.append(f"Negative vertex index detected: min index = {min_index}.")
    if max_index >= num_points:
        errors.append(
            f"Out-of-range vertex index: max index = {max_index} >= "
            f"num_points = {num_points}."
        )

    return CheckResult(ok=len(errors) == 0, errors=_as_tuple(errors), warnings=())


def check_triangle_quality(
    points: NDArray[np.float64], indices: NDArray[np.int64]
) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if indices.size == 0:
        return CheckResult(ok=True, errors=(), warnings=())

    if indices.size % 3 != 0:
        errors.append(
            f"Indices length {indices.size} is not divisible by 3; "
            f"cannot form triangles."
        )
        return CheckResult(ok=False, errors=_as_tuple(errors), warnings=())

    tris = indices.reshape(-1, 3)
    areas = triangle_areas(points, tris)
    min_area = float(np.min(areas)) if areas.size else 0.0

    degenerate_count = int(np.sum(areas < DEGENERATE_AREA_EPS))
    near_degenerate_count = int(np.sum(
        (areas >= DEGENERATE_AREA_EPS) & (areas < NEAR_DEGENERATE_AREA_EPS)
    ))

    if degenerate_count > 0:
        errors.append(
            f"Found {degenerate_count} degenerate triangles "
            f"(area < {DEGENERATE_AREA_EPS:g})."
        )
    if near_degenerate_count > 0:
        warnings.append(
            f"Found {near_degenerate_count} near-degenerate triangles "
            f"(area < {NEAR_DEGENERATE_AREA_EPS:g}); may cause instability."
        )
    warnings.append(
        f"Triangle area stats: "
        f"min={min_area:g}, mean={float(np.mean(areas)):g}"
    )

    return CheckResult(ok=len(errors) == 0, errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def check_duplicate_triangles(indices: NDArray[np.int64]) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if indices.size == 0 or indices.size % 3 != 0:
        return CheckResult(ok=True, errors=(), warnings=())

    tris = indices.reshape(-1, 3)
    sorted_tris = np.sort(tris, axis=1)
    _, counts = np.unique(sorted_tris, axis=0, return_counts=True)
    duplicate_groups = int(np.sum(counts > 1))

    if duplicate_groups > 0:
        message = (
            f"Found {duplicate_groups} duplicate triangle groups "
            f"(same 3 vertex indices). Frequently breaks PhysX mesh cooking."
        )
        if duplicate_groups > MAX_WARN_DUP_TRI_GROUPS:
            errors.append(message)
        else:
            warnings.append(message)

    return CheckResult(ok=len(errors) == 0, errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def check_nonmanifold_edges(indices: NDArray[np.int64]) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if indices.size == 0 or indices.size % 3 != 0:
        return CheckResult(ok=True, errors=(), warnings=())

    tris = indices.reshape(-1, 3)
    edge_pairs = np.vstack([
        np.sort(tris[:, [0, 1]], axis=1),
        np.sort(tris[:, [1, 2]], axis=1),
        np.sort(tris[:, [2, 0]], axis=1),
    ])

    _, edge_counts = np.unique(edge_pairs, axis=0, return_counts=True)
    nonmanifold_count = int(np.sum(edge_counts > 2))
    boundary_count = int(np.sum(edge_counts == 1))

    # Boundary edges are expected for garments (sleeves, neck, hem).
    warnings.append(
        f"Edge stats: boundary={boundary_count}, "
        f"nonmanifold(>2 faces)={nonmanifold_count}"
    )

    if nonmanifold_count > 0:
        message = (
            f"Non-manifold edges detected (used by >2 triangles): "
            f"{nonmanifold_count}. Commonly causes PxValidateTriangleMesh failure."
        )
        if nonmanifold_count > MAX_WARN_NONMANIFOLD_EDGES:
            errors.append(message)
        else:
            warnings.append(message)

    return CheckResult(ok=len(errors) == 0, errors=_as_tuple(errors), warnings=_as_tuple(warnings))


def check_vertex_spacing_uniformity(
    points: NDArray[np.float64], indices: NDArray[np.int64]
) -> CheckResult:
    """Warn if edge lengths vary widely (bad for PBD spring generation)."""
    warnings: List[str] = []

    if indices.size == 0 or indices.size % 3 != 0:
        return CheckResult(ok=True, errors=(), warnings=())

    tris = indices.reshape(-1, 3)
    lengths = edge_lengths(points, tris)
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    min_length = float(np.min(lengths))
    max_length = float(np.max(lengths))
    coefficient_of_variation = std_length / mean_length if mean_length > 0 else 0.0

    warnings.append(
        f"Edge length stats: "
        f"min={min_length:.6f}, max={max_length:.6f}, "
        f"mean={mean_length:.6f}, std={std_length:.6f}, "
        f"CV={coefficient_of_variation:.2f}"
    )

    if coefficient_of_variation > 1.0:
        warnings.append(
            "High edge-length variation (CV > 1.0). Very uneven triangles "
            "cause PBD simulation artifacts. Consider remeshing."
        )

    return CheckResult(ok=True, errors=(), warnings=_as_tuple(warnings))


def check_open_vs_closed(indices: NDArray[np.int64]) -> CheckResult:
    """Report whether the mesh is open (expected for garments) or closed."""
    warnings: List[str] = []

    if indices.size == 0 or indices.size % 3 != 0:
        return CheckResult(ok=True, errors=(), warnings=())

    tris = indices.reshape(-1, 3)
    edge_pairs = np.vstack([
        np.sort(tris[:, [0, 1]], axis=1),
        np.sort(tris[:, [1, 2]], axis=1),
        np.sort(tris[:, [2, 0]], axis=1),
    ])
    _, edge_counts = np.unique(edge_pairs, axis=0, return_counts=True)
    boundary_count = int(np.sum(edge_counts == 1))

    if boundary_count == 0:
        warnings.append(
            "Mesh is CLOSED (no boundary edges). Garments are typically "
            "open meshes. If this is a watertight shell, cloth simulation "
            "may behave unexpectedly."
        )
    else:
        warnings.append(
            f"Mesh is OPEN ({boundary_count} boundary edges). "
            f"Expected for garments (sleeves, neck, hem)."
        )

    return CheckResult(ok=True, errors=(), warnings=_as_tuple(warnings))


def check_kinematic_rigid_body(prim: Usd.Prim) -> CheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if UsdPhysics.RigidBodyAPI(prim):
        warnings.append(
            "UsdPhysics.RigidBodyAPI applied on the mesh. "
            "Deformables should generally NOT also be rigid bodies."
        )

    if PhysxSchema is not None:
        try:
            rigid_body = PhysxSchema.PhysxRigidBodyAPI(prim)
            if rigid_body:
                kinematic_attr = rigid_body.GetKinematicEnabledAttr()
                is_kinematic = bool(kinematic_attr.Get()) if kinematic_attr else False
                if is_kinematic:
                    errors.append(
                        "PhysxRigidBodyAPI.kinematicEnabled is True. "
                        "Kinematic deformables are not supported."
                    )
        except Exception:
            pass

    return CheckResult(ok=len(errors) == 0, errors=_as_tuple(errors), warnings=_as_tuple(warnings))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def merge_results(results: Sequence[CheckResult]) -> CheckResult:
    all_errors: List[str] = []
    all_warnings: List[str] = []
    ok = True
    for result in results:
        ok = ok and result.ok
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)
    return CheckResult(ok=ok, errors=_as_tuple(all_errors), warnings=_as_tuple(all_warnings))


# ---------------------------------------------------------------------------
# Performance estimation (research §2 / §6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerformanceEstimate:
    num_vertices: int
    num_triangles: int
    num_edges: int
    average_edge_length: float
    particle_state_bytes: int
    recommended_solid_rest_offset: float
    recommended_contact_offset: float

    def vram_estimate_gb(self, num_envs: int) -> float:
        """Estimated cloth VRAM in GB for *num_envs* parallel environments.

        Formula from research §6:
          particle_state = num_vertices × BYTES_PER_PARTICLE × num_envs
          solver_buffers ≈ SOLVER_OVERHEAD_FACTOR × particle_state
        """
        particle_bytes = self.num_vertices * BYTES_PER_PARTICLE * num_envs
        total_bytes = particle_bytes * SOLVER_OVERHEAD_FACTOR
        return total_bytes / (1024 ** 3)

    def total_vram_estimate_gb(self, num_envs: int) -> float:
        """Total estimated VRAM including robot + RL training overhead."""
        return (
            self.vram_estimate_gb(num_envs)
            + ROBOT_ARTICULATION_VRAM_GB
            + RL_TRAINING_VRAM_GB
        )


def estimate_performance(
    points: NDArray[np.float64], indices: NDArray[np.int64]
) -> PerformanceEstimate:
    num_vertices = points.shape[0]
    num_triangles = indices.size // 3 if indices.size >= 3 else 0

    tris = indices.reshape(-1, 3) if num_triangles > 0 else np.empty((0, 3), dtype=np.int64)
    edge_pairs = np.vstack([
        np.sort(tris[:, [0, 1]], axis=1),
        np.sort(tris[:, [1, 2]], axis=1),
        np.sort(tris[:, [2, 0]], axis=1),
    ]) if num_triangles > 0 else np.empty((0, 2), dtype=np.int64)
    unique_edges = np.unique(edge_pairs, axis=0) if edge_pairs.size > 0 else np.empty((0, 2), dtype=np.int64)
    num_edges = unique_edges.shape[0]

    lengths = edge_lengths(points, tris) if num_triangles > 0 else np.array([0.0])
    avg_edge = float(np.mean(lengths)) if lengths.size > 0 else 0.0

    # Research §3: solidRestOffset ≈ 0.5 × average_edge_length
    recommended_rest = 0.5 * avg_edge
    recommended_contact = recommended_rest * 1.5

    return PerformanceEstimate(
        num_vertices=num_vertices,
        num_triangles=num_triangles,
        num_edges=num_edges,
        average_edge_length=avg_edge,
        particle_state_bytes=num_vertices * BYTES_PER_PARTICLE,
        recommended_solid_rest_offset=recommended_rest,
        recommended_contact_offset=recommended_contact,
    )


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------

def validate_cloth_mesh(
    stage: Usd.Stage, prim_path: str
) -> Tuple[CheckResult, Optional[PerformanceEstimate]]:
    """Run all cloth-readiness checks on a USD mesh prim.

    Returns (merged_result, performance_estimate_or_None).
    """
    mesh, prim = load_mesh(stage, prim_path)

    if mesh is None:
        if prim is None:
            error = f"Prim does not exist at path: {prim_path}"
        else:
            error = (
                f"Prim at path is not a UsdGeom.Mesh: {prim_path} "
                f"(type={prim.GetTypeName()})"
            )
        return CheckResult(ok=False, errors=(error,), warnings=()), None

    # Structural checks first (triangulation, counts/indices)
    attribute_result = check_mesh_attributes(mesh)
    if not attribute_result.ok:
        return attribute_result, None

    # Load arrays
    points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)

    num_points = int(points.shape[0]) if points.ndim == 2 else 0

    results = [
        attribute_result,
        check_point_finiteness(points),
        check_index_ranges(indices, num_points),
        check_triangle_quality(points, indices),
        check_duplicate_triangles(indices),
        check_nonmanifold_edges(indices),
        check_vertex_spacing_uniformity(points, indices),
        check_open_vs_closed(indices),
        check_kinematic_rigid_body(prim),
    ]

    merged = merge_results(results)
    estimate = estimate_performance(points, indices) if merged.ok else None

    return merged, estimate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_block(label: str, items: Sequence[str]) -> None:
    if not items:
        return
    print(f"\n{label}:")
    for item in items:
        print(f"  - {item}")


def _print_performance(estimate: PerformanceEstimate) -> None:
    _print_header("Mesh Statistics & Performance Estimates")
    print(f"  Vertices:          {estimate.num_vertices:,}")
    print(f"  Triangles:         {estimate.num_triangles:,}")
    print(f"  Unique edges:      {estimate.num_edges:,}")
    print(f"  Avg edge length:   {estimate.average_edge_length:.6f} m")
    print(f"  Particle state:    {estimate.particle_state_bytes:,} bytes/env")
    print()
    print("  Recommended PhysX offsets:")
    print(f"    solidRestOffset:    {estimate.recommended_solid_rest_offset:.6f} m")
    print(f"    contactOffset:      {estimate.recommended_contact_offset:.6f} m")
    print()
    print("  VRAM estimates (cloth only | total with robot + RL):")
    for num_envs in (1024, 2048, 4096, 8192):
        cloth_gb = estimate.vram_estimate_gb(num_envs)
        total_gb = estimate.total_vram_estimate_gb(num_envs)
        fits_48gb = "OK" if total_gb <= 48.0 else "EXCEEDS 48 GB"
        print(
            f"    {num_envs:>5} envs: "
            f"{cloth_gb:6.2f} GB cloth | "
            f"{total_gb:6.2f} GB total  [{fits_48gb}]"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check USD mesh readiness for cloth simulation."
    )
    parser.add_argument("usd_path", help="Path to the USD file.")
    parser.add_argument(
        "--prim",
        default=None,
        help="Prim path to the Mesh prim. Auto-discovered if omitted.",
    )
    args = parser.parse_args(argv)

    # Open stage
    try:
        stage = Usd.Stage.Open(args.usd_path)
    except Exception as exc:
        print(f"ERROR: Cannot open USD file: {args.usd_path}\n  {exc}")
        return 2

    if stage is None:
        print(f"ERROR: Cannot open USD file: {args.usd_path}")
        return 2

    # Resolve prim path
    prim_path = args.prim
    if prim_path is None:
        prim_path = find_first_mesh(stage)
        if prim_path is None:
            print("ERROR: No UsdGeom.Mesh prim found in stage.")
            return 2
        print(f"Auto-discovered mesh prim: {prim_path}")

    _print_header("Cloth Readiness Check")
    print(f"  File: {args.usd_path}")
    print(f"  Prim: {prim_path}")

    result, estimate = validate_cloth_mesh(stage, prim_path)

    _print_block("Warnings", result.warnings)
    _print_block("Errors", result.errors)

    if estimate is not None:
        _print_performance(estimate)

    print()
    if result.ok:
        print("RESULT: PASS — mesh is cloth-ready.")
        return 0
    else:
        print("RESULT: FAIL — fix the errors above before using for cloth simulation.")
        print("Recommended fixes:")
        print("  1) Blender: Apply transforms → Merge by Distance → Delete Loose →"
              " Triangulate → Recalc normals → re-export USD.")
        print("  2) Isaac Sim: Use PhysxAutoDeformableMeshSimplificationAPI with"
              " remeshingEnabled + targetTriangleCount.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Preflight-checked cloth application: validate the mesh, then author cloth.

Combines ``check_cloth_readiness`` (geometry validation that frequently catches
the issues behind ``PxValidateTriangleMesh ... failed``) with the schema authors
in ``make_pbd_cloth`` / ``make_xpbd_cloth``.  It aborts *before* authoring any
schema when the mesh has errors that almost always break PhysX cooking.

Runs in plain Python (source ``Scripts/_usdenv.sh``) or in the Isaac Sim Script
Editor (it auto-detects the open stage).

Usage (standalone)::

    source Scripts/_usdenv.sh
    "$KPY" Scripts/make_cloth_checked.py \
        --input tshirt_mod.usdc --output tshirt_pbd.usd --backend pbd

Usage (Script Editor)::

    Set BACKEND / MESH_PATH below and run (Ctrl+Enter) on the open stage.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
from pxr import Usd

# Make sibling modules importable (for Script Editor usage).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from check_cloth_readiness import (  # noqa: E402
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
from make_pbd_cloth import create_pbd_cloth  # noqa: E402
from make_xpbd_cloth import create_xpbd_cloth  # noqa: E402

# Script Editor config (ignored in standalone CLI mode)
BACKEND = "pbd"                       # "pbd" or "xpbd"
MESH_PATH = "/root/World/mesh/Mesh"   # <- set to the *Mesh* prim path
ABORT_ON_WARNINGS = False


def _print_block(label, items):
    items = list(items)
    if not items:
        return
    print(f"\n{label}:")
    for it in items:
        print(f"  - {it}")


def preflight(stage: Usd.Stage, mesh_path: str) -> bool:
    """Return True if the mesh passes the cooking-relevant geometry checks."""
    mesh, prim = load_mesh(stage, mesh_path)
    if mesh is None:
        if prim is None:
            print(f"ERROR: prim does not exist: {mesh_path}")
        else:
            print(f"ERROR: prim is not a UsdGeom.Mesh: {mesh_path} (type={prim.GetTypeName()})")
        return False

    r_attr = check_mesh_attributes(mesh)
    _print_block("Warnings", r_attr.warnings)
    _print_block("Errors", r_attr.errors)
    if not r_attr.ok:
        print("\nABORT: fix the attribute errors above (triangulate / index mismatch).")
        return False

    points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
    npts = int(points.shape[0]) if points.ndim == 2 else 0

    merged = merge_results([
        check_point_finiteness(points),
        check_index_ranges(indices, npts),
        check_triangle_quality(points, indices),
        check_duplicate_triangles(indices),
        check_nonmanifold_edges(indices),
        check_kinematic_rigid_body(prim),
    ])
    _print_block("Warnings", merged.warnings)
    _print_block("Errors", merged.errors)

    if not merged.ok or (ABORT_ON_WARNINGS and merged.warnings):
        print("\nABORT: preflight indicates the mesh is likely to fail PhysX cooking.")
        print("  Blender fix: Apply transforms, Merge by Distance, Delete Loose,")
        print("  remove double-wall thickness, Triangulate, Recalc normals, re-export.")
        return False
    return True


def apply_backend(stage: Usd.Stage, mesh_path: str, backend: str) -> None:
    if backend == "pbd":
        create_pbd_cloth(stage, mesh_path)
    elif backend == "xpbd":
        create_xpbd_cloth(stage, mesh_path)
    else:
        raise ValueError(f"Unknown backend '{backend}' (use 'pbd' or 'xpbd').")


def _get_script_editor_stage():
    try:
        import omni.usd
        return omni.usd.get_context().get_stage()
    except Exception:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preflight-checked cloth application.")
    parser.add_argument("--input", help="Input USD (omit in the Script Editor).")
    parser.add_argument("--output", help="Output USD (defaults to overwriting --input).")
    parser.add_argument("--prim", default=MESH_PATH, help="Mesh prim path.")
    parser.add_argument("--backend", default=BACKEND, choices=["pbd", "xpbd"])
    args = parser.parse_args(argv)

    stage = _get_script_editor_stage()
    standalone = stage is None
    if standalone:
        if not args.input:
            print("ERROR: --input required when not in the Script Editor.")
            return 2
        stage = Usd.Stage.Open(args.input)
        if stage is None:
            print(f"ERROR: cannot open {args.input}")
            return 2

    print("=" * 72)
    print(f"  Preflight checks ({args.backend.upper()})")
    print("=" * 72)
    if not preflight(stage, args.prim):
        return 1

    print("\n" + "=" * 72)
    print("  Applying cloth schemas")
    print("=" * 72)
    apply_backend(stage, args.prim, args.backend)

    if standalone:
        out = args.output or args.input
        stage.GetRootLayer().Export(out)
        print(f"\nSaved to: {out}")
    else:
        print("\nDone. Press Play to simulate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

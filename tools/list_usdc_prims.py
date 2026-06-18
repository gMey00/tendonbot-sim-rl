#!/usr/bin/env python3
"""List all top-level prims and their children in the Blender USDC."""
from __future__ import annotations

import ctypes
import glob
import os
import sys

# Bootstrap pxr (same logic as build_tensegrity_arm_usd.py)
isaac = os.environ.get("ISAAC_PATH") or os.path.expanduser("~/Isaac/IsaacSim")
extscache = os.path.join(isaac, "extscache")

def _latest(pattern: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(extscache, pattern)))
    return matches[-1] if matches else None

usd_libs = _latest("omni.usd.libs-*")
if usd_libs:
    sys.path.insert(0, usd_libs)
    for so in sorted(glob.glob(os.path.join(usd_libs, "bin", "libusd_*.so"))):
        try:
            ctypes.CDLL(so, ctypes.RTLD_GLOBAL)
        except OSError:
            pass

from pxr import Usd, UsdGeom  # noqa: E402

USDC = "/home/robot/studentische-arbeiten/res/Tensegrity/meshes/tensegrity_arm_cad.usdc"
stage = Usd.Stage.Open(USDC)
root = stage.GetPseudoRoot()

print(f"=== Prims in {USDC} ===\n")
for child in root.GetChildren():
    mesh_count = sum(1 for p in Usd.PrimRange(child) if p.IsA(UsdGeom.Mesh))
    print(f"{child.GetName():<55} [{child.GetTypeName():<10}]  meshes={mesh_count}")
    for sub in child.GetChildren():
        sub_mesh = sum(1 for p in Usd.PrimRange(sub) if p.IsA(UsdGeom.Mesh))
        print(f"  {sub.GetName():<53} [{sub.GetTypeName():<10}]  meshes={sub_mesh}")

# Show transforms for rod meshes
print("\n=== Rod mesh transforms ===\n")
for name in ["joint_rods_left_visual", "joint_rods_right_visual",
             "joint_rods_left_collision", "joint_rods_right_collision",
             "elbow_approx_visual", "forearm_visual", "upper_arm_visual"]:
    prim = None
    for p in stage.Traverse():
        if p.GetName() == name:
            prim = p
            break
    if prim:
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        print(f"{name}:")
        for op in ops:
            print(f"  {op.GetOpName()}: {op.Get()}")
        # Also get bounding box
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        box = bbox.ComputeWorldBound(prim)
        rng = box.ComputeAlignedRange()
        print(f"  world bbox: min={rng.GetMin()}, max={rng.GetMax()}")
        print()

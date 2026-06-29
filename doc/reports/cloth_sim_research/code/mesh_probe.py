"""Probe the t-shirt mesh: raw extents, xform, open/closed, panel structure."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
from pxr import Usd, UsdGeom, Gf

USD = "/home/robot/studentische-arbeiten/res/Props/Cloth/tshirt_mod.usdc"
MESH = "/root/World/mesh/Mesh"

_RESF = open("/tmp/claude-1000/-home-robot-studentische-arbeiten/1975d042-f83b-4129-ac13-dfddbc8df8fa/scratchpad/probe_result.txt", "w")
def print(*a, **k):  # noqa
    import builtins
    builtins.print(*a, **k, file=_RESF, flush=True)

stage = Usd.Stage.Open(USD)
mesh = UsdGeom.Mesh(stage.GetPrimAtPath(MESH))
pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)

tc = Usd.TimeCode.Default()
l2w = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(tc), dtype=np.float64).reshape(4, 4)
world = (np.hstack([pts, np.ones((len(pts), 1))]) @ l2w)[:, :3]

def report(name, p):
    mn, mx = p.min(0), p.max(0)
    ext = mx - mn
    print(f"\n[{name}]  n={len(p)}")
    print(f"  min  = {np.round(mn,4)}")
    print(f"  max  = {np.round(mx,4)}")
    print(f"  ext  = {np.round(ext,4)}  (thinnest axis = {int(np.argmin(ext))})")
    print(f"  mean = {np.round(p.mean(0),4)}")
    # thickness distribution along thin axis: how 'flat' is it really?
    thin = int(np.argmin(ext))
    plane = [a for a in (0,1,2) if a != thin]
    # bin into a coarse grid on the plane, measure spread of thin coord per cell
    import collections
    g = collections.defaultdict(list)
    res = 0.05
    for q in p:
        key = (round(q[plane[0]]/res), round(q[plane[1]]/res))
        g[key].append(q[thin])
    spreads = [max(v)-min(v) for v in g.values() if len(v) > 1]
    if spreads:
        print(f"  per-cell thin-axis spread: mean={np.mean(spreads):.4f} "
              f"median={np.median(spreads):.4f} max={np.max(spreads):.4f}")
        print(f"  -> if spread ~ full ext ({ext[thin]:.3f}), the surface is a "
              f"DOUBLE layer (front+back panels stacked) = a real flat garment.")
        print(f"  -> if spread << ext, the thin axis spread comes from 3D curvature, not flatness.")

report("LOCAL points", pts)
report("WORLD (xform applied)", world)
print(f"\nxform L2W=\n{np.round(l2w,4)}")

# open/closed: count boundary edges
from collections import defaultdict
edge = defaultdict(int)
o = 0
for c in counts:
    face = idx[o:o+c]; o += c
    for i in range(c):
        a, b = face[i], face[(i+1)%c]
        edge[(min(a,b), max(a,b))] += 1
boundary = sum(1 for v in edge.values() if v == 1)
print(f"\nfaces={len(counts)}  edges={len(edge)}  boundary(open) edges={boundary} "
      f"-> {'OPEN surface' if boundary>0 else 'CLOSED shell'}")

app.close()

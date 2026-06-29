"""Boot Isaac Sim headless, load a cloth USD, simulate, and check behaviour.

This is the *dynamic* verification that complements the static schema check in
``verify_cloth_schemas.py``.  It loads the authored cloth, adds a ground plane,
runs the PhysX solver for a number of steps, and asserts the cloth:

  * stays finite (no NaN/Inf blow-up),
  * falls under gravity (mean Z decreases), and
  * does not explode (bounding box stays bounded).

Run with the Isaac Sim python launcher, e.g.::

    /home/robot/Isaac/IsaacLab/isaaclab.sh -p Scripts/simulate_cloth_check.py \
        --usd tshirt_pbd.usd --steps 150 --backend pbd
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--prim", default="/root/World/mesh/Mesh")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--ground-z", type=float, default=-1.0,
                    help="Z height of the ground plane (default below the shirt so it falls and drapes).")
parser.add_argument("--backend", default="pbd", choices=["pbd", "xpbd"])
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True})

# Enable the deformable beta feature flag (needed by FEM surface deformables).
import carb  # noqa: E402
carb.settings.get_settings().set_bool("/physics/enableDeformableBeta", True)

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics  # noqa: E402


def log(*a):
    print("[clothcheck]", *a, flush=True)


def main() -> int:
    ctx = omni.usd.get_context()
    ok = ctx.open_stage(args.usd)
    log(f"open_stage({args.usd}) ok={ok}")
    stage = ctx.get_stage()

    # Honour the stage up-axis so the test works for Y-up and Z-up assets.
    up_axis = UsdGeom.GetStageUpAxis(stage)
    up = 1 if up_axis == UsdGeom.Tokens.y else 2  # index of the "up" coordinate
    log(f"upAxis={up_axis} mpu={UsdGeom.GetStageMetersPerUnit(stage)}")

    # Ground plane (perpendicular to up) with a collider, at args.ground_z along up.
    ground_path = "/groundPlane"
    if not stage.GetPrimAtPath(ground_path):
        plane = UsdGeom.Mesh.Define(stage, ground_path + "/mesh")
        s = 5.0
        g = args.ground_z
        corners = [(-s, -s), (s, -s), (s, s), (-s, s)]
        if up == 1:  # Y-up: ground in the XZ plane
            pts = [(a, g, b) for (a, b) in corners]
            normal = (0, 1, 0)
        else:        # Z-up: ground in the XY plane
            pts = [(a, b, g) for (a, b) in corners]
            normal = (0, 0, 1)
        plane.CreatePointsAttr(pts)
        plane.CreateFaceVertexCountsAttr([4])
        plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        plane.CreateNormalsAttr([normal] * 4)
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(Sdf.Path(args.prim)))
    if not mesh:
        print(f"ERROR: no mesh at {args.prim}")
        return 1

    def world_points():
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        xform = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0)
        m = np.array(xform).reshape(4, 4)
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        return (homog @ m)[:, :3]

    p0 = world_points()
    z0 = float(p0[:, up].mean())
    log(f"Initial: n={len(p0)} meanUp={z0:.4f} bbox={p0.min(0)} .. {p0.max(0)}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    # Driving the app forward advances PhysX once the timeline is playing; this
    # also attaches the stage to the solver and cooks the cloth.
    for i in range(args.steps):
        sim_app.update()
        if (i + 1) % 25 == 0:
            zc = float(world_points()[:, up].mean())
            log(f"  step {i + 1:3d}: meanUp={zc:.4f}")
    timeline.stop()

    p1 = world_points()
    z1 = float(p1[:, up].mean())
    finite = bool(np.isfinite(p1).all())
    extent = float(np.linalg.norm(p1.max(0) - p1.min(0)))
    extent0 = float(np.linalg.norm(p0.max(0) - p0.min(0)))
    log(f"After {args.steps} steps: meanUp={z1:.4f} finite={finite} "
          f"extent {extent0:.3f}->{extent:.3f}")

    ok = True
    if not finite:
        print("FAIL: non-finite vertex positions (solver blew up).")
        ok = False
    if z1 >= z0 - 1e-4:
        print(f"FAIL: cloth did not fall under gravity (meanUp {z0:.4f} -> {z1:.4f}).")
        ok = False
    if extent > 5.0 * extent0:
        print(f"FAIL: cloth exploded (extent {extent0:.3f} -> {extent:.3f}).")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


rc = 1
try:
    rc = main()
except Exception:
    import traceback
    traceback.print_exc()
    rc = 1
finally:
    sys.stdout.flush()
    sim_app.close()
sys.exit(rc)

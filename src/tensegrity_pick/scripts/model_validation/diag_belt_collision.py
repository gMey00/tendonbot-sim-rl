"""Bug-B probe v2: does the gripper physically collide with the conveyor box?

Tracks the *actual collision-mesh* minimum Z of the gripper links (local bbox
captured at spawn, transformed by the live body pose) while driving the arm
down into the belt, plus base_z joint pos/target/torque.  If the mesh bottom
settles clearly below 0.8 m the box exerts no contact response on the robot.

Usage::
    python scripts/model_validation/diag_belt_collision.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-Tensegrity-Shirt-Place-Play-v0")
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402

IDX_ELBOW, IDX_WY, IDX_WX, IDX_GRIP, IDX_BY, IDX_BZ = range(6)
OPEN, CLOSE = 1.0, -1.0

TRACK_BODIES = [
    "robotiq_base_link", "left_outer_knuckle", "right_outer_knuckle",
    "left_outer_finger", "right_outer_finger", "left_inner_finger",
    "right_inner_finger", "left_inner_knuckle", "right_inner_knuckle",
    "wrist_link", "forearm_link",
]


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    robot = u.scene["robot"]
    names = robot.body_names
    stage = u.sim.stage
    env0 = u.scene.env_prim_paths[0]

    # ── ConveyorCollider box bbox at spawn ────────────────────────────
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=False)
    box_prim = stage.GetPrimAtPath(Sdf.Path(f"{env0}/ConveyorCollider"))
    box_range = bbox_cache.ComputeWorldBound(box_prim).ComputeAlignedRange()
    print(f"[AUDIT] ConveyorCollider world bbox: min={box_range.GetMin()} max={box_range.GetMax()}")

    # ── Capture local collider bboxes per tracked body (at spawn) ────
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    corners = {}
    for bname in TRACK_BODIES:
        if bname not in names:
            continue
        # find the body prim under the robot subtree
        found = None
        for prim in Usd.PrimRange(stage.GetPrimAtPath(Sdf.Path(f"{env0}/Robot"))):
            if prim.GetName() == bname and prim.IsA(UsdGeom.Xformable):
                found = prim
                break
        if found is None:
            print(f"[AUDIT] body prim not found for {bname}")
            continue
        # does the subtree have an enabled collider?
        has_col = any(
            p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(found)
        )
        wb = bbox_cache.ComputeWorldBound(found).ComputeAlignedRange()
        mn, mx = wb.GetMin(), wb.GetMax()
        # world corners -> body frame
        body_xf = xf_cache.GetLocalToWorldTransform(found)
        inv = body_xf.GetInverse()
        cs = []
        for x in (mn[0], mx[0]):
            for y in (mn[1], mx[1]):
                for z in (mn[2], mx[2]):
                    p = inv.Transform(Gf.Vec3d(x, y, z))
                    cs.append([p[0], p[1], p[2]])
        corners[bname] = torch.tensor(cs, device=dev, dtype=torch.float32)  # [8,3]
        print(f"[AUDIT] {bname}: collider={has_col} world_z=[{mn[2]:.3f},{mx[2]:.3f}]")

    body_idx = {b: names.index(b) for b in corners}
    jn = robot.joint_names
    limits = robot.data.soft_joint_pos_limits[0]
    for j in ("base_z_joint", "base_y_joint", "elbow_joint", "wrist_y_joint", "wrist_x_joint"):
        ji = jn.index(j)
        print(f"[AUDIT] joint {j}: limits=({float(limits[ji,0]):.3f}, {float(limits[ji,1]):.3f})")
    bz = jn.index("base_z_joint")

    def mesh_min_z() -> tuple[float, str]:
        best, who = 1e9, "?"
        for bname, cs in corners.items():
            bi = body_idx[bname]
            pos = robot.data.body_pos_w[0, bi]          # [3]
            quat = robot.data.body_quat_w[0, bi]        # [4]
            world = pos.unsqueeze(0) + quat_apply(quat.unsqueeze(0).expand(8, 4), cs)
            z = float(world[:, 2].min())
            if z < best:
                best, who = z, bname
        return best, who

    a = torch.zeros(n, 6, device=dev)

    def phase(tag: str, steps: int, act: torch.Tensor) -> None:
        for i in range(steps):
            env.step(act)
            if i % 30 == 29:
                mz, who = mesh_min_z()
                tip_z = float(u._finger_tip_pos()[0, 2])
                pos = float(robot.data.joint_pos[0, bz])
                tgt = float(robot.data.joint_pos_target[0, bz])
                tau = float(robot.data.applied_torque[0, bz])
                print(
                    f"[{tag}] step={i+1} mesh_min_z={mz:.3f}({who}) tip_z={tip_z:.3f} "
                    f"base_z pos={pos:.3f} tgt={tgt:.3f} tau={tau:.1f}",
                    flush=True,
                )

    # ── Phase D: free-air closure sweep — real fingertip z vs closure ──
    # Neutral pose (base_z = 0), slowly close the gripper and sample the
    # actual finger-mesh bottom relative to tool_link_0, to calibrate
    # FINGER_TIP_OPEN_Z / FINGER_TIP_CLOSED_Z.
    from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import FINGER_JOINT_CLOSE_POS
    ee_i = u._ee_body_idx
    fj = u._finger_joint_idx
    finger_bodies = [b for b in corners if "finger" in b or "knuckle" in b]

    def finger_mesh_min_z() -> float:
        best = 1e9
        for bname in finger_bodies:
            bi = body_idx[bname]
            pos = robot.data.body_pos_w[0, bi]
            quat = robot.data.body_quat_w[0, bi]
            world = pos.unsqueeze(0) + quat_apply(quat.unsqueeze(0).expand(8, 4), corners[bname])
            best = min(best, float(world[:, 2].min()))
        return best

    a.zero_(); a[:, IDX_GRIP] = OPEN
    for _ in range(30):
        env.step(a)
    a.zero_(); a[:, IDX_GRIP] = CLOSE
    for i in range(40):
        env.step(a)
        closure = float(robot.data.joint_pos[0, fj]) / FINGER_JOINT_CLOSE_POS
        ee_z = float(robot.data.body_pos_w[0, ee_i, 2])
        mz = finger_mesh_min_z()
        if i % 3 == 0 or i == 39:
            print(f"[D closure sweep] closure={closure:.2f} ee_z={ee_z:.3f} "
                  f"finger_mesh_min_z={mz:.3f} offset={mz-ee_z:+.4f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

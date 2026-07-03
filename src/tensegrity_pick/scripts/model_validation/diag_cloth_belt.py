"""Diagnostic: where does the cloth sit relative to the belt surface?

Prints the cloth particle Z distribution over time and renders a side view
(camera at belt height looking horizontally) + a top-down view, so we can see
whether the shirt rests ON the belt or sinks below it.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/tmp/shirt_render")
parser.add_argument("--steps", type=int, default=180)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import tensegrity_pick.tasks  # noqa: E402,F401
from tensegrity_pick.tasks.manager_based.shirt_place.shirt_place_scene_cfg import (  # noqa: E402
    CONVEYOR_SURFACE_HEIGHT_M,
)
from _recorder import add_record_camera  # noqa: E402

ENV_ID = "Template-Tensegrity-Shirt-Place-Play-v0"


def main() -> None:
    from isaaclab_tasks.utils import parse_env_cfg
    from PIL import Image

    env_cfg = parse_env_cfg(ENV_ID, num_envs=1)
    add_record_camera(env_cfg.scene, prim_path="/World/envs/env_0/DiagCam")
    env_cfg.scene.dome_light.spawn.intensity = 1500.0
    env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
    cam = env.scene["record_cam"]
    cloth = env.cloth
    belt = CONVEYOR_SURFACE_HEIGHT_M

    env.reset()
    zero = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    for i in range(args_cli.steps):
        env.step(zero)
        if (i + 1) % 30 == 0:
            cloth.update()
            p = cloth.nodal_pos_w[0].cpu().numpy()
            z = p[:, 2]
            pct = np.percentile(z, [1, 25, 50, 75, 99])
            below = float((z < belt).mean())
            c = p.mean(0)
            xext = p[:, 0].max() - p[:, 0].min()
            yext = p[:, 1].max() - p[:, 1].min()
            print(f"  step {i+1:3d}: belt={belt:.3f}  centroid=({c[0]:.2f},{c[1]:.2f},{c[2]:.3f})"
                  f"  footprint={xext:.2f}x{yext:.2f}m"
                  f"  z%[1,50,99]=[{pct[0]:.3f} {pct[2]:.3f} {pct[4]:.3f}]"
                  f"  frac_below={below:.2f}")

    cloth.update()
    c = cloth.centroid_pos_w[:1]
    os.makedirs(args_cli.out, exist_ok=True)

    # Probe: world AABB of the RENDER mesh points vs the particle-view positions.
    from pxr import UsdGeom, Sdf
    mesh_path = "/World/envs/env_0/Shirt/World/mesh/Mesh"
    mp = env.sim.stage.GetPrimAtPath(Sdf.Path(mesh_path))
    mgeom = UsdGeom.Mesh(mp)
    pts = np.asarray(mgeom.GetPointsAttr().Get(), dtype=np.float64)
    xf = np.array(UsdGeom.Xformable(mp).ComputeLocalToWorldTransform(0)).reshape(4, 4)
    wpts = (np.hstack([pts, np.ones((len(pts), 1))]) @ xf)[:, :3]
    vview = cloth.nodal_pos_w[0].cpu().numpy()
    print(f"[probe] render-mesh world  min={wpts.min(0)}  max={wpts.max(0)}")
    print(f"[probe] particle-view world min={vview.min(0)}  max={vview.max(0)}")
    print(f"[probe] L2W matrix translate row = {xf[3, :3]}")

    def shoot(eye, tgt, name):
        cam.set_world_poses_from_view(
            c + torch.tensor([eye], device=env.device),
            c + torch.tensor([tgt], device=env.device),
        )
        for _ in range(3):
            env.step(zero)
        rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(args_cli.out, name))
        print(f"  saved {name}")

    # Close 3/4 view aimed at the cloth centroid.
    shoot([0.45, -0.50, 0.35], [0.0, 0.0, 0.0], "diag_3q.png")
    # Near-top-down, offset so the gripper does not occlude the shirt.
    shoot([0.25, -0.25, 0.85], [0.0, 0.0, 0.0], "diag_top.png")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

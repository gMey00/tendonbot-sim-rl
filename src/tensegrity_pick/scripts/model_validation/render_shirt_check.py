"""Headless render diagnostic for the PBD shirt — answers "is it visible?".

Builds a minimal scene (ground + dome light + one PBD shirt), points a camera
at it, renders a few frames, and saves an RGB PNG.  Reports the fraction of
non-background pixels so visibility can be judged without a GUI.

Usage::

    conda run -n env_isaaclab python3 scripts/model_validation/render_shirt_check.py \
        --headless --enable_cameras --out /tmp/shirt_render.png
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/tmp/shirt_render.png")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--binding", default="strong", choices=["strong", "weak"],
                    help="Material binding strength for the cloth visual material.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
# Cameras must be enabled for offscreen rendering.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import tensegrity_pick.tasks.manager_based.shared.cloth_object as cloth_mod  # noqa: E402
from tensegrity_pick.tasks.manager_based.shirt_place.shirt_place_scene_cfg import (  # noqa: E402
    SHIRT_CLOTH_CFG,
)

# Optionally force the (old) weak binding to reproduce the invisibility bug.
if args_cli.binding == "weak":
    cloth_mod._VIS_BINDING_STRONG = False


@configclass
class _MiniSceneCfg(InteractiveSceneCfg):
    replicate_physics: bool = False
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=4000.0),
    )
    camera = CameraCfg(
        prim_path="/World/envs/env_0/Camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(pos=(1.0, -1.0, 1.4), rot=(0.0, 0.0, 0.0, 1.0)),
    )


def main() -> None:
    sim_cfg = SimulationCfg(dt=0.01)
    sim_cfg.physx.gpu_collision_stack_size = 2 ** 31
    sim = SimulationContext(sim_cfg)

    scene = InteractiveScene(_MiniSceneCfg(num_envs=1, env_spacing=5.0))

    mock_env = SimpleNamespace(sim=sim, scene=scene)
    cloth_mod.apply_cloth_startup_event(mock_env, None, SHIRT_CLOTH_CFG)

    # Aim the camera at the shirt spawn point.
    cam = scene["camera"]
    sim.reset()
    eye = torch.tensor([[1.0, -1.0, 1.35]], device=sim.device)
    target = torch.tensor([[0.15, -0.04, 0.83]], device=sim.device)
    cam.set_world_poses_from_view(eye, target)

    for _ in range(args_cli.steps):
        sim.step()
        scene.update(0.01)

    rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)

    # Background is the dome (roughly uniform).  Count pixels that differ from
    # the top-row background colour by a margin -> cloth/ground occupancy.
    bg = rgb[0:5].reshape(-1, 3).mean(axis=0)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).sum(axis=-1)
    occupied = float((diff > 30).mean())
    print(f"[render] saved {args_cli.out}  non-background fraction = {occupied:.3f}")
    print(f"[render] rgb mean = {rgb.reshape(-1,3).mean(axis=0)}")

    try:
        from PIL import Image
        Image.fromarray(rgb).save(args_cli.out)
    except Exception as exc:  # pragma: no cover
        np.save(args_cli.out + ".npy", rgb)
        print(f"[render] PIL unavailable ({exc}); saved npy instead")

    simulation_app.close()


if __name__ == "__main__":
    main()

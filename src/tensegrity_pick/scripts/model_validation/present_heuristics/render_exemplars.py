"""Render saved presentation states to annotated PNGs (camera + 3/4 view).

Builds the minimal render scene of ``render_shirt_check.py`` (ground + dome
light + ONE PBD garment), then for each saved particle state (from the study
run scripts' ``outputs/present_heuristics/states/*.pt`` dumps, anchor-relative
fp16 ``[P, 3]``) writes the state through
``ClothObject.write_nodal_state_to_sim`` and re-pins it every frame (zero
velocity) so the pose holds while the renderer converges.  The cloth MUST be
driven through the PhysX tensor path: with the sim playing, Hydra reads
Fabric, so plain USD point writes never reach the camera (first attempt —
blank renders — kept for the record in the study report's iteration log).

Each state produces

  * ``*_cam.png``  — the inspection-camera viewpoint (looking along −Y),
  * ``*_34.png``   — a 3/4 view showing the grasp configuration,

annotated (PIL) with the method + coverage.  States are rotated to their
best-coverage yaw first (the pose ``cov_yaw_max`` scores) unless
``--no_align_yaw``.

Usage (from src/tensegrity_pick, env_isaaclab active)::

    PYTHONUNBUFFERED=1 python scripts/model_validation/present_heuristics/render_exemplars.py \
        --headless --states "outputs/present_heuristics/states/h1/h1_s1_t00658.pt" \
        --out_dir ../../doc/reports/figures/present_heuristics/renders
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render saved presentation states.")
parser.add_argument("--states", type=str, nargs="+", required=True,
                    help=".pt state dumps (globs ok) from the study run scripts")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--label", type=str, default=None,
                    help="override the annotation label (default: method+coverage)")
parser.add_argument("--no_align_yaw", action="store_true",
                    help="skip rotating each state to its best-coverage yaw "
                         "(states are saved at an arbitrary swing phase; the "
                         "aligned pose is what cov_yaw_max scores)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import tensegrity_pick.tasks.manager_based.shared.cloth_object as cloth_mod  # noqa: E402
from tensegrity_pick.tasks.manager_based.shared.cloth_object import ClothObject  # noqa: E402
from tensegrity_pick.tasks.manager_based.shirt_place.shirt_place_scene_cfg import (  # noqa: E402
    SHIRT_CLOTH_CFG,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import load_metrics  # noqa: E402

ANCHOR_W = np.array([0.0, 0.0, 2.1])  # where the study anchor is placed
REPO = "/home/robot/studentische-arbeiten"


@configclass
class _RenderSceneCfg(InteractiveSceneCfg):
    replicate_physics: bool = False
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=4000.0),
    )
    camera = CameraCfg(
        prim_path="/World/envs/env_0/Camera",
        update_period=0.0, height=720, width=720, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=35.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 2.0, 1.4), rot=(0.0, 0.0, 0.0, 1.0)),
    )


def align_to_best_yaw(pos: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Rotate the anchor-relative state about +Z to its best-coverage yaw.

    Returns the rotated points and the best coverage (= the ``cov_yaw_max``
    the study reports; silhouettes repeat with period π).
    """
    cm = load_metrics()
    ref = torch.load(os.path.join(REPO, "doc", "reports", "data",
                                  "present_heuristics_flat_rest.pt"),
                     weights_only=False)["ref_area"]
    best_cov, best_q = -1.0, pos
    for k in range(24):
        a = torch.tensor(np.pi * k / 24)
        q = pos.clone()
        q[:, 0] = torch.cos(a) * pos[:, 0] - torch.sin(a) * pos[:, 1]
        q[:, 1] = torch.sin(a) * pos[:, 0] + torch.cos(a) * pos[:, 1]
        c = float(cm.silhouette_coverage(q, ref)[0])
        if c > best_cov:
            best_cov, best_q = c, q
    return best_q, best_cov


def annotate(png_path: str, label: str) -> None:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(png_path)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((pad, pad), label, font=font)
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
                   fill=(255, 255, 255, 220))
    draw.text((pad, pad), label, fill=(20, 20, 20), font=font)
    img.save(png_path)


def main() -> None:
    from PIL import Image

    sim_cfg = SimulationCfg(dt=0.01)
    sim_cfg.physx.gpu_collision_stack_size = 2 ** 31
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(_RenderSceneCfg(num_envs=1, env_spacing=5.0))
    mock_env = SimpleNamespace(sim=sim, scene=scene)
    cloth_mod.apply_cloth_startup_event(mock_env, None, SHIRT_CLOTH_CFG)
    cam = scene["camera"]
    sim.reset()
    cloth = ClothObject(SHIRT_CLOTH_CFG, num_envs=1, device=sim.device)
    ids = torch.zeros(1, dtype=torch.long, device=sim.device)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    files: list[str] = []
    for pat in args_cli.states:
        files.extend(sorted(glob.glob(pat)) or [pat])

    for f in files:
        d = torch.load(f, weights_only=False)
        rel = d["pos"].float()
        cov = d.get("cov_plane", float("nan"))
        if not args_cli.no_align_yaw:
            rel, cov = align_to_best_yaw(rel)
        pts = rel.to(sim.device) + torch.tensor(ANCHOR_W, dtype=torch.float32,
                                                device=sim.device)
        flat = pts.reshape(1, -1)
        zeros = torch.zeros_like(flat)
        center = pts.mean(dim=0).cpu().numpy()
        label = args_cli.label or (
            f"{d.get('method', '?')} stage {d.get('stage', '?')} — "
            f"coverage {cov:.2f}")
        base = os.path.splitext(os.path.basename(f))[0]
        for view, eye in (
            ("cam", center + np.array([0.0, 2.0, 0.15])),
            ("34", center + np.array([1.4, 1.4, 0.8])),
        ):
            cam.set_world_poses_from_view(
                torch.tensor([eye], dtype=torch.float32, device=sim.device),
                torch.tensor([center], dtype=torch.float32, device=sim.device))
            for _ in range(8):  # hold the pose while the renderer converges
                cloth.write_nodal_state_to_sim(flat, zeros, ids)
                sim.step()
                scene.update(0.01)
            rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            bg = rgb[0:5].reshape(-1, 3).mean(axis=0)
            diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).sum(-1)
            out = os.path.join(args_cli.out_dir, f"{base}_{view}.png")
            Image.fromarray(rgb.astype(np.uint8)).save(out)
            annotate(out, label)
            print(f"[render] {out}  non-bg={float((diff > 30).mean()):.3f}",
                  flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

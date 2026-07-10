"""Generate the cached holding-pose bank for shirt_distribute (UR5e-F140).

Runs the env's in-sim holding-pose sweep ONCE (guided rejection FK, see
``shirt_distribute_env._build_holding_pose_bank``) and saves the resulting
poses so every subsequent process — training AND evaluation, any seed, any
num_envs — samples the SAME initial-state distribution.

Why: without a cache the bank is re-swept per process from the global torch
RNG, so the init distribution itself depends on seed and num_envs.
Measured on 2026-07-05: the same checkpoint scored 0.208 vs 0.454
deterministic success on eval seeds 7/8 purely from that shift.

Output (torch .pt):
    res/Props/Cloth/banks/ur5e_f140_distribute_pose_bank.pt
    {
      "q":     float32 [K, A]   # arm joint positions (action-term order)
      "tip":   float32 [K, 3]   # env-local closed-gripper fingertip (PD equilibrium)
      "drape": float32 [K]      # pose-dependent admissible hang length (m)
      "meta":  {...}
    }

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/generate_distribute_pose_bank.py \
        --headless --num_envs 32 --seed 12345
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate the shirt_distribute holding-pose bank.")
parser.add_argument("--task", type=str, default="Template-Shirt-Distribute-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--out", type=str, default=None,
                    help="output .pt (default: res/Props/Cloth/banks/ur5e_f140_distribute_pose_bank.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import (  # noqa: E402
    PROJ_ASSETS_PATH,
)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    out_path = args_cli.out or os.path.join(
        PROJ_ASSETS_PATH, "Props", "Cloth", "banks", "ur5e_f140_distribute_pose_bank.pt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Force the sweep (ignore an existing cache so regeneration really regenerates).
    cfg.pose_bank_path = None
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped

    q = u._pose_bank_q.cpu()
    tip = u._pose_bank_tip.cpu()
    drape = u._pose_bank_drape.cpu()
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_rev = "unknown"
    meta = {
        "task": args_cli.task,
        "robot": "UR5e-F140 (second-robot pedestal mount)",
        "num_poses": q.shape[0],
        "arm_joints": list(u.cfg.actions.arm_action.joint_names),
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "git": git_rev,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/generate_distribute_pose_bank.py",
    }
    torch.save({"q": q, "tip": tip, "drape": drape, "meta": meta}, out_path)
    print(f"\nRESULT: saved {q.shape[0]} poses to {out_path} "
          f"(tip z [{tip[:, 2].min():.2f}, {tip[:, 2].max():.2f}] m, "
          f"drape [{drape.min():.2f}, {drape.max():.2f}] m)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

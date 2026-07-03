"""Diagnose a trained shirt_pick policy's presented-gate behaviour.

Runs one deterministic episode batch and prints, per env: the minimum and
final-second mean of (a) grasp-point distance to the presentation pose and
(b) EE speed — to see WHICH presented condition (dist < 0.15, speed < 0.2,
grasp held) fails when `present_rate` is 0 in deterministic eval.

Usage::

    cd src/tensegrity_pick
    PYTHONUNBUFFERED=1 python scripts/model_validation/diag_shirt_pick_policy.py \
        --headless --num_envs 8 --checkpoint <path/to/agent.pt>
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="shirt_pick policy presented-gate diagnosis.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (  # noqa: E402
    PRESENTATION_POS,
)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    u = env.unwrapped
    target = u.scene.env_origins + torch.tensor(
        PRESENTATION_POS, device=u.device, dtype=torch.float32).unsqueeze(0)
    obs, _ = env.reset()

    T = 420  # one full episode
    d_log, v_log, g_log, c_log = [], [], [], []
    for _ in range(T):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
        robot = u.scene["robot"]
        d_log.append((u.shirt_grasp_point_w - target).norm(dim=-1).clone())
        v_log.append(robot.data.body_lin_vel_w[:, u._ee_body_idx, :].norm(dim=-1).clone())
        c_log.append(u.cloth.centroid_vel_w.norm(dim=-1).clone())
        g_log.append(u.grasp_active.clone())

    d = torch.stack(d_log)      # [T, N]
    v = torch.stack(v_log)
    c = torch.stack(c_log)
    g = torch.stack(g_log)
    last = slice(-120, None)    # final 2 s
    print("\nper-env summary (min over episode / mean over final 2 s):")
    for i in range(u.num_envs):
        p_now = (d[:, i] < 0.15) & (v[:, i] < 0.20) & g[:, i]
        # longest consecutive presented streak
        streak = best = 0
        for x in p_now.tolist():
            streak = streak + 1 if x else 0
            best = max(best, streak)
        print(f"env {i}: min_d={d[:, i].min():.3f}  d_last2s={d[last, i].mean():.3f}  "
              f"v_last2s={v[last, i].mean():.3f}  cvel_last2s={c[last, i].mean():.3f}  "
              f"grasp_frac={g[:, i].float().mean():.2f}  "
              f"presented_steps={int(p_now.sum())}  longest_streak={best} (need 30)")
    print(f"\nfractions over all steps: dist<0.15: {(d < 0.15).float().mean():.3f}   "
          f"ee_speed<0.20: {(v < 0.20).float().mean():.3f}   "
          f"cloth_cvel<0.20: {(c < 0.20).float().mean():.3f}   "
          f"cloth_cvel<0.30: {(c < 0.30).float().mean():.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

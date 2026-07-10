"""Deterministic multi-checkpoint sweep eval for shirt_present (one Isaac boot).

Boots the env ONCE, then for every (checkpoint, eval-seed) pair reloads the
agent weights, re-seeds, and rolls out ``--num_episodes`` deterministic (mean-
action) episodes — accumulating the env-logged latched metrics. Far cheaper
than one Isaac boot per checkpoint when selecting among late checkpoints
(training reward does NOT predict deterministic quality; Phase-1 saw late-run
collapse — so eval several).

Usage (from the worktree, PYTHONPATH-pinned)::

    python scripts/skrl/evaluate_shirt_present_sweep.py --headless --num_envs 32 \
        --num_episodes 96 --eval_seeds 7 11 \
        --checkpoints <run>/checkpoints/agent_120000.pt <run>/checkpoints/best_agent.pt

Prints one RESULT row per (checkpoint, eval-seed) and a SELECTED line (highest
present_rate averaged over eval seeds, at coverage >= --cov_gate).
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="shirt_present multi-checkpoint deterministic eval.")
parser.add_argument("--task", type=str, default="Template-Shirt-Present-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=96)
parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
parser.add_argument("--eval_seeds", type=int, nargs="+", default=[7, 11])
parser.add_argument("--cov_gate", type=float, default=0.65)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import skrl  # noqa: F401, E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402


def run_one(env, runner, u, n_ep: int) -> dict:
    obs, _ = env.reset()
    acc = {"present": 0.0, "grasp": 0.0, "drop": 0.0, "cov": 0.0, "stretch": 0.0}
    episodes = 0
    while simulation_app.is_running() and episodes < n_ep:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
        done = int((terminated | truncated).sum())
        if done > 0:
            log = u.extras.get("log", {})
            acc["present"] += float(log.get("Metrics/present_rate", 0.0)) * done
            acc["grasp"] += float(log.get("Metrics/grasp_rate", 0.0)) * done
            acc["drop"] += float(log.get("Metrics/drop_rate", 0.0)) * done
            acc["cov"] += float(log.get("Metrics/final_coverage", 0.0)) * done
            acc["stretch"] += float(log.get("Metrics/final_stretch_ratio", 0.0)) * done
            episodes += done
    e = max(episodes, 1)
    return {k: v / e for k, v in acc.items()} | {"episodes": episodes}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    u = env.unwrapped
    runner = Runner(env, experiment_cfg)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    rows = []
    for ckpt in args_cli.checkpoints:
        path = os.path.abspath(ckpt)
        name = os.path.basename(ckpt)
        runner.agent.load(path)
        runner.agent.enable_training_mode(False, apply_to_models=True)
        for seed in args_cli.eval_seeds:
            torch.manual_seed(seed)
            u.cfg.seed = seed
            r = run_one(env, runner, u, args_cli.num_episodes)
            rows.append((name, seed, r))
            print(f"RESULT {name} seed{seed} ({r['episodes']} ep): "
                  f"present={r['present']:.3f} grasp={r['grasp']:.3f} "
                  f"drop={r['drop']:.3f} coverage={r['cov']:.3f} stretch={r['stretch']:.3f}",
                  flush=True)

    # aggregate present_rate across eval seeds per checkpoint (require cov gate)
    agg: dict[str, list] = {}
    for name, seed, r in rows:
        agg.setdefault(name, []).append(r)
    print("\n=== SUMMARY (mean over eval seeds) ===")
    best = (None, -1.0)
    for name, rs in agg.items():
        mp = sum(x["present"] for x in rs) / len(rs)
        mc = sum(x["cov"] for x in rs) / len(rs)
        mg = sum(x["grasp"] for x in rs) / len(rs)
        md = sum(x["drop"] for x in rs) / len(rs)
        print(f"  {name}: present={mp:.3f} grasp={mg:.3f} drop={md:.3f} coverage={mc:.3f}")
        if mp > best[1]:
            best = (name, mp)
    print(f"\nSELECTED (highest mean present_rate): {best[0]} = {best[1]:.3f} "
          f"(cov gate {args_cli.cov_gate})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

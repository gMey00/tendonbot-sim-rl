"""Diagnose a trained shirt_present policy's presented-gate behaviour.

Runs one deterministic episode batch and prints, per env: the per-gate
fractions of the presented predicate (both grasps, stretch band, coverage
threshold, cloth-centroid speed) plus tip→lowest-point reach stats — to see
WHICH gate fails when `present_rate` stalls (shirt_pick lesson: diagnose the
gate fractions BEFORE touching reward weights blindly).

Usage::

    cd src/tensegrity_pick
    PYTHONUNBUFFERED=1 python scripts/model_validation/diag_shirt_present_policy.py \
        --headless --num_envs 8 --checkpoint <path/to/agent.pt>
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="shirt_present policy presented-gate diagnosis.")
parser.add_argument("--task", type=str, default="Template-Shirt-Present-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--steps", type=int, default=480, help="one full episode at 8 s / 60 Hz")
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
from tensegrity_pick.tasks.manager_based.shirt_present.shirt_present_env import (  # noqa: E402
    COVERAGE_THRESHOLD,
    PRESENT_VEL_THRESHOLD,
    PRESENT_WINDOW,
    PRESENT_WINDOW_FRAC,
    STRETCH_BAND,
)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    torch.manual_seed(args_cli.seed)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    runner.agent.load(os.path.abspath(args_cli.checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    u = env.unwrapped
    obs, _ = env.reset()

    logs = {k: [] for k in ("d", "g0", "g1", "ratio", "cov", "cvel")}
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
        logs["d"].append((u._finger_tip_pos() - u.shirt_lowest_point_w).norm(dim=-1).clone())
        logs["g0"].append(u.grasp_active.clone())
        logs["g1"].append(u.holder_attached.clone())
        logs["ratio"].append(u.stretch_ratio_norm.clone())
        logs["cov"].append(u.coverage.clone())
        logs["cvel"].append(u.cloth.centroid_vel_w.norm(dim=-1).clone())

    d = torch.stack(logs["d"])          # [T, N]
    g0 = torch.stack(logs["g0"])
    g1 = torch.stack(logs["g1"])
    ratio = torch.stack(logs["ratio"])
    cov = torch.stack(logs["cov"])
    cvel = torch.stack(logs["cvel"])
    stretch_ok = (ratio >= STRETCH_BAND[0]) & (ratio <= STRETCH_BAND[1])
    cov_ok = cov >= COVERAGE_THRESHOLD
    still_ok = cvel < PRESENT_VEL_THRESHOLD
    p_now = g0 & g1 & stretch_ok & cov_ok & still_ok
    last = slice(-120, None)            # final 2 s

    print(f"\ngates: stretch ∈ [{STRETCH_BAND[0]}, {STRETCH_BAND[1]}], "
          f"coverage ≥ {COVERAGE_THRESHOLD}, cloth speed < {PRESENT_VEL_THRESHOLD}, "
          f"window {PRESENT_WINDOW} @ {PRESENT_WINDOW_FRAC}")
    print("per-env summary (fractions over the episode / final-2 s values):")
    for i in range(u.num_envs):
        # windowed latch check on the recorded trace
        pn = p_now[:, i].float()
        win = torch.nn.functional.avg_pool1d(
            pn.view(1, 1, -1), PRESENT_WINDOW, stride=1).squeeze()
        latched = bool((win >= PRESENT_WINDOW_FRAC).any())
        print(f"env {i}: min_reach={d[:, i].min():.3f}  grasp_frac={g0[:, i].float().mean():.2f}  "
              f"holder_frac={g1[:, i].float().mean():.2f}  "
              f"ratio_last2s={ratio[last, i].mean():.3f}  cov_last2s={cov[last, i].mean():.3f}  "
              f"cvel_last2s={cvel[last, i].mean():.3f}  "
              f"presented_steps={int(p_now[:, i].sum())}  latched={latched}")

    grasped_steps = g0 & g1
    n_gs = grasped_steps.float().sum().clamp(min=1.0)
    print("\ngate fractions over all steps (and conditioned on both-grasped):")
    print(f"  both grasps:      {grasped_steps.float().mean():.3f}")
    print(f"  stretch in band:  {stretch_ok.float().mean():.3f}   "
          f"| grasped: {(stretch_ok & grasped_steps).float().sum() / n_gs:.3f}")
    print(f"  coverage ≥ thr:   {cov_ok.float().mean():.3f}   "
          f"| grasped: {(cov_ok & grasped_steps).float().sum() / n_gs:.3f}")
    print(f"  cloth still:      {still_ok.float().mean():.3f}   "
          f"| grasped: {(still_ok & grasped_steps).float().sum() / n_gs:.3f}")
    print(f"  ALL (presented):  {p_now.float().mean():.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

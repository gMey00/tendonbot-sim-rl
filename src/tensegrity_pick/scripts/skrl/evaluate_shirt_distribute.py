"""Deterministic evaluation of shirt_distribute checkpoints (mean actions).

Checkpoint selection follows the shirt_place rule: training curves do NOT
predict deterministic quality — evaluate with deterministic (mean) actions
and select on the task metrics.

Counts COMPLETED episodes with an exact per-bin breakdown: the env stashes
each finished batch's (target bin, success, released) in
``last_finished_batch`` at reset, so the script accumulates per-episode
outcomes rather than batch means.

Prints per-bin and overall distribute_success_rate, release_rate and
mean |action| (saturated-noise indicator, shirt_place lesson) and a final
one-line RESULT.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/skrl/evaluate_shirt_distribute.py --headless \
        --task Template-Shirt-Distribute-UR5e-F140-v0 --num_envs 32 --seed 7 \
        --num_episodes 96 \
        --checkpoint logs/skrl/shirt_distribute/<run>/checkpoints/agent_20000.pt

    # stochastic actions instead of the mean (determinism-gap check):
    PLAY_STOCH=1 python scripts/skrl/evaluate_shirt_distribute.py ...
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic shirt_distribute evaluation.")
parser.add_argument("--task", type=str, default="Template-Shirt-Distribute-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=96)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import skrl  # noqa: F401, E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Always evaluate on the UNIFORM goal distribution (deployment) even for the
    # B5 variant that trains with difficulty-proportional goal sampling — per-bin
    # rates should reflect uniform commands, not the training over-sampling.
    if hasattr(env_cfg, "adaptive_goal_sampling"):
        env_cfg.adaptive_goal_sampling = False
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    # Explicit global torch seed: the pose/hang-bank sampling in the env reset
    # uses it, and cfg.seed alone gives identical rollouts (shirt_pick lesson).
    torch.manual_seed(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    # PerGoalRunner builds the multi-head critic + per-goal PPO for the per-goal
    # config, and falls back to stock Runner behaviour for a stock PPO yaml.
    from tensegrity_pick.tasks.manager_based.shirt_distribute.learning import PerGoalRunner
    runner = PerGoalRunner(env, experiment_cfg)
    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    u = env.unwrapped
    obs, _ = env.reset()
    # The initial reset stashes a bogus all-zero batch — discard it so only
    # genuinely completed episodes are counted.
    u.last_finished_batch = None
    episodes = 0
    n_bin = torch.zeros(3, dtype=torch.long)
    s_bin = torch.zeros(3)
    released_sum = 0.0
    act_abs_sum, steps = 0.0, 0
    stochastic = bool(os.environ.get("PLAY_STOCH"))

    while simulation_app.is_running() and episodes < args_cli.num_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
            actions = outputs[0] if stochastic else outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
        steps += 1
        act_abs_sum += float(actions.abs().mean())
        batch = getattr(u, "last_finished_batch", None)
        if int((terminated | truncated).sum()) > 0 and batch is not None:
            bins = batch["bin"].cpu()
            succ = batch["success"].cpu()
            released_sum += float(batch["released"].sum())
            for b in range(3):
                m = bins == b
                n_bin[b] += int(m.sum())
                s_bin[b] += float(succ[m].sum())
            episodes = int(n_bin.sum())
            u.last_finished_batch = None
            per_bin = " ".join(
                f"bin{b} {s_bin[b] / max(int(n_bin[b]), 1):.2f}({int(n_bin[b])})"
                for b in range(3)
            )
            print(f"[eval] episodes {episodes}/{args_cli.num_episodes}: "
                  f"success {s_bin.sum() / max(episodes, 1):.3f} | {per_bin}", flush=True)

    mode = "stochastic" if stochastic else "deterministic"
    per_bin = " ".join(
        f"bin{b}={s_bin[b] / max(int(n_bin[b]), 1):.3f}(n={int(n_bin[b])})" for b in range(3)
    )
    print(f"\nRESULT ({mode}, seed {args_cli.seed}, {episodes} episodes): "
          f"distribute_success_rate={s_bin.sum() / max(episodes, 1):.3f} "
          f"{per_bin} "
          f"release_rate={released_sum / max(episodes, 1):.3f} "
          f"mean|act|={act_abs_sum / max(steps, 1):.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Deterministic evaluation of shirt_pick checkpoints (mean actions).

Checkpoint selection follows the shirt_place rule: training curves do NOT
predict deterministic quality — evaluate each candidate checkpoint with
deterministic (mean) actions and select on the task metrics.

Counts COMPLETED episodes: on every env reset the env logs the latched
metrics (present/grasp/drop rate) over exactly the envs that just finished;
this script accumulates those batch means weighted by batch size until
``--num_episodes`` episodes are in.

Prints: present_rate (stable hold ≥0.5 s at the pose), grasp_rate,
drop_rate, mean |action| (saturated-noise indicator, shirt_place lesson) and
a final one-line RESULT.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/skrl/evaluate_shirt_pick.py --headless \
        --task Template-Shirt-Pick-Tensegrity-v0 --num_envs 32 --seed 7 \
        --num_episodes 96 \
        --checkpoint logs/skrl/shirt_pick/<run>/checkpoints/agent_20000.pt

    # stochastic actions instead of the mean (for the determinism-gap check):
    PLAY_STOCH=1 python scripts/skrl/evaluate_shirt_pick.py ...
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic shirt_pick checkpoint evaluation.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
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
from skrl.utils.runner.torch import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    # Seed the global torch RNG explicitly: the crumpled-bank sampling /
    # augmentation in the env reset uses it, and cfg.seed alone was observed
    # NOT to differentiate eval rollouts (two "seeds" gave identical runs).
    torch.manual_seed(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    u = env.unwrapped
    obs, _ = env.reset()
    episodes = 0
    acc = {"present": 0.0, "grasp": 0.0, "drop": 0.0}
    # S2 head metrics (logged only by the -Head- task variants; stay 0 otherwise)
    acc_head = {"high_value": 0.0, "pred_cov": 0.0}
    head_seen = False
    act_abs_sum, steps = 0.0, 0
    stochastic = bool(os.environ.get("PLAY_STOCH"))

    while simulation_app.is_running() and episodes < args_cli.num_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
            actions = outputs[0] if stochastic else outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
        steps += 1
        act_abs_sum += float(actions.abs().mean())
        done = int((terminated | truncated).sum())
        if done > 0:
            log = u.extras.get("log", {})
            acc["present"] += float(log.get("Metrics/present_rate", 0.0)) * done
            acc["grasp"] += float(log.get("Metrics/grasp_rate", 0.0)) * done
            acc["drop"] += float(log.get("Metrics/drop_rate", 0.0)) * done
            if "Metrics/high_value_hold_rate" in log:
                head_seen = True
                acc_head["high_value"] += float(log["Metrics/high_value_hold_rate"]) * done
                acc_head["pred_cov"] += float(log.get("Metrics/pred_coverage", 0.0)) * done
            episodes += done
            head_str = (f" high_value {acc_head['high_value']/episodes:.3f} "
                        f"pred_cov {acc_head['pred_cov']/episodes:.3f}") if head_seen else ""
            print(f"[eval] episodes {episodes}/{args_cli.num_episodes}: "
                  f"present {acc['present']/episodes:.3f} "
                  f"grasp {acc['grasp']/episodes:.3f} "
                  f"drop {acc['drop']/episodes:.3f}{head_str}", flush=True)

    mode = "stochastic" if stochastic else "deterministic"
    head_str = ""
    if head_seen:
        head_str = (f" high_value_hold_rate={acc_head['high_value']/max(episodes,1):.3f}"
                    f" pred_coverage={acc_head['pred_cov']/max(episodes,1):.3f}")
    print(f"\nRESULT ({mode}, seed {args_cli.seed}, {episodes} episodes): "
          f"present_rate={acc['present']/max(episodes,1):.3f} "
          f"grasp_rate={acc['grasp']/max(episodes,1):.3f} "
          f"drop_rate={acc['drop']/max(episodes,1):.3f} "
          f"mean|act|={act_abs_sum/max(steps,1):.3f}{head_str}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

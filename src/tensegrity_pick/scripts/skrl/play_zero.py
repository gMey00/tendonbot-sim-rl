# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-agent playback: step a task with ZERO actions to visually verify the
scene SETUP (initial poses, cloth hang anchor, fixtures) without any policy.

No checkpoint is loaded — the arm holds its reset pose while the sim runs, so
you can confirm the shirt's first-grasp/holder anchor and the tensegrity
holder-robot fixture position are where they should be.

Example (workstation, env_isaaclab conda env, from src/tensegrity_pick):
    PYTHONPATH="$PWD/source/tensegrity_pick:$PYTHONPATH" \
    python scripts/skrl/play_zero.py \
        --task Template-Shirt-Present-UR5e-F140-Play-v0 --num_envs 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-agent scene-setup playback.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time, if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import tensegrity_pick.tasks  # noqa: F401

# The task has no skrl agent dependency for a zero-action run, but hydra still
# needs an agent entry point to resolve; PPO's is present for every task here.
agent_cfg_entry_point = "skrl_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, experiment_cfg: dict):
    """Play the task with zero actions."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    try:
        dt = env.unwrapped.step_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.action_manager.total_action_dim
    device = env.unwrapped.device
    zero_actions = torch.zeros((num_envs, action_dim), device=device)
    print(f"[INFO] Zero-agent playback: task={args_cli.task}, num_envs={num_envs}, action_dim={action_dim}")

    env.reset()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            env.step(zero_actions)
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

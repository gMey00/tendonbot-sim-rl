# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified cube-place evaluation harness.

Scores ONE agent under ONE env seed on the cube-place Play env over a
fixed per-env episode budget and writes the aggregated metrics to JSON.
This is the place-task analogue of ``evaluate_reach.py`` and follows the
same seed-aggregated play-mode protocol that the project thesis
(Section 4.6 / 6.3) explicitly left as future work for the place task.

Supported agents
----------------
* ``zero``       — emits ``torch.zeros(action_space.shape)``     (all variants)
* ``random``     — emits ``2*U(0,1) - 1`` over ``action_space``  (all variants)
* ``checkpoint`` — skrl PPO policy loaded from a training run.   (all variants)

The ``heuristic`` baseline used by the reach harness is intentionally
omitted: a scripted pick-transport-place controller is a separate state
machine (grasp detection, drum approach, timed release) rather than a
single differential-IK step, and is out of scope for this baseline
study — consistent with the thesis omitting the place heuristic.

Success criterion (thesis Section 4.6.3.2, mirrored by the env latches in
``place_env.py``): an episode is successful when the green cube's center
enters the cylindrical drum success-detection volume (radius 0.2735 m,
height 0.30 m) at any point AFTER a valid grasp. Episodes run the full
horizon without early success termination.

Per-episode metrics aggregated (mean + std across all completed episodes):

* ``success_rate``     — fraction of episodes the cube was placed in the drum
* ``success_held``     — placed AND held inside the drum for >= --settle_steps steps
* ``grasp_rate``       — fraction of episodes a stable grasp was established
* ``red_safe_rate``    — fraction of episodes the red distractor stayed on the belt
* ``place_xy_error``   — last-step XY distance from cube to drum axis [m]
* ``place_time_mean``  — mean time-to-first-placement over successful episodes [s]
* ``action_rate_mean`` — mean per-step ||a_t - a_{t-1}||_1 (action smoothness)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Unified cube-place evaluation harness.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Tensegrity-Cube-Place-Tendon-Play-v0",
    help="Gym task id for the cube-place Play env to evaluate on.",
)
parser.add_argument(
    "--agent",
    type=str,
    required=True,
    choices=["zero", "random", "checkpoint"],
    help="Which agent / control policy to evaluate.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to skrl .pt (or run dir) (required iff --agent checkpoint).",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=10,
    help="Episodes per parallel env to aggregate over.",
)
parser.add_argument(
    "--seed",
    type=int,
    required=True,
    help="Env seed for this evaluation run.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Override num_envs (default: use the Play config's value, typically 50).",
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help=(
        "Output JSON path. Default: "
        "logs/skrl/cube_place/eval/<agent>_<variant>_seed<seed>.json"
    ),
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="ML framework for the checkpoint agent (matches play.py).",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO"],
    help="RL algorithm of the checkpoint (only PPO supported here).",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=10,
    help=(
        "Number of consecutive sim steps the cube must stay inside the drum"
        " volume for the 'success_held' metric to count."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# basic CLI validation that doesn't need the sim
if args_cli.agent == "checkpoint" and not args_cli.checkpoint:
    parser.error("--agent checkpoint requires --checkpoint <path>")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import tensegrity_pick.tasks  # noqa: F401
from tensegrity_pick.tasks.manager_based.cube_place.mdp.rewards import _get_world_pos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRUM_KEY = "drum_target"
GREEN_CUBE_KEY = "green_cube"


def _variant_from_task(task: str) -> str:
    """Short tag used in the JSON output and default file name."""
    t = task.lower()
    if "physical-tendon" in t or "physical_tendon" in t:
        return "tensegrity_physical_tendon"
    if "tendon" in t:
        return "tensegrity_tendon"
    return "tensegrity"


def _drum_pos_w(env) -> torch.Tensor:
    """Drum centre in world frame (N, 3)."""
    return _get_world_pos(env.unwrapped.scene[DRUM_KEY], env=env.unwrapped)


def _green_pos_w(env) -> torch.Tensor:
    """Green cube centre in world frame (N, 3)."""
    return env.unwrapped.scene[GREEN_CUBE_KEY].data.root_pos_w


def _place_xy_error(env) -> torch.Tensor:
    """Per-env XY distance from the green cube to the drum axis [m]."""
    g = _green_pos_w(env)
    d = _drum_pos_w(env)
    return torch.sqrt((g[:, 0] - d[:, 0]) ** 2 + (g[:, 1] - d[:, 1]) ** 2)


# ---------------------------------------------------------------------------
# Checkpoint agent (skrl PPO) — mirrors play.py / evaluate_reach.py
# ---------------------------------------------------------------------------

def _build_checkpoint_agent(env, task: str, experiment_cfg: dict, checkpoint_path: str):
    import skrl
    from packaging import version

    SKRL_VERSION = "1.4.3"
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        raise RuntimeError(f"skrl >= {SKRL_VERSION} required; got {skrl.__version__}")

    if args_cli.ml_framework.startswith("torch"):
        from skrl.utils.runner.torch import Runner
    else:
        from skrl.utils.runner.jax import Runner

    # disable any side-effect logging during eval
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(env, experiment_cfg)
    resume_path = os.path.abspath(checkpoint_path)
    if os.path.isdir(resume_path):
        # accept a run-dir; prefer best_agent.pt, else the latest numbered ckpt
        best = Path(resume_path) / "checkpoints" / "best_agent.pt"
        if best.exists():
            resume_path = str(best)
        else:
            ckpts = sorted(Path(resume_path).glob("checkpoints/agent_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints under {resume_path}/checkpoints/")
            resume_path = str(ckpts[-1])
    print(f"[evaluate][checkpoint] loading: {resume_path}")
    runner.agent.load(resume_path)
    # skrl 1.x saved the observation scaler as 'state_preprocessor'; skrl 2.x
    # renamed it to 'observation_preprocessor'.  agent.load() silently skips the
    # old key, leaving a fresh identity scaler so the policy sees raw obs.
    # Manually migrate it (identical to evaluate_reach.py).
    _ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    _obs_pre = runner.agent.checkpoint_modules.get("observation_preprocessor", None)
    if "state_preprocessor" in _ckpt and _obs_pre is not None and hasattr(_obs_pre, "load_state_dict"):
        _obs_pre.load_state_dict(_ckpt["state_preprocessor"])
        print("[evaluate] Migrated 'state_preprocessor' -> 'observation_preprocessor' from checkpoint")
    runner.agent.enable_training_mode(False, apply_to_models=True)
    return runner


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def _resolve_out_path(variant: str) -> Path:
    if args_cli.out:
        return Path(args_cli.out)
    base = Path("logs/skrl/cube_place/eval")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{args_cli.agent}_{variant}_seed{args_cli.seed}.json"


def _run_eval(env, agent_callable) -> dict:
    """Step ``env`` until every parallel env has logged ``--num_episodes`` episodes.

    ``agent_callable(obs)`` must return a (num_envs, action_dim) action tensor.
    Returns the aggregated per-episode metric records.

    Success/grasp/red-safe are read from the ``TensegrityPlaceEnv`` latches
    (``_was_placed`` / ``_was_grasped`` / ``_red_left_belt``).  Because Isaac
    Lab auto-resets done envs *inside* ``env.step()`` (which clears those
    latches), we snapshot every per-env quantity BEFORE each step and, for
    envs that become done, record the snapshot — the same lifecycle-safe
    pattern used by ``evaluate_reach.py`` (see its §8.1 note).
    """
    u = env.unwrapped
    num_envs = u.num_envs
    device = u.device
    n_target = args_cli.num_episodes

    completed = np.zeros(num_envs, dtype=np.int64)
    metrics_records: dict[str, list[float]] = {
        "success_rate": [],
        "success_held": [],
        "grasp_rate": [],
        "red_safe_rate": [],
        "place_xy_error": [],
        "place_time_mean": [],
        "action_rate_mean": [],
    }

    # Deterministic seeding (identical pattern to evaluate_reach.py): drive
    # torch / numpy / python random from --seed so the cube/target stream is
    # reproducible across agents at the same seed.
    import random as _py_random
    _py_random.seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    torch.manual_seed(int(args_cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args_cli.seed))

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    step_dt = float(u.step_dt)
    settle_steps = int(max(1, args_cli.settle_steps))
    print(
        f"[evaluate] seed={args_cli.seed} task={args_cli.task} agent={args_cli.agent} "
        f"num_envs={num_envs} num_episodes_per_env={n_target}"
    )

    # Pre-step snapshots of the latched per-env state.
    prev_placed = u.was_placed.detach().clone()
    prev_grasped = u.was_grasped.detach().clone()
    prev_red_safe = (~u._red_left_belt).detach().clone()
    prev_xy_err = _place_xy_error(env).detach().clone()

    # Local trackers (lifecycle-independent).
    consec_in_drum = torch.zeros(num_envs, dtype=torch.long, device=device)
    held_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_place_step = torch.full((num_envs,), float("inf"), device=device)
    step_counter = torch.zeros(num_envs, device=device)
    prev_actions = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    action_rate_sum = torch.zeros(num_envs, device=device)
    action_rate_count = torch.zeros(num_envs, device=device)

    while (completed < n_target).any() and simulation_app.is_running():
        with torch.inference_mode():
            actions = agent_callable(obs)
        # Step OUTSIDE inference_mode: TensegrityPlaceEnv.step() performs in-place
        # updates to latch tensors (_was_grasped / _was_placed / _task_done /
        # _red_left_belt) that were created at init outside inference mode, which
        # PyTorch forbids from within an inference_mode block. Clone the action to
        # a normal tensor first so the env never receives an inference tensor.
        actions = actions.detach().clone()
        with torch.no_grad():
            step_out = env.step(actions)
        obs, _, terminated, truncated, _ = step_out

        done_t = (
            torch.as_tensor(terminated).view(num_envs)
            | torch.as_tensor(truncated).view(num_envs)
        ).to(device)
        not_done = ~done_t

        # Action-rate (mean |a_t - a_{t-1}|) per non-done step.
        a_t = actions.detach().to(device)
        a_diff = (a_t - prev_actions).abs().mean(dim=-1)
        action_rate_sum[not_done] += a_diff[not_done]
        action_rate_count[not_done] += 1.0
        prev_actions = a_t

        # Current in-drum state (post-step; valid for non-done envs).
        in_drum_now = u._green_in_drum()
        step_counter[not_done] += 1.0

        # First-placement time (local): first step the cube is in-drum while
        # having been grasped, for non-done envs.
        just_placed = (
            (first_place_step == float("inf")) & not_done & in_drum_now & u.was_grasped
        )
        first_place_step[just_placed] = step_counter[just_placed]

        # Settle tracking: consecutive in-drum steps (reset on exit).
        consec_in_drum = torch.where(
            not_done & in_drum_now, consec_in_drum + 1, torch.zeros_like(consec_in_drum)
        )
        held_success |= consec_in_drum >= settle_steps

        done = done_t.cpu().numpy().astype(bool)
        placed_done = prev_placed.cpu().numpy()
        grasped_done = prev_grasped.cpu().numpy()
        red_safe_done = prev_red_safe.cpu().numpy()
        xy_done = prev_xy_err.cpu().numpy()
        held_done = held_success.cpu().numpy()
        first_place_done = first_place_step.cpu().numpy()
        ar_sum_np = action_rate_sum.cpu().numpy()
        ar_cnt_np = action_rate_count.cpu().numpy()

        for i in np.flatnonzero(done):
            if completed[i] >= n_target:
                continue
            metrics_records["success_rate"].append(float(placed_done[i]))
            metrics_records["success_held"].append(float(held_done[i]))
            metrics_records["grasp_rate"].append(float(grasped_done[i]))
            metrics_records["red_safe_rate"].append(float(red_safe_done[i]))
            metrics_records["place_xy_error"].append(float(xy_done[i]))
            if placed_done[i] and np.isfinite(first_place_done[i]):
                metrics_records["place_time_mean"].append(float(first_place_done[i]) * step_dt)
            else:
                metrics_records["place_time_mean"].append(float("nan"))
            if ar_cnt_np[i] > 0:
                metrics_records["action_rate_mean"].append(float(ar_sum_np[i] / ar_cnt_np[i]))
            else:
                metrics_records["action_rate_mean"].append(float("nan"))
            completed[i] += 1
            # reset local trackers for this env (new episode starts next step)
            held_success[i] = False
            consec_in_drum[i] = 0
            first_place_step[i] = float("inf")
            step_counter[i] = 0.0
            action_rate_sum[i] = 0.0
            action_rate_count[i] = 0.0
            prev_actions[i] = 0.0

        # Refresh snapshots for the next step.
        prev_placed = u.was_placed.detach().clone()
        prev_grasped = u.was_grasped.detach().clone()
        prev_red_safe = (~u._red_left_belt).detach().clone()
        prev_xy_err = _place_xy_error(env).detach().clone()

    return metrics_records


def _aggregate(records: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k, v in records.items():
        arr = np.asarray(v, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[k] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite)) if finite.size else float("nan"),
            "n": int(finite.size),
        }
    return out


def _write_result(variant: str, agent_callable_factory, env) -> None:
    """Run the eval loop and dump JSON."""
    agent_callable = agent_callable_factory(env)
    records = _run_eval(env, agent_callable)
    agg = _aggregate(records)

    out_path = _resolve_out_path(variant)
    payload = {
        "agent": args_cli.agent,
        "task": args_cli.task,
        "variant": variant,
        "seed": int(args_cli.seed),
        "num_episodes": int(args_cli.num_episodes),
        "settle_steps": int(args_cli.settle_steps),
        "n_envs": int(env.unwrapped.num_envs),
        "n_episodes_total": agg["success_rate"]["n"],
        "metrics": agg,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[evaluate] wrote {out_path}  ({payload['n_episodes_total']} episodes)")


def _action_zero_factory(env):
    shape = env.action_space.shape
    device = env.unwrapped.device

    def _act(obs=None):
        return torch.zeros(shape, device=device)

    return _act


def _action_random_factory(env):
    shape = env.action_space.shape
    device = env.unwrapped.device

    def _act(obs=None):
        return 2 * torch.rand(shape, device=device) - 1

    return _act


def _main_simple(make_factory) -> None:
    """For zero / random — no skrl agent config needed."""
    variant = _variant_from_task(args_cli.task)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = int(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        _write_result(variant, make_factory, env)
    finally:
        env.close()


def _main_checkpoint() -> None:
    """Hydra-wrapped entry point for --agent checkpoint."""
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"

    @hydra_task_config(args_cli.task, agent_cfg_entry_point)
    def _inner(env_cfg: ManagerBasedRLEnvCfg, experiment_cfg: dict):
        variant = _variant_from_task(args_cli.task)

        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        experiment_cfg["seed"] = int(args_cli.seed)
        env_cfg.seed = int(args_cli.seed)

        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env_wrapped = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
            runner = _build_checkpoint_agent(env_wrapped, args_cli.task, experiment_cfg, args_cli.checkpoint)

            def _factory(_env):
                def _act(obs):
                    outputs = runner.agent.act(obs, _env.state(), timestep=0, timesteps=0)
                    return outputs[-1].get("mean_actions", outputs[0])

                return _act

            # Drive the run via the *wrapped* env so observations stay in sync.
            _write_result(variant, _factory, env_wrapped)
        finally:
            env.close()

    _inner()


def main() -> None:
    if args_cli.agent == "zero":
        _main_simple(_action_zero_factory)
    elif args_cli.agent == "random":
        _main_simple(_action_random_factory)
    elif args_cli.agent == "checkpoint":
        _main_checkpoint()
    else:  # pragma: no cover — argparse already restricts choices
        raise ValueError(f"unknown agent: {args_cli.agent}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

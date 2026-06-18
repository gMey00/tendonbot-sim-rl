# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified reach-task evaluation harness.

Scores ONE agent under ONE env seed on the reach Play env over a fixed
per-env episode budget and writes the aggregated metrics to JSON.

Supported agents
----------------
* ``zero``       — emits ``torch.zeros(action_space.shape)``        (all variants)
* ``random``     — emits ``2*U(0,1) - 1`` over ``action_space``     (all variants)
* ``heuristic``  — damped-least-squares differential IK, position
                   priority, mapped into the
                   ``JointPositionToLimitsAction`` convention.       (PD variant only)
* ``checkpoint`` — skrl policy loaded from a training run.           (all variants)

The four metrics aggregated per episode (from the ``ee_pose``
:class:`FKSampledPoseCommand` term):

* ``success_rate``       — fraction of envs that reached target before truncation
* ``position_error``     — last-step L2 distance to target [m]
* ``orientation_error``  — last-step quaternion error norm
* ``reach_time_mean``    — mean time-to-first-reach over reached envs [s]

The script writes mean + std across all completed episodes to a JSON file.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Unified reach evaluation harness.")
parser.add_argument(
    "--task",
    type=str,
    default="Template-Reach-Tensegrity-Play-v0",
    help="Gym task id for the reach Play env to evaluate on.",
)
parser.add_argument(
    "--agent",
    type=str,
    required=True,
    choices=["zero", "random", "heuristic", "checkpoint"],
    help="Which agent / control policy to evaluate.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to skrl .pt (required iff --agent checkpoint).",
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
        "logs/skrl/reach/eval/<agent>_<variant>_seed<seed>.json"
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
        "Number of consecutive sim steps the EE position must stay within"
        " the success threshold for the 'success_held' metric to count."
        " At 60 Hz, 10 steps \u2248 167 ms."
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

import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import tensegrity_pick.tasks  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _variant_from_task(task: str) -> str:
    """Short tag used in the JSON output and default file name."""
    t = task.lower()
    if "physical-tendon" in t or "phystendon" in t:
        return "tensegrity_physical_tendon"
    if "tendon" in t:
        return "tensegrity_tendon"
    return "tensegrity"


def _ee_pose_metrics(env) -> dict[str, torch.Tensor]:
    """Per-env tensors from the FK-sampled pose command term."""
    return env.unwrapped.command_manager.get_term("ee_pose").metrics


def _ee_pose_command_b(env) -> torch.Tensor:
    """Current ee_pose target in the robot root frame (N, 7)."""
    return env.unwrapped.command_manager.get_command("ee_pose")


# ---------------------------------------------------------------------------
# Heuristic DLS-IK agent (PD variant only)
# ---------------------------------------------------------------------------

class HeuristicIKAgent:
    """Damped-least-squares differential IK, position-priority.

    Emits actions in the ``JointPositionToLimitsAction`` convention
    (i.e. ``a = scale_transform(joint_pos_des, lower, upper)`` in
    [-1, 1]).  Only the PD reach variant is supported, because the
    tendon variants act in cable-tension space.
    """

    def __init__(self, env, task: str):
        if "tendon" in task.lower():
            raise SystemExit(
                f"[evaluate] --agent heuristic only supports the PD reach variant; "
                f"got task={task!r}. The tendon variants act in cable-tension "
                f"space and require a separate torque->tension controller (out of scope)."
            )

        u = env.unwrapped
        self.device = u.device
        self.num_envs = u.num_envs

        # Resolve robot articulation + the EE body and controlled joints from
        # the action term (so we always agree with the env config).
        action_term = u.action_manager.get_term("arm_action")
        self._action_term = action_term
        self.robot = action_term._asset

        # Joints driven by the action term (preserve order, NOT the slice form).
        self.joint_ids, _ = self.robot.find_joints(action_term.cfg.joint_names)

        # EE body name comes from the ee_pose command term so we follow the env.
        ee_body_name = u.command_manager.get_term("ee_pose").cfg.body_name
        body_ids, _ = self.robot.find_bodies(ee_body_name)
        self.body_idx = body_ids[0]

        # Map our body / joint indices to the Jacobian view.
        # (fixed-base => drop one body row; floating-base => offset joint cols by 6)
        if self.robot.is_fixed_base:
            self._jacobi_body_idx = self.body_idx - 1
            self._jacobi_joint_ids = self.joint_ids
        else:
            self._jacobi_body_idx = self.body_idx
            self._jacobi_joint_ids = [i + 6 for i in self.joint_ids]

        # Soft joint limits for the action-space inversion.
        soft_lim = self.robot.data.soft_joint_pos_limits  # (N, J, 2)
        self.lower = soft_lim[:, self.joint_ids, 0]
        self.upper = soft_lim[:, self.joint_ids, 1]

        # Build the DLS IK controller (position priority).
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
        )
        self.ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # Sanity / verification gate buffers (printed on first step).
        self._zero_error_check_done = False

    def reset(self, env_ids=None):
        self.ik.reset(env_ids)

    def act(self, env) -> torch.Tensor:
        cmd_b = _ee_pose_command_b(env)
        target_pos_b = cmd_b[:, :3]
        target_quat_b = cmd_b[:, 3:7]

        # convert root-frame target -> world frame
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            root_pos_w, root_quat_w, target_pos_b, target_quat_b
        )

        # 2) Current EE state + Jacobian
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]
        joint_pos_cur = self.robot.data.joint_pos[:, self.joint_ids]

        # 3) Solve for desired joint positions (position priority).
        self.ik.set_command(target_pos_w, ee_pos=ee_pos_w, ee_quat=ee_quat_w)
        joint_pos_des = self.ik.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos_cur)

        # 4) Verification gate (first call only): zero-error -> action ~ 0
        if not self._zero_error_check_done:
            self._zero_error_check_done = True
            # synthesise: target == current
            self.ik.set_command(ee_pos_w, ee_pos=ee_pos_w, ee_quat=ee_quat_w)
            joint_des_zero = self.ik.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos_cur)
            zero_action = self._joint_pos_to_action(joint_des_zero)
            max_abs = float(zero_action.abs().max().item())
            print(
                f"[evaluate][heuristic] zero-error sanity check: "
                f"|action|_max = {max_abs:.3e}  (expected ~0)"
            )
            # re-issue the real command for this step
            self.ik.set_command(target_pos_w, ee_pos=ee_pos_w, ee_quat=ee_quat_w)
            joint_pos_des = self.ik.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos_cur)

        return self._joint_pos_to_action(joint_pos_des)

    def _joint_pos_to_action(self, joint_pos_des: torch.Tensor) -> torch.Tensor:
        """Invert ``JointPositionToLimitsAction``.

        The forward map (with scale=1 and rescale_to_limits=True) is::

            processed = unscale_transform(clamp(a, -1, 1), lower, upper)

        So the inverse over the unsaturated range is::

            a = scale_transform(joint_pos_des, lower, upper)
        """
        # Clamp into joint limits so the [-1, 1] mapping is well-defined.
        clamped = torch.maximum(torch.minimum(joint_pos_des, self.upper), self.lower)
        a = math_utils.scale_transform(clamped, self.lower, self.upper)
        return a.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Checkpoint agent (skrl PPO) — mirrors play.py
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
        # accept a run-dir; pick the latest checkpoint inside
        ckpts = sorted(Path(resume_path).glob("checkpoints/agent_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints under {resume_path}/checkpoints/")
        resume_path = str(ckpts[-1])
    print(f"[evaluate][checkpoint] loading: {resume_path}")
    runner.agent.load(resume_path)
    # skrl 1.x saved the observation scaler as 'state_preprocessor'; skrl 2.x
    # renamed it to 'observation_preprocessor'.  agent.load() iterates the
    # checkpoint keys against agent.checkpoint_modules and silently skips the
    # old 'state_preprocessor' key, leaving a fresh identity scaler so the
    # policy sees raw un-normalized observations.  Manually migrate it.
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

# Pre-decide whether we need Hydra (only the checkpoint agent does, since it
# needs skrl_cfg_entry_point parsed).  For zero / random / heuristic we use
# parse_env_cfg directly to avoid the Hydra mode constraints.

def _resolve_out_path(variant: str) -> Path:
    if args_cli.out:
        return Path(args_cli.out)
    base = Path("logs/skrl/reach/eval")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{args_cli.agent}_{variant}_seed{args_cli.seed}.json"


def _run_eval(env, agent_callable, target_first_per_env: list[float | None]) -> dict:
    """Step ``env`` until every parallel env has logged ``--num_episodes`` episodes.

    ``agent_callable()`` must return a (num_envs, action_dim) action tensor.
    Returns the aggregated metrics dict.
    """
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    n_target = args_cli.num_episodes

    completed = np.zeros(num_envs, dtype=np.int64)
    metrics_records: dict[str, list[float]] = {
        "success_rate": [],
        "success_held": [],
        "position_error": [],
        "orientation_error": [],
        "reach_time_mean": [],
        "action_rate_mean": [],
    }

    # Deterministic seeding: torch / numpy / python random are all driven
    # from --seed so the FK target stream is identical across agents that
    # use the same seed (zero, random, heuristic AND checkpoint).
    import random as _py_random
    _py_random.seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    torch.manual_seed(int(args_cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args_cli.seed))

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    # log first target per env (for pairing verification across agents)
    cmd = _ee_pose_command_b(env).detach().cpu().numpy()
    for i in range(num_envs):
        target_first_per_env[i] = float(cmd[i, 0])  # x of first target
    print(
        f"[evaluate] seed={args_cli.seed} task={args_cli.task} agent={args_cli.agent} "
        f"num_envs={num_envs} num_episodes_per_env={n_target} "
        f"first_target_x[0]={target_first_per_env[0]:.4f}"
    )

    metric_term = _ee_pose_metrics(env)
    command_term = env.unwrapped.command_manager.get_term("ee_pose")
    threshold = float(command_term.cfg.success_threshold)
    step_dt = float(env.unwrapped.step_dt)
    settle_steps = int(max(1, args_cli.settle_steps))

    # Pre-step snapshot of pos/ori error so we can read them at episode end
    # (Isaac Lab auto-resets done envs INSIDE env.step(), which overwrites
    # position_error/orientation_error with the new episode's first-frame
    # distance.)  We also track success / reach time / settle / action-rate
    # LOCALLY — even after the FKSampledPoseCommand.reset() bug is fixed,
    # tracking here keeps the harness self-contained and independent of the
    # env's metric tensor lifecycle.
    prev_pe = metric_term["position_error"].detach().clone()
    prev_oe = metric_term["orientation_error"].detach().clone()
    cum_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
    held_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    consec_in_thresh = torch.zeros(num_envs, dtype=torch.long, device=device)
    first_reach_step_local = torch.full((num_envs,), float("inf"), device=device)
    step_counter_local = torch.zeros(num_envs, device=device)
    prev_actions = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    action_rate_sum = torch.zeros(num_envs, device=device)
    action_rate_count = torch.zeros(num_envs, device=device)

    while (completed < n_target).any() and simulation_app.is_running():
        with torch.inference_mode():
            actions = agent_callable(obs)
            step_out = env.step(actions)
            if len(step_out) == 5:
                obs, _, terminated, truncated, _ = step_out
            else:
                obs, _, terminated, truncated, _ = step_out

        # post-step pos_err: for non-done envs this is the new state; for
        # done envs it has already been overwritten by the auto-reset.
        pe_now = metric_term["position_error"].detach()

        done_t = (
            torch.as_tensor(terminated).view(num_envs)
            | torch.as_tensor(truncated).view(num_envs)
        ).to(device)
        not_done = ~done_t

        # Action-rate (||a_t - a_{t-1}||) per non-done step.
        a_t = actions.detach().to(device)
        a_diff = (a_t - prev_actions).abs().mean(dim=-1)
        action_rate_sum[not_done] += a_diff[not_done]
        action_rate_count[not_done] += 1.0
        prev_actions = a_t

        # Update local success tracking — only for non-done envs, since for
        # done envs pe_now reflects the post-reset state.
        in_thresh = pe_now < threshold
        step_counter_local[not_done] += 1.0
        just_reached = (~cum_reached) & not_done & in_thresh
        first_reach_step_local[just_reached] = step_counter_local[just_reached]
        cum_reached |= just_reached

        # Settle tracking: count consecutive in-threshold steps. Reset on
        # any out-of-threshold step (still for non-done envs only).
        consec_in_thresh = torch.where(
            not_done & in_thresh, consec_in_thresh + 1, torch.zeros_like(consec_in_thresh)
        )
        held_success |= consec_in_thresh >= settle_steps

        done = done_t.cpu().numpy().astype(bool)
        pe_done = prev_pe.cpu().numpy()
        oe_done = prev_oe.cpu().numpy()
        cum_done = cum_reached.cpu().numpy()
        held_done = held_success.cpu().numpy()
        first_reach_done = first_reach_step_local.cpu().numpy()
        ar_sum_np = action_rate_sum.cpu().numpy()
        ar_cnt_np = action_rate_count.cpu().numpy()

        for i in np.flatnonzero(done):
            if completed[i] >= n_target:
                continue
            metrics_records["success_rate"].append(float(cum_done[i]))
            metrics_records["success_held"].append(float(held_done[i]))
            metrics_records["position_error"].append(float(pe_done[i]))
            metrics_records["orientation_error"].append(float(oe_done[i]))
            if cum_done[i]:
                metrics_records["reach_time_mean"].append(
                    float(first_reach_done[i]) * step_dt
                )
            else:
                metrics_records["reach_time_mean"].append(float("nan"))
            if ar_cnt_np[i] > 0:
                metrics_records["action_rate_mean"].append(
                    float(ar_sum_np[i] / ar_cnt_np[i])
                )
            else:
                metrics_records["action_rate_mean"].append(float("nan"))
            completed[i] += 1
            # reset local tracking for this env (new episode starts next step)
            cum_reached[i] = False
            held_success[i] = False
            consec_in_thresh[i] = 0
            first_reach_step_local[i] = float("inf")
            step_counter_local[i] = 0.0
            action_rate_sum[i] = 0.0
            action_rate_count[i] = 0.0
            prev_actions[i] = 0.0

        prev_pe = metric_term["position_error"].detach().clone()
        prev_oe = metric_term["orientation_error"].detach().clone()

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


# ---------------------------------------------------------------------------
# Two entry points: one for checkpoint (Hydra-wrapped), one for the others.
# ---------------------------------------------------------------------------

def _write_result(variant: str, agent_callable_factory, env, n_envs_hint: int) -> None:
    """Run the eval loop and dump JSON."""
    first_targets: list[float | None] = [None] * env.unwrapped.num_envs

    # Build the per-step action callable.
    agent_callable = agent_callable_factory(env)

    records = _run_eval(env, agent_callable, first_targets)
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
        "n_episodes_total": sum(agg[k]["n"] for k in ["success_rate"]),
        "first_target_x": first_targets,
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


def _action_heuristic_factory(env):
    agent = HeuristicIKAgent(env, args_cli.task)

    def _act(obs=None):
        return agent.act(env)

    return _act


def _main_simple(make_factory) -> None:
    """For zero / random / heuristic — no skrl agent config needed."""
    variant = _variant_from_task(args_cli.task)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = int(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        _write_result(variant, make_factory, env, env_cfg.scene.num_envs)
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
            _write_result(variant, _factory, env_wrapped, env_cfg.scene.num_envs)
        finally:
            env.close()

    _inner()


def main() -> None:
    if args_cli.agent == "zero":
        _main_simple(_action_zero_factory)
    elif args_cli.agent == "random":
        _main_simple(_action_random_factory)
    elif args_cli.agent == "heuristic":
        _main_simple(_action_heuristic_factory)
    elif args_cli.agent == "checkpoint":
        _main_checkpoint()
    else:  # pragma: no cover  — argparse already restricts choices
        raise ValueError(f"unknown agent: {args_cli.agent}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

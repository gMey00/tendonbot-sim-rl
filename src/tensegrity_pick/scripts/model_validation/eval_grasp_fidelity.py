"""Ranking-preservation harness: idealized vs gated grasp evaluation (S5/M4).

Evaluates >= 2 checkpoints of ONE task deterministically (mean actions, fixed
seed) under two grasp models:

  (a) idealized  — the deterministic attachment all training uses (gates off);
  (b) gated      — grasp-fidelity gates ON (pre-condition + stochastic
                   misgrasp/slip, GraspFidelityCfg defaults: DRAPER-calibrated
                   16-22 % parallel-jaw misgrasp, hang-weight slip model).

and prints per-checkpoint success, the success drop, and the ranking verdict
with the pre-registered decision rule (research report §2-Q5 / gate G2):

    PROMOTE the gates into the training loop ONLY IF the gated evaluation
    drops success by > 0.15 absolute AND the checkpoint ranking reorders;
    otherwise the idealized training grasp is validated and the gates stay
    evaluation-only.

One Isaac boot per invocation (one task); every (checkpoint, condition) run
re-seeds the global RNG, the env and the gate RNG identically, so runs see
the same reset/bank draws.

Usage::

    cd src/tensegrity_pick
    # env_isaaclab (conda activate, keep PYTHONPATH)
    python scripts/model_validation/eval_grasp_fidelity.py --headless \
        --task Template-Shirt-Pick-Tensegrity-v0 --num_envs 32 --seed 7 \
        --num_episodes 96 \
        --checkpoints logs/skrl/shirt_pick/<run>/checkpoints/agent_12000.pt \
                      logs/skrl/shirt_pick/<run>/checkpoints/agent_20000.pt

    # shirt_present (success key defaults to Metrics/present_rate for both):
    python scripts/model_validation/eval_grasp_fidelity.py --headless \
        --task Template-Shirt-Present-UR5e-F140-v0 ... --checkpoints ...
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Idealized vs gated grasp ranking eval.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                    help=">= 2 checkpoints of the SAME task")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=96)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--success_key", type=str, default="Metrics/present_rate",
                    help="extras-log metric that defines task success")
parser.add_argument("--drop_threshold", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402

if len(args_cli.checkpoints) < 2:
    raise SystemExit("Need >= 2 checkpoints for a ranking verdict.")


def run_eval(gym_env, experiment_cfg, checkpoint: str, gated: bool) -> dict:
    """One deterministic eval run; returns accumulated Metrics/* means."""
    u = gym_env.unwrapped
    gf = u.grasp_fidelity
    gf.precondition_gate = gated
    gf.stochastic_gate = gated
    u._gf_rng.manual_seed(int(gf.seed))
    for k in u.gf_event_counts:
        u.gf_event_counts[k] = 0
    torch.manual_seed(args_cli.seed)

    env = SkrlVecEnvWrapper(gym_env, ml_framework="torch")
    env._seed = args_cli.seed             # identical reset draws across runs
    runner = Runner(env, experiment_cfg)
    runner.agent.load(os.path.abspath(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    episodes, steps = 0, 0
    acc: dict[str, float] = {}
    # The ENTIRE env interaction (reset included) runs inside inference_mode:
    # stepping under inference_mode turns the env's internal buffers into
    # inference tensors, and a later reset OUTSIDE the context may not write
    # them in-place ("Inplace update to inference tensor outside
    # InferenceMode") — hit on the harness's second run.
    with torch.inference_mode():
        obs, _ = env.reset()
        while simulation_app.is_running() and episodes < args_cli.num_episodes:
            outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
            steps += 1
            done = int((terminated | truncated).sum())
            if done > 0:
                log = u.extras.get("log", {})
                for k, v in log.items():
                    if k.startswith("Metrics/"):
                        acc[k] = acc.get(k, 0.0) + float(v) * done
                episodes += done
    out = {k: v / max(episodes, 1) for k, v in acc.items()}
    out["_episodes"] = episodes
    out["_gate_events"] = dict(u.gf_event_counts)
    return out


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    experiment_cfg["seed"] = args_cli.seed
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    gym_env = gym.make(args_cli.task, cfg=env_cfg)

    results: dict[str, dict[bool, dict]] = {}
    for ckpt in args_cli.checkpoints:
        results[ckpt] = {}
        for gated in (False, True):
            label = "gated" if gated else "ideal"
            print(f"\n[eval] {os.path.basename(ckpt)} / {label} ...", flush=True)
            r = run_eval(gym_env, experiment_cfg, ckpt, gated)
            results[ckpt][gated] = r
            ev = r["_gate_events"]
            metrics = " ".join(
                f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(r.items())
                if k.startswith("Metrics/")
            )
            print(f"[eval] {os.path.basename(ckpt)} / {label}: {metrics} "
                  f"({r['_episodes']} eps; gate events {ev})", flush=True)

    # ── Ranking table + verdict ─────────────────────────────────────────
    key = args_cli.success_key
    names = [os.path.basename(c) for c in args_cli.checkpoints]
    succ_i = [results[c][False].get(key, float("nan")) for c in args_cli.checkpoints]
    succ_g = [results[c][True].get(key, float("nan")) for c in args_cli.checkpoints]
    drops = [i - g for i, g in zip(succ_i, succ_g)]
    rank_i = sorted(range(len(names)), key=lambda k: -succ_i[k])
    rank_g = sorted(range(len(names)), key=lambda k: -succ_g[k])
    reorder = rank_i != rank_g
    max_drop = max(drops)

    print("\n" + "=" * 72)
    print(f" GRASP-FIDELITY RANKING STUDY — {args_cli.task}")
    print(f" success = {key}, seed {args_cli.seed}, "
          f"{args_cli.num_episodes} episodes/run")
    print("=" * 72)
    print(f" {'checkpoint':<22} {'ideal':>7} {'gated':>7} {'drop':>7} "
          f"{'misgr':>6} {'slips':>6}")
    for k, c in enumerate(args_cli.checkpoints):
        ev = results[c][True]["_gate_events"]
        print(f" {names[k]:<22} {succ_i[k]:>7.3f} {succ_g[k]:>7.3f} "
              f"{drops[k]:>7.3f} {ev['misgrasps']:>6} {ev['slips']:>6}")
    print("-" * 72)
    print(f" ranking idealized : {' > '.join(names[k] for k in rank_i)}")
    print(f" ranking gated     : {' > '.join(names[k] for k in rank_g)}")
    print(f" reordered: {reorder}   max drop: {max_drop:.3f} "
          f"(threshold {args_cli.drop_threshold})")
    promote = reorder and max_drop > args_cli.drop_threshold
    print("-" * 72)
    print(" DECISION RULE: promote gates into TRAINING only if drop > "
          f"{args_cli.drop_threshold} absolute AND the ranking reorders.")
    print(f" VERDICT: {'PROMOTE gates into training' if promote else 'KEEP gates evaluation-only (idealized training grasp validated)'}")
    print("=" * 72)

    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

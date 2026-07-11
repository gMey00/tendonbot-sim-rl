"""Runtime probe: does skrl 2.1.0 truly separate policy vs critic observations?

Verdict script for pipeline Stage-2 S4 (asymmetric actor-critic).  Static
reading of the installed skrl 2.1.0 says YES (the 2.x observation/state
refactor: ``IsaacLabWrapper.state()`` returns the manager-based env's
``"critic"`` observation group, trainers pass it into ``agent.act`` /
``record_transition``, PPO stores a separate ``states`` memory tensor, and
yaml ``input: STATES`` compiles to ``inputs["states"]``).  This script proves
it AT RUNTIME — don't trust docs — by:

  1. booting a cheap no-cloth task (reach) and injecting a ``critic``
     observation group = policy terms + a 7-wide constant probe term (42.0),
     so critic dim != policy dim and critic content is recognizable;
  2. training a few hundred PPO steps with an asymmetric agent config
     (policy ``input: OBSERVATIONS``, value ``input: STATES``, separate
     models, preprocessors OFF so raw tensors reach the models);
  3. spying on both models' ``act`` calls and checking, on every call:
     the value net consumed a tensor of the CRITIC dim whose last 7 columns
     are exactly 42.0, while the policy consumed the policy dim without them.

PASS  → skrl asymmetric AC works; use the config pattern documented in
        doc/reports/pipeline_infra_tracking.md.
FAIL  → states aliased to observations; use the RSL-RL fallback runner.

Usage::

    cd src/tensegrity_pick
    # env_isaaclab (conda activate, keep PYTHONPATH)
    python scripts/model_validation/check_skrl_asymmetric_ac.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="skrl asymmetric-AC runtime probe.")
parser.add_argument("--task", type=str, default="Template-Reach-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--timesteps", type=int, default=64,
                    help="trainer timesteps (rollouts=16 -> 4 PPO updates at 64)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402

PROBE_WIDTH = 7
PROBE_VALUE = 42.0


def critic_probe_constant(env) -> torch.Tensor:
    """Recognizable privileged-only term: [N, 7] of exactly 42.0."""
    return torch.full((env.num_envs, PROBE_WIDTH), PROBE_VALUE, device=env.device)


def build_agent_cfg(log_dir: str, timesteps: int) -> dict:
    """Minimal asymmetric PPO runner config (preprocessors OFF for raw spying)."""
    return {
        "models": {
            "separate": True,  # asymmetric AC requires separate networks
            "policy": {
                "class": "GaussianMixin",
                "clip_actions": False,
                "clip_log_std": True,
                "min_log_std": -3.0,
                "max_log_std": 0.5,
                "initial_log_std": -0.5,
                "network": [{"name": "net", "input": "OBSERVATIONS",
                             "layers": [64, 64], "activations": "elu"}],
                "output": "ACTIONS",
            },
            "value": {
                "class": "DeterministicMixin",
                "clip_actions": False,
                "network": [{"name": "net", "input": "STATES",
                             "layers": [64, 64], "activations": "elu"}],
                "output": "ONE",
            },
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO",
            "rollouts": 16,
            "learning_epochs": 2,
            "mini_batches": 2,
            "discount_factor": 0.99,
            "learning_rate": 3.0e-04,
            # NOTE: preprocessors deliberately absent -> raw tensors reach the
            # models, so the 42.0 content check below is exact.  In a real
            # config set BOTH observation_preprocessor AND state_preprocessor
            # (legacy 'state_preprocessor'-only yamls get REMAPPED to
            # observation_preprocessor by Runner._check_cfg_compatibility).
            "experiment": {
                "directory": log_dir,
                "experiment_name": "asym_probe",
                "write_interval": 0,
                "checkpoint_interval": 0,
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": timesteps,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)

    # Inject the critic group: policy terms + the constant probe term.
    critic_group = copy.deepcopy(env_cfg.observations.policy)
    if hasattr(critic_group, "enable_corruption"):
        critic_group.enable_corruption = False
    critic_group.probe_constant = ObsTerm(func=critic_probe_constant)
    env_cfg.observations.critic = critic_group

    env = gym.make(args_cli.task, cfg=env_cfg)
    single_obs_space = env.unwrapped.single_observation_space
    print(f"[probe] observation groups: {list(single_obs_space.keys())}")

    wrapped = SkrlVecEnvWrapper(env)
    obs_dim = gym.spaces.flatdim(wrapped.observation_space)
    state_dim = gym.spaces.flatdim(wrapped.state_space)
    print(f"[probe] policy obs dim = {obs_dim}, critic state dim = {state_dim} "
          f"(expected critic = policy + {PROBE_WIDTH})")

    log_dir = os.path.join("logs", "skrl", "asym_probe")
    runner = Runner(wrapped, build_agent_cfg(log_dir, args_cli.timesteps))

    # ── Spy on both models' act() calls ────────────────────────────────
    records: dict[str, list[dict]] = {"policy": [], "value": []}

    def attach_spy(model, name: str) -> None:
        orig_act = model.act

        def spy(inputs, role=""):
            obs = inputs.get("observations")
            states = inputs.get("states")
            rec = {
                "obs_dim": None if obs is None else int(obs.shape[-1]),
                "state_dim": None if states is None else int(states.shape[-1]),
                "state_probe_ok": (
                    states is not None
                    and states.shape[-1] == state_dim
                    and bool((states[..., -PROBE_WIDTH:] == PROBE_VALUE).all())
                ),
                "obs_has_probe": (
                    obs is not None
                    and bool((obs == PROBE_VALUE).any())
                ),
            }
            records[name].append(rec)
            return orig_act(inputs, role=role)

        model.act = spy

    attach_spy(runner.agent.models["policy"], "policy")
    attach_spy(runner.agent.models["value"], "value")

    runner.run()

    # ── Analysis ────────────────────────────────────────────────────────
    value_model = runner.agent.models["value"]
    policy_model = runner.agent.models["policy"]
    first_linear_in = {
        name: next(m for m in model.modules() if isinstance(m, nn.Linear)).in_features
        for name, model in (("policy", policy_model), ("value", value_model))
    }

    n_val = len(records["value"])
    val_probe_ok = all(r["state_probe_ok"] for r in records["value"])
    val_dims = {r["state_dim"] for r in records["value"]}
    pol_clean = not any(r["obs_has_probe"] for r in records["policy"])
    pol_dims = {r["obs_dim"] for r in records["policy"]}

    separated = (
        state_dim == obs_dim + PROBE_WIDTH
        and n_val > 0
        and val_probe_ok
        and val_dims == {state_dim}
        and pol_clean
        and pol_dims == {obs_dim}
        and first_linear_in["value"] == state_dim
        and first_linear_in["policy"] == obs_dim
    )

    print("\n" + "=" * 66)
    print(" SKRL 2.1.0 ASYMMETRIC ACTOR-CRITIC RUNTIME PROBE")
    print("=" * 66)
    print(f" spaces          : policy {obs_dim}  critic {state_dim}")
    print(f" model act calls : policy {len(records['policy'])}  value {n_val}")
    print(f" value inputs    : state dims seen {val_dims}, probe cols == "
          f"{PROBE_VALUE} on all calls: {val_probe_ok}")
    print(f" policy inputs   : obs dims seen {pol_dims}, free of probe value: "
          f"{pol_clean}")
    print(f" first Linear in : policy {first_linear_in['policy']} "
          f"(exp {obs_dim}), value {first_linear_in['value']} (exp {state_dim})")
    print("-" * 66)
    print(f" VERDICT: {'SEPARATED — asymmetric AC works' if separated else 'ALIASED / BROKEN — use RSL-RL fallback'}")
    print("=" * 66)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

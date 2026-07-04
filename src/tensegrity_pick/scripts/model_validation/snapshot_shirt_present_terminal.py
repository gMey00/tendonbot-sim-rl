"""Validate + exercise the Task-2 → Task-3 terminal-state snapshot hook.

Rolls the selected shirt_present checkpoint deterministically to just before
the episode timeout, calls ``ShirtPresentEnv.snapshot_terminal_states`` and
(a) sanity-checks every field (shapes, finiteness, attachment masks,
presented flags), (b) writes the presented-filtered states to a bank ``.pt``
in the shirt_distribute initial-state format (pos/vel per particle +
both grasp states + metadata).

Usage::

    cd src/tensegrity_pick
    PYTHONUNBUFFERED=1 python scripts/model_validation/snapshot_shirt_present_terminal.py \
        --headless --num_envs 32 --rounds 2 \
        --checkpoint logs/skrl/shirt_present/<run>/checkpoints/agent_96000.pt \
        --out /tmp/tshirt_present_terminal_bank.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="shirt_present terminal-state snapshot validation.")
parser.add_argument("--task", type=str, default="Template-Shirt-Present-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--rounds", type=int, default=2, help="episodes to roll per env")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", type=str, default="/tmp/tshirt_present_terminal_bank.pt")
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
    runner.agent.load(args_cli.checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    u = env.unwrapped
    ep_steps = int(u.max_episode_length)
    obs, _ = env.reset()

    banks: list[dict] = []
    for rnd in range(args_cli.rounds):
        # Roll to 2 steps BEFORE timeout so the snapshot sees live episodes.
        for _ in range(ep_steps - 2 if rnd == 0 else ep_steps):
            with torch.inference_mode():
                outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = env.step(actions)
        snap = u.snapshot_terminal_states()

        # ── Field validation ─────────────────────────────────────────
        n, p = u.num_envs, u.cloth.num_particles
        expect = {
            "pos": (n, p, 3), "vel": (n, p, 3),
            "attach_mask_hand": (n, p), "attach_mask_holder": (n, p),
            "grasp_point_local": (n, 3), "anchor_local": (n, 3),
            "stretch_ratio": (n,), "coverage": (n,), "presented": (n,),
        }
        for k, shape in expect.items():
            t = snap[k]
            assert tuple(t.shape) == shape, f"{k}: {tuple(t.shape)} != {shape}"
            assert torch.isfinite(t.float()).all(), f"{k}: non-finite values"
        held = snap["attach_mask_hand"].any(dim=1)
        pinned = snap["attach_mask_holder"].any(dim=1)
        pres = snap["presented"].bool()
        print(f"[round {rnd}] held {int(held.sum())}/{n}  holder {int(pinned.sum())}/{n}  "
              f"presented {int(pres.sum())}/{n}  "
              f"coverage {snap['coverage'][pres].mean() if pres.any() else float('nan'):.3f}")
        assert bool(pinned.all()), "holder anchor lost in some env"
        banks.append(snap)

    # ── Write the presented-filtered bank ────────────────────────────
    keep_pos, keep_vel, keep_hand, keep_holder, meta = [], [], [], [], []
    for snap in banks:
        m = snap["presented"].bool()
        keep_pos.append(snap["pos"][m])
        keep_vel.append(snap["vel"][m])
        keep_hand.append(snap["attach_mask_hand"][m])
        keep_holder.append(snap["attach_mask_holder"][m])
        meta.append(torch.stack([snap["stretch_ratio"][m], snap["coverage"][m]], dim=1))
    bank = {
        "pos": torch.cat(keep_pos), "vel": torch.cat(keep_vel),
        "attach_mask_hand": torch.cat(keep_hand),
        "attach_mask_holder": torch.cat(keep_holder),
        "stretch_coverage": torch.cat(meta),
        "source_checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
    }
    torch.save(bank, args_cli.out)
    print(f"\nRESULT: snapshot hook VALID — saved {bank['pos'].shape[0]} presented "
          f"terminal states to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

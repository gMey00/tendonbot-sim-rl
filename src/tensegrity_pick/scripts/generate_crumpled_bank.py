"""Generate the cached crumpled-shirt state bank (Task-1 initial states).

Implements the SoftGym-style drop-and-settle protocol (the field standard for
crumpled cloth initial states, see doc/reports/RESEARCH_cloth_sorting_pipeline.md §1)
in parallel envs:

  per round:
    1. lay the shirt flat on the FREE upstream belt (away from the robot)
    2. crumple: attach a random cloth particle (slot-0 anchor), lift it to a
       random height, optionally drag it a random lateral offset, release
    3. settle until max particle |v| < SETTLE_VEL (≤ SETTLE_MAX_STEPS)
    4. capture the particle state → one bank entry per env

Bank entries are stored ENV-LOCAL and XY-CENTRED on the state's centroid
(z stays belt-relative), so a consumer can place a state at any belt position
and yaw.  Cheap augmentation (yaw rotation + mirror) is applied at LOAD time
by ``ClothSortingEnvBase``, not stored.

Output (torch .pt):
    res/Props/Cloth/banks/tshirt_crumpled_bank.pt
    {
      "pos":  float32 [K, P, 3]   # env-local, xy-centred, z above ground
      "vel":  float32 [K, P, 3]   # settled residual velocities (≈ 0)
      "meta": {...}               # protocol params, mesh, date, stats
    }

The deterministic restore path this bank relies on is validated by
``scripts/model_validation/test_cache_restore_reset.py`` (Stage-0 de-risk).

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    # 64 envs x 4 rounds = 256 states (~10 min warm):
    PYTHONUNBUFFERED=1 python scripts/generate_crumpled_bank.py --headless \
        --num_envs 64 --rounds 4

    # quick smoke bank (16 states):
    PYTHONUNBUFFERED=1 python scripts/generate_crumpled_bank.py --headless \
        --num_envs 8 --rounds 2 --out /tmp/smoke_bank.pt
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate the crumpled-shirt state bank.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", type=str, default=None,
                    help="output .pt path (default: res/Props/Cloth/banks/tshirt_crumpled_bank.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (  # noqa: E402
    CONVEYOR_SURFACE_HEIGHT_M,
    PROJ_ASSETS_PATH,
    UPSTREAM_SETTLE_POS,
)

# ── Drop-and-settle protocol parameters (SoftGym-style) ──────────────────
LIFT_HEIGHT_RANGE = (0.15, 0.45)   # m above the belt for the picked particle
LIFT_DRAG_RANGE = (0.0, 0.15)      # random lateral drag while lifted (m)
LIFT_SPEED = 0.5                   # m/s pick-point speed (fast, dynamic crumple)
PICK_RADIUS = 0.03                 # small pinch (a few particles)
SETTLE_VEL = 0.01                  # m/s max particle speed = settled (SoftGym)
SETTLE_MAX_STEPS = 300             # SoftGym cap
FLAT_SETTLE_STEPS = 60             # settle after the flat lay before crumpling


def main() -> None:
    torch.manual_seed(args_cli.seed)
    out_path = args_cli.out or os.path.join(
        PROJ_ASSETS_PATH, "Props", "Cloth", "banks", "tshirt_crumpled_bank.pt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.episode_length_s = 1.0e6  # no manager resets during generation
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    cloth = u.cloth
    dev = u.device
    n = u.num_envs
    p = cloth.num_particles
    all_ids = torch.arange(n, device=dev)
    dt = u.cfg.sim.dt

    # Generate away from the robot, on the free upstream belt.
    origins = u.scene.env_origins
    settle_xy = torch.tensor(UPSTREAM_SETTLE_POS, device=dev)

    def sim_steps(k: int, hold: torch.Tensor | None = None) -> None:
        for _ in range(k):
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt)
            cloth.update()
            if hold is not None:
                cloth.hold(hold, slot=0)

    def settle() -> int:
        """Step until max particle |v| < SETTLE_VEL (SoftGym criterion)."""
        for i in range(SETTLE_MAX_STEPS):
            sim_steps(1)
            if cloth.nodal_vel_w.norm(dim=-1).max().item() < SETTLE_VEL:
                return i + 1
        return SETTLE_MAX_STEPS

    bank_pos, bank_vel = [], []
    t_start = time.perf_counter()

    for rnd in range(args_cli.rounds):
        # 1) flat lay on the upstream belt
        z_local = CONVEYOR_SURFACE_HEIGHT_M + 0.02 - cloth.flat_rest_min_z
        centroids = origins.clone()
        centroids[:, 0] += settle_xy[0]
        centroids[:, 1] += settle_xy[1]
        centroids[:, 2] += z_local
        yaws = (torch.rand(n, device=dev) - 0.5) * 2.0 * torch.pi
        cloth.reset_randomized(all_ids, centroids, yaws)
        sim_steps(FLAT_SETTLE_STEPS)

        # 2) crumple: pick a random particle per env, lift + drag, release
        pick_idx = torch.randint(0, p, (n,), device=dev)
        pick_pos = cloth.nodal_pos_w[all_ids, pick_idx]              # [N, 3]
        cloth.attach(all_ids, pick_pos, PICK_RADIUS, slot=0)

        lift_h = LIFT_HEIGHT_RANGE[0] + torch.rand(n, device=dev) * (
            LIFT_HEIGHT_RANGE[1] - LIFT_HEIGHT_RANGE[0])
        drag_ang = torch.rand(n, device=dev) * 2.0 * torch.pi
        drag_len = LIFT_DRAG_RANGE[0] + torch.rand(n, device=dev) * (
            LIFT_DRAG_RANGE[1] - LIFT_DRAG_RANGE[0])
        target = pick_pos.clone()
        target[:, 0] += drag_len * torch.cos(drag_ang)
        target[:, 1] += drag_len * torch.sin(drag_ang)
        target[:, 2] = origins[:, 2] + CONVEYOR_SURFACE_HEIGHT_M + lift_h

        move = target - pick_pos
        steps = max(int((move.norm(dim=-1).max() / (LIFT_SPEED * dt)).item()), 1)
        hold = pick_pos.clone()
        for s in range(steps):
            hold = pick_pos + move * float(s + 1) / steps
            sim_steps(1, hold=hold)
        cloth.detach(all_ids, slot=0)

        # 3) settle
        n_settle = settle()
        vmax = cloth.nodal_vel_w.norm(dim=-1).max().item()

        # 4) capture env-local, xy-centred states
        pos = cloth.nodal_pos_w - origins.unsqueeze(1)               # env-local
        cxy = pos[:, :, :2].mean(dim=1, keepdim=True)                # [N,1,2]
        pos = pos.clone()
        pos[:, :, :2] -= cxy
        bank_pos.append(pos.cpu())
        bank_vel.append(cloth.nodal_vel_w.clone().cpu())

        heights = (pos[:, :, 2].max(dim=1).values
                   - torch.full((n,), CONVEYOR_SURFACE_HEIGHT_M).to(pos.device))
        print(f"[round {rnd}] settled in {n_settle} steps (max|v| {vmax:.4f}); "
              f"pile height above belt: mean {heights.mean():.3f} m, "
              f"max {heights.max():.3f} m")

    pos = torch.cat(bank_pos, dim=0)
    vel = torch.cat(bank_vel, dim=0)
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_rev = "unknown"
    meta = {
        "mesh": "tshirt_clothesnet.usd",
        "num_states": pos.shape[0],
        "num_particles": p,
        "protocol": {
            "lift_height_range": LIFT_HEIGHT_RANGE,
            "lift_drag_range": LIFT_DRAG_RANGE,
            "lift_speed": LIFT_SPEED,
            "pick_radius": PICK_RADIUS,
            "settle_vel": SETTLE_VEL,
            "settle_max_steps": SETTLE_MAX_STEPS,
        },
        "frame": "env-local, xy-centred on centroid, z above ground (belt at "
                 f"{CONVEYOR_SURFACE_HEIGHT_M})",
        "seed": args_cli.seed,
        "git": git_rev,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/generate_crumpled_bank.py",
    }
    torch.save({"pos": pos, "vel": vel, "meta": meta}, out_path)
    dt_total = time.perf_counter() - t_start
    print(f"\nRESULT: saved {pos.shape[0]} states x {p} particles to {out_path} "
          f"({os.path.getsize(out_path)/1024**2:.1f} MB) in {dt_total:.0f}s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

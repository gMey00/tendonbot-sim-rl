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

# ── Trash-toss drop-and-settle protocol parameters ───────────────────────
# v2 (2026-07-03): the v1 SoftGym-style partial lift (0.15–0.45 m) left most
# of the shirt on the belt → near-flat piles (⌀0.07 m) that did not look like
# garments tossed onto a trash conveyor (user GUI review).  v2 lifts the
# picked point past FULL lift-off (shirt is ~0.7 m long), drags fast, and
# releases with momentum — plus an optional half-fold before the toss and an
# optional second toss — producing properly wadded piles.
LIFT_HEIGHT_RANGE = (0.45, 0.80)   # m above the belt — full lift-off
LIFT_DRAG_RANGE = (0.05, 0.30)     # random lateral drag while lifted (m)
LIFT_SPEED = 1.2                   # m/s pick-point speed (dynamic toss)
PICK_RADIUS = 0.03                 # small pinch (a few particles)
FOLD_PROB = 0.5                    # chance of a half-fold in the flat lay
FOLD_FRAC_RANGE = (0.25, 0.75)     # fold hinge position (fraction of length)
SECOND_TOSS_PROB = 0.5             # chance of a second pick-toss round
SETTLE_VEL = 0.01                  # m/s max particle speed = settled (SoftGym)
SETTLE_MAX_STEPS = 300             # step cap (the velocity gate rarely fires —
                                   # persistent contact jitter, Stage-0 §1)
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

    # Toss targets are biased toward the belt centre and clamped to the belt
    # interior — an unconstrained 1.2 m/s toss throws the shirt clear off the
    # 0.9 m-wide belt (v2.0 lesson: mean pile top ended up BELOW belt height
    # because most shirts landed on the floor).
    belt_cx = origins[:, 0] + settle_xy[0]
    belt_cy = origins[:, 1] + settle_xy[1]

    def toss() -> None:
        """One pick-lift-drag-release toss (attach slot 0, dynamic release)."""
        pick_idx = torch.randint(0, p, (n,), device=dev)
        pick_pos = cloth.nodal_pos_w[all_ids, pick_idx]              # [N, 3]
        cloth.attach(all_ids, pick_pos, PICK_RADIUS, slot=0)

        lift_h = LIFT_HEIGHT_RANGE[0] + torch.rand(n, device=dev) * (
            LIFT_HEIGHT_RANGE[1] - LIFT_HEIGHT_RANGE[0])
        # Drag direction: toward the belt centre ± ~60° (keeps landings on-belt).
        to_c = torch.atan2(belt_cy - pick_pos[:, 1], belt_cx - pick_pos[:, 0])
        drag_ang = to_c + (torch.rand(n, device=dev) - 0.5) * 2.0
        drag_len = LIFT_DRAG_RANGE[0] + torch.rand(n, device=dev) * (
            LIFT_DRAG_RANGE[1] - LIFT_DRAG_RANGE[0])
        target = pick_pos.clone()
        target[:, 0] += drag_len * torch.cos(drag_ang)
        target[:, 1] += drag_len * torch.sin(drag_ang)
        # Clamp the release point over the belt interior.
        target[:, 0] = torch.clamp(target[:, 0], belt_cx - 0.35, belt_cx + 0.35)
        target[:, 1] = torch.clamp(target[:, 1], belt_cy - 0.25, belt_cy + 0.25)
        target[:, 2] = origins[:, 2] + CONVEYOR_SURFACE_HEIGHT_M + lift_h

        move = target - pick_pos
        steps = max(int((move.norm(dim=-1).max() / (LIFT_SPEED * dt)).item()), 1)
        for s in range(steps):
            hold = pick_pos + move * float(s + 1) / steps
            sim_steps(1, hold=hold)
        # Release at full speed — the dangling cloth keeps its momentum and
        # tumbles/wads on landing instead of settling from rest.
        cloth.detach(all_ids, slot=0)

    flat_diag = None  # xy footprint diagonal of the flat lay (crumple metric)
    for rnd in range(args_cli.rounds):
        # 1) flat lay on the upstream belt (random yaw, optional half-fold)
        z_local = CONVEYOR_SURFACE_HEIGHT_M + 0.02 - cloth.flat_rest_min_z
        centroids = origins.clone()
        centroids[:, 0] += settle_xy[0]
        centroids[:, 1] += settle_xy[1]
        centroids[:, 2] += z_local
        yaws = (torch.rand(n, device=dev) - 0.5) * 2.0 * torch.pi
        fold = torch.where(
            torch.rand(n, device=dev) < FOLD_PROB,
            FOLD_FRAC_RANGE[0] + torch.rand(n, device=dev)
            * (FOLD_FRAC_RANGE[1] - FOLD_FRAC_RANGE[0]),
            torch.full((n,), -1.0, device=dev),
        )
        cloth.reset_randomized(all_ids, centroids, yaws, fold_fracs=fold)
        sim_steps(FLAT_SETTLE_STEPS)
        if flat_diag is None:
            pts = cloth.nodal_pos_w
            ext = pts[:, :, :2].max(dim=1).values - pts[:, :, :2].min(dim=1).values
            flat_diag = ext.norm(dim=-1).mean().item()

        # 2) toss (once, plus a second toss for ~half the rounds' envs —
        #    applied round-wise to keep the loop batched)
        toss()
        if torch.rand(1).item() < SECOND_TOSS_PROB:
            sim_steps(60)  # brief settle between tosses
            toss()

        # 3) settle
        n_settle = settle()

        # 4) validate: settled, and the whole shirt still ON the belt —
        #    a toss off the edge lands the shirt on the floor (invalid state).
        top_z = cloth.nodal_pos_w[:, :, 2].max(dim=1).values - origins[:, 2]
        min_z = cloth.nodal_pos_w[:, :, 2].min(dim=1).values - origins[:, 2]
        env_vmax = cloth.nodal_vel_w.norm(dim=-1).max(dim=1).values
        valid = (
            (top_z > CONVEYOR_SURFACE_HEIGHT_M + 0.02)
            & (min_z > CONVEYOR_SURFACE_HEIGHT_M - 0.10)
            & (env_vmax < 1.0)
        )

        # 5) capture env-local, xy-centred states (valid envs only)
        pos = cloth.nodal_pos_w - origins.unsqueeze(1)               # env-local
        cxy = pos[:, :, :2].mean(dim=1, keepdim=True)                # [N,1,2]
        pos = pos.clone()
        pos[:, :, :2] -= cxy
        bank_pos.append(pos[valid].cpu())
        bank_vel.append(cloth.nodal_vel_w[valid].clone().cpu())

        # crumpledness stats (valid envs): pile height + footprint shrink
        heights = top_z[valid] - CONVEYOR_SURFACE_HEIGHT_M
        ext = pos[valid][:, :, :2].max(dim=1).values - pos[valid][:, :, :2].min(dim=1).values
        diag = ext.norm(dim=-1)
        print(f"[round {rnd}] settled in {n_settle} steps; valid {int(valid.sum())}/{n} "
              f"(rejected: off-belt/unsettled); pile height above belt: "
              f"mean {heights.mean():.3f} m, max {heights.max():.3f} m; "
              f"xy footprint: mean {diag.mean():.3f} m "
              f"({diag.mean() / max(flat_diag, 1e-6):.2f}x flat)")

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

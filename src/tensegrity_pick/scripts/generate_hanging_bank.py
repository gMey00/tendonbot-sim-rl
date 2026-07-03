"""Generate the cached hanging-shirt state bank (Task-2/3 initial states).

Produces relaxed hang shapes of the shirt pinned at ONE RANDOM particle patch
— the initial-state distribution for shirt_present (shirt held at the
presentation pose by the retriever) and shirt_distribute (shirt held in the
learning robot's own gripper), decoupled from actually executing the upstream
pipeline tasks.  In parallel envs, per round:

    1. lay the shirt flat on the FREE upstream belt (random yaw)
    2. attach a random particle patch (pad-sized radius, slot-0 anchor)
    3. lift the pick point to the hang height at moderate speed (like the
       retriever would — the garment unfolds off the belt instead of being
       teleported mid-air)
    4. hold static and settle: a DAMPED phase (per-step velocity scale
       SETTLE_DRAG — synthetic air drag, since PBD cloth pendulums for tens
       of seconds in vacuum) until the ROBUST criteria fire (p95 particle
       speed + centroid speed; the max never converges — a few particles
       jitter at ~1 m/s from constraint-solver re-excitation, the hanging
       analogue of the crumpled bank's contact jitter), then a FREE
       re-equilibration phase so the captured state continues naturally
       when restored
    5. validate (hangs below the anchor, clear of the belt, settled) and
       capture the particle state ANCHOR-CENTRED (pinned patch at the local
       origin) → one bank entry per valid env

Anchor-centred states are translation-invariant: the consumer restores them
at ANY world anchor point via
``ClothSortingEnvBase._reset_cloth_hanging_from_bank`` (yaw + mirror
augmentation applied at load time), then re-attaches the wanted slot at the
anchor with the same pad-sized capture radius.

Output (torch .pt):
    res/Props/Cloth/banks/tshirt_hanging_bank.pt
    {
      "pos":        float32 [K, P, 3]  # anchor-centred (pinned patch at origin)
      "vel":        float32 [K, P, 3]  # settled residual velocities (≈ 0)
      "anchor_idx": int64   [K]        # picked particle index (metadata)
      "drape":      float32 [K]        # hang length below the anchor (m) —
                                       # consumers filter via max_drape to keep
                                       # long hangs clear of scene geometry
      "meta":       {...}              # protocol params, mesh, date, stats
    }

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    # 32 envs x 8 rounds = 256 states (~10 min warm):
    PYTHONUNBUFFERED=1 python scripts/generate_hanging_bank.py --headless \
        --num_envs 32 --rounds 8

    # quick smoke bank (16 states):
    PYTHONUNBUFFERED=1 python scripts/generate_hanging_bank.py --headless \
        --num_envs 8 --rounds 2 --out /tmp/smoke_hang_bank.pt
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate the hanging-shirt state bank.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--rounds", type=int, default=8)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", type=str, default=None,
                    help="output .pt path (default: res/Props/Cloth/banks/tshirt_hanging_bank.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_env import (  # noqa: E402
    HANG_ANCHOR_RADIUS,
)
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (  # noqa: E402
    CONVEYOR_SURFACE_HEIGHT_M,
    PROJ_ASSETS_PATH,
    UPSTREAM_SETTLE_POS,
)

# ── Random-point hang protocol parameters ─────────────────────────────────
HANG_HEIGHT = 1.0        # anchor height above the belt — the ~0.75 m garment
                         # drape stays well clear of the belt surface
LIFT_SPEED = 0.6         # m/s pick-point lift speed (gentle, retriever-like)
FLAT_SETTLE_STEPS = 60   # settle after the flat lay before picking
# Settle criteria are ROBUST statistics, not the max particle speed: a
# handful of particles jitter at ~1 m/s indefinitely (constraint-solver
# re-excitation — the hanging-cloth analogue of the crumpled bank's
# persistent contact jitter), while the bulk is settled after ~2 s
# (measured: p95 speed 0.08, centroid 0.05 m/s plateau).
SETTLE_P95_VEL = 0.10    # m/s 95th-percentile particle speed = settled
SETTLE_CENTROID_VEL = 0.08  # m/s centroid speed = settled
SETTLE_MAX_STEPS = 420   # damped-phase step cap (7 s @ 60 Hz)
SETTLE_DRAG = 0.97       # per-step velocity scale during the damped phase —
                         # synthetic air drag: PBD cloth has no medium, so a
                         # free-hanging garment pendulums for tens of seconds
                         # (measured: vmax ≈ 0.5 m/s after 7 s undamped)
FREE_SETTLE_STEPS = 60   # undamped re-equilibration before capture
VALID_P95_VEL = 0.15     # m/s p95 accepted at capture
VALID_CENTROID_VEL = 0.10  # m/s centroid speed accepted at capture


def main() -> None:
    torch.manual_seed(args_cli.seed)
    out_path = args_cli.out or os.path.join(
        PROJ_ASSETS_PATH, "Props", "Cloth", "banks", "tshirt_hanging_bank.pt"
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

    # Generate away from the robots, over the free upstream belt.
    origins = u.scene.env_origins
    settle_xy = torch.tensor(UPSTREAM_SETTLE_POS, device=dev)
    anchor = origins.clone()
    anchor[:, 0] += settle_xy[0]
    anchor[:, 1] += settle_xy[1]
    anchor[:, 2] += CONVEYOR_SURFACE_HEIGHT_M + HANG_HEIGHT

    def sim_steps(k: int, hold: torch.Tensor | None = None) -> None:
        for _ in range(k):
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt)
            cloth.update()
            if hold is not None:
                cloth.hold(hold, slot=0)

    all_i32 = all_ids.to(torch.int32)

    def settle_hanging() -> int:
        # Damped phase: bleed the pendulum swing with synthetic air drag.
        steps = SETTLE_MAX_STEPS
        for i in range(SETTLE_MAX_STEPS):
            sim_steps(1, hold=anchor)
            cloth._vel_flat[:] = (cloth.nodal_vel_w * SETTLE_DRAG).reshape(n, -1)
            cloth._set_velocities(cloth._vel_flat, all_i32)
            speeds = cloth.nodal_vel_w.norm(dim=-1)
            p95 = speeds.quantile(0.95, dim=1).max().item()
            cvel = cloth.centroid_vel_w.norm(dim=-1).max().item()
            if p95 < SETTLE_P95_VEL and cvel < SETTLE_CENTROID_VEL:
                steps = i + 1
                break
        # Free phase: let the PBD solver re-equilibrate without the drag so
        # the captured state continues naturally when restored.
        sim_steps(FREE_SETTLE_STEPS, hold=anchor)
        return steps

    bank_pos, bank_vel, bank_idx, bank_drape = [], [], [], []
    t_start = time.perf_counter()

    for rnd in range(args_cli.rounds):
        # 1) flat lay on the upstream belt (random yaw)
        z_local = CONVEYOR_SURFACE_HEIGHT_M + 0.02 - cloth.flat_rest_min_z
        centroids = origins.clone()
        centroids[:, 0] += settle_xy[0]
        centroids[:, 1] += settle_xy[1]
        centroids[:, 2] += z_local
        yaws = (torch.rand(n, device=dev) - 0.5) * 2.0 * torch.pi
        cloth.reset_randomized(all_ids, centroids, yaws)
        sim_steps(FLAT_SETTLE_STEPS)

        # 2) attach a random particle patch (the future hang anchor)
        pick_idx = torch.randint(0, p, (n,), device=dev)
        pick_pos = cloth.nodal_pos_w[all_ids, pick_idx]          # [N, 3]
        cloth.attach(all_ids, pick_pos, HANG_ANCHOR_RADIUS, slot=0)

        # 3) lift the pick point up to the anchor at moderate speed
        move = anchor - pick_pos
        steps = max(int((move.norm(dim=-1).max() / (LIFT_SPEED * dt)).item()), 1)
        for s in range(steps):
            sim_steps(1, hold=pick_pos + move * float(s + 1) / steps)

        # 4) hold static and settle into the relaxed hang
        n_settle = settle_hanging()

        # 5) validate + capture anchor-centred states
        rel = cloth.nodal_pos_w - anchor.unsqueeze(1)            # [N, P, 3]
        top_rel = rel[:, :, 2].max(dim=1).values
        min_z = cloth.nodal_pos_w[:, :, 2].min(dim=1).values - origins[:, 2]
        speeds = cloth.nodal_vel_w.norm(dim=-1)
        env_p95 = speeds.quantile(0.95, dim=1)
        env_cvel = cloth.centroid_vel_w.norm(dim=-1)
        below = top_rel < 0.10                                 # hangs below the anchor
        clear = min_z > CONVEYOR_SURFACE_HEIGHT_M + 0.05       # clear of the belt
        still = (env_p95 < VALID_P95_VEL) & (env_cvel < VALID_CENTROID_VEL)
        valid = below & clear & still
        if not valid.all():
            print(f"    rejects: above-anchor {int((~below).sum())}, "
                  f"belt-contact {int((~clear).sum())}, "
                  f"unsettled {int((~still).sum())} "
                  f"(p95 max {env_p95.max():.2f}, centroid max {env_cvel.max():.2f}; "
                  f"top_rel max {top_rel.max():.3f})")
        bank_pos.append(rel[valid].clone().cpu())
        bank_vel.append(cloth.nodal_vel_w[valid].clone().cpu())
        bank_idx.append(pick_idx[valid].clone().cpu())
        bank_drape.append((-rel[valid][:, :, 2].min(dim=1).values).clone().cpu())

        drape = -rel[valid][:, :, 2].min(dim=1).values
        ext = rel[valid][:, :, :2].max(dim=1).values - rel[valid][:, :, :2].min(dim=1).values
        print(f"[round {rnd}] settled in {n_settle} steps; valid {int(valid.sum())}/{n}; "
              f"drape below anchor: mean {drape.mean():.3f} m, max {drape.max():.3f} m; "
              f"xy extent: mean {ext.norm(dim=-1).mean():.3f} m")

        cloth.detach(all_ids, slot=0)

    pos = torch.cat(bank_pos, dim=0)
    vel = torch.cat(bank_vel, dim=0)
    anchor_idx = torch.cat(bank_idx, dim=0)
    drape_all = torch.cat(bank_drape, dim=0)
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
            "hang_height": HANG_HEIGHT,
            "lift_speed": LIFT_SPEED,
            "anchor_radius": HANG_ANCHOR_RADIUS,
            "settle_p95_vel": SETTLE_P95_VEL,
            "settle_centroid_vel": SETTLE_CENTROID_VEL,
            "settle_max_steps": SETTLE_MAX_STEPS,
            "settle_drag": SETTLE_DRAG,
            "free_settle_steps": FREE_SETTLE_STEPS,
            "valid_p95_vel": VALID_P95_VEL,
            "valid_centroid_vel": VALID_CENTROID_VEL,
        },
        "frame": "anchor-centred (pinned patch centroid at origin), gravity -z; "
                 "restore via ClothSortingEnvBase._reset_cloth_hanging_from_bank",
        "seed": args_cli.seed,
        "git": git_rev,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/generate_hanging_bank.py",
    }
    torch.save({"pos": pos, "vel": vel, "anchor_idx": anchor_idx,
                "drape": drape_all, "meta": meta}, out_path)
    dt_total = time.perf_counter() - t_start
    print(f"\nRESULT: saved {pos.shape[0]} states x {p} particles to {out_path} "
          f"({os.path.getsize(out_path)/1024**2:.1f} MB) in {dt_total:.0f}s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

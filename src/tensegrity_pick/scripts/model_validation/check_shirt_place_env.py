"""Zero / random-agent sanity check for the shirt-place RL env.

Builds the registered ``Template-Tensegrity-Shirt-Place-Play-v0`` environment and
runs two rollouts to confirm — before any training — that:

* the shirt is laid **flat and fixed** directly under the gripper at reset,
* the cloth is **stable** (no NaN / explosion) under a zero (idle) policy,
* the **deterministic attachment grasp** fires under a random policy (the gripper
  occasionally closes on the cloth and welds/lifts it), then releases on open,
* observations stay finite throughout.

This replaces ``evaluate.py`` for this task (that harness is reach-only — it reads
an ``ee_pose`` command term the shirt task does not have).

Usage::

    cd src/tensegrity_pick
    conda run -n env_isaaclab python3 scripts/model_validation/check_shirt_place_env.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero/random-agent check for shirt-place env.")
parser.add_argument("--task", type=str, default="Template-Tensegrity-Shirt-Place-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--zero_steps", type=int, default=180)
parser.add_argument("--random_steps", type=int, default=360)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shirt_place.shirt_place_scene_cfg import (  # noqa: E402
    CONVEYOR_SURFACE_HEIGHT_M,
    SHIRT_REST_XY,
)


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def _cloth_report(env) -> tuple[float, float, float, float, int]:
    """(centroid_x_local, centroid_z, thickness, max_grasp_dist, n_attached)."""
    u = env.unwrapped
    cloth = u._cloth
    pts = cloth.nodal_pos_w[0]                       # env 0
    origin = u.scene.env_origins[0]
    cx = float(pts[:, 0].mean() - origin[0])
    cz = float(pts[:, 2].mean())
    thick = float(pts[:, 2].max() - pts[:, 2].min())
    n_attached = int(u.grasp_active.sum())
    return cx, cz, thick, n_attached


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset(seed=args_cli.seed)
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    adim = env.action_space.shape[-1]

    print("=" * 64)
    print(f" Shirt-place env check — {args_cli.task}  ({n} envs, action_dim={adim})")
    print("=" * 64)

    # ── Reset placement check ─────────────────────────────────────────
    cx, cz, thick, _ = _cloth_report(env)
    belt = CONVEYOR_SURFACE_HEIGHT_M
    print("\n[reset] cloth laid flat under the gripper")
    print(f"  centroid X (local) = {cx:+.3f}  (target {SHIRT_REST_XY[0]:+.3f})")
    print(f"  centroid Z         = {cz:.3f}   (belt {belt:.3f})")
    print(f"  thickness          = {thick:.3f} m")
    place_ok = abs(cx - SHIRT_REST_XY[0]) < 0.10 and thick < 0.12 and (belt - 0.05) < cz < (belt + 0.20)
    print(f"  placement {'OK' if place_ok else 'SUSPECT'}")

    # ── Phase 1: zero agent (idle, gripper open) ──────────────────────
    print(f"\n[zero] {args_cli.zero_steps} idle steps (gripper open) — expect stable, no grasp")
    zero_act = torch.zeros(n, adim, device=dev)
    z_finite = True
    z_max_attach = 0
    z_max_z = 0.0
    for _ in range(args_cli.zero_steps):
        obs, _, term, trunc, _ = env.step(zero_act)
        z_finite &= _finite(obs["policy"] if isinstance(obs, dict) else obs)
        z_finite &= _finite(u._cloth.nodal_pos_w)
        cx, cz, thick, na = _cloth_report(env)
        z_max_attach = max(z_max_attach, na)
        z_max_z = max(z_max_z, cz)
    print(f"  obs/cloth finite   = {z_finite}")
    print(f"  max attached envs  = {z_max_attach}  (expect 0 — gripper open)")
    print(f"  cloth max centroidZ= {z_max_z:.3f}  (expect ~belt, not rising/exploding)")
    zero_ok = z_finite and z_max_attach == 0 and z_max_z < belt + 0.25

    # ── Phase 2: random agent ─────────────────────────────────────────
    print(f"\n[random] {args_cli.random_steps} random steps — expect occasional attach/detach, stable")
    torch.manual_seed(args_cli.seed)
    r_finite = True
    r_ever_attached = 0
    r_detach_events = 0
    prev_attached = u.grasp_active.clone()
    r_max_grasp_z = 0.0
    for _ in range(args_cli.random_steps):
        act = 2.0 * torch.rand(n, adim, device=dev) - 1.0
        obs, _, term, trunc, _ = env.step(act)
        r_finite &= _finite(obs["policy"] if isinstance(obs, dict) else obs)
        r_finite &= _finite(u._cloth.nodal_pos_w)
        attached = u.grasp_active
        r_ever_attached = max(r_ever_attached, int(attached.sum()))
        r_detach_events += int((prev_attached & ~attached).sum())
        prev_attached = attached.clone()
        # grasp-point height when attached (did a weld lift the cloth?)
        gp = u.shirt_grasp_point_w
        local_z = gp[:, 2] - u.scene.env_origins[:, 2]
        if attached.any():
            r_max_grasp_z = max(r_max_grasp_z, float(local_z[attached].max()))
    print(f"  obs/cloth finite      = {r_finite}")
    print(f"  peak attached envs    = {r_ever_attached} / {n}")
    print(f"  detach events         = {r_detach_events}")
    print(f"  max grasp-point Z     = {r_max_grasp_z:.3f}  (belt {belt:.3f}; >belt ⇒ a weld lifted cloth)")
    random_ok = r_finite  # attach is stochastic; the hard requirement is stability

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    all_ok = place_ok and zero_ok and random_ok
    print(f" Summary: {'ALL OK' if all_ok else 'CHECK FAILURES ABOVE'}")
    print(f"   placement={'OK' if place_ok else 'FAIL'}  "
          f"zero={'OK' if zero_ok else 'FAIL'}  random={'OK' if random_ok else 'FAIL'}")
    if r_ever_attached == 0:
        print("   note: random policy never triggered a grasp (stochastic) — the")
        print("   deterministic weld is validated separately by run_shirt_validation.py")
    print("=" * 64)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

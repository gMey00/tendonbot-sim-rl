"""Validate the shirt_distribute grasped-hang reset + grasp persistence.

Checks (headless, GPU):
  1. Env constructs; holding-pose sweep yields a bank (size + tip box printed).
  2. After reset: grasp_active everywhere, cloth welded at the fingertip
     (anchor↔tip distance), whole hang clear of the drum tops.
  3. HOLD: 120 steps with arm zero-action + gripper CLOSE (−1) — the hang
     must survive (no detach, cloth stays at the tip, no explosion).
     NOTE the plain zero_agent DROPS by design: on the binary gripper term
     any action ≥ 0 commands OPEN, so all-zero actions release immediately.
  4. RELEASE: command OPEN (+1) — grasp must detach and the cloth fall.
  5. Reset diversity: several resets, print tip-position spread + target-bin
     distribution + a per-env sample.

PASS/FAIL is printed per check and as a final RESULT line.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/check_shirt_distribute_env.py \
        --headless --num_envs 8
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="shirt_distribute env validation.")
parser.add_argument("--task", type=str, default="Template-Shirt-Distribute-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--resets", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import (  # noqa: E402
    DRUM_HEIGHT_M,
)

HOLD_STEPS = 120
RELEASE_STEPS = 60
CLOSE, OPEN = -1.0, 1.0  # binary gripper term: action < 0 ⇒ close


def main() -> None:
    torch.manual_seed(3)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.episode_length_s = 1.0e6  # no timeout resets during the scripted checks
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    n = u.num_envs
    dev = u.device
    results: list[tuple[str, bool, str]] = []

    # ── 1. pose bank ──────────────────────────────────────────────────
    k = u._pose_bank_q.shape[0]
    tips = u._pose_bank_tip
    ok = k >= 32
    results.append(("pose_bank", ok,
                    f"{k} poses; tip x[{tips[:,0].min():.2f},{tips[:,0].max():.2f}] "
                    f"y[{tips[:,1].min():.2f},{tips[:,1].max():.2f}] "
                    f"z[{tips[:,2].min():.2f},{tips[:,2].max():.2f}]"))

    def act(grip: float) -> torch.Tensor:
        a = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
        a[:, -1] = grip
        return a

    # one settle step so robot.data is fresh after the wrapper reset
    with torch.inference_mode():
        env.step(act(CLOSE))

    # ── 2. reset integrity ────────────────────────────────────────────
    attached = u.grasp_active
    tip = u._finger_tip_pos()
    gp = u.shirt_grasp_point_w
    d_tip = torch.norm(gp - tip, dim=-1)
    bottom = u.shirt_min_z_w - u.scene.env_origins[:, 2]
    ok_att = bool(attached.all())
    ok_tip = bool((d_tip < 0.15).all())
    ok_clear = bool((bottom > DRUM_HEIGHT_M).all())
    results.append(("reset_attached", ok_att, f"{int(attached.sum())}/{n} attached"))
    results.append(("reset_at_tip", ok_tip,
                    f"grasp-point↔tip dist max {d_tip.max():.3f} m"))
    results.append(("reset_drape_clear", ok_clear,
                    f"cloth bottom min {bottom.min():.3f} m (drum top {DRUM_HEIGHT_M})"))

    # ── 3. hold persistence ───────────────────────────────────────────
    drops = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.inference_mode():
        for _ in range(HOLD_STEPS):
            env.step(act(CLOSE))
            drops |= ~u.grasp_active
    tip = u._finger_tip_pos()
    d_tip = torch.norm(u.shirt_grasp_point_w - tip, dim=-1)
    speeds = u.cloth.nodal_vel_w.norm(dim=-1)
    ok_hold = bool((~drops).all()) and bool((d_tip < 0.15).all())
    results.append(("hold_120_steps", ok_hold,
                    f"drops {int(drops.sum())}/{n}; tip dist max {d_tip.max():.3f}; "
                    f"p95 particle speed {speeds.quantile(0.95):.3f} m/s"))

    # ── 4. release ────────────────────────────────────────────────────
    with torch.inference_mode():
        for _ in range(RELEASE_STEPS):
            env.step(act(OPEN))
    released = ~u.grasp_active
    fell = (u.shirt_min_z_w - u.scene.env_origins[:, 2]) < DRUM_HEIGHT_M
    ok_rel = bool(released.all()) and bool(fell.all())
    results.append(("release_drops", ok_rel,
                    f"released {int(released.sum())}/{n}, fell {int(fell.sum())}/{n}"))

    # ── 5. reset diversity ────────────────────────────────────────────
    tip_list, bins = [], []
    with torch.inference_mode():
        for _ in range(args_cli.resets):
            env.reset()
            env.step(act(CLOSE))
            tip_list.append((u._finger_tip_pos() - u.scene.env_origins).clone())
            bins.append(u._target_bin.clone())
    tips_r = torch.cat(tip_list, dim=0)
    bins_r = torch.cat(bins, dim=0)
    spread = tips_r.std(dim=0)
    counts = torch.bincount(bins_r, minlength=3)
    ok_div = bool((spread > 0.02).all()) and bool((counts > 0).all())
    results.append(("reset_diversity", ok_div,
                    f"tip std ({spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f}) m; "
                    f"bins {counts.tolist()}"))

    print("\n── shirt_distribute env validation ──────────────────────────")
    all_ok = True
    for name, ok, info in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s} {info}")
        all_ok &= ok
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'} "
          f"({sum(o for _, o, _ in results)}/{len(results)} checks)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Scripted heuristic baseline for the shirt_pick task (no RL).

Two purposes (both research-report recommendations):
  1. a hand-engineered pick baseline BEFORE training (ICRA-2024 cloth
     competition: heuristics are competitive) — align over the shirt's
     highest point, descend, close, lift, servo the grasp point to the
     presentation pose, hold;
  2. an empirical REACHABILITY check of ``PRESENTATION_POS`` for the
     retriever (shirt_place lesson: gate every goal predicate on reachable
     geometry).  The first version of this script caught that the original
     pose (0.15, 0.45, 1.25) was outside the tensegrity carry envelope.

The controller is a simple per-axis proportional servo on the persistent
action vector (actions are joint-position targets relative to defaults, so
integrating errors into the action is a position servo):
  base_y ← y-error, base_z ← z-error, elbow ← x-error (sign auto-calibrated
  by a nudge at startup — the elbow-to-tip-x sign depends on the mount).

Reports per env: grasp success (latched), minimum / final hold distance of
the grasp point to the presentation pose, stable-hold fraction over the last
second, and the aggregate baseline present rate.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/baseline_shirt_pick.py --headless --num_envs 8

    # crumpled-bank robustness (bank auto-loads when generated):
    PYTHONUNBUFFERED=1 python scripts/model_validation/baseline_shirt_pick.py --headless --num_envs 16
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted shirt_pick baseline + reachability check.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=8)
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
    PRESENTATION_POS,
)

# action layout (tensegrity variant): [elbow, wrist_y, wrist_x, gripper, base_y, base_z]
IDX_ELBOW, IDX_WY, IDX_WX, IDX_GRIP, IDX_BY, IDX_BZ = range(6)
OPEN, CLOSE = 1.0, -1.0  # BinaryJointPositionAction: positive = open

DIST_THR = 0.15   # "presented" distance threshold (matches ShirtPickEnv)
VEL_THR = 0.20    # m/s EE speed for a "stable" hold

ALIGN_STEPS = 60
DESCEND_STEPS = 120
CLOSE_STEPS = 40
LIFT_STEPS = 60
CARRY_STEPS = 180
HOLD_STEPS = 60

K_Y = 6.0   # base_y action gain on y error (action unit ≈ 0.5 m)
K_Z = 6.0   # base_z gain
K_X = 3.0   # elbow gain on x error (sign auto-calibrated)


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.episode_length_s = 1.0e6  # scripted rollout, no timeout resets
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    dt = u.cfg.sim.dt

    target = u.scene.env_origins + torch.tensor(
        PRESENTATION_POS, device=dev, dtype=torch.float32
    ).unsqueeze(0)

    a = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
    a[:, IDX_GRIP] = OPEN

    def step_once() -> None:
        with torch.inference_mode():
            env.step(a)

    def servo_xy(goal: torch.Tensor, elbow_sign: float) -> None:
        """One P-servo update of base_y (y) and elbow (x) toward *goal*."""
        tip = u._finger_tip_pos()
        err = goal - tip
        a[:, IDX_BY] = torch.clamp(a[:, IDX_BY] + K_Y * err[:, 1] * dt, -1.0, 1.0)
        a[:, IDX_ELBOW] = torch.clamp(
            a[:, IDX_ELBOW] + elbow_sign * K_X * err[:, 0] * dt, -1.0, 1.0)

    # ── auto-calibrate the elbow → tip-x sign (mount-dependent) ──────────
    tip0 = u._finger_tip_pos()[:, 0].mean()
    a[:, IDX_ELBOW] = 0.2
    for _ in range(15):
        step_once()
    elbow_sign = 1.0 if (u._finger_tip_pos()[:, 0].mean() - tip0) > 0 else -1.0
    a[:, IDX_ELBOW] = 0.0
    for _ in range(15):
        step_once()
    print(f"[calib] elbow → tip-x sign: {elbow_sign:+.0f}")

    # Phase 1: align the tip over the shirt's highest point
    for _ in range(ALIGN_STEPS):
        servo_xy(u.shirt_grasp_point_w, elbow_sign)
        step_once()

    # Phase 2: descend (base_z down) while keeping the alignment
    for _ in range(DESCEND_STEPS):
        servo_xy(u.shirt_grasp_point_w, elbow_sign)
        a[:, IDX_BZ] = torch.clamp(a[:, IDX_BZ] - 0.03, -1.0, 1.0)
        step_once()
        d = (u._finger_tip_pos() - u.shirt_grasp_point_w).norm(dim=-1)
        if (d < 0.08).all():
            break
    d = (u._finger_tip_pos() - u.shirt_grasp_point_w).norm(dim=-1)
    print(f"[descend] tip→highest-point: {[f'{x:.3f}' for x in d.tolist()]}")

    # Phase 3: close (attach fires when the tip reaches the cloth)
    a[:, IDX_GRIP] = CLOSE
    for _ in range(CLOSE_STEPS):
        servo_xy(u.shirt_grasp_point_w, elbow_sign)
        step_once()
    print(f"[grasp] attached after close: {int(u.grasp_active.sum())}/{n}")

    # Phase 4: lift straight up
    for _ in range(LIFT_STEPS):
        a[:, IDX_BZ] = torch.clamp(a[:, IDX_BZ] + 0.05, -1.0, 1.0)
        step_once()
    gp = u.shirt_grasp_point_w - u.scene.env_origins
    print(f"[lift] grasp-point z: mean {gp[:, 2].mean():.3f} m, max {gp[:, 2].max():.3f} m "
          f"(attached {int(u.grasp_active.sum())}/{n})")

    # Phase 5+6: servo the grasp point to the presentation pose, then hold
    dists, speeds = [], []
    for _ in range(CARRY_STEPS + HOLD_STEPS):
        gp = u.shirt_grasp_point_w
        err = target - gp
        a[:, IDX_BY] = torch.clamp(a[:, IDX_BY] + K_Y * err[:, 1] * dt, -1.0, 1.0)
        a[:, IDX_BZ] = torch.clamp(a[:, IDX_BZ] + K_Z * err[:, 2] * dt, -1.0, 1.0)
        a[:, IDX_ELBOW] = torch.clamp(
            a[:, IDX_ELBOW] + elbow_sign * K_X * err[:, 0] * dt, -1.0, 1.0)
        step_once()
        robot = u.scene["robot"]
        dists.append((u.shirt_grasp_point_w - target).norm(dim=-1).clone())
        speeds.append(robot.data.body_lin_vel_w[:, u._ee_body_idx, :].norm(dim=-1).clone())

    dist_t = torch.stack(dists)
    speed_t = torch.stack(speeds)
    grasped = u.was_grasped & u.grasp_active  # grasped and never dropped
    min_d = dist_t.min(dim=0).values
    final_d = dist_t[-HOLD_STEPS:].mean(dim=0)
    stable = (
        (dist_t[-HOLD_STEPS:] < DIST_THR)
        & (speed_t[-HOLD_STEPS:] < VEL_THR)
        & grasped.unsqueeze(0)
    ).float().mean(dim=0)

    print("\n── per-env results ──────────────────────────────────────")
    for i in range(n):
        print(f"env {i}: grasped={bool(grasped[i])}  min_dist={min_d[i]:.3f} m  "
              f"hold_dist={final_d[i]:.3f} m  stable_hold_frac={stable[i]:.2f}")
    ok = grasped & (final_d < DIST_THR)
    reach = min_d[grasped].min().item() if grasped.any() else float("nan")
    print(f"\n[reachability] PRESENTATION_POS {PRESENTATION_POS}: "
          f"min grasped-carry dist {reach:.3f} m "
          f"({'REACHABLE' if reach < DIST_THR else 'NOT reachable — move the pose!'})")
    print(f"RESULT: baseline present rate {int(ok.sum())}/{n} "
          f"(ever-grasped {int(u.was_grasped.sum())}/{n}, still-held {int(grasped.sum())}/{n}, "
          f"mean stable-hold fraction {stable.mean():.2f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

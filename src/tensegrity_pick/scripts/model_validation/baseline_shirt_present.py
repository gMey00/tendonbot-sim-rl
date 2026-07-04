"""Scripted heuristic baseline for the shirt_present task (no RL).

Three purposes (research-report + shirt_pick-lesson recommendations):
  1. prove the naive two-grasp presentation heuristic (second grab at the
     LOWEST hanging point + stretch between the grasps) is physically
     achievable from the hanging-bank initial states BEFORE any training;
  2. MEASURE the coverage / tautness distributions of (a) the raw hang and
     (b) the scripted stretch — the empirical basis for the success
     thresholds (``COVERAGE_THRESHOLD``, ``STRETCH_BAND`` in
     shirt_present_env.py);
  3. catch kinematic-reach problems of the UR5e around the hanging cloth
     early (shirt_place lesson: gate every goal predicate on reachable
     geometry).

Controller: the UR5e variant uses joint-space actions, so the Cartesian
servo runs through a finite-difference Jacobian estimated IN ACTION SPACE
(columns = tip displacement per action unit).  Per the shirt_pick lesson,
the Jacobian is re-estimated every ``--jac_every`` steps during the approach
— NEVER trust an "unreachable" verdict from a stale-Jacobian servo.

Phases: settle → estimate J → approach the lowest point → close (the
deterministic attach fires within 0.10 m) → stretch along the anchor→grasp
line, feedback-stopped on the live tautness ratio → hold + measure.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/baseline_shirt_present.py \
        --headless --num_envs 16

    # different stretch target / pull direction study:
    ... baseline_shirt_present.py --headless --num_envs 16 --target_ratio 1.05
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted shirt_present baseline + calibration.")
parser.add_argument("--task", type=str, default="Template-Shirt-Present-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--target_ratio", type=float, default=1.00,
                    help="tautness ratio the stretch phase servos to")
parser.add_argument("--jac_every", type=int, default=200,
                    help="re-estimate the FD Jacobian every N approach steps")
parser.add_argument("--act_clamp", type=float, default=1.0,
                    help="absolute arm-action clamp.  v3: the task uses EMA "
                         "joint-position-to-limits actions, so the full joint range "
                         "lives inside [-1, 1] (v1/v2 history: delta-from-default "
                         "scale 0.5 saturated at every clamp tried - the measured "
                         "justification for the action-space switch)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shirt_present.shirt_present_env import (  # noqa: E402
    COVERAGE_THRESHOLD,
    PRESENT_VEL_THRESHOLD,
    STRETCH_BAND,
)

N_ARM = 6                 # UR5e arm action dims (gripper binary is the last dim)
OPEN, CLOSE = 1.0, -1.0   # BinaryJointPositionAction: positive = open

SETTLE_STEPS = 90
JAC_NUDGE = 0.10          # action-units nudge per joint for the FD Jacobian
JAC_STEPS = 18            # sim steps for the nudge to settle into the PD arm
APPROACH_MAX_STEPS = 480
CLOSE_STEPS = 50
STRETCH_MAX_STEPS = 300
HOLD_STEPS = 120

K_SERVO = 0.15            # fraction of the Cartesian error commanded per step
MAX_DA = 0.05             # per-step action-delta clamp (action units)


def pct(t: torch.Tensor, q: float) -> float:
    return float(torch.quantile(t, q))


def stats(name: str, t: torch.Tensor) -> str:
    return (f"{name}: mean {t.mean():.3f}  p10 {pct(t, 0.10):.3f}  "
            f"p50 {pct(t, 0.50):.3f}  p90 {pct(t, 0.90):.3f}  "
            f"min {t.min():.3f}  max {t.max():.3f}")


def main() -> None:
    torch.manual_seed(args_cli.seed)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.episode_length_s = 1.0e6  # scripted rollout, no timeout resets
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    dev = u.device
    n = u.num_envs

    a = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
    a[:, N_ARM] = OPEN

    def step_once() -> None:
        with torch.inference_mode():
            env.step(a)

    def tip() -> torch.Tensor:
        return u._finger_tip_pos()

    # ── FD Jacobian in action space: J[e, :, j] = d tip / d action_j ─────
    def estimate_jacobian() -> torch.Tensor:
        J = torch.zeros(n, 3, N_ARM, device=dev)
        for _ in range(10):
            step_once()
        base = tip().clone()
        for j in range(N_ARM):
            a[:, j] += JAC_NUDGE
            for _ in range(JAC_STEPS):
                step_once()
            J[:, :, j] = (tip() - base) / JAC_NUDGE
            a[:, j] -= JAC_NUDGE
            for _ in range(JAC_STEPS):
                step_once()
        return J

    def servo_to(J_pinv: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> None:
        """One resolved-rate update of the arm actions toward *target*."""
        err = (target - tip()) * K_SERVO
        da = torch.bmm(J_pinv, err.unsqueeze(-1)).squeeze(-1)
        da = torch.clamp(da, -MAX_DA, MAX_DA) * active.float().unsqueeze(-1)
        c = args_cli.act_clamp
        a[:, :N_ARM] = torch.clamp(a[:, :N_ARM] + da, -c, c)

    # ── Phase 0: settle the restored hang, measure the RAW distributions ─
    for _ in range(SETTLE_STEPS):
        step_once()
    raw_cov = u.coverage.clone()
    raw_low = u.shirt_lowest_point_w.clone()
    cloth_speed = u.cloth.centroid_vel_w.norm(dim=-1)
    print("\n── raw hang (bank states, settled) ──────────────────────")
    print(stats("[raw] coverage", raw_cov))
    print(stats("[raw] cloth speed (m/s)", cloth_speed))
    print(stats("[raw] lowest-point z (m)", raw_low[:, 2] - u.scene.env_origins[:, 2]))

    # ── Phase 1: Jacobian + approach the lowest hanging point ────────────
    print("\n[jac] estimating FD action-space Jacobian "
          f"({N_ARM} joints × {2 * JAC_STEPS} steps)...")
    J = estimate_jacobian()
    J_pinv = torch.linalg.pinv(J)
    print(f"[jac] column norms (m/action-unit): "
          f"{[f'{x:.3f}' for x in J.norm(dim=1).mean(dim=0).tolist()]}")

    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    min_d = torch.full((n,), float("inf"), device=dev)
    for it in range(APPROACH_MAX_STEPS):
        if it > 0 and it % args_cli.jac_every == 0 and not reached.all():
            J = estimate_jacobian()
            J_pinv = torch.linalg.pinv(J)
            print(f"[jac] re-estimated at approach step {it}")
        low = u.shirt_lowest_point_w
        d = (tip() - low).norm(dim=-1)
        min_d = torch.minimum(min_d, d)
        reached |= d < 0.07
        if reached.all():
            break
        # Staged: aim 12 cm above the lowest point until close (approach from
        # above disturbs the hang less), then the true target.
        stage_off = torch.zeros_like(low)
        stage_off[:, 2] = torch.where(d > 0.15, 0.12, 0.0)
        servo_to(J_pinv, low + stage_off, active=~reached)
        step_once()
        if (it + 1) % 120 == 0:
            sat = (a[:, :N_ARM].abs() > 0.98 * args_cli.act_clamp).float().mean()
            print(f"[approach] step {it + 1}: mean dist {d.mean():.3f} m, "
                  f"reached {int(reached.sum())}/{n}, action-sat {sat:.2f}")
    d = (tip() - u.shirt_lowest_point_w).norm(dim=-1)
    sat = (a[:, :N_ARM].abs() > 0.98 * args_cli.act_clamp).float().mean()
    max_act = a[:, :N_ARM].abs().max()
    print(f"\n[approach] reached (<7 cm): {int(reached.sum())}/{n} "
          f"after {it + 1} steps (action-sat {sat:.2f}, max |a| {max_act:.2f} "
          f"of clamp {args_cli.act_clamp})")
    print(stats("[approach] final tip→lowest dist (m)", d))
    print(stats("[approach] min tip→lowest dist (m)", min_d))

    # ── Phase 2: close → deterministic attach at the lowest point ────────
    a[:, N_ARM] = CLOSE
    for _ in range(CLOSE_STEPS):
        target = u.shirt_lowest_point_w
        servo_to(J_pinv, target, active=~u.grasp_active)
        step_once()
    grasped = u.grasp_active.clone()
    print(f"\n[grasp] attached after close: {int(grasped.sum())}/{n}")

    # ── Phase 3: stretch along the anchor→grasp direction ────────────────
    # Fixed Cartesian pull target on the anchor→tip ray; the live tautness
    # ratio gives the stop feedback (freeze when ratio ≥ target).
    anchor = u._anchor_pos
    dvec = tip() - anchor
    dhat = dvec / dvec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ratio_hist, cov_hist = [], []
    for it in range(STRETCH_MAX_STEPS):
        ratio = u.stretch_ratio
        need_pull = grasped & u.grasp_active & (ratio < args_cli.target_ratio)
        # pull target: current tip pushed outward along the ray
        target = tip() + dhat * 0.05
        servo_to(J_pinv, target, active=need_pull)
        step_once()
        ratio_hist.append(u.stretch_ratio.clone())
        cov_hist.append(u.coverage.clone())
        if not bool(need_pull.any()):
            break
    print(f"\n[stretch] pull phase ended after {it + 1} steps "
          f"(target ratio {args_cli.target_ratio})")

    # ── Phase 4: hold still + measure ─────────────────────────────────────
    pres_frac = torch.zeros(n, device=dev)
    hold_cov, hold_ratio, hold_speed = [], [], []
    for _ in range(HOLD_STEPS):
        step_once()
        hold_cov.append(u.coverage.clone())
        hold_ratio.append(u.stretch_ratio.clone())
        hold_speed.append(u.cloth.centroid_vel_w.norm(dim=-1).clone())
        pres_frac += u.presented_now.float()
    pres_frac /= HOLD_STEPS
    hold_cov_t = torch.stack(hold_cov).mean(dim=0)
    hold_ratio_t = torch.stack(hold_ratio).mean(dim=0)
    hold_speed_t = torch.stack(hold_speed).mean(dim=0)
    still_held = grasped & u.grasp_active

    print("\n── stretched hold (last "
          f"{HOLD_STEPS} steps ≈ {HOLD_STEPS / 60:.1f} s) ────────────────")
    print(stats("[hold] coverage", hold_cov_t))
    print(stats("[hold] coverage (still-held only)",
                hold_cov_t[still_held]) if still_held.any() else "[hold] no held envs")
    print(stats("[hold] stretch ratio", hold_ratio_t))
    print(stats("[hold] cloth speed (m/s)", hold_speed_t))
    print(stats("[hold] presented_now fraction", pres_frac))

    print("\n── per-env results ──────────────────────────────────────")
    for i in range(n):
        print(f"env {i}: grasped={bool(grasped[i])} held={bool(still_held[i])} "
              f"min_reach={min_d[i]:.3f} m  ratio={hold_ratio_t[i]:.3f}  "
              f"cov(raw→hold)={raw_cov[i]:.3f}→{hold_cov_t[i]:.3f}  "
              f"presented_frac={pres_frac[i]:.2f}")

    cov_gain = hold_cov_t - raw_cov
    ok = still_held & (pres_frac >= 0.8)
    print("\n── calibration summary ──────────────────────────────────")
    print(f"current thresholds: coverage ≥ {COVERAGE_THRESHOLD}, "
          f"stretch ∈ [{STRETCH_BAND[0]}, {STRETCH_BAND[1]}], "
          f"cloth speed < {PRESENT_VEL_THRESHOLD}")
    print(stats("[calib] raw-hang coverage", raw_cov))
    if still_held.any():
        print(stats("[calib] held stretched coverage", hold_cov_t[still_held]))
        print(stats("[calib] held stretch ratio", hold_ratio_t[still_held]))
    print(stats("[calib] coverage gain (hold − raw)", cov_gain))
    print(f"\nRESULT: baseline present rate {int(ok.sum())}/{n} "
          f"(grasped {int(grasped.sum())}/{n}, still-held {int(still_held.sum())}/{n}, "
          f"reach<7cm {int(reached.sum())}/{n})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

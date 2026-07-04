"""Scripted heuristic baseline for the shirt_present task (no RL).

Three purposes (research-report + shirt_pick-lesson recommendations):
  1. prove the naive two-grasp presentation heuristic (second grab at the
     LOWEST hanging point + stretch between the grasps) is physically
     achievable from the hanging-bank initial states BEFORE any training;
  2. MEASURE the coverage / tautness distributions of (a) the raw hang and
     (b) the scripted stretch — the empirical basis for the success
     thresholds (``COVERAGE_THRESHOLD``, ``STRETCH_BAND`` in
     shirt_present_env.py), plus a geodesic sanity check (straight vertical
     hang ⇒ anchor→lowest Euclidean ≈ geodesic);
  3. catch kinematic-reach problems of the UR5e around the hanging cloth
     early (shirt_place lesson: gate every goal predicate on reachable
     geometry).

Controller history (tracking report Phase 1):
  * v1/v2 — finite-difference action-space Jacobian: saturated the stub's
    delta-from-default action space (the measured justification for the
    EMA to-limits switch) and went stale between re-estimates (3–5/16
    reached).
  * v4 (this) — resolved-rate servo on the ANALYTIC PhysX Jacobian, fresh
    every step (the DifferentialIK data path), acting in joint space and
    converted to to-limits actions by inverting the normalisation; while a
    grasp holds, the arm is commanded to its CURRENT posture (v3 lesson:
    frozen saturated actions dragged the grasped cloth to ratio 2.9).

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/baseline_shirt_present.py \
        --headless --num_envs 16

    # different stretch target:
    ... baseline_shirt_present.py --headless --num_envs 16 --target_ratio 1.05
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted shirt_present baseline + calibration.")
parser.add_argument("--task", type=str, default="Template-Shirt-Present-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--target_ratio", type=float, default=1.02,
                    help="AT-GRASP-NORMALISED tautness the stretch phase servos to")
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
from tensegrity_pick.robots.ur5e_robot_cfg import CONTROLLED_JOINT_NAMES  # noqa: E402
from tensegrity_pick.tasks.manager_based.shirt_present.shirt_present_env import (  # noqa: E402
    COVERAGE_THRESHOLD,
    PRESENT_VEL_THRESHOLD,
    STRETCH_BAND,
)

N_ARM = 6                 # UR5e arm action dims (gripper binary is the last dim)
OPEN, CLOSE = 1.0, -1.0   # BinaryJointPositionAction: positive = open

SETTLE_STEPS = 90
APPROACH_MAX_STEPS = 420
CLOSE_STEPS = 50
STRETCH_MAX_STEPS = 300
HOLD_STEPS = 120

K_SERVO = 3.0             # Cartesian error gain (1/s): dx = K * err * dt
DQ_MAX = 0.04             # per-step joint-target step clamp (rad)
A_LIM = 0.98              # keep commanded normalised targets off the exact limits


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
    dt = u.cfg.sim.dt

    robot = u.scene["robot"]
    arm_ids = [robot.joint_names.index(j) for j in CONTROLLED_JOINT_NAMES]
    limits = robot.data.joint_pos_limits[:, arm_ids, :]          # [N, 6, 2]
    lo, hi = limits[..., 0], limits[..., 1]

    a = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
    a[:, N_ARM] = OPEN
    ALL = torch.ones(n, dtype=torch.bool, device=dev)

    def q_arm() -> torch.Tensor:
        return robot.data.joint_pos[:, arm_ids]

    def to_action(q_des: torch.Tensor) -> torch.Tensor:
        """Invert the to-limits normalisation: joint targets → actions [-1, 1]."""
        return torch.clamp(2.0 * (q_des - lo) / (hi - lo) - 1.0, -A_LIM, A_LIM)

    def hold_current(mask: torch.Tensor) -> None:
        """Command the CURRENT posture for envs in *mask* (stationary hold)."""
        if mask.any():
            a[mask, :N_ARM] = to_action(q_arm())[mask]

    def arm_jacobian() -> torch.Tensor:
        """Positional analytic Jacobian of the EE body wrt the arm joints [N, 3, 6]."""
        jac = robot.root_physx_view.get_jacobians()              # [N, B-1, 6, D]
        return jac[:, u._ee_body_idx - 1, :3, :][:, :, arm_ids]

    def tip() -> torch.Tensor:
        return u._finger_tip_pos()

    def servo_to(target: torch.Tensor, active: torch.Tensor) -> None:
        """One resolved-rate update toward *target* (fresh analytic Jacobian)."""
        v = K_SERVO * (target - tip()) * dt                      # desired dx this step (m)
        J = arm_jacobian()
        dq = torch.linalg.lstsq(J, v.unsqueeze(-1)).solution.squeeze(-1)
        dq = torch.clamp(dq, -DQ_MAX, DQ_MAX) * active.float().unsqueeze(-1)
        a[:, :N_ARM] = to_action(q_arm() + dq)

    def step_once() -> None:
        with torch.inference_mode():
            env.step(a)

    # ── Phase 0: settle the restored hang, measure the RAW distributions ─
    for _ in range(SETTLE_STEPS):
        hold_current(ALL)
        step_once()
    raw_cov = u.coverage.clone()
    raw_low = u.shirt_lowest_point_w.clone()
    cloth_speed = u.cloth.centroid_vel_w.norm(dim=-1)
    print("\n── raw hang (bank states, settled) ──────────────────────")
    print(stats("[raw] coverage", raw_cov))
    print(stats("[raw] cloth speed (m/s)", cloth_speed))
    print(stats("[raw] lowest-point z (m)", raw_low[:, 2] - u.scene.env_origins[:, 2]))

    # Geodesic sanity: on a gravity-straightened hang the anchor→lowest
    # Euclidean distance ≈ the fabric path (geodesic), so euclid/geodesic
    # should sit at or just below 1 (folded sleeves → lower).  Values > ~1.05
    # break the metric (index-space mismatch — tracking report Phase 1).
    if u._geo_edges is not None:
        pts = u.cloth.nodal_pos_w
        low_idx = pts[:, :, 2].argmin(dim=1)
        geo_low = u._geo_dist.gather(1, low_idx.unsqueeze(1)).squeeze(1)
        euc_low = (raw_low - u._anchor_pos).norm(dim=-1)
        print(stats("[geo] anchor→lowest euclid (m)", euc_low))
        print(stats("[geo] anchor→lowest geodesic (m)", geo_low))
        print(stats("[geo] euclid/geodesic (expect ≤ ~1)", euc_low / geo_low.clamp(min=1e-6)))
    else:
        print("[geo] WARNING: no spring graph — flat-Euclidean fallback in use")

    # ── Phase 1: approach the lowest hanging point ────────────────────────
    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    min_d = torch.full((n,), float("inf"), device=dev)
    it = 0
    for it in range(APPROACH_MAX_STEPS):
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
        servo_to(low + stage_off, active=~reached)
        hold_current(reached)
        step_once()
        if (it + 1) % 120 == 0:
            print(f"[approach] step {it + 1}: mean dist {d.mean():.3f} m, "
                  f"reached {int(reached.sum())}/{n}")
    d = (tip() - u.shirt_lowest_point_w).norm(dim=-1)
    print(f"\n[approach] reached (<7 cm): {int(reached.sum())}/{n} after {it + 1} steps")
    print(stats("[approach] final tip→lowest dist (m)", d))
    print(stats("[approach] min tip→lowest dist (m)", min_d))

    # ── Phase 2: close → deterministic attach at the lowest point ────────
    a[:, N_ARM] = CLOSE
    for _ in range(CLOSE_STEPS):
        servo_to(u.shirt_lowest_point_w, active=~u.grasp_active)
        hold_current(u.grasp_active)
        step_once()
    grasped = u.grasp_active.clone()
    print(f"\n[grasp] attached after close: {int(grasped.sum())}/{n}")
    if grasped.any():
        print(stats("[grasp] at-grasp RAW stretch ratio (= r0, held envs)",
                    u.stretch_ratio[grasped]))

    # ── Phase 3: stretch along the anchor→grasp direction ────────────────
    anchor = u._anchor_pos
    dvec = tip() - anchor
    dhat = dvec / dvec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    for it in range(STRETCH_MAX_STEPS):
        ratio = u.stretch_ratio_norm
        need_pull = grasped & u.grasp_active & (ratio < args_cli.target_ratio)
        servo_to(tip() + dhat * 0.05, active=need_pull)
        hold_current(~need_pull)
        step_once()
        if not bool(need_pull.any()):
            break
    print(f"\n[stretch] pull phase ended after {it + 1} steps "
          f"(target ratio {args_cli.target_ratio})")

    # ── Phase 4: hold still + measure ─────────────────────────────────────
    pres_frac = torch.zeros(n, device=dev)
    hold_cov, hold_ratio, hold_speed = [], [], []
    for _ in range(HOLD_STEPS):
        hold_current(ALL)
        step_once()
        hold_cov.append(u.coverage.clone())
        hold_ratio.append(u.stretch_ratio_norm.clone())
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
    if still_held.any():
        print(stats("[hold] coverage (still-held only)", hold_cov_t[still_held]))
    print(stats("[hold] stretch ratio (at-grasp-normalised)", hold_ratio_t))
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
        print(stats("[calib] held stretch ratio (normalised)", hold_ratio_t[still_held]))
    print(stats("[calib] coverage gain (hold − raw)", cov_gain))
    print(f"\nRESULT: baseline present rate {int(ok.sum())}/{n} "
          f"(grasped {int(grasped.sum())}/{n}, still-held {int(still_held.sum())}/{n}, "
          f"reach<7cm {int(reached.sum())}/{n})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

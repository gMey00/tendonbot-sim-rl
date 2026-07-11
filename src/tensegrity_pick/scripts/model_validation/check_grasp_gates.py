"""Robot-free validation rig for the grasp-fidelity gates (Stage-2 S5/M3).

Reuses the presentation-study harness (``present_heuristics/study_common.py``:
boot + hanging-bank restore + scripted attachments) to exercise the gate code
in ``shared/cloth_sorting_env.py`` WITHOUT a policy or scripted robot motion:

  1. flags-off inertness — with both gates off (the default), stepping never
     touches the gate counters (the deterministic path stays untouched;
     bit-identity of actual grasps is covered by test_shirt_fixes.py 5/5 and
     the M4 checkpoint evals);
  2. per-region gate rates, each gate measured SEPARATELY through the REAL
     ``_apply_grasp_gates`` path on hanging bank states:
       2a. stochastic gate only — misgrasp rate per region must match the
           border/mid-panel probability mixture (DRAPER calibration:
           p_border 0.16, p_mid 0.22) given the region's measured border
           fraction;
       2b. pre-condition gate only, under two approach axes — straight-down
           (physically wrong on a hanging garment: panels are vertical, so
           alignment must block most attempts) vs surface-aligned (the
           realistic side pinch: block rate must drop sharply; remaining
           blocks are genuine multi-layer/clearance detections, reported per
           region);
  3. slip gate — hang the garment from the hand slot, draw pinch capacities,
     and check the static-load slip fraction against the analytic
     Φ((m·g − mean)/std) prediction, plus a dynamic upward-acceleration pull
     that must slip strictly more often.

Usage::

    cd src/tensegrity_pick
    # env_isaaclab (conda activate, keep PYTHONPATH)
    python scripts/model_validation/check_grasp_gates.py --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grasp-fidelity gate rig.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--rounds", type=int, default=12,
                    help="hang-restore rounds per region (samples = rounds × envs)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "present_heuristics"))
import markers as mk  # noqa: E402
from study_common import StudyRig  # noqa: E402

from tensegrity_pick.tasks.manager_based.shared.cloth_metrics import (  # noqa: E402
    depth_layer_count_in_ball,
)

SETTLE_STEPS = 20  # bank states are pre-relaxed; short damped settle suffices


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}: {name}{'  (' + detail + ')' if detail else ''}")
    return bool(cond)


def main() -> None:
    ok = True
    rig = StudyRig(args_cli, seed=0)
    u = rig.u
    n, dev = rig.n, rig.dev
    gf = u.grasp_fidelity
    down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3)

    # ── 1. flags-off inertness ─────────────────────────────────────────
    print("flags-off inertness:")
    ok &= check("gates default OFF",
                not gf.precondition_gate and not gf.stochastic_gate)
    rig.sim_steps(20)
    ok &= check("no gate events while stepping",
                all(v == 0 for v in u.gf_event_counts.values()),
                str(u.gf_event_counts))

    # ── 2. per-region gate rates on hanging states ─────────────────────
    eye3 = torch.eye(3, device=dev)

    def local_normal(tip: torch.Tensor) -> torch.Tensor:
        """PCA surface normal of the weld-ball particles (rig-side copy)."""
        pos = u.cloth.nodal_pos_w
        rel = pos - tip.unsqueeze(1)
        w = (rel.norm(dim=-1) <= gf.layer_radius_m).float()
        cnt = w.sum(dim=1).clamp(min=1.0)
        mean = (pos * w.unsqueeze(-1)).sum(dim=1) / cnt.unsqueeze(-1)
        d = (pos - mean.unsqueeze(1)) * w.unsqueeze(-1)
        cov = d.transpose(1, 2) @ d / cnt.view(-1, 1, 1) + 1e-12 * eye3
        return torch.linalg.eigh(cov).eigenvectors[..., 0]

    def region_sweep(label: str, precondition: bool, stochastic: bool,
                     aligned_approach: bool) -> list[dict]:
        gf.precondition_gate = precondition
        gf.stochastic_gate = stochastic
        rows = []
        print(f"{label} ({args_cli.rounds} rounds × {n} envs):")
        print(f"  {'region':<12} {'n':>4} {'pre_block':>9} {'layers>2':>8} "
              f"{'border':>6} {'misgrasp':>8} {'attach':>6}")
        for r, rname in enumerate(mk.SYM_CLASSES):
            pool = torch.as_tensor(mk.region_particles(r), device=dev)
            if pool.numel() == 0:
                continue
            att = pre_block = mis = attach = 0
            layers_hi = border_cnt = 0
            for _ in range(args_cli.rounds):
                bank_idx = torch.randint(0, rig.bank_size, (n,), device=dev)
                yaw = torch.rand(n, device=dev) * 2.0 * torch.pi
                mirror = torch.rand(n, device=dev) < 0.5
                rig.restore_hang(bank_idx, yaw, mirror)
                rig.sim_steps(SETTLE_STEPS, drag=0.97)
                idx = pool[torch.randint(0, pool.numel(), (n,), device=dev)]
                tip = u.cloth.nodal_pos_w[rig.all_ids, idx]
                approach = local_normal(tip) if aligned_approach else down
                u._misgrasp_latch[:] = False
                c0 = dict(u.gf_event_counts)
                allow = u._apply_grasp_gates(
                    torch.ones(n, dtype=torch.bool, device=dev), tip, None,
                    approach=approach,
                )
                att += int(u.gf_event_counts["attempts"] - c0["attempts"])
                pre_block += int(u.gf_event_counts["blocked_precondition"]
                                 - c0["blocked_precondition"])
                mis += int(u.gf_event_counts["misgrasps"] - c0["misgrasps"])
                attach += int(allow.sum())
                layers = depth_layer_count_in_ball(
                    u.cloth.nodal_pos_w, tip, gf.layer_radius_m, axis=approach,
                    layer_gap=gf.layer_gap_m)
                layers_hi += int((layers > gf.max_layers).sum())
                border_cnt += int((u._grasp_border_dist(idx)
                                   <= gf.border_dist_threshold_m).sum())
            eligible = att - pre_block
            row = dict(region=rname, n=att, pre_block=pre_block / max(att, 1),
                       multi=layers_hi / max(att, 1),
                       border=border_cnt / max(att, 1),
                       misgrasp=mis / max(eligible, 1), eligible=eligible,
                       attach=attach / max(att, 1))
            rows.append(row)
            print(f"  {row['region']:<12} {att:>4} {row['pre_block']:>9.2f} "
                  f"{row['multi']:>8.2f} {row['border']:>6.2f} "
                  f"{row['misgrasp']:>8.2f} {row['attach']:>6.2f}")
        return rows

    # 2a. stochastic gate alone: per-region misgrasp rate must match the
    # border/mid-panel mixture given the measured border fraction (n = 96
    # per region → binomial σ ≈ 0.04; allow 3σ).
    rows_s = region_sweep("stochastic gate only (top-down)", False, True, False)
    mix_ok = True
    worst = 0.0
    for row in rows_s:
        expected = (row["border"] * gf.p_misgrasp_border
                    + (1.0 - row["border"]) * gf.p_misgrasp_mid_panel)
        err = abs(row["misgrasp"] - expected)
        worst = max(worst, err)
        mix_ok &= err < 0.12
    ok &= check("misgrasp rates match border/mid mixture", mix_ok,
                f"max |rate − expected| = {worst:.3f}")

    # 2b. pre-condition gate alone: a straight-down pinch on a hanging
    # garment must be blocked far more often than a surface-aligned pinch
    # (vertical panels ⊥ downward approach); aligned-approach blocks are the
    # genuine multi-layer/clearance detections.
    rows_down = region_sweep("pre-condition gate only (top-down)", True, False, False)
    rows_al = region_sweep("pre-condition gate only (surface-aligned)", True, False, True)
    mean_down = sum(r["pre_block"] for r in rows_down) / len(rows_down)
    mean_al = sum(r["pre_block"] for r in rows_al) / len(rows_al)
    ok &= check("down-approach blocked ≫ aligned",
                mean_down > mean_al + 0.3,
                f"block(down) {mean_down:.2f} vs block(aligned) {mean_al:.2f}")
    ok &= check("aligned approach mostly attachable", mean_al < 0.45,
                f"mean block {mean_al:.2f}")

    # ── 3. slip gate: static analytic check + dynamic pull ─────────────
    print("slip gate:")
    mass = float(u.cloth.cfg.pbd_params.mass_kg)
    static_load = mass * gf.gravity_m_s2   # full garment hanging below the tip
    p_analytic = 0.5 * (1.0 + math.erf(
        (static_load - gf.slip_capacity_mean_n)
        / (gf.slip_capacity_std_n * math.sqrt(2.0))))
    slips = trials = 0
    slips_dyn = 0
    for _ in range(args_cli.rounds * 2):
        bank_idx = torch.randint(0, rig.bank_size, (n,), device=dev)
        yaw = torch.rand(n, device=dev) * 2.0 * torch.pi
        mirror = torch.rand(n, device=dev) < 0.5
        rig.restore_hang(bank_idx, yaw, mirror)   # slot 0 = the "hand" grasp
        rig.sim_steps(SETTLE_STEPS, drag=0.97)
        tip = rig.anchor.clone()
        # draw fresh capacities exactly like the attach path
        cap = gf.slip_capacity_mean_n + gf.slip_capacity_std_n * torch.randn(
            n, device=dev, generator=u._gf_rng)
        u._slip_capacity[:] = cap.clamp(min=gf.slip_capacity_min_n)
        u._misgrasp_latch[:] = False
        u._gf_prev_valid[:] = False
        u._gf_vel_valid[:] = False
        c0 = u.gf_event_counts["slips"]
        u._apply_slip_gate(tip)                  # primes prev buffers, acc=0
        u._apply_slip_gate(tip)                  # static check
        slips += int(u.gf_event_counts["slips"] - c0)
        trials += n
        # dynamic: re-attach survivors' load with an accelerating upward pull
        c1 = u.gf_event_counts["slips"]
        dt = u.cfg.sim.dt * u.cfg.decimation
        v = 0.0
        for k in range(6):
            v += 8.0 * dt                         # 8 m/s² upward ramp
            tip = tip.clone()
            tip[:, 2] += v * dt
            u._apply_slip_gate(tip)
        slips_dyn += int(u.gf_event_counts["slips"] - c1)
    static_rate = slips / trials
    ok &= check("static slip rate ≈ analytic Φ",
                abs(static_rate - p_analytic) < 0.10,
                f"{static_rate:.3f} vs Φ = {p_analytic:.3f} "
                f"(load {static_load:.2f} N)")
    ok &= check("dynamic pull slips more", slips_dyn > 0,
                f"{slips_dyn} extra slips on accelerating pulls")

    print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES — see above'}")
    rig.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

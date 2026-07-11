"""Heuristic 1 — naive 2-grasp presentation study (stretch-ratio sweep).

Protocol per trial (batched over envs): restore a random hanging-bank state
pinned at the study anchor (first grasp = the bank's random particle) →
brief settle → second grasp at the LOWEST hanging point → HORIZONTAL
presentation stretch (bring the grasped point up to the first grasp's height
and pull outward along the camera-plane x axis to a target tautness ratio;
gravity extends the garment below the taut chord) → damped settle → measure
silhouette coverage + secondary metrics.  This is exactly the shirt_present
task's heuristic; the coverage distribution calibrates that task's success
threshold.

Usage (from src/tensegrity_pick, env_isaaclab active)::

    PYTHONUNBUFFERED=1 python scripts/model_validation/present_heuristics/run_two_grasp.py \
        --headless --num_envs 48 --rounds_per_ratio 5 \
        --ratios 0.95 1.0 1.05 1.10 --seed 11

    # smoke test:
    ... run_two_grasp.py --headless --num_envs 8 --rounds_per_ratio 1 --ratios 1.05 \
        --csv /tmp/h1_smoke.csv --tag smoke

Outputs: doc/reports/data/present_h1_two_grasp.csv (+ per-trial final states
under outputs/present_heuristics/states/h1/ for render_exemplars.py, and the
flat rest shape dump used by all offline analysis).
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H1: naive 2-grasp study.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--rounds_per_ratio", type=int, default=5)
parser.add_argument("--ratios", type=float, nargs="+",
                    default=[0.95, 1.0, 1.05, 1.10])
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--settle_after_restore", type=int, default=45,
                    help="damped steps after the bank restore before regrasp")
parser.add_argument("--csv", type=str, default=None)
parser.add_argument("--tag", type=str, default="h1_two_grasp")
parser.add_argument("--method", type=str, default="h1")
parser.add_argument("--shake_amp", type=float, default=0.0,
                    help="extra method: out-of-plane (±Y) shake amplitude (m) "
                         "applied to the moving anchor after the stretch, to "
                         "release folds before the final settle")
parser.add_argument("--shake_cycles", type=int, default=3)
parser.add_argument("--pulse_ratio", type=float, default=0.0,
                    help="extra method: tautness pulse — overstretch to this "
                         "ratio first (brief hold, no settle), then relax to "
                         "the target ratio and settle (the human 'snap' that "
                         "pops folds open without paying the ringing settle "
                         "of holding the higher tautness)")
parser.add_argument("--pulse_hold_steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as sc  # noqa: E402

REPO = "/home/robot/studentische-arbeiten"


def main() -> None:
    rig = sc.StudyRig(args_cli, seed=args_cli.seed)
    csv_path = args_cli.csv or os.path.join(
        REPO, "doc", "reports", "data", "present_h1_two_grasp.csv")
    states_dir = os.path.join("outputs", "present_heuristics", "states",
                              args_cli.method)
    log = sc.TrialLog(csv_path, states_dir)
    # The flat-rest dump is the study's FIXED offline reference frame (regions,
    # markers, border distances were built in it).  Each boot re-adopts a
    # slightly different settled rest shape (per-particle drift ~1.6 mm median,
    # up to ~15 mm, see report §1.2), so NEVER overwrite an existing dump —
    # later runs must analyze in the original frame.
    flat_rest_path = os.path.join(
        REPO, "doc", "reports", "data", "present_heuristics_flat_rest.pt")
    if not os.path.exists(flat_rest_path):
        sc.dump_flat_rest(rig, flat_rest_path)

    n = rig.n
    trial = 0
    for ratio in args_cli.ratios:
        for rnd in range(args_cli.rounds_per_ratio):
            bank_idx = torch.randint(0, rig.bank_size, (n,), device=rig.dev)
            yaw = torch.rand(n, device=rig.dev) * 2.0 * torch.pi
            mirror = torch.rand(n, device=rig.dev) < 0.5
            idx_top = rig.restore_hang(bank_idx, yaw, mirror)
            rig.sim_steps(args_cli.settle_after_restore, drag=sc.SETTLE_DRAG)

            idx_move = rig.lowest_particle()
            attached = rig.attach_at_particles(idx_move, slot=1)
            ratios = torch.full((n,), ratio, device=rig.dev)
            if args_cli.pulse_ratio > 0.0:
                # Tautness pulse: overstretch (no settle), brief hold, relax
                # to the target ratio, then the one canonical settle.
                pulse = torch.full((n,), args_cli.pulse_ratio, device=rig.dev)
                pre = rig.stretch(top_slot=0, move_slot=1, idx_top=idx_top,
                                  idx_move=idx_move, target_ratio=pulse,
                                  settle=False)
                rig.sim_steps(args_cli.pulse_hold_steps)
                info = rig.stretch(top_slot=0, move_slot=1, idx_top=idx_top,
                                   idx_move=idx_move, target_ratio=ratios,
                                   x_sign=pre["x_sign"])
                info["stretch_steps"] += (pre["stretch_steps"]
                                          + args_cli.pulse_hold_steps)
            else:
                info = rig.stretch(top_slot=0, move_slot=1,
                                   idx_top=idx_top, idx_move=idx_move,
                                   target_ratio=ratios)
            if args_cli.shake_amp > 0.0:
                # Sinusoidal ±Y (out-of-plane) shake of the moving anchor to
                # flap open residual folds, then a fresh damped settle.
                base = rig.hold_pos[1].clone()
                steps_per_cycle = 24  # 0.4 s @ 60 Hz per cycle
                for s in range(args_cli.shake_cycles * steps_per_cycle):
                    off = args_cli.shake_amp * torch.sin(
                        torch.tensor(2.0 * torch.pi * s / steps_per_cycle))
                    b = base.clone()
                    b[:, 1] += off
                    rig.hold_pos[1] = b
                    rig.sim_steps(1)
                rig.hold_pos[1] = base
                info["settle_steps"] += rig.settle()
            meas = rig.measure(idx_top, idx_move, rig.anchor[:, 2])
            log.add_round(
                args_cli.method, args_cli.seed, 1, trial,
                {"bank_idx": bank_idx, "yaw": yaw, "mirror": mirror,
                 "idx_top": idx_top, "idx_move": idx_move,
                 "target_ratio": ratios,
                 "settle_steps": info["settle_steps"],
                 "stretch_steps": info["stretch_steps"],
                 "x_sign": info["x_sign"],
                 "attached_ok": attached,
                 "valid": attached & torch.isfinite(meas["cov_plane"])},
                meas, rig)
            cov = meas["cov_plane"]
            print(f"[{args_cli.method} ratio={ratio:.2f} round={rnd}] coverage "
                  f"median {cov.median():.3f} IQR "
                  f"[{cov.quantile(0.25):.3f}, {cov.quantile(0.75):.3f}] "
                  f"meas_ratio med {meas['meas_ratio'].median():.3f}", flush=True)
            log.status(args_cli.tag)
            trial += n

    rig.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

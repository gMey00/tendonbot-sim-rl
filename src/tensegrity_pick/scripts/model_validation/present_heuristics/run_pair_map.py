"""Heuristic 3 — random-grasp-pair quality map (the brute-force oracle).

Protocol per trial (batched over envs): restore a random hanging-bank state
at the study anchor — the FIRST grasp is the bank's pinned particle, which is
uniform over the garment surface by construction — then attach the second
grasp at a UNIFORM-RANDOM particle of the hanging shirt, HORIZONTAL
presentation stretch (up to the holder's height, out along the camera-plane
x axis) to a fixed tautness, settle, measure.  Keyed by both particles'
flat-rest coordinates, the trials build the grasp-pair quality map
(doc/TODO.md Task 2, "grasp-pair oracle"): it upper-bounds all heuristics and
labels the good second-grasp regions.

Sampling rationale: uniform over particles ≈ uniform over garment area (the
mesh is near-uniform, ~5 mm spacing), so region aggregates are area-weighted
means with region-proportional trial counts; no stratification needed at
≥ 2 k trials for an 8×8 region grid (analysis reports per-cell n).

Usage (from src/tensegrity_pick, env_isaaclab active)::

    PYTHONUNBUFFERED=1 python scripts/model_validation/present_heuristics/run_pair_map.py \
        --headless --num_envs 64 --rounds 40 --ratio 1.05 --seed 33

    # smoke test:
    ... run_pair_map.py --headless --num_envs 8 --rounds 1 --csv /tmp/h3_smoke.csv

Output: doc/reports/data/present_h3_pair_map.csv (+ states saved only for the
running best/worst trials — 5 k full states would be ~350 MB).
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H3: random-pair oracle map.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rounds", type=int, default=40)
parser.add_argument("--ratio", type=float, default=1.05)
parser.add_argument("--seed", type=int, default=33)
parser.add_argument("--settle_after_restore", type=int, default=45)
parser.add_argument("--csv", type=str, default=None)
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
        REPO, "doc", "reports", "data", "present_h3_pair_map.csv")
    log = sc.TrialLog(csv_path, states_dir=None)
    states_dir = os.path.join("outputs", "present_heuristics", "states", "h3")
    os.makedirs(states_dir, exist_ok=True)

    n = rig.n
    ratios = torch.full((n,), args_cli.ratio, device=rig.dev)
    best = {"cov": -1.0}
    worst = {"cov": 2.0}
    trial = 0
    for rnd in range(args_cli.rounds):
        bank_idx = torch.randint(0, rig.bank_size, (n,), device=rig.dev)
        yaw = torch.rand(n, device=rig.dev) * 2.0 * torch.pi
        mirror = torch.rand(n, device=rig.dev) < 0.5
        idx_top = rig.restore_hang(bank_idx, yaw, mirror)
        rig.sim_steps(args_cli.settle_after_restore, drag=sc.SETTLE_DRAG)

        idx_move = torch.randint(0, rig.p, (n,), device=rig.dev)
        attached = rig.attach_at_particles(idx_move, slot=1)
        info = rig.stretch(top_slot=0, move_slot=1,
                           idx_top=idx_top, idx_move=idx_move,
                           target_ratio=ratios)
        meas = rig.measure(idx_top, idx_move, rig.anchor[:, 2])
        log.add_round(
            "h3", args_cli.seed, 1, trial,
            {"bank_idx": bank_idx, "yaw": yaw, "mirror": mirror,
             "idx_top": idx_top, "idx_move": idx_move,
             "target_ratio": ratios,
             "settle_steps": info["settle_steps"],
             "stretch_steps": info["stretch_steps"],
             "x_sign": info["x_sign"],
             "attached_ok": attached,
             "valid": attached & torch.isfinite(meas["cov_plane"])},
            meas, rig, save_states=False)

        # Keep only the running best / worst full states for the renders.
        cov = meas["cov_plane"]
        for tag, sel, store in (("best", cov.argmax(), best),
                                ("worst", cov.argmin(), worst)):
            c = float(cov[sel])
            if (tag == "best" and c > store["cov"] and bool(attached[sel])) or \
               (tag == "worst" and c < store["cov"] and bool(attached[sel])):
                store["cov"] = c
                torch.save(
                    {"pos": (rig.cloth.nodal_pos_w[sel]
                             - rig.anchor[sel]).half().cpu(),
                     "cov_plane": c, "method": "h3", "stage": 1,
                     "trial": trial + int(sel)},
                    os.path.join(states_dir, f"h3_{tag}.pt"))

        print(f"[h3 round={rnd}] coverage median {cov.median():.3f} IQR "
              f"[{cov.quantile(0.25):.3f}, {cov.quantile(0.75):.3f}] "
              f"running best {best['cov']:.3f} worst {worst['cov']:.3f}",
              flush=True)
        log.status("h3")
        trial += n

    rig.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

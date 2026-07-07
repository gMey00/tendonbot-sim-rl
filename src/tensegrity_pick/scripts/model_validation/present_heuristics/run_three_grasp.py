"""Heuristic 2 — 3-grasp with regrasp (+ optional iterated regrasp, extras).

Protocol per trial (batched over envs):

  stage 1 : restore a random hanging-bank state at the study anchor (first
            grasp = bank particle) → second grasp at the LOWEST point →
            HORIZONTAL presentation stretch (up to the holder's height, out
            along camera-plane x) → settle → measure  (== heuristic 1; the
            stage-1 rows give a PAIRED H1 sample for the comparison)
  stage k : the OLDER grasp releases; the newer grasp KEEPS HOLDING where the
            previous stretch left it (already at height) → damped settle
            (shirt re-hangs from it) → regrasp at the NEW lowest point →
            horizontal stretch in the OPPOSITE x direction of the previous
            stage → measure

``--regrasps 1`` is the task-prompt heuristic 2; ``--regrasps 3`` iterates
the cycle, alternating stretch directions (extra method).

Usage (from src/tensegrity_pick, env_isaaclab active)::

    PYTHONUNBUFFERED=1 python scripts/model_validation/present_heuristics/run_three_grasp.py \
        --headless --num_envs 48 --rounds 5 --ratio 1.05 --regrasps 1 --seed 22

    # iterated-regrasp extra (measures every stage):
    ... run_three_grasp.py --headless --num_envs 48 --rounds 3 --regrasps 3 \
        --csv .../present_h4_iterated.csv --method h4_iter --seed 44

Output: doc/reports/data/present_h2_three_grasp.csv (stage column: 1 = the
paired naive result, 2 = after the regrasp, 3+ = further iterations) + final
states per stage under outputs/present_heuristics/states/<method>/.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H2: 3-grasp regrasp study.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--ratio", type=float, default=1.05)
parser.add_argument("--regrasps", type=int, default=1,
                    help="number of drop→re-hang→lowest-point regrasp cycles")
parser.add_argument("--seed", type=int, default=22)
parser.add_argument("--settle_after_restore", type=int, default=45)
parser.add_argument("--method", type=str, default="h2")
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
        REPO, "doc", "reports", "data", "present_h2_three_grasp.csv")
    states_dir = os.path.join("outputs", "present_heuristics", "states",
                              args_cli.method)
    log = sc.TrialLog(csv_path, states_dir)

    n = rig.n
    ratios = torch.full((n,), args_cli.ratio, device=rig.dev)
    trial = 0
    for rnd in range(args_cli.rounds):
        bank_idx = torch.randint(0, rig.bank_size, (n,), device=rig.dev)
        yaw = torch.rand(n, device=rig.dev) * 2.0 * torch.pi
        mirror = torch.rand(n, device=rig.dev) < 0.5
        idx_top = rig.restore_hang(bank_idx, yaw, mirror)
        rig.sim_steps(args_cli.settle_after_restore, drag=sc.SETTLE_DRAG)
        top_slot, move_slot = 0, 1

        idx_move = rig.lowest_particle()
        attached = rig.attach_at_particles(idx_move, slot=move_slot)
        info = rig.stretch(top_slot, move_slot, idx_top, idx_move, ratios)
        meas = rig.measure(idx_top, idx_move, rig.anchor[:, 2])
        valid = attached & torch.isfinite(meas["cov_plane"])

        def log_stage(stage: int) -> None:
            log.add_round(
                args_cli.method, args_cli.seed, stage, trial,
                {"bank_idx": bank_idx, "yaw": yaw, "mirror": mirror,
                 "idx_top": idx_top, "idx_move": idx_move,
                 "target_ratio": ratios,
                 "settle_steps": info["settle_steps"],
                 "stretch_steps": info["stretch_steps"],
                 "x_sign": info["x_sign"],
                 "attached_ok": attached, "valid": valid},
                meas, rig)
            cov = meas["cov_plane"]
            print(f"[{args_cli.method} round={rnd} stage={stage}] coverage "
                  f"median {cov.median():.3f} IQR "
                  f"[{cov.quantile(0.25):.3f}, {cov.quantile(0.75):.3f}]",
                  flush=True)

        log_stage(1)

        for it in range(args_cli.regrasps):
            # The OLDER grasp releases; the newer grasp keeps holding where
            # the previous stretch left it (already at anchor height).
            rig.cloth.detach(rig.all_ids, slot=top_slot)
            rig.hold_pos[top_slot] = None
            resettle = rig.settle()
            top_slot, move_slot = move_slot, top_slot
            idx_top = idx_move
            # Regrasp at the new lowest point; stretch the OTHER way.
            idx_move = rig.lowest_particle()
            attached = attached & rig.attach_at_particles(idx_move, slot=move_slot)
            info = rig.stretch(top_slot, move_slot, idx_top, idx_move, ratios,
                               x_sign=-info["x_sign"])
            info["settle_steps"] += resettle
            meas = rig.measure(idx_top, idx_move, rig.anchor[:, 2])
            valid = attached & torch.isfinite(meas["cov_plane"])
            log_stage(2 + it)

        log.status(args_cli.method)
        trial += n

    rig.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

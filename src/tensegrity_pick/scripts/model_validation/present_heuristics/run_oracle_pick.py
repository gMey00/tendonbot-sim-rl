"""Extra method — oracle-guided second grasp (pair-map policy playback).

Protocol per trial (batched over envs): restore a random hanging-bank state at
the study anchor (first grasp = bank particle, a random garment point exactly
like H1) → classify the first grasp's garment region → look up the BEST
second-grasp region from the pair-map oracle (``--pair_json``, produced by the
H3 analysis) → grasp the particle at that region's landmark (choosing the
left/right variant farther from the first grasp in rest coords) →
HORIZONTAL presentation stretch → settle → measure.

This measures how much of the oracle upper bound a REGION-TARGETING policy
recovers vs the lowest-point heuristic — ground-truth particle lookup stands
in for perception, so it is an upper bound for any real second-grasp chooser.

Usage (from src/tensegrity_pick, env_isaaclab active)::

    PYTHONUNBUFFERED=1 python scripts/model_validation/present_heuristics/run_oracle_pick.py \
        --headless --num_envs 48 --rounds 5 --ratio 1.05 --seed 55 \
        --pair_json ../../doc/reports/data/present_oracle_policy.json

``pair_json`` maps first-grasp mirror-class → best second-grasp mirror-class,
e.g. {"collar": "hem_c", "shoulder": "hem_corner", ...}.

Output: doc/reports/data/present_h5_oracle_pick.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Oracle-guided second grasp.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--ratio", type=float, default=1.05)
parser.add_argument("--seed", type=int, default=55)
parser.add_argument("--settle_after_restore", type=int, default=45)
parser.add_argument("--pair_json", type=str, default=None,
                    help="pair-map policy (required unless --fixed_target)")
parser.add_argument("--fixed_target", type=str, default=None,
                    help="override the policy with one fixed second-grasp "
                         "mirror-class (e.g. 'shoulder' for the scripted "
                         "shoulder↔shoulder presentation baseline)")
parser.add_argument("--first_regions", type=str, nargs="+", default=None,
                    help="restrict FIRST grasps to these mirror-classes by "
                         "filtering the bank on its anchor particle (e.g. "
                         "'shoulder' to always start hanging from a shoulder)")
parser.add_argument("--method", type=str, default="h5_oracle")
parser.add_argument("--csv", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as sc  # noqa: E402
from regions import REGION_LANDMARKS, REGION_NAMES, assign_regions  # noqa: E402

REPO = "/home/robot/studentische-arbeiten"

_SYM_OF = {"collar": "collar", "shoulder_l": "shoulder", "shoulder_r": "shoulder",
           "sleeve_l": "sleeve", "sleeve_r": "sleeve", "chest": "chest",
           "side_l": "side", "side_r": "side", "belly": "belly",
           "hem_l": "hem_corner", "hem_r": "hem_corner", "hem_c": "hem_c"}


def main() -> None:
    if args_cli.fixed_target is None:
        assert args_cli.pair_json, "--pair_json required unless --fixed_target"
        with open(args_cli.pair_json) as fh:
            policy = json.load(fh)  # mirror-class -> mirror-class
    else:
        policy = {cls: args_cli.fixed_target for cls in set(_SYM_OF.values())}

    rig = sc.StudyRig(args_cli, seed=args_cli.seed)
    csv_path = args_cli.csv or os.path.join(
        REPO, "doc", "reports", "data", "present_h5_oracle_pick.csv")
    states_dir = os.path.join("outputs", "present_heuristics", "states",
                              args_cli.method)
    log = sc.TrialLog(csv_path, states_dir)

    # Landmark-nearest particle per CONCRETE region (the deterministic
    # "grasp the middle of that region" target).
    fr = rig.flat_rest.cpu().numpy()
    reg = assign_regions(fr[:, 0], fr[:, 1])
    lm_particle: dict[str, int] = {}
    for i, name in enumerate(REGION_NAMES):
        pts = fr[:, :2]
        d = ((pts - np.array(REGION_LANDMARKS[name])) ** 2).sum(axis=1)
        d[reg != i] = np.inf
        lm_particle[name] = int(d.argmin())
    # Candidate target particles per mirror-class.
    class_candidates: dict[str, list[int]] = {}
    for name, cls in _SYM_OF.items():
        class_candidates.setdefault(cls, []).append(lm_particle[name])

    flat = rig.flat_rest
    n = rig.n
    ratios = torch.full((n,), args_cli.ratio, device=rig.dev)

    # Optional first-grasp restriction: keep only bank states whose pinned
    # particle lies in the requested mirror-classes.
    bank_pool = torch.arange(rig.bank_size, device=rig.dev)
    if args_cli.first_regions:
        axy = rig.rest_xy(rig.bank_anchor_idx).cpu().numpy()
        cls = [_SYM_OF[REGION_NAMES[r]]
               for r in assign_regions(axy[:, 0], axy[:, 1])]
        keep = torch.tensor([c in args_cli.first_regions for c in cls],
                            device=rig.dev)
        bank_pool = bank_pool[keep]
        print(f"[h5] first-grasp filter {args_cli.first_regions}: "
              f"{bank_pool.numel()}/{rig.bank_size} bank states")
        assert bank_pool.numel() > 0

    trial = 0
    for rnd in range(args_cli.rounds):
        bank_idx = bank_pool[torch.randint(0, bank_pool.numel(), (n,),
                                           device=rig.dev)]
        yaw = torch.rand(n, device=rig.dev) * 2.0 * torch.pi
        mirror = torch.rand(n, device=rig.dev) < 0.5
        idx_top = rig.restore_hang(bank_idx, yaw, mirror)
        rig.sim_steps(args_cli.settle_after_restore, drag=sc.SETTLE_DRAG)

        # Policy lookup: first-grasp class -> target class -> the concrete
        # candidate particle with the larger rest distance to the first grasp.
        top_xy = rig.rest_xy(idx_top).cpu().numpy()
        top_cls = [
            _SYM_OF[REGION_NAMES[r]]
            for r in assign_regions(top_xy[:, 0], top_xy[:, 1])
        ]
        idx_move = torch.empty(n, dtype=torch.long, device=rig.dev)
        for e in range(n):
            cands = class_candidates[policy[top_cls[e]]]
            dists = [float((flat[idx_top[e]] - flat[c]).norm()) for c in cands]
            idx_move[e] = cands[int(np.argmax(dists))]

        attached = rig.attach_at_particles(idx_move, slot=1)
        info = rig.stretch(top_slot=0, move_slot=1,
                           idx_top=idx_top, idx_move=idx_move,
                           target_ratio=ratios)
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
        print(f"[{args_cli.method} round={rnd}] coverage median {cov.median():.3f} IQR "
              f"[{cov.quantile(0.25):.3f}, {cov.quantile(0.75):.3f}]", flush=True)
        log.status(args_cli.method)
        trial += n

    rig.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Keypoint-guided presentation grasps — can the ClothesNet keypoints be good
grasp targets?

The garment ships 10 semantic keypoints (``markers.py``: collar / shoulders /
sleeve ends / armpits / hem corners / centre).  This script measures whether
grasping AT those keypoints yields good presentations, in two modes:

* ``--mode second`` — the actionable case: FIRST grasp a random bank hang
  (as H1/H3), SECOND grasp a keypoint (cycled uniformly), horizontal stretch.
  Answers "which keypoint is a good second-grasp / regrasp target", directly
  comparable to H1's lowest-point rule and the oracle-guided pick.

* ``--mode pairs`` — pure "keypoint sampling": grasp TWO keypoints.  The first
  grasp is the bank hang whose pinned anchor is nearest keypoint A
  (``KEYPOINT_NEAREST_BANK``); the second is keypoint B.  Sampling A, B
  uniformly over the 10 keypoints (A != B) builds a keypoint-pair map — does a
  chooser restricted to the 10 keypoints reach the hem↔hem / oracle quality?

Grasping the second point BY PARTICLE INDEX is invariant to the bank's yaw +
mirror augmentation (the same material point is captured wherever it moved),
so both modes keep full augmentation.  Regions/keypoint ids are recovered
offline from the logged rest coordinates + ``markers.py``.

Usage (from src/tensegrity_pick, env_isaaclab active)::

    PYTHONUNBUFFERED=1 python scripts/model_validation/present_heuristics/run_keypoint_pick.py \
        --headless --num_envs 64 --mode second --rounds 12 --ratio 1.05 --seed 44

    ... run_keypoint_pick.py --headless --num_envs 64 --mode pairs --rounds 24 --seed 45

    # smoke: ... --mode second --num_envs 8 --rounds 1 --csv /tmp/hkp_smoke.csv

Output: doc/reports/data/present_hk_second.csv / present_hk_pairs.csv.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keypoint-guided presentation grasps.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--mode", choices=["second", "pairs"], default="second")
parser.add_argument("--rounds", type=int, default=12)
parser.add_argument("--ratio", type=float, default=1.05)
parser.add_argument("--seed", type=int, default=44)
parser.add_argument("--settle_after_restore", type=int, default=45)
parser.add_argument("--csv", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markers as mk  # noqa: E402
import study_common as sc  # noqa: E402
from regions import X_CENTER  # noqa: E402  (garment symmetry axis)

REPO = "/home/robot/studentische-arbeiten"


def main() -> None:
    rig = sc.StudyRig(args_cli, seed=args_cli.seed)
    method = "hk_second" if args_cli.mode == "second" else "hk_pairs"
    default = f"present_{'hk_second' if args_cli.mode == 'second' else 'hk_pairs'}.csv"
    csv_path = args_cli.csv or os.path.join(REPO, "doc", "reports", "data", default)
    log = sc.TrialLog(csv_path, states_dir=None)
    states_dir = os.path.join("outputs", "present_heuristics", "states", method)
    os.makedirs(states_dir, exist_ok=True)

    n, dev = rig.n, rig.dev
    ratios = torch.full((n,), args_cli.ratio, device=dev)
    fx = rig.flat_rest[:, 0]                                     # flat-rest x
    kp_idx = torch.tensor(mk.KEYPOINT_IDX, device=dev)          # [10] shipped
    kp_mir = torch.tensor(mk.KEYPOINT_MIRROR_IDX, device=dev)   # [10] mirror
    kp_bank = torch.tensor(mk.KEYPOINT_NEAREST_BANK, device=dev)  # [10] bank ids
    # Left/right variant of each keypoint (shipped or its mirror partner), so
    # the second grasp is taken from the side OPPOSITE the first grasp — same
    # convention as the stratified pair map, and a fair keypoint-sampling test.
    kp_left = torch.where(fx[kp_idx] < X_CENTER, kp_idx, kp_mir)   # [12]
    kp_right = torch.where(fx[kp_idx] >= X_CENTER, kp_idx, kp_mir)  # [12]
    nkp = len(kp_idx)
    best = {"cov": -1.0}
    worst = {"cov": 2.0}
    trial = 0

    for rnd in range(args_cli.rounds):
        yaw = torch.rand(n, device=dev) * 2.0 * torch.pi
        mirror = torch.rand(n, device=dev) < 0.5
        if args_cli.mode == "second":
            bank_idx = torch.randint(0, rig.bank_size, (n,), device=dev)
            kb = torch.randint(0, nkp, (n,), device=dev)         # second keypoint
        else:
            ka = torch.randint(0, nkp, (n,), device=dev)         # first keypoint
            kb = (ka + torch.randint(1, nkp, (n,), device=dev)) % nkp  # != ka
            bank_idx = kp_bank[ka]
        # second grasp = keypoint kb on the side opposite the first grasp.
        s1 = fx[rig.bank_anchor_idx[bank_idx]]                    # [n] first side
        idx_move = torch.where(s1 >= X_CENTER, kp_left[kb], kp_right[kb])

        idx_top = rig.restore_hang(bank_idx, yaw, mirror)
        rig.sim_steps(args_cli.settle_after_restore, drag=sc.SETTLE_DRAG)
        attached = rig.attach_at_particles(idx_move, slot=1)
        info = rig.stretch(top_slot=0, move_slot=1, idx_top=idx_top,
                           idx_move=idx_move, target_ratio=ratios)
        meas = rig.measure(idx_top, idx_move, rig.anchor[:, 2])
        valid = attached & torch.isfinite(meas["cov_plane"])
        log.add_round(
            method, args_cli.seed, 1, trial,
            {"bank_idx": bank_idx, "yaw": yaw, "mirror": mirror,
             "idx_top": idx_top, "idx_move": idx_move, "target_ratio": ratios,
             "settle_steps": info["settle_steps"],
             "stretch_steps": info["stretch_steps"], "x_sign": info["x_sign"],
             "attached_ok": attached, "valid": valid},
            meas, rig, save_states=False)

        cov = meas["cov_plane"]
        for tag, sel, store in (("best", cov.argmax(), best),
                                ("worst", cov.argmin(), worst)):
            c = float(cov[sel])
            if (tag == "best" and c > store["cov"] and bool(valid[sel])) or \
               (tag == "worst" and c < store["cov"] and bool(valid[sel])):
                store["cov"] = c
                torch.save(
                    {"pos": (rig.cloth.nodal_pos_w[sel] - rig.anchor[sel]).half().cpu(),
                     "cov_plane": c, "method": method, "stage": 1,
                     "trial": trial + int(sel)},
                    os.path.join(states_dir, f"{method}_{tag}.pt"))
        print(f"[{method} round={rnd}] median {cov.median():.3f} "
              f"best {best['cov']:.3f}", flush=True)
        log.status(method)
        trial += n

    rig.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

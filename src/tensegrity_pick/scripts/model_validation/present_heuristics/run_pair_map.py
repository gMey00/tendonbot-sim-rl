"""Heuristic 3 — grasp-pair quality map (the brute-force oracle).

Protocol per trial (batched over envs): restore a hanging-bank state at the
study anchor — the FIRST grasp is the bank's pinned particle — then attach the
second grasp at another particle of the hanging shirt, HORIZONTAL presentation
stretch (up to the holder's height, out along the camera-plane x axis) to a
fixed tautness, settle, measure.  Keyed by both particles' flat-rest
coordinates, the trials build the grasp-pair quality map (doc/TODO.md Task 2,
"grasp-pair oracle"): it upper-bounds all heuristics and labels the good
second-grasp regions.

Two sampling modes:

* ``--random`` (the original oracle): first grasp = a random bank state (its
  anchor uniform over the garment), second grasp = a UNIFORM-RANDOM particle.
  Uniform over particles ≈ uniform over area, so region aggregates are
  area-weighted — but per-cell counts are region-proportional, so rare
  region→region cells (collar-first, belly-first, …) end up thin.

* ``--stratified`` (default): sample ``--per_cell`` trials for EVERY
  first-region → second-region cell of the 8×8 folded-region grid.  The first
  grasp is a bank state whose pinned anchor falls in the target first region;
  the second grasp is a random particle of the target second region (pools from
  ``markers.py``).  For a two-sided first region (shoulder / sleeve / side /
  hem_corner) the second grasp is drawn from the OPPOSITE left/right half of
  the second region — never the same side, which would be a degenerate
  same-corner short chord (matching §5.2's oracle-pick "l/r variant farther
  from the first grasp").  Guarantees ≥ ``per_cell`` valid trials per cell —
  the balanced map the study's per-cell comparison needs.

Regions are recovered OFFLINE from the logged rest coordinates (identical
schema to the random run), so both CSVs feed the same analysis.

Usage (from src/tensegrity_pick, env_isaaclab active)::

    # stratified map, 100 valid trials per region→region cell:
    PYTHONUNBUFFERED=1 python scripts/model_validation/present_heuristics/run_pair_map.py \
        --headless --num_envs 64 --stratified --per_cell 100 --ratio 1.05 --seed 33

    # original random oracle (2880 trials):
    ... run_pair_map.py --headless --num_envs 64 --random --rounds 45 --seed 33

    # smoke test:
    ... run_pair_map.py --headless --num_envs 8 --stratified --per_cell 8 \
        --csv /tmp/h3s_smoke.csv

Output: doc/reports/data/present_h3s_stratified.csv (stratified) or
present_h3_pair_map.csv (random), + full states only for the running
best/worst trials.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H3: grasp-pair oracle map.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rounds", type=int, default=40,
                    help="random mode: number of batched rounds")
parser.add_argument("--random", dest="stratified", action="store_false",
                    help="original uniform-random pair sampling")
parser.add_argument("--stratified", dest="stratified", action="store_true",
                    help="balanced per region→region sampling (default)")
parser.set_defaults(stratified=True)
parser.add_argument("--per_cell", type=int, default=100,
                    help="stratified mode: min valid trials per region→region cell")
parser.add_argument("--ratio", type=float, default=1.05)
parser.add_argument("--seed", type=int, default=33)
parser.add_argument("--settle_after_restore", type=int, default=45)
parser.add_argument("--resume", action="store_true",
                    help="stratified mode: reuse cells already complete in --csv")
parser.add_argument("--csv", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import markers as mk  # noqa: E402
import study_common as sc  # noqa: E402
from regions import X_CENTER  # noqa: E402  (garment symmetry axis)

REPO = "/home/robot/studentische-arbeiten"


def one_round(rig, log, method, trial, ratios, best, worst, states_dir,
              bank_idx, yaw, mirror, idx_move):
    """Restore → attach second grasp → horizontal stretch → settle → measure →
    log one batched round.  Returns the per-env valid mask."""
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
    return valid, cov


def run_random(rig, log, ratios, best, worst, states_dir):
    n = rig.n
    trial = 0
    for rnd in range(args_cli.rounds):
        bank_idx = torch.randint(0, rig.bank_size, (n,), device=rig.dev)
        yaw = torch.rand(n, device=rig.dev) * 2.0 * torch.pi
        mirror = torch.rand(n, device=rig.dev) < 0.5
        idx_move = torch.randint(0, rig.p, (n,), device=rig.dev)
        valid, cov = one_round(rig, log, "h3", trial, ratios, best, worst,
                               states_dir, bank_idx, yaw, mirror, idx_move)
        print(f"[h3 round={rnd}] median {cov.median():.3f} "
              f"best {best['cov']:.3f} worst {worst['cov']:.3f}", flush=True)
        log.status("h3")
        trial += n


# Two-sided folded classes: a grasp there has a definite left/right side, so
# the PARTNER grasp is taken from the OPPOSITE side (never the same side — that
# would be a degenerate same-corner/same-sleeve short chord).  The central
# classes have no side, so their partner is unrestricted.  This matches the
# oracle-pick convention (§5.2: "the l/r variant farther from the first grasp").
_CENTRAL = {"collar", "chest", "belly", "hem_c"}


def run_stratified(rig, log, ratios, best, worst, states_dir):
    """Fill every folded first-region → second-region cell to --per_cell valid.

    For a two-sided first region the second grasp is drawn from the opposite
    left/right half of the second region (mirror-invariant: the anchor's flat-
    rest x-sign flips with the bank mirror exactly as the candidate's does, so
    "opposite flat-rest x-sign" == "opposite physical side" under any mirror).
    """
    n = rig.n
    dev = rig.dev
    bank_anchor = rig.bank_anchor_idx.cpu().numpy()
    fx = rig.flat_rest[:, 0]                              # [P] flat-rest x
    K = len(mk.SYM_CLASSES)
    # Per-region pools (numpy → device tensors), split by flat-rest side.
    bank_pool = {r: mk.bank_states_in_region(r, bank_anchor) for r in range(K)}
    part_pool = {r: torch.tensor(mk.region_particles(r), device=dev)
                 for r in range(K)}
    left_pool, right_pool = {}, {}
    for r in range(K):
        p = part_pool[r]
        left_pool[r] = p[fx[p] < X_CENTER]
        right_pool[r] = p[fx[p] >= X_CENTER]
    # Resume: count already-valid trials per cell in the existing CSV.
    done = np.zeros((K, K), dtype=int)
    if args_cli.resume and log.rows:
        import pandas as pd
        df = pd.DataFrame(log.rows)
        df = df[df["valid"].astype(int) == 1]
        r1 = mk.PARTICLE_REGION[df["idx_top"].astype(int).values]
        r2 = mk.PARTICLE_REGION[df["idx_move"].astype(int).values]
        for a, b in zip(r1, r2):
            done[a, b] += 1
        print(f"[h3s] resume: {int(done.sum())} valid trials already logged",
              flush=True)
    trial = len(log.rows)
    for r1 in range(K):
        pool = bank_pool[r1]
        if len(pool) == 0:
            print(f"[h3s] no bank states in first-region "
                  f"{mk.SYM_CLASSES[r1]} — skipping row", flush=True)
            continue
        pool_t = torch.tensor(pool, device=dev)
        r1_sided = (mk.SYM_CLASSES[r1] not in _CENTRAL
                    and len(left_pool[r1]) and len(right_pool[r1]))
        for r2 in range(K):
            p2 = part_pool[r2]
            l2, r2p = left_pool[r2], right_pool[r2]
            opp = r1_sided and len(l2) > 0 and len(r2p) > 0
            target = args_cli.per_cell
            got = int(done[r1, r2])
            while got < target:
                bank_idx = pool_t[torch.randint(0, len(pool_t), (n,), device=dev)]
                yaw = torch.rand(n, device=dev) * 2.0 * torch.pi
                mirror = torch.rand(n, device=dev) < 0.5
                if opp:
                    # first grasp side = which side of X_CENTER its anchor is;
                    # take the second grasp from the OPPOSITE side of region r2.
                    s1 = fx[rig.bank_anchor_idx[bank_idx]]        # [n]
                    li = l2[torch.randint(0, len(l2), (n,), device=dev)]
                    ri = r2p[torch.randint(0, len(r2p), (n,), device=dev)]
                    idx_move = torch.where(s1 >= X_CENTER, li, ri)  # opposite side
                else:
                    idx_move = p2[torch.randint(0, len(p2), (n,), device=dev)]
                valid, cov = one_round(rig, log, "h3s", trial, ratios, best,
                                       worst, states_dir, bank_idx, yaw,
                                       mirror, idx_move)
                got += int(valid.sum())
                trial += n
            done[r1, r2] = got
            print(f"[h3s] cell {mk.SYM_CLASSES[r1]:11s}->"
                  f"{mk.SYM_CLASSES[r2]:11s} valid={got} "
                  f"(running best {best['cov']:.3f})", flush=True)
        log.status("h3s")


def main() -> None:
    rig = sc.StudyRig(args_cli, seed=args_cli.seed)
    default = "present_h3s_stratified.csv" if args_cli.stratified \
        else "present_h3_pair_map.csv"
    csv_path = args_cli.csv or os.path.join(REPO, "doc", "reports", "data", default)
    log = sc.TrialLog(csv_path, states_dir=None)
    if args_cli.resume and os.path.exists(csv_path):
        import pandas as pd
        log.rows = pd.read_csv(csv_path).to_dict("records")
    tag = "h3s" if args_cli.stratified else "h3"
    states_dir = os.path.join("outputs", "present_heuristics", "states", tag)
    os.makedirs(states_dir, exist_ok=True)

    ratios = torch.full((rig.n,), args_cli.ratio, device=rig.dev)
    best = {"cov": -1.0}
    worst = {"cov": 2.0}
    if args_cli.stratified:
        run_stratified(rig, log, ratios, best, worst, states_dir)
    else:
        run_random(rig, log, ratios, best, worst, states_dir)
    rig.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

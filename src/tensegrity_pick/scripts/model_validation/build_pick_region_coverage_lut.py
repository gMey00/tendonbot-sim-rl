"""Region → expected-Task-2-coverage lookup table for the shirt_pick S2 head.

Derives, OFFLINE and without any simulation, the terminal-bonus lookup used by
the learned grasp-point refinement head (Stage-2 roadmap, S2/T1): for each
FIRST-grasp region (8 symmetrized study classes), the presentation coverage
achievable with the study's best SECOND-grasp partner region — the "best
partner per first grasp" from the stratified pair map
(`doc/reports/data/present_h3s_stratified.csv`, heuristics study §4).

Method (matches `present_heuristics/plot_study.py::fig_best_partner` exactly):
  * valid == 1 trials only;
  * first/second grasp region = `regions.assign_regions` on the logged
    flat-rest coords (`rest_x_top/rest_y_top`, `rest_x_move/rest_y_move`),
    folded left/right into the 8 sym classes;
  * cell statistic = MEDIAN `cov_plane` (camera-plane coverage, the study's
    primary metric); cells with < --min_n trials are ignored;
  * LUT[first] = max over second-grasp cells (the oracle best partner).

Prints the table + provenance and the python constant for
`shirt_pick/mdp/grasp_head.py`; with the constant already present there, it is
re-validated against the CSV-derived values (drift check).

Usage (plain python, no isaaclab needed)::

    cd src/tensegrity_pick
    python scripts/model_validation/build_pick_region_coverage_lut.py [--min_n 25]
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA = os.path.normpath(os.path.join(REPO, "..", "..", "doc", "reports", "data"))
CSV = os.path.join(DATA, "present_h3s_stratified.csv")

# Same folding as the study (plot_study._SYM_OF / present_markers.sym_classes).
SYM_CLASSES = ["collar", "shoulder", "sleeve", "chest", "side", "belly", "hem_corner", "hem_c"]
SYM_OF = {
    "collar": "collar", "shoulder_l": "shoulder", "shoulder_r": "shoulder",
    "sleeve_l": "sleeve", "sleeve_r": "sleeve", "chest": "chest",
    "side_l": "side", "side_r": "side", "belly": "belly",
    "hem_l": "hem_corner", "hem_r": "hem_corner", "hem_c": "hem_c",
}


def _load_regions():
    path = os.path.join(HERE, "present_heuristics", "regions.py")
    spec = importlib.util.spec_from_file_location("regions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_lut(min_n: int) -> tuple[dict, pd.DataFrame]:
    regions = _load_regions()
    sym_lut = np.array([SYM_CLASSES.index(SYM_OF[n]) for n in regions.REGION_NAMES])

    df = pd.read_csv(CSV)
    n_raw = len(df)
    df = df[df.valid == 1].copy()
    top = sym_lut[regions.assign_regions(df.rest_x_top.values, df.rest_y_top.values)]
    mov = sym_lut[regions.assign_regions(df.rest_x_move.values, df.rest_y_move.values)]
    cov = df.cov_plane.values

    k = len(SYM_CLASSES)
    rows = []
    lut = {}
    for i in range(k):
        med = [np.median(cov[(top == i) & (mov == j)])
               if ((top == i) & (mov == j)).sum() >= min_n else np.nan
               for j in range(k)]
        counts = [int(((top == i) & (mov == j)).sum()) for j in range(k)]
        if np.all(np.isnan(med)):
            lut[SYM_CLASSES[i]] = float("nan")
            rows.append({"first": SYM_CLASSES[i], "best_second": "-", "cell_n": 0,
                         "coverage": float("nan"), "n_first": int((top == i).sum())})
            continue
        best = int(np.nanargmax(med))
        lut[SYM_CLASSES[i]] = round(float(med[best]), 4)
        rows.append({"first": SYM_CLASSES[i], "best_second": SYM_CLASSES[best],
                     "cell_n": counts[best], "coverage": round(float(med[best]), 4),
                     "n_first": int((top == i).sum())})
    table = pd.DataFrame(rows)
    table.attrs["provenance"] = {
        "csv": os.path.relpath(CSV, REPO), "rows_raw": n_raw, "rows_valid": len(df),
        "metric": "median cov_plane per (first, second) sym-class cell",
        "min_n": min_n, "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return lut, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_n", type=int, default=25,
                    help="min trials per cell (25 = plot_study stratified setting)")
    args = ap.parse_args()

    lut, table = build_lut(args.min_n)
    prov = table.attrs["provenance"]
    print("Provenance:", prov)
    print("\nBest-partner coverage per FIRST-grasp region "
          "(= S2 terminal-bonus lookup):\n")
    print(table.to_string(index=False))
    print("\nPython constant for shirt_pick/mdp/grasp_head.py:\n")
    print("REGION_COVERAGE_LUT = {")
    for k, v in lut.items():
        print(f'    "{k}": {v},')
    print("}")

    # Drift check against the constant embedded in the head module, if any.
    rewards_py = os.path.join(
        REPO, "source", "tensegrity_pick", "tensegrity_pick", "tasks",
        "manager_based", "shirt_pick", "mdp", "grasp_head.py")
    src = open(rewards_py).read() if os.path.isfile(rewards_py) else ""
    if "REGION_COVERAGE_LUT" in src:
        ns: dict = {}
        block = src.split("REGION_COVERAGE_LUT = ", 1)[1]
        block = block[: block.index("}") + 1]
        exec("REGION_COVERAGE_LUT = " + block, ns)  # noqa: S102 — own repo constant
        ok = all(
            (np.isnan(v) and np.isnan(ns["REGION_COVERAGE_LUT"][k]))
            or abs(ns["REGION_COVERAGE_LUT"][k] - v) < 5e-4
            for k, v in lut.items()
        )
        print("\ngrasp_head.py REGION_COVERAGE_LUT check:", "MATCH" if ok else "*** DRIFT ***")
    else:
        print("\n(grasp_head.py has no REGION_COVERAGE_LUT yet — paste the constant above.)")


if __name__ == "__main__":
    main()

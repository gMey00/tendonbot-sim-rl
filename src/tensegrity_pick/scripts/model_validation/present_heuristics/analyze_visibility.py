"""Occlusion-aware visibility check over the saved per-trial states (offline).

``silhouette_coverage`` counts overlapping fabric layers ONCE: a sleeve draped
in front of the torso does not lower coverage, although the torso patch behind
it is not inspectable from either the front or the back camera.  This script
computes ``two_sided_visible_fraction`` (cloth_metrics.py: per raster cell the
front camera sees the nearest depth layer, the back camera the farthest;
middle layers are hidden) for EVERY saved final state of the scripted methods
and answers: does the method ranking of the study survive an occlusion-aware
metric — with only the front + back inspection cameras the task already has?

Pure offline (torch + the saved ``outputs/present_heuristics/states/`` dumps,
no Isaac).  Run from src/tensegrity_pick in the conda env::

    python scripts/model_validation/present_heuristics/analyze_visibility.py

Outputs: doc/reports/data/present_visibility.csv (method, stage, trial,
cov_plane, vis_frac per saved state), the figure
figures/present_heuristics/visibility_check.png, and a markdown summary table
on stdout.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import load_metrics  # noqa: E402

REPO = "/home/robot/studentische-arbeiten"
DATA = os.path.join(REPO, "doc", "reports", "data")
FIGS = os.path.join(REPO, "doc", "reports", "figures", "present_heuristics")
STATES = os.path.join(REPO, "src", "tensegrity_pick", "outputs",
                      "present_heuristics", "states")
BANK = os.path.join(REPO, "res", "Props", "Cloth", "banks",
                    "tshirt_hanging_bank.pt")

# Method dirs with per-trial final states (measured stage per file) and the
# display label/stage filter: (dir, label, stage or None=all).
METHOD_DIRS = [
    ("h7_shoulder", "shoulder↔shoulder", None),
    ("h1", "H1 naive 2-grasp (all ratios)", None),
    ("h2", "H2 regrasp (stage 2)", 2),
    ("h2o", "H2 oracle regrasp (stage 2)", 2),
    ("h6_shake", "H1 + shake", None),
    ("h6_pulse", "H1 + tautness pulse", None),
    ("h5_oracle", "oracle-guided 2nd grasp", None),
    ("h8_hem", "hem↔hem @1.05", None),
    ("h8_hem110", "hem↔hem @1.10", None),
]

CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
       "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 170, "savefig.bbox": "tight",
})


def batched_vis(cm, pos_list: list[torch.Tensor], batch: int = 256):
    """cov_plane + vis_frac for a list of [P, 3] states (fp16 dumps)."""
    covs, viss = [], []
    ref = None
    fr = torch.load(os.path.join(DATA, "present_heuristics_flat_rest.pt"),
                    weights_only=False)
    ref = fr["ref_area"]
    for i in range(0, len(pos_list), batch):
        chunk = torch.stack([p.float() for p in pos_list[i:i + batch]])
        covs.append(cm.silhouette_coverage(chunk, ref))
        viss.append(cm.two_sided_visible_fraction(chunk))
    return torch.cat(covs).numpy(), torch.cat(viss).numpy()


def main() -> None:
    cm = load_metrics()
    os.makedirs(FIGS, exist_ok=True)
    rows = []
    summary = []
    for d, label, stage in METHOD_DIRS:
        files = sorted(glob.glob(os.path.join(STATES, d, "*.pt")))
        metas = [torch.load(f, weights_only=False) for f in files]
        if stage is not None:
            metas = [m for m in metas if m.get("stage") == stage]
        if not metas:
            print(f"[skip] {d}: no states")
            continue
        cov, vis = batched_vis(cm, [m["pos"] for m in metas])
        for m, c, v in zip(metas, cov, vis):
            rows.append({"dir": d, "label": label, "method": m["method"],
                         "stage": m["stage"], "trial": m["trial"],
                         "cov_saved": m["cov_plane"],
                         "cov_plane": float(c), "vis_frac": float(v)})
        summary.append((label, cov, vis))
        print(f"[{d}] n={len(metas)}  cov med {np.median(cov):.3f} "
              f"(saved {np.median([m['cov_plane'] for m in metas]):.3f})  "
              f"vis_frac med {np.median(vis):.3f}", flush=True)

    # Free-hang reference from the bank (single-point hangs).
    bank_pos = torch.load(BANK, weights_only=False)["pos"]
    cov_h, vis_h = batched_vis(cm, list(bank_pos))
    summary.insert(0, ("free 1-point hang", cov_h, vis_h))
    print(f"[hang] n={len(bank_pos)}  cov med {np.median(cov_h):.3f}  "
          f"vis_frac med {np.median(vis_h):.3f}")

    import csv
    out_csv = os.path.join(DATA, "present_visibility.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[vis] {len(rows)} rows → {out_csv}")

    # Figure: per-method vis_frac boxes + cov-vs-vis scatter.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 0.42 * len(summary) + 2.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.grid(axis="y", visible=False)
    for i, (label, cov, vis) in enumerate(summary):
        ax.boxplot([vis], positions=[i], widths=0.55, vert=False,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, lw=1.4),
                   whiskerprops=dict(color=INK2, lw=1.0),
                   capprops=dict(color=INK2, lw=1.0),
                   boxprops=dict(facecolor=CAT[0], edgecolor=INK2, lw=0.8,
                                 alpha=0.85))
        ax.text(np.median(vis), i + 0.36, f"{np.median(vis):.2f}",
                ha="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(summary)), [s[0] for s in summary])
    ax.set_xlabel("two-sided visible fraction (front + back cameras)")
    ax.set_title("Occlusion-aware visibility by method", fontsize=9.5)
    ax = axes[1]
    allc = np.concatenate([s[1] for s in summary[1:]])
    allv = np.concatenate([s[2] for s in summary[1:]])
    ax.scatter(allc, allv, s=4, color=CAT[0], alpha=0.2, edgecolors="none",
               rasterized=True)
    r = np.corrcoef(allc, allv)[0, 1]
    ax.set_xlabel("silhouette coverage (camera plane)")
    ax.set_ylabel("two-sided visible fraction")
    ax.set_title(f"coverage vs visibility, all saved trials (r = {r:.2f})",
                 fontsize=9.5)
    fig.savefig(os.path.join(FIGS, "visibility_check.png"))
    plt.close(fig)

    print("\n| method | n | vis_frac median | IQR | coverage median |")
    print("|---|---|---|---|---|")
    for label, cov, vis in summary:
        print(f"| {label} | {len(vis)} | {np.median(vis):.3f} "
              f"| {np.quantile(vis, .25):.3f}–{np.quantile(vis, .75):.3f} "
              f"| {np.median(cov):.3f} |")
    ranks_c = np.argsort([-np.median(s[1]) for s in summary])
    ranks_v = np.argsort([-np.median(s[2]) for s in summary])
    print(f"\nranking identical under both metrics: "
          f"{bool((ranks_c == ranks_v).all())}")
    print(f"figure → {os.path.join(FIGS, 'visibility_check.png')}")


if __name__ == "__main__":
    main()

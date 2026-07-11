"""Figures + stats for the Task-1→2 terminal bank (Stage-2 S0, offline).

Reads ``res/Props/Cloth/banks/tshirt_pick_terminal_bank.pt`` and produces the
thesis figure quantifying the gap the S2 grasp head must close: the
first-grasp-region distribution of the trained highest-point policy vs the
best-partner Task-2 coverage each region enables (offline LUT from the
stratified pair map — ``build_pick_region_coverage_lut.py``).

Writes ``doc/reports/figures/shirt_pick/first_grasp_region_distribution.png``
and prints the summary table (markdown).  No Isaac Sim required:

    cd src/tensegrity_pick
    python scripts/model_validation/plot_pick_terminal_bank.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_pick_region_coverage_lut import SYM_CLASSES, build_lut  # noqa: E402

REGION_COVERAGE_LUT, _lut_table = build_lut(min_n=25)  # recomputed from the CSV

REPO = "/home/robot/studentische-arbeiten"
BANK = os.path.join(REPO, "res", "Props", "Cloth", "banks", "tshirt_pick_terminal_bank.pt")
FIGS = os.path.join(REPO, "doc", "reports", "figures", "shirt_pick")
ORACLE_ARBITRARY = 0.713  # oracle-guided 2nd grasp, arbitrary 1st (study §6)

# Same reference palette as plot_study.py (dataviz skill instance).
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
       "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
NEUTRAL_BAR = "#b9b7b0"  # non-high-value bars (identity is on the axis)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 170, "savefig.bbox": "tight",
})

HIGH_VALUE = ("side", "hem_corner")


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    d = torch.load(BANK, map_location="cpu", weights_only=False)
    presented = d["presented"].bool()
    sym8 = d["sym8"][presented].numpy()
    n = presented.sum().item()

    counts = np.array([(sym8 == i).sum() for i in range(len(SYM_CLASSES))])
    frac = counts / max(n, 1)
    cov = np.array([REGION_COVERAGE_LUT[c] for c in SYM_CLASSES])
    exp_cov = float((frac * cov).sum())
    hv = sum(frac[SYM_CLASSES.index(c)] for c in HIGH_VALUE)

    # ── Summary table (markdown) ──────────────────────────────────────────
    print(f"Terminal bank: {len(d['presented'])} states, {n} presented "
          f"({d['meta']['checkpoint'].split('/')[-1]}, seed {d['meta']['seed']})")
    print(f"High-value-hold share (side + hem_corner): {hv:.3f}")
    print(f"Expected best-partner coverage of the distribution: {exp_cov:.3f} "
          f"(arbitrary-first oracle 0.713; hem-corner-first bound 0.820)\n")
    print("| first-grasp region | holds | share | best-partner coverage |")
    print("|---|---|---|---|")
    for i, c in enumerate(SYM_CLASSES):
        print(f"| {c} | {counts[i]} | {frac[i]:.3f} | {cov[i]:.3f} |")

    # ── Figure: two panels, shared region axis ────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.6, 3.2), sharey=False,
        gridspec_kw={"wspace": 0.28},
    )
    x = np.arange(len(SYM_CLASSES))
    colors = [CAT[0] if c in HIGH_VALUE else NEUTRAL_BAR for c in SYM_CLASSES]

    ax1.bar(x, frac, color=colors, width=0.62, zorder=3)
    for i, f in enumerate(frac):
        ax1.text(i, f + 0.006, f"{f:.2f}", ha="center", va="bottom",
                 fontsize=7.5, color=INK2)
    ax1.set_xticks(x, SYM_CLASSES, rotation=45, ha="right")
    ax1.set_ylabel("share of presented holds")
    ax1.set_title(f"Where the highest-point policy grasps first\n"
                  f"({n} presented terminal states, agent_12000)", fontsize=9)
    ax1.set_axisbelow(True)

    # Dot plot (not bars): the coverage axis is truncated to the data band,
    # and bar LENGTH would lie there — point POSITION does not.
    ax2.scatter(x, cov, s=52, color=colors, zorder=4)
    ax2.vlines(x, 0.6, cov, color=GRID, lw=1.0, zorder=2)
    for i, v in enumerate(cov):
        ax2.text(i, v + 0.007, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=7.5, color=INK2)
    ax2.axhline(ORACLE_ARBITRARY, color=INK2, lw=1.0, ls="--", zorder=3)
    ax2.text(len(x) - 0.4, ORACLE_ARBITRARY + 0.004,
             "arbitrary-first oracle 0.713", ha="right", va="bottom",
             fontsize=7.5, color=INK2)
    ax2.set_xticks(x, SYM_CLASSES, rotation=45, ha="right")
    ax2.set_ylim(0.6, 0.86)
    ax2.set_ylabel("best-partner Task-2 coverage")
    ax2.set_title("What each first-grasp region enables\n"
                  "(stratified pair-map LUT, best second grasp)", fontsize=9)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for i, c in enumerate(SYM_CLASSES):
            if c in HIGH_VALUE:
                ax.get_xticklabels()[i].set_color(CAT[0])

    fig.suptitle(
        f"Task-1 first-grasp distribution vs downstream value — high-value share "
        f"{hv:.2f}, expected coverage {exp_cov:.3f}",
        fontsize=9.5, y=1.04,
    )
    out = os.path.join(FIGS, "first_grasp_region_distribution.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nFigure: {out}")


if __name__ == "__main__":
    main()

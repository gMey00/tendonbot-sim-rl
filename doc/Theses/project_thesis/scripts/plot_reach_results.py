"""Reach-task figures for Results §5.2.

Produces:

* ``reach_convergence.png`` — single 2×2 figure showing total reward,
  position tracking, orientation tracking, and policy std deviation
  for all three tensegrity variants over training.
* ``reach_comparison.png``  — final-performance grouped bar chart
  (one bar per metric, three bars per group: PD / Tendon / Physical).
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

REACH_X_MAX: Final = 48_000  # last PD timestep — used as common x-cutoff


def _convergence_figure() -> None:
    fig, axes = plt.subplots(
        2, 2,
        figsize=c.pcfg.scaled(11, 7.0, scale=1.0),
        constrained_layout=True,
    )

    panels = [
        ("Reward_Totalrewardmean",
         "(a) Total reward (mean)",      "Reward",            True),
        ("Info_Episode_Reward_end_effector_position_tracking",
         "(b) Position tracking",         "Reward / step",     False),
        ("Info_Episode_Reward_end_effector_orientation_tracking",
         "(c) Orientation tracking",      "Reward / step",     False),
        ("Policy_Standarddeviation",
         "(d) Policy std deviation σ",    "σ",                 False),
    ]

    for ax, (metric, title, ylabel, show_legend) in zip(axes.flat, panels):
        c.plot_variants_curve(
            ax, "reach", metric,
            x_max=REACH_X_MAX,
            y_label=ylabel,
            title=title,
            show_legend=show_legend,
        )

    c.save(fig, c.RESULTS_FIG_DIR / "reach_convergence.png", tight=False)


def _comparison_figure() -> None:
    """Final converged-metric bar chart, one panel.

    Numbers are quoted from `tab:reach_results` in the thesis.
    """
    metrics = ("Total Reward", "Position\ntracking", "Orientation\ntracking",
               "Fine-grained")
    pd_vals  = ( 0.79, -0.006, -0.009, 0.087)
    tn_vals  = ( 0.76, -0.007, -0.012, 0.085)
    ph_vals  = (-0.46, -0.020, -0.044, 0.041)

    fig, ax = plt.subplots(
        figsize=c.pcfg.scaled(11, 4.0, scale=1.0),
        constrained_layout=True,
    )
    n = len(metrics)
    base = np.arange(n)
    bar_w = 0.27

    for offset, vals, label, color in (
        (-bar_w, pd_vals, "PD",       c.pcfg.FAPS_BLUE),
        (    0,  tn_vals, "Tendon",   c.pcfg.FAPS_GREEN),
        ( bar_w, ph_vals, "Physical", c.pcfg.PURPLE),
    ):
        bars = ax.bar(base + offset, vals, width=bar_w * 0.95,
                      color=color, edgecolor="white", linewidth=0.4,
                      label=label)
        for x, v in zip(base + offset, vals):
            va = "bottom" if v >= 0 else "top"
            dy = 0.01 if v >= 0 else -0.01
            ax.text(x, v + dy * abs(ax.get_ylim()[1] or 1),
                    f"{v:+.2f}" if abs(v) >= 0.05 else f"{v:+.3f}",
                    ha="center", va=va, fontsize=7, color="#333333")

    ax.axhline(0, color=c.pcfg.FAPS_DARK_GREY, linewidth=0.6)
    ax.set_xticks(base)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Reward / score")
    ax.set_title("Final reach-task performance — three tensegrity variants",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.set_axisbelow(True)

    c.save(fig, c.RESULTS_FIG_DIR / "reach_comparison.png", tight=False)


def main() -> None:
    c.init()
    _convergence_figure()
    _comparison_figure()


if __name__ == "__main__":
    main()

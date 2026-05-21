"""Reach-task figures for Results §5.2.

Reads metrics directly from the skrl TensorBoard event files declared in
:data:`common.RUNS`. Variants whose tfevents file is missing are skipped.

Produces:

* ``reach_convergence.png`` — 4-panel training-progress overview
  (total reward with min/max envelope, position-reached success,
  pose-reached success, policy σ). A single legend appears underneath
  the figure.
* ``reach_comparison.png``  — final-performance grouped bar chart over
  the same success metrics (last-10 % mean per run).
"""

from __future__ import annotations

from typing import Callable, Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

REACH_X_MAX: Final = 48_000  # common cutoff (PD variant training length)

# Binary reach-success rewards are logged as Episode_Reward terms with
# weight = 1e-6. Multiply by 1e6 to recover the underlying success rate.
SUCCESS_WEIGHT: Final = 1e-6


def _success_rate_panel(ax: plt.Axes, tag: str, *, title: str) -> bool:
    has = c.plot_variants_curve(
        ax, "reach", tag,
        x_max=REACH_X_MAX, y_label="Rate", title=title,
        value_scale=1.0 / SUCCESS_WEIGHT,
    )
    ax.set_ylim(0, 1.0)
    return has


def _convergence_figure() -> None:
    fig, axes = plt.subplots(
        4, 1,
        figsize=c.pcfg.scaled(11, 14.0, scale=1.0),
        constrained_layout=False,
    )

    # (a) total reward — mean line + min/max envelope per variant
    c.plot_envelope(
        axes[0], "reach",
        "Reward / Total reward (mean)",
        "Reward / Total reward (min)",
        "Reward / Total reward (max)",
        x_max=REACH_X_MAX,
    )
    axes[0].set_title(
        "(a) Total reward (mean ± min/max envelope)",
        fontsize=10, fontweight="bold", loc="left",
    )
    axes[0].set_ylabel("Reward")

    # (b) position-reached success rate
    _success_rate_panel(
        axes[1], "Info / Episode_Reward/position_reached",
        title=r"(b) Position-reached rate ($d_\mathrm{pos} < 5$ cm)",
    )

    # (c) full pose-reached success rate
    _success_rate_panel(
        axes[2], "Info / Episode_Reward/pose_reached",
        title="(c) Full pose-reached rate (position + orientation)",
    )

    # (d) policy std deviation
    c.plot_variants_curve(
        axes[3], "reach", "Policy / Standard deviation",
        x_max=REACH_X_MAX, y_label="σ",
        title="(d) Policy standard deviation",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    c.figure_bottom_legend(fig, axes, ncol=3, y=0.005)
    c.save(fig, c.RESULTS_FIG_DIR / "reach_convergence.png", tight=False)


def _comparison_figure() -> None:
    """Final-performance bar chart — total reward + three success rates."""
    metric_specs: list[tuple[str, str, Callable[[float], float]]] = [
        ("Total reward",  "Reward / Total reward (mean)",                 lambda v: v),
        ("Position\nreached",    "Info / Episode_Reward/position_reached",    lambda v: v / SUCCESS_WEIGHT),
        ("Orientation\nreached", "Info / Episode_Reward/orientation_reached", lambda v: v / SUCCESS_WEIGHT),
        ("Full pose\nreached",   "Info / Episode_Reward/pose_reached",        lambda v: v / SUCCESS_WEIGHT),
    ]
    rows: dict[str, list[float | None]] = {v: [] for v in c.VARIANTS}
    for _, tag, fn in metric_specs:
        for variant in c.VARIANTS:
            mean = c.final_mean("reach", variant, tag, x_max=REACH_X_MAX)
            rows[variant].append(None if mean is None else fn(mean))

    metric_labels = [m[0] for m in metric_specs]
    bar_w = 0.27

    fig, (ax_r, ax_p) = plt.subplots(
        1, 2,
        figsize=c.pcfg.scaled(11, 4.2, scale=1.0),
        gridspec_kw={"width_ratios": (1, 3)},
        constrained_layout=True,
    )

    for ax, slice_ in ((ax_r, slice(0, 1)), (ax_p, slice(1, None))):
        sub_labels = metric_labels[slice_]
        sub_base = np.arange(len(sub_labels))
        for offset, variant in zip((-bar_w, 0.0, bar_w), c.VARIANTS):
            vals = rows[variant][slice_]
            for x_pos, val in zip(sub_base + offset, vals):
                if val is None:
                    c.hatched_missing_bar(ax, x_pos, height=0.05, width=bar_w * 0.95)
                    continue
                ax.bar(x_pos, val, width=bar_w * 0.95,
                       color=c.VARIANT_COLOR[variant], edgecolor="white",
                       linewidth=0.4,
                       label=c.VARIANT_LABEL[variant] if ax is ax_r else None)
                txt = f"{val:+.2f}" if ax is ax_r else f"{100 * val:.1f} %"
                ax.text(x_pos, val + (0.02 if val >= 0 else -0.04), txt,
                        ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=7, color="#333333")
        ax.set_xticks(sub_base); ax.set_xticklabels(sub_labels)
        ax.set_axisbelow(True)

    ax_r.set_ylabel("Total reward")
    ax_r.axhline(0, color=c.pcfg.FAPS_DARK_GREY, linewidth=0.6)
    ax_p.set_ylabel("Success rate"); ax_p.set_ylim(0, 1.10)
    ax_r.set_title("Final reach-task performance — three tensegrity variants",
                   fontsize=10, fontweight="bold", loc="left")
    ax_r.legend(loc="lower left", ncol=3, fontsize=8)
    c.save(fig, c.RESULTS_FIG_DIR / "reach_comparison.png", tight=False)


def main() -> None:
    c.init()
    _convergence_figure()
    _comparison_figure()


if __name__ == "__main__":
    main()

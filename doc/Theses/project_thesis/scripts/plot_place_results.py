"""Cube-place figures for Results §5.3.

Produces three figures:

* ``place_convergence.png`` — 2×2 training-progress overview
  (total reward, grasp rate, place success rate, policy σ).
* ``place_curriculum.png``  — 2×2 panel highlighting the effect of
  the two curriculum events (red-cube introduction at 100k, regularisation
  ramp at 200k) on the PD variant.
* ``place_comparison.png``  — final-performance grouped bar chart.
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

PLACE_X_MAX: Final = 300_000
RED_CUBE_STEP: Final = 100_000
REG_RAMP_STEP: Final = 200_000


def _convergence_figure() -> None:
    fig, axes = plt.subplots(
        4, 1,
        figsize=c.pcfg.scaled(11, 15.0, scale=1.0),
        constrained_layout=True,
    )
    panels = [
        ("Reward_Totalrewardmean",
         "(a) Total reward (mean)", "Reward",  True),
        ("Info_Metrics_grasp_rate",
         "(b) Grasp success rate",  "Rate",    True),
        ("Info_Metrics_place_success_rate",
         "(c) Place success rate",  "Rate",    True),
        ("Policy_Standarddeviation",
         "(d) Policy std σ",        "σ",       True),
    ]
    for ax, (metric, title, ylabel, show_legend) in zip(axes, panels):
        c.plot_variants_curve(
            ax, "cube_place", metric,
            x_max=PLACE_X_MAX,
            y_label=ylabel,
            title=title,
            show_legend=show_legend,
        )
        if metric.endswith("_rate"):
            ax.set_ylim(0, 1.0)

    c.save(fig, c.RESULTS_FIG_DIR / "place_convergence.png", tight=False)


def _curriculum_figure() -> None:
    """Highlight curriculum effect on the PD variant."""
    fig, axes = plt.subplots(
        4, 1,
        figsize=c.pcfg.scaled(11, 15.0, scale=1.0),
        constrained_layout=True,
    )
    variant = "tensegrity_pd"

    # (a) Total reward
    steps, vals = c.load_rl_csv("cube_place", variant, "Reward_Totalrewardmean")
    axes[0].plot(c.steps_to_k(steps), vals,
                 color=c.pcfg.FAPS_BLUE, alpha=0.25, linewidth=0.7)
    axes[0].plot(c.steps_to_k(steps), c.smooth(vals),
                 color=c.pcfg.FAPS_BLUE, linewidth=1.4, label="Total reward")
    c.add_curriculum_marker(axes[0], RED_CUBE_STEP, "Red cube in")
    c.add_curriculum_marker(axes[0], REG_RAMP_STEP, "Reg. ramp")
    axes[0].set_title("(a) Total reward — PD variant",
                      fontsize=10, fontweight="bold", loc="left")
    axes[0].set_xlabel("Timesteps / k"); axes[0].set_ylabel("Reward")
    axes[0].set_xlim(0, c.steps_to_k(PLACE_X_MAX))
    axes[0].legend(loc="upper left", fontsize=8)

    # (b) Task metrics
    for metric, label, color in (
        ("Info_Metrics_grasp_rate",         "Grasp",    c.pcfg.FAPS_BLUE),
        ("Info_Metrics_place_success_rate", "Place",    c.pcfg.FAPS_GREEN),
        ("Info_Metrics_red_on_conveyor_rate", "Red safe", c.pcfg.MUTED_RED),
    ):
        steps, vals = c.load_rl_csv("cube_place", variant, metric)
        axes[1].plot(c.steps_to_k(steps), c.smooth(vals, 0.85),
                     color=color, linewidth=1.3, label=label)
    c.add_curriculum_marker(axes[1], RED_CUBE_STEP)
    c.add_curriculum_marker(axes[1], REG_RAMP_STEP)
    axes[1].set_title("(b) Task metrics — PD variant",
                      fontsize=10, fontweight="bold", loc="left")
    axes[1].set_xlabel("Timesteps / k"); axes[1].set_ylabel("Rate")
    axes[1].set_xlim(0, c.steps_to_k(PLACE_X_MAX)); axes[1].set_ylim(0, 1.0)
    axes[1].legend(loc="lower right", fontsize=8)

    # (c) Regularisation penalties
    for metric, label, color in (
        ("Info_Episode_Reward_action_rate",  "Action rate",    c.pcfg.AMBER),
        ("Info_Episode_Reward_joint_vel",    "Joint velocity", c.pcfg.TEAL),
    ):
        steps, vals = c.load_rl_csv("cube_place", variant, metric)
        axes[2].plot(c.steps_to_k(steps), c.smooth(vals),
                     color=color, linewidth=1.3, label=label)
    c.add_curriculum_marker(axes[2], REG_RAMP_STEP, "Reg. ramp")
    axes[2].set_title("(c) Regularisation penalties",
                      fontsize=10, fontweight="bold", loc="left")
    axes[2].set_xlabel("Timesteps / k"); axes[2].set_ylabel("Penalty / step")
    axes[2].set_xlim(0, c.steps_to_k(PLACE_X_MAX))
    axes[2].legend(loc="upper right", fontsize=8)

    # (d) Red-cube safety reward
    steps, vals = c.load_rl_csv(
        "cube_place", variant, "Info_Episode_Reward_cube_off_conveyor")
    axes[3].plot(c.steps_to_k(steps), c.smooth(vals, 0.85),
                 color=c.pcfg.MUTED_RED, linewidth=1.3,
                 label="Cube off conveyor")
    c.add_curriculum_marker(axes[3], RED_CUBE_STEP, "Red cube in")
    axes[3].set_title("(d) Red-cube penalty",
                      fontsize=10, fontweight="bold", loc="left")
    axes[3].set_xlabel("Timesteps / k"); axes[3].set_ylabel("Penalty / step")
    axes[3].set_xlim(0, c.steps_to_k(PLACE_X_MAX))
    axes[3].legend(loc="upper right", fontsize=8)

    c.save(fig, c.RESULTS_FIG_DIR / "place_curriculum.png", tight=False)


def _comparison_figure() -> None:
    """Final converged metrics — bar chart of grasp/place/red-safe rates."""
    metrics = ("Grasp rate", "Place success", "Red safe")
    pd_vals = (0.987, 0.906, 0.980)
    tn_vals = (0.947, 0.934, 0.952)
    ph_vals = (0.005, 0.000, 0.978)

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
        ax.bar(base + offset, vals, width=bar_w * 0.95,
               color=color, edgecolor="white", linewidth=0.4, label=label)
        for x, v in zip(base + offset, vals):
            ax.text(x, v + 0.015, f"{100 * v:.1f} %",
                    ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(base); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.10); ax.set_ylabel("Final rate")
    ax.set_title("Final cube-place performance — three tensegrity variants",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.set_axisbelow(True)

    c.save(fig, c.RESULTS_FIG_DIR / "place_comparison.png", tight=False)


def main() -> None:
    c.init()
    _convergence_figure()
    _curriculum_figure()
    _comparison_figure()


if __name__ == "__main__":
    main()

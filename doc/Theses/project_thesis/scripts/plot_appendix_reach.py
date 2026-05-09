"""Appendix A.4 — complete reach-task training curves.

Generates four detailed figures that consolidate the per-section
sub-figures previously rendered in cetz inline:

* ``reach_reward_overview.png`` — combined mean/max/min reward + policy σ
* ``reach_components.png``      — task reward components (4-panel)
* ``reach_diagnostics.png``     — regularisation + losses (4-panel)
* ``reach_diagnostics_extra.png`` — entropy loss + learning rate (2-panel)
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

REACH_X_MAX: Final = 48_000
TASK: Final = "reach"


def _plot_envelope(ax: plt.Axes, metric_mean: str,
                   metric_max: str, metric_min: str) -> None:
    """Mean (solid) + min/max (filled envelope) for all three variants."""
    for variant in c.VARIANTS:
        s_mean, v_mean = c.load_rl_csv(TASK, variant, metric_mean)
        if len(s_mean) == 0:
            continue
        mask = s_mean <= REACH_X_MAX
        s_mean, v_mean = s_mean[mask], v_mean[mask]

        _, v_max = c.load_rl_csv(TASK, variant, metric_max)
        _, v_min = c.load_rl_csv(TASK, variant, metric_min)
        v_max = v_max[: len(v_mean)]
        v_min = v_min[: len(v_mean)]

        x = c.steps_to_k(s_mean)
        color = c.VARIANT_COLOR[variant]
        if len(v_min) == len(v_mean) and len(v_max) == len(v_mean):
            ax.fill_between(x, v_min, v_max, color=color, alpha=0.15)
        ax.plot(x, c.smooth(v_mean), color=color, linewidth=1.4,
                label=c.VARIANT_LABEL[variant])

    ax.set_xlabel("Timesteps / k"); ax.set_ylabel("Reward")
    ax.set_xlim(0, c.steps_to_k(REACH_X_MAX))
    ax.legend(loc="lower right", fontsize=8)


def _reward_overview() -> None:
    fig, axes = plt.subplots(
        2, 1, sharex=True,
        figsize=c.pcfg.scaled(11, 7.5, scale=1.0),
        constrained_layout=True,
    )
    _plot_envelope(axes[0],
                   "Reward_Totalrewardmean",
                   "Reward_Totalrewardmax",
                   "Reward_Totalrewardmin")
    axes[0].set_title("(a) Total reward (mean ± min/max envelope)",
                      fontsize=10, fontweight="bold", loc="left")

    c.plot_variants_curve(axes[1], TASK, "Policy_Standarddeviation",
                          x_max=REACH_X_MAX, y_label="σ",
                          title="(b) Policy standard deviation",
                          show_legend=True)

    c.save(fig, c.APPENDIX_FIG_DIR / "reach_reward_overview.png", tight=False)


def _grid_figure(out_name: str, panels: list[tuple[str, str, str]]) -> None:
    """Generic 2×2 grid of variant-curve panels (panel = (metric, title, y_label))."""
    fig, axes = plt.subplots(
        2, 2,
        figsize=c.pcfg.scaled(11, 7.0, scale=1.0),
        constrained_layout=True,
    )
    for ax, (metric, title, ylabel) in zip(axes.flat, panels):
        c.plot_variants_curve(ax, TASK, metric,
                              x_max=REACH_X_MAX,
                              y_label=ylabel, title=title)
    # Single legend at bottom
    handles = [plt.Line2D([0], [0], color=c.VARIANT_COLOR[v], linewidth=1.4,
                           label=c.VARIANT_LABEL[v]) for v in c.VARIANTS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    c.save(fig, c.APPENDIX_FIG_DIR / out_name, tight=False)


def main() -> None:
    c.init()

    _reward_overview()

    _grid_figure("reach_components.png", [
        ("Info_Episode_Reward_end_effector_position_tracking",
            "(a) Position tracking", "Reward / step"),
        ("Info_Episode_Reward_end_effector_orientation_tracking",
            "(b) Orientation tracking", "Reward / step"),
        ("Info_Episode_Reward_end_effector_position_tracking_fine_grained",
            "(c) Fine-grained position", "Reward / step"),
        ("Info_Episode_Reward_pose_reached",
            "(d) Pose reached bonus", "Reward / step"),
    ])

    _grid_figure("reach_diagnostics.png", [
        ("Info_Episode_Reward_action_rate",
            "(a) Action-rate penalty",  "Penalty"),
        ("Info_Episode_Reward_joint_vel",
            "(b) Joint-velocity penalty", "Penalty"),
        ("Loss_Policyloss",  "(c) Policy loss",  "Loss"),
        ("Loss_Valueloss",   "(d) Value loss",   "Loss"),
    ])

    # Two extra panels (no full grid)
    fig, axes = plt.subplots(
        1, 2,
        figsize=c.pcfg.scaled(11, 3.5, scale=1.0),
        constrained_layout=True,
    )
    c.plot_variants_curve(axes[0], TASK, "Loss_Entropyloss",
                          x_max=REACH_X_MAX, y_label="Loss",
                          title="(a) Entropy loss", show_legend=True)
    c.plot_variants_curve(axes[1], TASK, "Learning_Learningrate",
                          x_max=REACH_X_MAX, y_label="LR",
                          title="(b) Learning rate")
    c.save(fig, c.APPENDIX_FIG_DIR / "reach_diagnostics_extra.png", tight=False)


if __name__ == "__main__":
    main()

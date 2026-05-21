"""Appendix A.5 — complete reach-task training curves.

Loads scalars directly from the TensorBoard event files declared in
:data:`common.RUNS`. Variants with missing tfevents are skipped silently.

All figures use a single shared variant legend at the bottom. Only the
total-reward overview panel draws the min/max envelope — every other
panel shows the smoothed mean line per variant without a ghost raw line.

Generates:

* ``reach_reward_overview.png``    — total-reward envelope + policy σ
* ``reach_components.png``         — task reward components (4-panel)
* ``reach_diagnostics.png``        — regularisation + losses (4-panel)
* ``reach_diagnostics_extra.png``  — entropy loss + learning rate (2-panel)
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt

import common as c

REACH_X_MAX: Final = 48_000
TASK: Final = "reach"


def _stacked_figure(out_name: str, panels: list[tuple[str, str, str]],
                    *, panel_h: float = 3.5) -> None:
    n = len(panels)
    fig, axes = plt.subplots(
        n, 1,
        figsize=c.pcfg.scaled(11, panel_h * n + 0.6, scale=1.0),
        constrained_layout=False,
    )
    axes_iter = axes if n > 1 else [axes]
    for ax, (metric, title, ylabel) in zip(axes_iter, panels):
        c.plot_variants_curve(ax, TASK, metric,
                              x_max=REACH_X_MAX, y_label=ylabel, title=title)
    fig.tight_layout(rect=(0, 0.06 if n <= 2 else 0.04, 1, 1))
    c.figure_bottom_legend(fig, axes_iter, ncol=3,
                           y=0.01 if n <= 2 else 0.005)
    c.save(fig, c.APPENDIX_FIG_DIR / out_name, tight=False)


def _reward_overview() -> None:
    fig, axes = plt.subplots(
        2, 1, sharex=True,
        figsize=c.pcfg.scaled(11, 7, scale=1.0),
        constrained_layout=False,
    )
    c.plot_envelope(axes[0], TASK,
                    "Reward / Total reward (mean)",
                    "Reward / Total reward (min)",
                    "Reward / Total reward (max)",
                    x_max=REACH_X_MAX)
    axes[0].set_title("(a) Total reward (mean ± min/max envelope)",
                      fontsize=10, fontweight="bold", loc="left")
    axes[0].set_ylabel("Reward")

    c.plot_variants_curve(axes[1], TASK, "Policy / Standard deviation",
                          x_max=REACH_X_MAX, y_label="σ",
                          title="(b) Policy standard deviation")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    # Pin the left margin to match the stacked figures (tight_layout sets a
    # wider margin here because "Reward" is longer than "Rate"/"Steps").
    fig.subplots_adjust(left=0.0957)
    c.figure_bottom_legend(fig, axes, ncol=3, y=0.01)
    c.save(fig, c.APPENDIX_FIG_DIR / "reach_reward_overview.png", tight=False)


def main() -> None:
    c.init()
    _reward_overview()

    _stacked_figure("reach_components.png", [
        ("Info / Episode_Reward/end_effector_position_tracking",
            "(a) Position tracking", "Reward"),
        ("Info / Episode_Reward/end_effector_orientation_tracking",
            "(b) Orientation tracking", "Reward"),
        ("Info / Episode_Reward/end_effector_position_tracking_fine_grained",
            "(c) Fine-grained position", "Reward"),
        ("Info / Episode_Reward/pose_reached",
            "(d) Pose-reached bonus (binary × 1e-6)", "Reward"),
    ])

    _stacked_figure("reach_diagnostics.png", [
        ("Info / Episode_Reward/action_rate", "(a) Action-rate penalty",  "Penalty"),
        ("Info / Episode_Reward/joint_vel",   "(b) Joint-velocity penalty","Penalty"),
        ("Loss / Policy loss",                "(c) Policy loss",           "Loss"),
        ("Loss / Value loss",                 "(d) Value loss",            "Loss"),
    ])

    _stacked_figure("reach_diagnostics_extra.png", [
        ("Loss / Entropy loss",         "(a) Entropy loss",  "Loss"),
        ("Learning / Learning rate",    "(b) Learning rate", "LR"),
    ], panel_h=3.2)


if __name__ == "__main__":
    main()

"""Appendix A.7 — complete cube-place training curves.

All figures show all three tensegrity variants (PD, Tendon, Physical).
Only the total-reward overview panel draws the min/max envelope — every
other panel renders only the smoothed mean line per variant with no
ghost raw line. Each multi-panel figure has a single shared variant
legend at the bottom.

The cube-place curriculum is annotated where relevant:

* ``Red cube in``  — red distractor activated at 125 k steps.
* ``Reg. ramp``    — ``action_rate`` / ``joint_vel`` weights ramp up at 150 k.

The 5-term manipulation, placement and regularisation groups are split
into two figures each so every figure fits on one A4 page without
overlapping the page margins.

Generates:

* ``place_reward_overview.png``    — total reward envelope + policy σ
* ``place_metrics.png``            — task metrics (4-panel)
* ``place_manipulation_a.png``     — reaching + grasping (3-panel)
* ``place_manipulation_b.png``     — lifting + height bonus (2-panel)
* ``place_placement.png``          — placement reward terms (4-panel,
                                     ``return_to_neutral`` removed)
* ``place_regularization_a.png``   — action / vel / torque (3-panel)
* ``place_regularization_b.png``   — belt + arm utilisation (2-panel)
* ``place_safety.png``             — safety / red-cube terms (3-panel,
                                     ``red_in_target`` removed)
* ``place_diagnostics.png``        — losses + learning rate (4-panel)
"""

from __future__ import annotations

from typing import Final, Sequence

import matplotlib.pyplot as plt

import common as c

PLACE_X_MAX: Final = 300_000
TASK: Final = "cube_place"
RED_CUBE_STEP: Final = c.PLACE_RED_CUBE_STEP   # 125 000
REG_RAMP_STEP: Final = c.PLACE_REG_RAMP_STEP   # 150 000

# Per-panel height (inches) — tuned so even 4-panel stacks comfortably fit on
# one A4 page (textheight ≈ 24.3 cm ≈ 9.6 in) once the figure is rendered at
# the 16 cm text width.
PANEL_H: Final = 3.5 #2.7


def _add_curriculum(ax: plt.Axes, kinds: Sequence[str]) -> None:
    """Draw vertical markers for the requested curriculum events."""
    if "red" in kinds:
        c.add_curriculum_marker(ax, RED_CUBE_STEP, "Red cube in", y_frac=0.95)
    if "reg" in kinds:
        c.add_curriculum_marker(ax, REG_RAMP_STEP, "Reg. ramp", y_frac=0.85)


def _stacked_figure(
    out_name: str,
    panels: list[tuple[str, str, str]],
    *,
    curriculum: Sequence[str] = (),
    y_lim: tuple[float, float] | None = None,
) -> None:
    n = len(panels)
    fig, axes = plt.subplots(
        n, 1,
        figsize=c.pcfg.scaled(11, PANEL_H * n + 0.6, scale=1.0),
        constrained_layout=False,
    )
    axes_iter = list(axes) if n > 1 else [axes]
    for ax, (metric, title, ylabel) in zip(axes_iter, panels):
        c.plot_variants_curve(ax, TASK, metric,
                              x_max=PLACE_X_MAX, y_label=ylabel, title=title)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
        if curriculum:
            _add_curriculum(ax, curriculum)
    fig.tight_layout(rect=(0, 0.05 if n <= 2 else 0.035, 1, 1))
    c.figure_bottom_legend(fig, axes_iter, ncol=3,
                           y=0.01 if n <= 2 else 0.005)
    c.save(fig, c.APPENDIX_FIG_DIR / out_name, tight=False)


def _reward_overview() -> None:
    fig, axes = plt.subplots(
        2, 1, sharex=True,
        figsize=c.pcfg.scaled(11, 6.6, scale=1.0),
        constrained_layout=False,
    )
    c.plot_envelope(axes[0], TASK,
                    "Reward / Total reward (mean)",
                    "Reward / Total reward (min)",
                    "Reward / Total reward (max)",
                    x_max=PLACE_X_MAX)
    axes[0].set_title("(a) Total reward (mean ± min/max envelope)",
                      fontsize=10, fontweight="bold", loc="left")
    axes[0].set_ylabel("Reward")
    _add_curriculum(axes[0], ("red", "reg"))

    c.plot_variants_curve(axes[1], TASK, "Policy / Standard deviation",
                          x_max=PLACE_X_MAX, y_label="σ",
                          title="(b) Policy standard deviation")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    # Pin the left margin to match the stacked figures (tight_layout sets a
    # wider margin here because "Reward" is longer than "Rate"/"Steps").
    fig.subplots_adjust(left=0.0957)
    c.figure_bottom_legend(fig, axes, ncol=3, y=0.01)
    c.save(fig, c.APPENDIX_FIG_DIR / "place_reward_overview.png", tight=False)


def main() -> None:
    c.init()
    _reward_overview()

    _stacked_figure("place_metrics.png", [
        ("Info / Metrics/grasp_rate",          "(a) Grasp rate",          "Rate"),
        ("Info / Metrics/place_success_rate",  "(b) Place success rate",  "Rate"),
        ("Info / Metrics/red_on_conveyor_rate","(c) Red-cube safe",       "Rate"),
        ("Info / Metrics/mean_episode_length", "(d) Mean episode length", "Steps"),
    ], curriculum=("red", "reg"))

    # Manipulation pipeline (5 terms) — split 3 + 2 so each fits one page.
    _stacked_figure("place_manipulation_a.png", [
        ("Info / Episode_Reward/reaching_object",
            "(a) Reaching object (coarse)", "Reward"),
        ("Info / Episode_Reward/reaching_object_fine",
            "(b) Reaching object (fine)",   "Reward"),
        ("Info / Episode_Reward/grasping",
            "(c) Grasping",                 "Reward"),
    ], curriculum=("red", "reg"))
    _stacked_figure("place_manipulation_b.png", [
        ("Info / Episode_Reward/lifting_object",
            "(d) Lifting object", "Reward"),
        ("Info / Episode_Reward/height_bonus",
            "(e) Height bonus",   "Reward"),
    ], curriculum=("red", "reg"))

    # Placement (4 terms — ``return_to_neutral`` removed).
    _stacked_figure("place_placement.png", [
        ("Info / Episode_Reward/goal_tracking",
            "(a) Goal tracking (coarse)", "Reward"),
        ("Info / Episode_Reward/goal_tracking_fine",
            "(b) Goal tracking (fine)",   "Reward"),
        ("Info / Episode_Reward/release",
            "(c) Release",                "Reward"),
        ("Info / Episode_Reward/green_in_target",
            "(d) Green-in-target bonus",  "Reward"),
    ], curriculum=("red", "reg"))

    # Regularisation (5 terms) — split 3 + 2.
    _stacked_figure("place_regularization_a.png", [
        ("Info / Episode_Reward/action_rate",  "(a) Action-rate penalty",    "Penalty"),
        ("Info / Episode_Reward/joint_vel",    "(b) Joint-velocity penalty", "Penalty"),
        ("Info / Episode_Reward/joint_torque", "(c) Joint-torque penalty",   "Penalty"),
    ], curriculum=("reg",))
    _stacked_figure("place_regularization_b.png", [
        ("Info / Episode_Reward/belt_contact",    "(d) Belt-contact penalty",    "Penalty"),
        ("Info / Episode_Reward/arm_utilization", "(e) Arm-utilization penalty", "Penalty"),
    ], curriculum=("reg",))

    # Safety (3 terms — ``red_in_target`` removed).
    _stacked_figure("place_safety.png", [
        ("Info / Episode_Reward/cube_off_conveyor",
            "(a) Cube-off-conveyor penalty",  "Penalty"),
        ("Info / Episode_Reward/red_green_separation",
            "(b) Red/green separation",       "Reward"),
        ("Info / Episode_Reward/red_clearance",
            "(c) Red-cube clearance",         "Reward"),
    ], curriculum=("red",))

    _stacked_figure("place_diagnostics.png", [
        ("Loss / Policy loss",       "(a) Policy loss",   "Loss"),
        ("Loss / Value loss",        "(b) Value loss",    "Loss"),
        ("Loss / Entropy loss",      "(c) Entropy loss",  "Loss"),
        ("Learning / Learning rate", "(d) Learning rate", "LR"),
    ])


if __name__ == "__main__":
    main()

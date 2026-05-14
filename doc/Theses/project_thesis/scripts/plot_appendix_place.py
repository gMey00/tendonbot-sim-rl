"""Appendix A.5 — complete cube-place training curves.

Six figures mirroring the previously-inline cetz subfigures:

* ``place_reward_overview.png``   — total reward envelope + policy σ
* ``place_metrics.png``           — task metrics 4-panel
* ``place_manipulation.png``      — manipulation reward components 4-panel
* ``place_placement.png``         — placement reward components 4-panel
* ``place_regularization.png``    — regularisation terms 4-panel
* ``place_diagnostics.png``       — losses + learning rate 4-panel
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt

import common as c

PLACE_X_MAX: Final = 300_000
TASK: Final = "cube_place"


def _envelope_panel(ax: plt.Axes,
                    metric_mean: str, metric_max: str, metric_min: str) -> None:
    for variant in c.VARIANTS:
        s_mean, v_mean = c.load_rl_csv(TASK, variant, metric_mean)
        if len(s_mean) == 0:
            continue
        mask = s_mean <= PLACE_X_MAX
        s_mean, v_mean = s_mean[mask], v_mean[mask]
        _, v_max = c.load_rl_csv(TASK, variant, metric_max)
        _, v_min = c.load_rl_csv(TASK, variant, metric_min)
        v_max = v_max[: len(v_mean)]; v_min = v_min[: len(v_mean)]
        x = c.steps_to_k(s_mean); color = c.VARIANT_COLOR[variant]
        if len(v_min) == len(v_mean) and len(v_max) == len(v_mean):
            ax.fill_between(x, v_min, v_max, color=color, alpha=0.15)
        ax.plot(x, c.smooth(v_mean), color=color, linewidth=1.4,
                label=c.VARIANT_LABEL[variant])
    ax.set_xlabel("Timesteps / k"); ax.set_ylabel("Reward")
    ax.set_xlim(0, c.steps_to_k(PLACE_X_MAX))
    ax.legend(loc="upper left", fontsize=8)


def _grid_figure(out_name: str, panels: list[tuple[str, str, str]],
                 *, y_lim: tuple[float, float] | None = None) -> None:
    fig, axes = plt.subplots(
        4, 1,
        figsize=c.pcfg.scaled(11, 15.0, scale=1.0),
        constrained_layout=True,
    )
    for ax, (metric, title, ylabel) in zip(axes, panels):
        c.plot_variants_curve(ax, TASK, metric,
                              x_max=PLACE_X_MAX, y_label=ylabel, title=title,
                              show_legend=True)
        if y_lim is not None:
            ax.set_ylim(*y_lim)
    c.save(fig, c.APPENDIX_FIG_DIR / out_name, tight=False)


def _reward_overview() -> None:
    fig, axes = plt.subplots(
        2, 1, sharex=True,
        figsize=c.pcfg.scaled(11, 7.5, scale=1.0),
        constrained_layout=True,
    )
    _envelope_panel(axes[0],
                    "Reward_Totalrewardmean",
                    "Reward_Totalrewardmax",
                    "Reward_Totalrewardmin")
    axes[0].set_title("(a) Total reward (mean ± min/max envelope)",
                      fontsize=10, fontweight="bold", loc="left")
    c.plot_variants_curve(axes[1], TASK, "Policy_Standarddeviation",
                          x_max=PLACE_X_MAX, y_label="σ",
                          title="(b) Policy standard deviation",
                          show_legend=True)
    c.save(fig, c.APPENDIX_FIG_DIR / "place_reward_overview.png", tight=False)


def main() -> None:
    c.init()
    _reward_overview()

    _grid_figure("place_metrics.png", [
        ("Info_Metrics_grasp_rate",         "(a) Grasp rate",         "Rate"),
        ("Info_Metrics_place_success_rate", "(b) Place success rate", "Rate"),
        ("Info_Metrics_red_on_conveyor_rate", "(c) Red-cube safe",    "Rate"),
        ("Info_Metrics_mean_episode_length", "(d) Mean episode length", "Steps"),
    ])

    _grid_figure("place_manipulation.png", [
        ("Info_Episode_Reward_reaching_object", "(a) Reaching object",   "Reward / step"),
        ("Info_Episode_Reward_grasping",        "(b) Grasping",          "Reward / step"),
        ("Info_Episode_Reward_lifting_object",  "(c) Lifting object",    "Reward / step"),
        ("Info_Episode_Reward_height_bonus",    "(d) Height bonus",      "Reward / step"),
    ])

    _grid_figure("place_placement.png", [
        ("Info_Episode_Reward_goal_tracking",      "(a) Goal tracking (coarse)", "Reward / step"),
        ("Info_Episode_Reward_goal_tracking_fine", "(b) Goal tracking (fine)",   "Reward / step"),
        ("Info_Episode_Reward_release",            "(c) Release",                "Reward / step"),
        ("Info_Episode_Reward_green_in_target",    "(d) Green-in-target",        "Reward / step"),
    ])

    _grid_figure("place_regularization.png", [
        ("Info_Episode_Reward_action_rate",  "(a) Action-rate penalty",  "Penalty"),
        ("Info_Episode_Reward_joint_vel",    "(b) Joint-velocity penalty", "Penalty"),
        ("Info_Episode_Reward_joint_torque", "(c) Joint-torque penalty",   "Penalty"),
        ("Info_Episode_Reward_belt_contact", "(d) Belt-contact penalty",   "Penalty"),
    ])

    _grid_figure("place_diagnostics.png", [
        ("Loss_Policyloss",        "(a) Policy loss",   "Loss"),
        ("Loss_Valueloss",         "(b) Value loss",    "Loss"),
        ("Loss_Entropyloss",       "(c) Entropy loss",  "Loss"),
        ("Learning_Learningrate",  "(d) Learning rate", "LR"),
    ])


if __name__ == "__main__":
    main()

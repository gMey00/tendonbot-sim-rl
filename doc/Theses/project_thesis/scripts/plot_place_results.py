"""Cube-place figures for Results §5.3.

Reads metrics directly from the skrl TensorBoard event files declared in
:data:`common.RUNS`. Variants whose tfevents file is missing are skipped
gracefully (hatched bars in the comparison figure; the curriculum panel
falls back from the PD to the Tendon variant when PD logs are absent).

Produces:

* ``place_convergence.png`` — 4-panel training-progress overview
  (total reward with min/max envelope, grasp rate, place success rate,
  policy σ) with a single shared legend underneath.
* ``place_curriculum.png``  — 4-panel illustration of the curriculum
  events (red-cube introduction at 125 k, regularisation ramp at 150 k).
* ``place_comparison.png``  — final-performance grouped bar chart.
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

PLACE_X_MAX: Final = 300_000
RED_CUBE_STEP: Final = c.PLACE_RED_CUBE_STEP    # 125 000
REG_RAMP_STEP: Final = c.PLACE_REG_RAMP_STEP    # 150 000

# Variant to demonstrate the curriculum: prefer PD (used in the thesis text);
# fall back to the next available variant if its tfevents are missing.
_CURRICULUM_PREFERENCE: Final = ("tensegrity_pd", "tensegrity_tendon",
                                 "tensegrity_physical")


def _pick_curriculum_variant() -> str | None:
    for v in _CURRICULUM_PREFERENCE:
        if c.run_dir("cube_place", v) is not None:
            return v
    return None


def _convergence_figure() -> None:
    fig, axes = plt.subplots(
        4, 1,
        figsize=c.pcfg.scaled(11, 14.5, scale=1.0),
        constrained_layout=False,
    )

    # (a) Total reward + min/max envelope
    c.plot_envelope(
        axes[0], "cube_place",
        "Reward / Total reward (mean)",
        "Reward / Total reward (min)",
        "Reward / Total reward (max)",
        x_max=PLACE_X_MAX,
    )
    axes[0].set_title("(a) Total reward (mean ± min/max envelope)",
                      fontsize=10, fontweight="bold", loc="left")
    axes[0].set_ylabel("Reward")

    # (b) Grasp rate
    c.plot_variants_curve(axes[1], "cube_place", "Info / Metrics/grasp_rate",
                          x_max=PLACE_X_MAX, y_label="Rate",
                          title="(b) Grasp success rate")
    axes[1].set_ylim(0, 1.0)

    # (c) Place success rate
    c.plot_variants_curve(axes[2], "cube_place",
                          "Info / Metrics/place_success_rate",
                          x_max=PLACE_X_MAX, y_label="Rate",
                          title="(c) Place success rate")
    axes[2].set_ylim(0, 1.0)

    # (d) Policy std
    c.plot_variants_curve(axes[3], "cube_place", "Policy / Standard deviation",
                          x_max=PLACE_X_MAX, y_label="σ",
                          title="(d) Policy standard deviation")

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    c.figure_bottom_legend(fig, axes, ncol=3, y=0.005)
    c.save(fig, c.RESULTS_FIG_DIR / "place_convergence.png", tight=False)


def _curriculum_figure() -> None:
    variant = _pick_curriculum_variant()
    fig, axes = plt.subplots(
        4, 1,
        figsize=c.pcfg.scaled(11, 14.5, scale=1.0),
        constrained_layout=True,
    )
    if variant is None:
        for ax in axes:
            c.missing_panel(ax, "no cube_place logs available")
        c.save(fig, c.RESULTS_FIG_DIR / "place_curriculum.png", tight=False)
        return

    label = c.VARIANT_LABEL[variant]

    def _plot(ax, tag, *, color, plot_label, smoothing=0.92):
        steps, vals = c.load_rl("cube_place", variant, tag)
        if len(steps) == 0:
            return False
        mask = steps <= PLACE_X_MAX
        steps, vals = steps[mask], vals[mask]
        if len(vals) == 0:
            return False
        x = c.steps_to_k(steps)
        ax.plot(x, c.smooth(vals, smoothing), color=color, linewidth=1.5,
                label=plot_label)
        return True

    # (a) Total reward
    _plot(axes[0], "Reward / Total reward (mean)",
          color=c.VARIANT_COLOR[variant], plot_label="Total reward")
    c.add_curriculum_marker(axes[0], RED_CUBE_STEP, "Red cube in", y_frac=0.30)
    c.add_curriculum_marker(axes[0], REG_RAMP_STEP, "Reg. ramp", y_frac=0.20)
    axes[0].set_title(f"(a) Total reward — {label} variant",
                      fontsize=10, fontweight="bold", loc="left")
    axes[0].set_xlabel("Timesteps / k"); axes[0].set_ylabel("Reward")
    axes[0].set_xlim(0, c.steps_to_k(PLACE_X_MAX))
    axes[0].legend(loc="lower right", fontsize=8)

    # (b) Task metrics
    metrics_b = (
        ("Info / Metrics/grasp_rate",          "Grasp",    c.pcfg.FAPS_BLUE),
        ("Info / Metrics/place_success_rate",  "Place",    c.pcfg.FAPS_GREEN),
        ("Info / Metrics/red_on_conveyor_rate","Red safe", c.pcfg.MUTED_RED),
    )
    for tag, lab, col in metrics_b:
        _plot(axes[1], tag, color=col, plot_label=lab, smoothing=0.85)
    c.add_curriculum_marker(axes[1], RED_CUBE_STEP, "Red cube in", y_frac=0.70)
    c.add_curriculum_marker(axes[1], REG_RAMP_STEP, "Reg. ramp", y_frac=0.60)
    axes[1].set_title(f"(b) Task metrics — {label} variant",
                      fontsize=10, fontweight="bold", loc="left")
    axes[1].set_xlabel("Timesteps / k"); axes[1].set_ylabel("Rate")
    axes[1].set_xlim(0, c.steps_to_k(PLACE_X_MAX)); axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="center right", fontsize=8)

    # (c) Regularisation penalties
    metrics_c = (
        ("Info / Episode_Reward/action_rate", "Action rate",    c.pcfg.AMBER),
        ("Info / Episode_Reward/joint_vel",   "Joint velocity", c.pcfg.TEAL),
    )
    for tag, lab, col in metrics_c:
        _plot(axes[2], tag, color=col, plot_label=lab)
    c.add_curriculum_marker(axes[2], REG_RAMP_STEP, "Reg. ramp", y_frac=0.30)
    axes[2].set_title("(c) Regularisation penalties",
                      fontsize=10, fontweight="bold", loc="left")
    axes[2].set_xlabel("Timesteps / k"); axes[2].set_ylabel("Penalty")
    axes[2].set_xlim(0, c.steps_to_k(PLACE_X_MAX))
    axes[2].legend(loc="lower left", fontsize=8)

    # (d) Red-cube safety reward (zero before 125k, negative after)
    _plot(axes[3], "Info / Episode_Reward/cube_off_conveyor",
          color=c.pcfg.MUTED_RED, plot_label="Cube off conveyor",
          smoothing=0.85)
    c.add_curriculum_marker(axes[3], RED_CUBE_STEP, "Red cube in", y_frac=0.30)
    axes[3].set_title("(d) Red-cube safety penalty",
                      fontsize=10, fontweight="bold", loc="left")
    axes[3].set_xlabel("Timesteps / k"); axes[3].set_ylabel("Penalty")
    axes[3].set_xlim(0, c.steps_to_k(PLACE_X_MAX))
    axes[3].legend(loc="lower left", fontsize=8)

    c.save(fig, c.RESULTS_FIG_DIR / "place_curriculum.png", tight=False)


def _comparison_figure() -> None:
    """Final converged metrics — bar chart of grasp/place/red-safe rates."""
    metric_specs = (
        ("Grasp rate",     "Info / Metrics/grasp_rate"),
        ("Place success",  "Info / Metrics/place_success_rate"),
        ("Red safe",       "Info / Metrics/red_on_conveyor_rate"),
    )
    rows: dict[str, list[float | None]] = {v: [] for v in c.VARIANTS}
    for _, tag in metric_specs:
        for variant in c.VARIANTS:
            rows[variant].append(
                c.final_mean("cube_place", variant, tag, x_max=PLACE_X_MAX)
            )

    metrics = [m[0] for m in metric_specs]
    fig, ax = plt.subplots(
        figsize=c.pcfg.scaled(11, 4.0, scale=1.0),
        constrained_layout=True,
    )
    n = len(metrics); base = np.arange(n); bar_w = 0.27

    for offset, variant in zip((-bar_w, 0.0, bar_w), c.VARIANTS):
        vals = rows[variant]
        labeled = False
        for x_pos, val in zip(base + offset, vals):
            if val is None:
                c.hatched_missing_bar(ax, x_pos, height=0.05, width=bar_w * 0.95)
                continue
            ax.bar(x_pos, val, width=bar_w * 0.95,
                   color=c.VARIANT_COLOR[variant], edgecolor="white",
                   linewidth=0.4,
                   label=None if labeled else c.VARIANT_LABEL[variant])
            labeled = True
            ax.text(x_pos, val + 0.015, f"{100 * val:.1f} %",
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

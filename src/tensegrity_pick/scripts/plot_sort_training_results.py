#!/usr/bin/env python3
"""Generate training result plots for the Cube Sort task README.

Reads TensorBoard event files from the latest training run and produces
publication-quality figures saved into the task's ``figures/`` directory.

Usage
-----
    cd src/tensegrity_pick
    python scripts/plot_sort_training_results.py [--run DIR_NAME]

If ``--run`` is omitted the script auto-selects the latest ``*_ppo_torch``
directory under ``logs/skrl/cube_sort/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / ".config"))
import plot_config as pcfg
pcfg.apply_style()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGS_ROOT: Final = Path("logs/skrl/cube_sort")
FIGURES_DIR: Final = Path(
    "src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/"
    "manager_based/cube_sort/figures"
)

CURRICULUM_REGULARISATION_STEP: Final = 200_000

SMOOTHING_WEIGHT: Final = 0.92  # EMA smoothing factor for noisy curves


# Semantic colour aliases for this task
_C_BLUE = pcfg.FAPS_BLUE
_C_GREEN = pcfg.FAPS_GREEN
_C_AMBER = pcfg.AMBER
_C_RED = pcfg.MUTED_RED
_C_PURPLE = pcfg.PURPLE
_C_TEAL = pcfg.TEAL
_C_GREY = pcfg.FAPS_DARK_GREY
_C_ORANGE = pcfg.DARK_ORANGE
_C_DARK = "#333333"

FIGSIZE_WIDE: Final = (11, 4.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_latest_run(root: Path) -> Path:
    """Return the path to the most recent ``*_ppo_torch`` run directory."""
    candidates = sorted(root.glob("*_ppo_torch"))
    if not candidates:
        raise FileNotFoundError(f"No PPO runs found under {root}")
    return candidates[-1]


def load_scalars(
    accumulator: EventAccumulator,
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (steps, values) arrays for *tag*, or empty arrays if absent."""
    if tag not in accumulator.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = accumulator.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values: np.ndarray, weight: float = SMOOTHING_WEIGHT) -> np.ndarray:
    """Exponential moving average with *weight* in (0, 1)."""
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def add_curriculum_markers(axis: plt.Axes) -> None:
    """Add vertical dashed lines for curriculum events."""
    axis.axvline(
        steps_to_k(CURRICULUM_REGULARISATION_STEP),
        color=_C_GREY,
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label="_nolegend_",
    )


def annotate_curriculum(axis: plt.Axes, y_frac: float = 0.95) -> None:
    """Place text labels for curriculum events at *y_frac* of the axis."""
    transform = axis.get_xaxis_transform()
    axis.text(
        steps_to_k(CURRICULUM_REGULARISATION_STEP),
        y_frac,
        " Reg. ramp",
        transform=transform,
        fontsize=7.5,
        color=_C_GREY,
        va="top",
        ha="left",
        alpha=0.85,
    )


def steps_to_k(steps: np.ndarray | int | float) -> np.ndarray | float:
    """Convert steps to thousands for cleaner x-axis labels."""
    return steps / 1000.0


def finalise(
    figure: plt.Figure,
    path: Path,
    tight: bool = True,
    *,
    title: str | None = None,
) -> None:
    """Save figure and close."""
    pcfg.finalize(figure, path, title=title, tight=tight)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_total_reward(
    accumulator: EventAccumulator,
    output: Path,
) -> None:
    """Fig 1 — Total episode reward with min/max envelope."""
    steps_mean, values_mean = load_scalars(accumulator, "Reward / Total reward (mean)")
    _, values_max = load_scalars(accumulator, "Reward / Total reward (max)")
    _, values_min = load_scalars(accumulator, "Reward / Total reward (min)")

    x = steps_to_k(steps_mean)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.fill_between(
        x,
        smooth(values_min, 0.85),
        smooth(values_max, 0.85),
        alpha=0.15,
        color=_C_BLUE,
        linewidth=0,
    )
    ax.plot(x, smooth(values_mean), color=_C_BLUE, linewidth=1.8, label="Mean")
    ax.plot(x, smooth(values_max, 0.85), color=_C_BLUE, linewidth=0.6, alpha=0.5, linestyle="--", label="Max")
    ax.plot(x, smooth(values_min, 0.85), color=_C_BLUE, linewidth=0.6, alpha=0.5, linestyle=":", label="Min")

    add_curriculum_markers(ax)
    annotate_curriculum(ax)

    ax.set_xlabel("Training Steps (×1 000)")
    ax.set_ylabel("Episode Return")
    ax.set_title("Total Episode Reward", )
    ax.legend(loc="lower right", framealpha=0.9)
    # grid from style
    ax.set_xlim(left=0)

    finalise(fig, output)


def plot_task_success(
    accumulator: EventAccumulator,
    output: Path,
) -> None:
    """Fig 2 — Grasp rate, placed count, and missed count."""
    steps_grasp, values_grasp = load_scalars(accumulator, "Info / Metrics/grasp_rate")
    steps_placed, values_placed = load_scalars(accumulator, "Info / Metrics/mean_targets_placed")
    steps_missed, values_missed = load_scalars(accumulator, "Info / Metrics/mean_targets_missed")

    fig, ax_left = plt.subplots(figsize=FIGSIZE_WIDE)
    ax_right = ax_left.twinx()

    # Grasp rate (left axis, %)
    ax_left.plot(
        steps_to_k(steps_grasp),
        smooth(values_grasp * 100),
        color=_C_GREEN,
        linewidth=1.8,
        label="Grasp Rate (%)",
    )
    ax_left.set_ylabel("Grasp Rate (%)", fontsize=10, color=_C_GREEN)
    ax_left.tick_params(axis="y", labelcolor=_C_GREEN)
    ax_left.set_ylim(-5, 105)

    # Placed count (right axis)
    if len(steps_placed) > 0:
        ax_right.plot(
            steps_to_k(steps_placed),
            smooth(values_placed),
            color=_C_BLUE,
            linewidth=1.8,
            label="Mean Targets Placed",
        )
    if len(steps_missed) > 0:
        ax_right.plot(
            steps_to_k(steps_missed),
            smooth(values_missed),
            color=_C_RED,
            linewidth=1.8,
            linestyle="--",
            label="Mean Targets Missed",
        )
    ax_right.set_ylabel("Count per Episode", fontsize=10, color=_C_BLUE)
    ax_right.tick_params(axis="y", labelcolor=_C_BLUE)

    add_curriculum_markers(ax_left)
    annotate_curriculum(ax_left)

    # Combined legend
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="center right",
        fontsize=9,
        framealpha=0.9,
    )

    ax_left.set_xlabel("Training Steps (×1 000)")
    ax_left.set_title("Task Success Metrics", )
    # grid from style
    ax_left.set_xlim(left=0)

    finalise(fig, output)


def plot_reward_decomposition(
    accumulator: EventAccumulator,
    output: Path,
) -> None:
    """Fig 3 — Sequential skill acquisition shown through reward components."""
    reward_terms: list[tuple[str, str, str]] = [
        ("Info / Episode_Reward/reaching_object", "Reach (coarse)", _C_TEAL),
        ("Info / Episode_Reward/reaching_object_fine", "Reach (fine)", _C_BLUE),
        ("Info / Episode_Reward/grasping", "Grasp", _C_AMBER),
        ("Info / Episode_Reward/lifting_object", "Lift", _C_ORANGE),
        ("Info / Episode_Reward/height_bonus", "Height", _C_RED),
        ("Info / Episode_Reward/goal_tracking", "Transport", _C_GREEN),
        ("Info / Episode_Reward/goal_tracking_fine", "Transport (fine)", _C_DARK),
        ("Info / Episode_Reward/release", "Release", _C_PURPLE),
    ]

    fig, axes = plt.subplots(
        len(reward_terms), 1,
        figsize=(11, 1.4 * len(reward_terms) + 1.2),
        sharex=True,
    )

    for ax, (tag, label, colour) in zip(axes, reward_terms):
        steps, values = load_scalars(accumulator, tag)
        if len(steps) == 0:
            continue

        x = steps_to_k(steps)
        ax.fill_between(x, 0, smooth(values), alpha=0.25, color=colour, linewidth=0)
        ax.plot(x, smooth(values), color=colour, linewidth=1.4)

        add_curriculum_markers(ax)

        ax.set_ylabel(label, fontsize=9, fontweight="bold", rotation=0, labelpad=70, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        # grid from style
        ax.set_xlim(left=0)

        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    axes[-1].set_xlabel("Training Steps (×1 000)")
    fig.subplots_adjust(hspace=0.15)

    finalise(fig, output, tight=False,
             title="Reward Decomposition — Sequential Skill Acquisition")


def plot_penalties(
    accumulator: EventAccumulator,
    output: Path,
) -> None:
    """Fig 4 — Penalty and regularisation terms with curriculum markers."""
    penalty_terms: list[tuple[str, str, str]] = [
        ("Info / Episode_Reward/base_velocity", "Base Velocity", _C_BLUE),
        ("Info / Episode_Reward/cube_off_conveyor", "Cube Off Conveyor", _C_RED),
        ("Info / Episode_Reward/belt_contact", "Belt Contact", _C_PURPLE),
        ("Info / Episode_Reward/action_rate", "Action Rate", _C_TEAL),
        ("Info / Episode_Reward/joint_vel", "Joint Velocity", _C_GREY),
        ("Info / Episode_Reward/joint_torque", "Joint Torque", _C_AMBER),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for tag, label, colour in penalty_terms:
        steps, values = load_scalars(accumulator, tag)
        if len(steps) == 0:
            continue
        ax.plot(steps_to_k(steps), smooth(values), color=colour, linewidth=1.3, label=label)

    add_curriculum_markers(ax)
    annotate_curriculum(ax, y_frac=0.08)

    ax.set_xlabel("Training Steps (×1 000)")
    ax.set_ylabel("Episode Penalty")
    ax.set_title("Penalties & Regularisation", )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)
    # grid from style
    ax.set_xlim(left=0)

    finalise(fig, output)


def plot_policy_diagnostics(
    accumulator: EventAccumulator,
    output: Path,
) -> None:
    """Fig 5 — Policy standard deviation, losses, and learning rate."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # (a) Policy std
    ax = axes[0, 0]
    steps, values = load_scalars(accumulator, "Policy / Standard deviation")
    ax.plot(steps_to_k(steps), smooth(values), color=_C_BLUE, linewidth=1.4)
    add_curriculum_markers(ax)
    ax.set_title("Policy Std. Deviation", )
    ax.set_ylabel("σ")
    # grid from style

    # (b) Policy loss
    ax = axes[0, 1]
    steps, values = load_scalars(accumulator, "Loss / Policy loss")
    ax.plot(steps_to_k(steps), smooth(values), color=_C_GREEN, linewidth=1.4)
    add_curriculum_markers(ax)
    ax.set_title("Policy (Surrogate) Loss", )
    # grid from style

    # (c) Value loss
    ax = axes[1, 0]
    steps, values = load_scalars(accumulator, "Loss / Value loss")
    ax.plot(steps_to_k(steps), smooth(values), color=_C_ORANGE, linewidth=1.4)
    add_curriculum_markers(ax)
    ax.set_title("Value Function Loss", )
    ax.set_xlabel("Steps (×1 000)")
    ax.set_ylabel("MSE")
    # grid from style

    # (d) Learning rate
    ax = axes[1, 1]
    steps, values = load_scalars(accumulator, "Learning / Learning rate")
    ax.plot(steps_to_k(steps), values, color=_C_PURPLE, linewidth=1.4)
    add_curriculum_markers(ax)
    ax.set_title("Learning Rate (KL-adaptive)", )
    ax.set_xlabel("Steps (×1 000)")
    # grid from style

    for row in axes:
        for a in row:
            a.set_xlim(left=0)

    finalise(fig, output, title="Policy & Training Diagnostics")


def plot_converged_summary(
    accumulator: EventAccumulator,
    output: Path,
) -> None:
    """Fig 6 — Final converged performance bar chart."""
    reward_tags: list[tuple[str, str]] = [
        ("Info / Episode_Reward/goal_tracking", "Transport"),
        ("Info / Episode_Reward/reaching_object_fine", "Reach (fine)"),
        ("Info / Episode_Reward/reaching_object", "Reach (coarse)"),
        ("Info / Episode_Reward/arm_utilization", "Arm Use"),
        ("Info / Episode_Reward/goal_tracking_fine", "Fine Pos."),
        ("Info / Episode_Reward/grasping", "Grasp"),
        ("Info / Episode_Reward/reorient", "Reorient"),
        ("Info / Episode_Reward/height_bonus", "Height"),
        ("Info / Episode_Reward/lifting_object", "Lift"),
        ("Info / Episode_Reward/release", "Release"),
    ]
    penalty_tags: list[tuple[str, str]] = [
        ("Info / Episode_Reward/cube_off_conveyor", "Cube Off"),
        ("Info / Episode_Reward/base_velocity", "Base Vel."),
        ("Info / Episode_Reward/belt_contact", "Belt"),
        ("Info / Episode_Reward/joint_torque", "Torque"),
        ("Info / Episode_Reward/action_rate", "Act. Rate"),
        ("Info / Episode_Reward/joint_vel", "Jnt. Vel"),
    ]

    def final_mean(tag: str) -> float:
        steps, values = load_scalars(accumulator, tag)
        if len(values) == 0:
            return 0.0
        cutoff = int(len(values) * 0.9)
        return float(np.mean(values[cutoff:]))

    reward_labels = [label for _, label in reward_tags]
    reward_values = [final_mean(tag) for tag, _ in reward_tags]
    penalty_labels = [label for _, label in penalty_tags]
    penalty_values = [final_mean(tag) for tag, _ in penalty_tags]

    fig, (ax_r, ax_p) = plt.subplots(
        1, 2,
        figsize=(12, 4.5),
        gridspec_kw={"width_ratios": [1.3, 1]},
    )

    # Positive rewards
    y_pos = np.arange(len(reward_labels))
    bars_r = ax_r.barh(y_pos, reward_values, color=_C_GREEN, alpha=0.8, height=0.65)
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(reward_labels)
    ax_r.set_xlabel("Mean Episode Reward (last 10%)")
    ax_r.set_title("Reward Components", )
    ax_r.grid(True, alpha=0.3, axis="x")
    ax_r.invert_yaxis()

    for bar, val in zip(bars_r, reward_values):
        ax_r.text(
            bar.get_width() + max(reward_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=8,
            color=_C_DARK,
        )

    # Penalties (negative values)
    y_neg = np.arange(len(penalty_labels))
    bars_p = ax_p.barh(y_neg, penalty_values, color=_C_RED, alpha=0.8, height=0.65)
    ax_p.set_yticks(y_neg)
    ax_p.set_yticklabels(penalty_labels)
    ax_p.set_xlabel("Mean Episode Penalty (last 10%)")
    ax_p.set_title("Penalty Components", )
    ax_p.grid(True, alpha=0.3, axis="x")
    ax_p.invert_yaxis()

    for bar, val in zip(bars_p, penalty_values):
        offset = min(penalty_values) * 0.02 if penalty_values else 0
        ax_p.text(
            bar.get_width() + offset if val >= 0 else bar.get_width() - abs(offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
            color=_C_DARK,
        )

    finalise(fig, output, title="Converged Reward Breakdown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run directory name (e.g. 2026-03-08_04-40-18_ppo_torch)",
    )
    args = parser.parse_args()

    if args.run:
        run_dir = LOGS_ROOT / args.run
    else:
        run_dir = resolve_latest_run(LOGS_ROOT)

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard events in {run_dir}")

    print(f"Loading run: {run_dir.name}")
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()

    total_tags = len(accumulator.Tags().get("scalars", []))
    total_events = sum(
        len(accumulator.Scalars(tag))
        for tag in accumulator.Tags().get("scalars", [])
    )
    print(f"  {total_tags} metrics, {total_events:,} data points")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {FIGURES_DIR}/")

    plot_total_reward(accumulator, FIGURES_DIR / "01_total_reward.png")
    plot_task_success(accumulator, FIGURES_DIR / "02_task_success.png")
    plot_reward_decomposition(accumulator, FIGURES_DIR / "03_reward_decomposition.png")
    plot_penalties(accumulator, FIGURES_DIR / "04_penalties.png")
    plot_policy_diagnostics(accumulator, FIGURES_DIR / "05_policy_diagnostics.png")
    plot_converged_summary(accumulator, FIGURES_DIR / "06_converged_breakdown.png")

    print("Done — 6 figures generated.")


if __name__ == "__main__":
    main()

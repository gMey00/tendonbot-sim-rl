#!/usr/bin/env python3
"""Generate training result plots for the manager-based reach task."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

VARIANT_LOG_DIRS: Final[dict[str, Path]] = {
    "tensegrity": Path("logs/skrl/reach/tensegrity"),
    "tensegrity_tendon": Path("logs/skrl/reach/tensegrity_tendon"),
    "ur10e": Path("logs/skrl/reach/ur10e"),
    "kinova": Path("logs/skrl/reach/kinova"),
    "kinova_frankenstein": Path("logs/skrl/reach/kinova_frankenstein"),
}
FIGURES_BASE: Final = Path(
    "source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/figures"
)
REPORTS_BASE: Final = Path(
    "source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/reports"
)

DEFAULT_CURRICULUM_STEP: Final = 4500
SMOOTHING_WEIGHT: Final = 0.85
DPI: Final = 180
FIGSIZE_WIDE: Final = (11, 4.5)

VARIANT_LABELS: Final[dict[str, str]] = {
    "tensegrity": "Tensegrity 5-DOF (PD)",
    "tensegrity_tendon": "Tensegrity 5-DOF (Tendon)",
    "ur10e": "UR10e 6-DOF (PD)",
    "kinova": "Kinova Gen3 7-DOF (PD)",
    "kinova_frankenstein": "Kinova Frankenstein 9-DOF (PD)",
}


class Palette:
    blue: str = "#4477AA"
    cyan: str = "#66CCEE"
    green: str = "#228833"
    yellow: str = "#CCBB44"
    red: str = "#EE6677"
    purple: str = "#AA3377"
    grey: str = "#BBBBBB"
    orange: str = "#EE8866"
    dark: str = "#333333"


PALETTE = Palette()


def write_report(variant: str, run_name: str, stats: dict, report_path: Path) -> None:
    scalars = stats["scalars"]

    def metric(tag: str) -> float:
        entry = scalars.get(tag)
        return float(entry["final_mean"]) if entry else 0.0

    success_tag = (
        "Info / Episode_Reward/pose_goal_reached"
        if "Info / Episode_Reward/pose_goal_reached" in scalars
        else "Info / Episode_Reward/goal_reached"
    )

    report = f"""# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `{variant}` |
| Run ID | `{run_name}` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | {metric('Reward / Total reward (mean)') if 'Reward / Total reward (mean)' in scalars else metric('Reward / Instantaneous reward (mean)'):.4f} |
| Position tracking reward | {metric('Info / Episode_Reward/end_effector_position_tracking'):.4f} |
| Orientation tracking reward | {metric('Info / Episode_Reward/end_effector_orientation_tracking'):.4f} |
| Position fine-grained reward | {metric('Info / Episode_Reward/end_effector_position_tracking_fine_grained'):.6f} |
| Orientation fine-grained reward | {metric('Info / Episode_Reward/end_effector_orientation_tracking_fine_grained'):.6f} |
| Pose goal reached reward | {metric(success_tag):.6f} |
| Action rate penalty | {metric('Info / Episode_Reward/action_rate'):.6f} |
| Joint velocity penalty | {metric('Info / Episode_Reward/joint_vel'):.6f} |
| Policy std deviation | {metric('Policy / Standard deviation'):.4f} |
| Learning rate (final) | {metric('Learning / Learning rate'):.2e} |

## Figures

### Total Reward

![Total Reward](../../figures/{variant}/01_total_reward.png)

### Task Success

![Task Success](../../figures/{variant}/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/{variant}/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/{variant}/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/{variant}/05_converged_breakdown.png)
"""

    report_path.write_text(report)


def resolve_latest_run(root: Path) -> Path:
    candidates = sorted(root.glob("*_ppo_torch"))
    if not candidates:
        raise FileNotFoundError(f"No PPO runs found under {root}")
    return candidates[-1]


def read_run_status(run_dir: Path) -> dict:
    status_file = run_dir / "run_status.json"
    if not status_file.exists():
        return {}
    try:
        return json.loads(status_file.read_text())
    except json.JSONDecodeError:
        return {}


def read_target_timesteps(run_dir: Path) -> int | None:
    agent_yaml = run_dir / "params" / "agent.yaml"
    if not agent_yaml.exists():
        return None
    match = re.search(r"^\s*timesteps:\s*(\d+)\s*$", agent_yaml.read_text(), flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def read_checkpoint_max_step(run_dir: Path) -> int:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return -1
    checkpoint_steps: list[int] = []
    for file in checkpoints_dir.glob("agent_*.pt"):
        match = re.match(r"agent_(\d+)\.pt", file.name)
        if match:
            checkpoint_steps.append(int(match.group(1)))
    return max(checkpoint_steps) if checkpoint_steps else -1


def read_max_logged_step(accumulator: EventAccumulator) -> int:
    max_step = -1
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if events:
            max_step = max(max_step, int(events[-1].step))
    return max_step


def validate_run_completion(
    run_dir: Path,
    accumulator: EventAccumulator,
    min_completion_ratio: float,
    allow_incomplete: bool,
) -> dict:
    status_data = read_run_status(run_dir)
    target_timesteps = status_data.get("target_timesteps") or read_target_timesteps(run_dir)
    checkpoint_max = read_checkpoint_max_step(run_dir)
    event_max = read_max_logged_step(accumulator)

    completed_by_status = status_data.get("status") == "completed"
    completed_by_steps = False
    if target_timesteps and target_timesteps > 0:
        step_ratio = min(checkpoint_max, event_max) / float(target_timesteps)
        completed_by_steps = step_ratio >= min_completion_ratio

    is_complete = completed_by_status or completed_by_steps
    summary = {
        "is_complete": is_complete,
        "status": status_data.get("status", "unknown"),
        "target_timesteps": int(target_timesteps) if target_timesteps else None,
        "checkpoint_max_step": int(checkpoint_max),
        "event_max_step": int(event_max),
    }

    if not is_complete and not allow_incomplete:
        raise RuntimeError(
            "Run appears incomplete. "
            f"status={summary['status']}, target={summary['target_timesteps']}, "
            f"checkpoint_max={summary['checkpoint_max_step']}, event_max={summary['event_max_step']}. "
            "Use --allow-incomplete to force plotting partial runs."
        )

    return summary


def load_scalars(accumulator: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    if tag not in accumulator.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = accumulator.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_event_accumulator(run_dir: Path) -> EventAccumulator:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard events in {run_dir}")

    if len(event_files) == 1:
        accumulator = EventAccumulator(str(event_files[0]))
        accumulator.Reload()
        return accumulator

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    return accumulator


def smooth(values: np.ndarray, weight: float = SMOOTHING_WEIGHT) -> np.ndarray:
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def steps_to_k(steps: np.ndarray | int) -> np.ndarray | float:
    return steps / 1000.0


def add_curriculum_marker(axis: plt.Axes, curriculum_step: int) -> None:
    axis.axvline(
        steps_to_k(curriculum_step),
        color=PALETTE.grey,
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label="_nolegend_",
    )


def annotate_curriculum(axis: plt.Axes, curriculum_step: int, y_frac: float = 0.95) -> None:
    transform = axis.get_xaxis_transform()
    axis.text(
        steps_to_k(curriculum_step),
        y_frac,
        " Reg. ramp",
        transform=transform,
        fontsize=7.5,
        color=PALETTE.grey,
        va="top",
        ha="left",
        alpha=0.85,
    )


def resolve_tag(accumulator: EventAccumulator, *candidates: str) -> tuple[np.ndarray, np.ndarray]:
    available = set(accumulator.Tags().get("scalars", []))
    for tag in candidates:
        if tag in available:
            return load_scalars(accumulator, tag)
    return np.array([]), np.array([])


def finalise(figure: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        figure.tight_layout()
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"  ✓ {path.name}")


def plot_total_reward(
    accumulator: EventAccumulator, output: Path, curriculum_step: int, variant_label: str
) -> None:
    steps_mean, values_mean = resolve_tag(
        accumulator,
        "Reward / Total reward (mean)",
        "Reward / Instantaneous reward (mean)",
    )
    _, values_max = resolve_tag(
        accumulator,
        "Reward / Total reward (max)",
        "Reward / Instantaneous reward (max)",
    )
    _, values_min = resolve_tag(
        accumulator,
        "Reward / Total reward (min)",
        "Reward / Instantaneous reward (min)",
    )

    x = steps_to_k(steps_mean)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    if len(x):
        if len(values_max) == 0:
            values_max = values_mean
        if len(values_min) == 0:
            values_min = values_mean
        ax.fill_between(
            x,
            smooth(values_min, 0.80),
            smooth(values_max, 0.80),
            alpha=0.15,
            color=PALETTE.blue,
            linewidth=0,
        )
        ax.plot(x, smooth(values_mean), color=PALETTE.blue, linewidth=1.8, label="Mean")
        ax.plot(x, smooth(values_max, 0.80), color=PALETTE.blue, linewidth=0.6, alpha=0.5, linestyle="--", label="Max")
        ax.plot(x, smooth(values_min, 0.80), color=PALETTE.blue, linewidth=0.6, alpha=0.5, linestyle=":", label="Min")

    add_curriculum_marker(ax, curriculum_step)
    annotate_curriculum(ax, curriculum_step)

    ax.set_xlabel("Training Steps (×1 000)", fontsize=10)
    using_total = "Reward / Total reward (mean)" in accumulator.Tags().get("scalars", [])
    ax.set_ylabel("Episode Return" if using_total else "Instantaneous Reward", fontsize=10)
    ax.set_title(f"Total Episode Reward — {variant_label}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    finalise(fig, output)


def plot_task_success(
    accumulator: EventAccumulator, output: Path, curriculum_step: int, variant_label: str
) -> None:
    steps_goal, values_goal = resolve_tag(
        accumulator,
        "Info / Episode_Reward/pose_goal_reached",
        "Info / Episode_Reward/goal_reached",
    )
    goal_label = (
        "Pose Goal Reached"
        if "Info / Episode_Reward/pose_goal_reached" in accumulator.Tags().get("scalars", [])
        else "Goal Reached (pos only)"
    )
    steps_prox, values_prox = load_scalars(
        accumulator, "Info / Episode_Reward/end_effector_position_tracking_fine_grained"
    )

    fig, ax_left = plt.subplots(figsize=FIGSIZE_WIDE)
    ax_right = ax_left.twinx()

    if len(steps_goal):
        ax_left.plot(
            steps_to_k(steps_goal),
            smooth(values_goal),
            color=PALETTE.green,
            linewidth=1.8,
            label=goal_label,
        )
    ax_left.set_ylabel("Goal Reached Reward", fontsize=10, color=PALETTE.green)
    ax_left.tick_params(axis="y", labelcolor=PALETTE.green)

    if len(steps_prox):
        ax_right.plot(
            steps_to_k(steps_prox),
            smooth(values_prox),
            color=PALETTE.blue,
            linewidth=1.8,
            label="Position Fine-Grained",
        )
    ax_right.set_ylabel("Fine-Grained Position Reward", fontsize=10, color=PALETTE.blue)
    ax_right.tick_params(axis="y", labelcolor=PALETTE.blue)

    add_curriculum_marker(ax_left, curriculum_step)
    annotate_curriculum(ax_left, curriculum_step)

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="center right",
        fontsize=9,
        framealpha=0.9,
    )

    ax_left.set_xlabel("Training Steps (×1 000)", fontsize=10)
    ax_left.set_title(f"Task Success Metrics — {variant_label}", fontsize=12, fontweight="bold")
    ax_left.grid(True, alpha=0.3)
    ax_left.set_xlim(left=0)

    finalise(fig, output)


def plot_reward_decomposition(
    accumulator: EventAccumulator, output: Path, curriculum_step: int, variant_label: str
) -> None:
    available = set(accumulator.Tags().get("scalars", []))

    success_tag = (
        "Info / Episode_Reward/pose_goal_reached"
        if "Info / Episode_Reward/pose_goal_reached" in available
        else "Info / Episode_Reward/goal_reached"
    )
    success_label = "Pose Goal Reached" if "pose_goal_reached" in success_tag else "Goal Reached"

    candidate_terms: list[tuple[str, str, str]] = [
        ("Info / Episode_Reward/end_effector_position_tracking", "Pos. Error", PALETTE.red),
        ("Info / Episode_Reward/end_effector_position_tracking_fine_grained", "Pos. Fine", PALETTE.cyan),
        ("Info / Episode_Reward/end_effector_orientation_tracking", "Orient. Error", PALETTE.orange),
        ("Info / Episode_Reward/end_effector_orientation_tracking_fine_grained", "Orient. Fine", PALETTE.yellow),
        (success_tag, success_label, PALETTE.green),
        ("Info / Episode_Reward/action_rate", "Action Rate", PALETTE.grey),
        ("Info / Episode_Reward/joint_vel", "Joint Velocity", PALETTE.purple),
    ]
    reward_terms = [(t, l, c) for t, l, c in candidate_terms if t in available]

    fig, axes = plt.subplots(
        len(reward_terms), 1,
        figsize=(11, 1.4 * max(1, len(reward_terms)) + 1.2),
        sharex=True,
    )
    if len(reward_terms) == 1:
        axes = [axes]

    for ax, (tag, label, colour) in zip(axes, reward_terms):
        steps, values = load_scalars(accumulator, tag)
        if len(steps) == 0:
            continue

        x = steps_to_k(steps)
        ax.fill_between(x, 0, smooth(values), alpha=0.25, color=colour, linewidth=0)
        ax.plot(x, smooth(values), color=colour, linewidth=1.4)

        add_curriculum_marker(ax, curriculum_step)

        ax.set_ylabel(label, fontsize=9, fontweight="bold", rotation=0, labelpad=65, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(left=0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    axes[-1].set_xlabel("Training Steps (×1 000)", fontsize=10)
    fig.suptitle(
        f"Reward Decomposition — {variant_label}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(hspace=0.15)

    finalise(fig, output, tight=False)


def plot_policy_diagnostics(
    accumulator: EventAccumulator, output: Path, curriculum_step: int, variant_label: str
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    steps, values = load_scalars(accumulator, "Policy / Standard deviation")
    if len(steps):
        ax.plot(steps_to_k(steps), smooth(values), color=PALETTE.blue, linewidth=1.4)
    add_curriculum_marker(ax, curriculum_step)
    ax.set_title("Policy Std. Deviation", fontsize=10, fontweight="bold")
    ax.set_ylabel("σ", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    steps, values = load_scalars(accumulator, "Loss / Policy loss")
    if len(steps):
        ax.plot(steps_to_k(steps), smooth(values), color=PALETTE.green, linewidth=1.4)
    add_curriculum_marker(ax, curriculum_step)
    ax.set_title("Policy (Surrogate) Loss", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    steps, values = load_scalars(accumulator, "Loss / Value loss")
    if len(steps):
        ax.plot(steps_to_k(steps), smooth(values), color=PALETTE.orange, linewidth=1.4)
    add_curriculum_marker(ax, curriculum_step)
    ax.set_title("Value Function Loss", fontsize=10, fontweight="bold")
    ax.set_xlabel("Steps (×1 000)", fontsize=10)
    ax.set_ylabel("MSE", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    steps, values = load_scalars(accumulator, "Learning / Learning rate")
    if len(steps):
        ax.plot(steps_to_k(steps), values, color=PALETTE.purple, linewidth=1.4)
    add_curriculum_marker(ax, curriculum_step)
    ax.set_title("Learning Rate (KL-adaptive)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Steps (×1 000)", fontsize=10)
    ax.grid(True, alpha=0.3)

    for row in axes:
        for axis in row:
            axis.set_xlim(left=0)

    fig.suptitle(
        f"Policy & Training Diagnostics — {variant_label}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    finalise(fig, output)


def compute_stats(accumulator: EventAccumulator) -> dict:
    stats: dict = {}
    for tag in accumulator.Tags().get("scalars", []):
        evs = accumulator.Scalars(tag)
        if not evs:
            continue
        vals = np.asarray([e.value for e in evs], dtype=float)
        cutoff = int(len(vals) * 0.9)
        final_window = vals[cutoff:]
        finite_window = final_window[np.isfinite(final_window)]
        stats[tag] = {
            "final_mean": float(np.mean(finite_window)) if finite_window.size else 0.0,
            "final_std": float(np.std(finite_window)) if finite_window.size else 0.0,
            "max_step": int(evs[-1].step),
        }
    return stats


def plot_converged_summary(accumulator: EventAccumulator, output: Path, variant_label: str) -> None:
    available = set(accumulator.Tags().get("scalars", []))

    success_tag = (
        "Info / Episode_Reward/pose_goal_reached"
        if "Info / Episode_Reward/pose_goal_reached" in available
        else "Info / Episode_Reward/goal_reached"
    )
    success_label = "Pose Goal Reached" if "pose_goal_reached" in success_tag else "Goal Reached"

    candidate_reward_tags: list[tuple[str, str]] = [
        (success_tag, success_label),
        ("Info / Episode_Reward/end_effector_position_tracking_fine_grained", "Pos. Fine"),
        ("Info / Episode_Reward/end_effector_orientation_tracking_fine_grained", "Orient. Fine"),
    ]
    reward_tags = [(tag, label) for tag, label in candidate_reward_tags if tag in available]

    penalty_tags: list[tuple[str, str]] = [
        ("Info / Episode_Reward/end_effector_position_tracking", "Pos. Error"),
        ("Info / Episode_Reward/end_effector_orientation_tracking", "Orient. Error"),
        ("Info / Episode_Reward/action_rate", "Action Rate"),
        ("Info / Episode_Reward/joint_vel", "Joint Velocity"),
    ]

    def final_mean(tag: str) -> float:
        _, values = load_scalars(accumulator, tag)
        if len(values) == 0:
            return 0.0
        cutoff = int(len(values) * 0.9)
        final_window = np.asarray(values[cutoff:], dtype=float)
        finite_window = final_window[np.isfinite(final_window)]
        return float(np.mean(finite_window)) if finite_window.size else 0.0

    reward_labels = [label for _, label in reward_tags]
    reward_values = [final_mean(tag) for tag, _ in reward_tags]
    penalty_labels = [label for _, label in penalty_tags]
    penalty_values = [final_mean(tag) for tag, _ in penalty_tags]

    fig, (ax_r, ax_p) = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.3]},
    )

    y_pos = np.arange(len(reward_labels))
    bars_r = ax_r.barh(y_pos, reward_values, color=PALETTE.green, alpha=0.8, height=0.55)
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(reward_labels, fontsize=9)
    ax_r.set_xlabel("Mean Episode Reward (last 10%)", fontsize=9)
    ax_r.set_title("Reward Components", fontsize=10, fontweight="bold")
    ax_r.grid(True, alpha=0.3, axis="x")
    ax_r.invert_yaxis()

    for bar, val in zip(bars_r, reward_values):
        ax_r.text(
            bar.get_width() + max(max(reward_values, default=0.01), 0.01) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
            color=PALETTE.dark,
        )

    y_neg = np.arange(len(penalty_labels))
    bars_p = ax_p.barh(y_neg, penalty_values, color=PALETTE.red, alpha=0.8, height=0.55)
    ax_p.set_yticks(y_neg)
    ax_p.set_yticklabels(penalty_labels, fontsize=9)
    ax_p.set_xlabel("Mean Episode Penalty (last 10%)", fontsize=9)
    ax_p.set_title("Penalty Components", fontsize=10, fontweight="bold")
    ax_p.grid(True, alpha=0.3, axis="x")
    ax_p.invert_yaxis()

    for bar, val in zip(bars_p, penalty_values):
        offset = min(min(penalty_values, default=-0.01), -0.01) * 0.02
        ax_p.text(
            bar.get_width() + offset if val >= 0 else bar.get_width() - abs(offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
            color=PALETTE.dark,
        )

    fig.suptitle(
        f"Converged Reward Breakdown — {variant_label}",
        fontsize=12,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.14, top=0.84, wspace=0.42)

    finalise(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run directory name (e.g. 2026-03-13_21-01-58_ppo_torch)",
    )
    parser.add_argument("--variant", type=str, required=True, choices=list(VARIANT_LABELS.keys()))
    parser.add_argument(
        "--curriculum-step",
        type=int,
        default=DEFAULT_CURRICULUM_STEP,
        help=f"Step at which curriculum ramp completes (default: {DEFAULT_CURRICULUM_STEP})",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow plotting partial runs that have not reached configured timesteps.",
    )
    parser.add_argument(
        "--min-completion-ratio",
        type=float,
        default=0.98,
        help="Minimum ratio of logged/checkpoint steps to configured timesteps to consider a run complete.",
    )
    args = parser.parse_args()

    logs_root = VARIANT_LOG_DIRS[args.variant]
    run_dir = logs_root / args.run if args.run else resolve_latest_run(logs_root)

    print(f"Loading run: {run_dir.name}")
    accumulator = load_event_accumulator(run_dir)
    completion = validate_run_completion(
        run_dir=run_dir,
        accumulator=accumulator,
        min_completion_ratio=args.min_completion_ratio,
        allow_incomplete=args.allow_incomplete,
    )

    total_tags = len(accumulator.Tags().get("scalars", []))
    total_events = sum(len(accumulator.Scalars(tag)) for tag in accumulator.Tags().get("scalars", []))
    print(f"  {total_tags} metrics, {total_events:,} data points")
    print(
        "  completion:",
        json.dumps(
            {
                "status": completion["status"],
                "target_timesteps": completion["target_timesteps"],
                "checkpoint_max_step": completion["checkpoint_max_step"],
                "event_max_step": completion["event_max_step"],
            }
        ),
    )

    variant_label = VARIANT_LABELS[args.variant]
    curriculum_step = args.curriculum_step

    figures_dir = FIGURES_BASE / args.variant
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = REPORTS_BASE / args.variant
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {figures_dir}/")

    plot_total_reward(accumulator, figures_dir / "01_total_reward.png", curriculum_step, variant_label)
    plot_task_success(accumulator, figures_dir / "02_task_success.png", curriculum_step, variant_label)
    plot_reward_decomposition(accumulator, figures_dir / "03_reward_decomposition.png", curriculum_step, variant_label)
    plot_policy_diagnostics(accumulator, figures_dir / "04_policy_diagnostics.png", curriculum_step, variant_label)
    plot_converged_summary(accumulator, figures_dir / "05_converged_breakdown.png", variant_label)

    stats = compute_stats(accumulator)
    stats_path = figures_dir / "training_stats.json"
    with open(stats_path, "w") as file:
        json.dump(
            {
                "run": run_dir.name,
                "variant": args.variant,
                "completion": completion,
                "scalars": stats,
            },
            file,
            indent=2,
        )
    print("  ✓ training_stats.json")

    report_name = f"reach_results_{args.variant}_{run_dir.name}.md"
    write_report(args.variant, run_dir.name, {"scalars": stats}, reports_dir / report_name)
    print(f"  ✓ {report_name}")

    print("Done — 5 figures + stats + report generated.")


if __name__ == "__main__":
    main()

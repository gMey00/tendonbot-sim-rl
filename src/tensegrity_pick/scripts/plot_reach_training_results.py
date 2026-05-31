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

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / ".config"))
import plot_config as pcfg
pcfg.apply_style()

# Relative sub-paths (joined with --root at runtime). The canonical extension
# tree lives under src/tensegrity_pick/source/... — older versions of this
# script wrote to a stray top-level source/ tree, which is the wrong location.
_LOG_SUBDIR: Final = "logs/skrl/reach"
_FIGURES_SUBDIR: Final = (
    "src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/figures"
)
_REPORTS_SUBDIR: Final = (
    "src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/reports"
)

VARIANT_NAMES: Final[list[str]] = [
    "tensegrity",
    "tensegrity_tendon",
    "tensegrity_physical_tendon",
    "ur10e",
    "kinova",
    "kinova_frankenstein",
]

DEFAULT_CURRICULUM_STEP: Final = 4500
SMOOTHING_WEIGHT: Final = 0.85

# Reward weights used to scale binary success metrics for percentage display
SUCCESS_WEIGHT: Final = 1e-6
POSITION_ERROR_WEIGHT: Final = -0.2

VARIANT_LABELS: Final[dict[str, str]] = {
    "tensegrity": "Tensegrity 5-DOF (PD)",
    "tensegrity_tendon": "Tensegrity 5-DOF (Tendon)",
    "tensegrity_physical_tendon": "Tensegrity 5-DOF (Physical Tendon)",
    "ur10e": "UR10e 6-DOF (PD)",
    "kinova": "Kinova Gen3 7-DOF (PD)",
    "kinova_frankenstein": "Kinova Frankenstein 9-DOF (PD)",
}

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
| Position reached (< 5cm) | {metric('Info / Episode_Reward/position_reached'):.4f} |
| Orientation reached (< 0.3 rad) | {metric('Info / Episode_Reward/orientation_reached'):.4f} |
| Pose reached (both) | {metric('Info / Episode_Reward/pose_reached'):.4f} |
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

### Episode Length

![Episode Length](../../figures/{variant}/06_episode_length.png)
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
    finite_mask = np.isfinite(values)
    smoothed = np.full_like(values, np.nan)
    last_valid = np.nan
    for i in range(len(values)):
        if finite_mask[i]:
            last_valid = values[i] if np.isnan(last_valid) else weight * last_valid + (1 - weight) * values[i]
            smoothed[i] = last_valid
    return smoothed


def steps_to_k(steps: np.ndarray | int) -> np.ndarray | float:
    return steps / 1000.0


def add_curriculum_marker(axis: plt.Axes, curriculum_step: int) -> None:
    axis.axvline(
        steps_to_k(curriculum_step),
        color=_C_GREY,
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
        color=_C_GREY,
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


def finalise(figure: plt.Figure, path: Path, tight: bool = True, *, title: str | None = None) -> None:
    pcfg.finalize(figure, path, title=title, tight=tight)


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
            color=_C_BLUE,
            linewidth=0,
        )
        ax.plot(x, smooth(values_mean), color=_C_BLUE, linewidth=1.8, label="Mean")
        ax.plot(x, smooth(values_max, 0.80), color=_C_BLUE, linewidth=0.6, alpha=0.5, linestyle="--", label="Max")
        ax.plot(x, smooth(values_min, 0.80), color=_C_BLUE, linewidth=0.6, alpha=0.5, linestyle=":", label="Min")

    add_curriculum_marker(ax, curriculum_step)
    annotate_curriculum(ax, curriculum_step)

    ax.set_xlabel("Training Steps (×1 000)")
    using_total = "Reward / Total reward (mean)" in accumulator.Tags().get("scalars", [])
    ax.set_ylabel("Episode Return" if using_total else "Instantaneous Reward")
    ax.set_title(f"Total Episode Reward — {variant_label}", )
    ax.legend(loc="lower right", framealpha=0.9)
    # grid from style
    ax.set_xlim(left=0)

    finalise(fig, output)


def plot_task_success(
    accumulator: EventAccumulator, output: Path, curriculum_step: int, variant_label: str
) -> None:
    # Note: Episode_Reward tags are per-step means (weighted), NOT episode sums.
    # To recover the underlying metric: divide by the reward weight.

    # Success metrics (binary, weight = SUCCESS_WEIGHT)
    steps_pos, values_pos = resolve_tag(accumulator, "Info / Episode_Reward/position_reached")
    steps_orient, values_orient = resolve_tag(accumulator, "Info / Episode_Reward/orientation_reached")
    steps_pose, values_pose = resolve_tag(accumulator, "Info / Episode_Reward/pose_reached")

    if len(steps_pos) == 0:
        steps_pos, values_pos = resolve_tag(accumulator, "Info / Episode_Reward/goal_reached")
    if len(steps_pose) == 0:
        steps_pose, values_pose = resolve_tag(accumulator, "Info / Episode_Reward/pose_goal_reached")

    # Mean L2 distance to goal for proximity (right axis, in metres)
    steps_dist, values_dist = resolve_tag(
        accumulator, "Info / Episode_Reward/end_effector_position_tracking",
    )

    fig, ax_left = plt.subplots(figsize=FIGSIZE_WIDE)
    has_success = False

    if len(steps_pos):
        pct = values_pos / SUCCESS_WEIGHT * 100.0
        ax_left.plot(steps_to_k(steps_pos), smooth(pct), color=_C_GREEN, linewidth=1.8, label="Position < 5 cm")
        has_success = True
    if len(steps_orient):
        pct = values_orient / SUCCESS_WEIGHT * 100.0
        ax_left.plot(steps_to_k(steps_orient), smooth(pct), color=_C_ORANGE, linewidth=1.8, label="Orientation < 0.3 rad")
        has_success = True
    if len(steps_pose):
        pct = values_pose / SUCCESS_WEIGHT * 100.0
        ax_left.plot(steps_to_k(steps_pose), smooth(pct), color=_C_BLUE, linewidth=1.8, label="Full Pose")
        has_success = True

    ax_left.set_xlabel("Training Steps (×1 000)")
    ax_left.set_ylabel("Success Rate (%)")
    ax_left.set_title(f"Task Success Metrics — {variant_label}", )
    # grid from style
    ax_left.set_xlim(left=0)
    ax_left.set_ylim(bottom=0)

    # Right axis: mean distance to goal in metres
    ax_right = ax_left.twinx()
    if len(steps_dist):
        dist_m = values_dist / POSITION_ERROR_WEIGHT  # undo weight → positive metres
        ax_right.plot(
            steps_to_k(steps_dist), smooth(dist_m),
            color=_C_PURPLE, linewidth=1.4, linestyle="--", label="Mean Distance",
        )
    ax_right.set_ylabel("Mean Distance to Goal (m)", fontsize=10, color=_C_PURPLE)
    ax_right.tick_params(axis="y", labelcolor=_C_PURPLE)

    add_curriculum_marker(ax_left, curriculum_step)
    annotate_curriculum(ax_left, curriculum_step)

    # Combined legend
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    if lines_l or lines_r:
        ax_left.legend(
            lines_l + lines_r, labels_l + labels_r,
            loc="center right",
            fontsize=9,
            framealpha=0.9,
        )

    finalise(fig, output)


def plot_episode_length(
    accumulator: EventAccumulator, output: Path, curriculum_step: int, variant_label: str
) -> None:
    steps, values = resolve_tag(accumulator, "Episode / Total timesteps (mean)")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    if len(steps):
        ax.plot(steps_to_k(steps), smooth(values, 0.7), color=_C_TEAL, linewidth=1.8)

    add_curriculum_marker(ax, curriculum_step)
    annotate_curriculum(ax, curriculum_step)

    ax.set_xlabel("Training Steps (×1 000)")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title(f"Episode Length — {variant_label}", )
    # grid from style
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    finalise(fig, output)


def plot_reward_decomposition(
    accumulator: EventAccumulator, output: Path, curriculum_step: int, variant_label: str
) -> None:
    available = set(accumulator.Tags().get("scalars", []))

    candidate_terms: list[tuple[str, str, str]] = [
        ("Info / Episode_Reward/end_effector_position_tracking", "Pos. Error", _C_RED),
        ("Info / Episode_Reward/end_effector_position_tracking_fine_grained", "Pos. Fine", _C_TEAL),
        ("Info / Episode_Reward/end_effector_position_tracking_proximity", "Pos. Proximity", _C_AMBER),
        ("Info / Episode_Reward/end_effector_orientation_tracking", "Orient. Error", _C_ORANGE),
        ("Info / Episode_Reward/action_rate", "Action Rate", _C_GREY),
        ("Info / Episode_Reward/joint_vel", "Joint Velocity", _C_PURPLE),
    ]
    reward_terms = [(t, l, c) for t, l, c in candidate_terms if t in available]

    if not reward_terms:
        print("  ⚠ No reward terms available — skipping reward decomposition")
        return

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
        # grid from style
        ax.set_xlim(left=0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    axes[-1].set_xlabel("Training Steps (×1 000)")
    fig.subplots_adjust(hspace=0.15)

    finalise(fig, output, tight=False,
             title=f"Reward Decomposition — {variant_label}")


def plot_policy_diagnostics(
    accumulator: EventAccumulator, output: Path, curriculum_step: int, variant_label: str
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    steps, values = load_scalars(accumulator, "Policy / Standard deviation")
    if len(steps):
        ax.plot(steps_to_k(steps), smooth(values), color=_C_BLUE, linewidth=1.4)
    add_curriculum_marker(ax, curriculum_step)
    ax.set_title("Policy Std. Deviation", )
    ax.set_ylabel("σ")
    # grid from style

    ax = axes[0, 1]
    steps, values = load_scalars(accumulator, "Loss / Policy loss")
    if len(steps):
        ax.plot(steps_to_k(steps), smooth(values), color=_C_GREEN, linewidth=1.4)
    add_curriculum_marker(ax, curriculum_step)
    ax.set_title("Policy (Surrogate) Loss", )
    # grid from style

    ax = axes[1, 0]
    steps, values = load_scalars(accumulator, "Loss / Value loss")
    if len(steps):
        ax.plot(steps_to_k(steps), smooth(values), color=_C_ORANGE, linewidth=1.4)
    add_curriculum_marker(ax, curriculum_step)
    ax.set_title("Value Function Loss", )
    ax.set_xlabel("Steps (×1 000)")
    ax.set_ylabel("MSE")
    # grid from style

    ax = axes[1, 1]
    steps, values = load_scalars(accumulator, "Learning / Learning rate")
    if len(steps):
        ax.plot(steps_to_k(steps), values, color=_C_PURPLE, linewidth=1.4)
    add_curriculum_marker(ax, curriculum_step)
    ax.set_title("Learning Rate (KL-adaptive)", )
    ax.set_xlabel("Steps (×1 000)")
    # grid from style

    for row in axes:
        for axis in row:
            axis.set_xlim(left=0)

    finalise(fig, output, title=f"Policy & Training Diagnostics — {variant_label}")


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


def _fmt(val: float) -> str:
    """Format a scalar for bar-chart annotations, using SI-like concise form."""
    if val == 0.0:
        return "0"
    abs_v = abs(val)
    if abs_v >= 0.1:
        return f"{val:.3f}"
    return f"{val:.3e}"


def plot_converged_summary(accumulator: EventAccumulator, output: Path, variant_label: str) -> None:
    available = set(accumulator.Tags().get("scalars", []))

    candidate_reward_tags: list[tuple[str, str]] = [
        ("Info / Episode_Reward/end_effector_position_tracking_proximity", "Pos. Proximity"),
        ("Info / Episode_Reward/end_effector_position_tracking_fine_grained", "Pos. Fine"),
    ]
    reward_tags = [(tag, label) for tag, label in candidate_reward_tags if tag in available]

    candidate_penalty_tags: list[tuple[str, str]] = [
        ("Info / Episode_Reward/end_effector_position_tracking", "Pos. Error"),
        ("Info / Episode_Reward/end_effector_orientation_tracking", "Orient. Error"),
        ("Info / Episode_Reward/action_rate", "Action Rate"),
        ("Info / Episode_Reward/joint_vel", "Joint Velocity"),
    ]
    penalty_tags = [(tag, label) for tag, label in candidate_penalty_tags if tag in available]

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

    # --- Reward panel ---
    y_pos = np.arange(len(reward_labels))
    bars_r = ax_r.barh(y_pos, reward_values, color=_C_GREEN, alpha=0.8, height=0.55)
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(reward_labels)
    ax_r.set_xlabel("Mean Episode Reward (last 10%)")
    ax_r.set_title("Reward Components", )
    ax_r.grid(True, alpha=0.3, axis="x")
    ax_r.invert_yaxis()

    if reward_values:
        max_r = max(reward_values) if max(reward_values) > 0 else 1.0
        ax_r.set_xlim(0, max_r * 1.35)
        text_offset_r = max_r * 0.03
        for bar, val in zip(bars_r, reward_values):
            ax_r.text(
                bar.get_width() + text_offset_r,
                bar.get_y() + bar.get_height() / 2,
                _fmt(val),
                va="center",
                ha="left",
                fontsize=8,
                color=_C_DARK,
            )

    # --- Penalty panel ---
    y_neg = np.arange(len(penalty_labels))
    bars_p = ax_p.barh(y_neg, penalty_values, color=_C_RED, alpha=0.8, height=0.55)
    ax_p.set_yticks(y_neg)
    ax_p.set_yticklabels(penalty_labels)
    ax_p.set_xlabel("Mean Episode Penalty (last 10%)")
    ax_p.set_title("Penalty Components", )
    ax_p.grid(True, alpha=0.3, axis="x")
    ax_p.invert_yaxis()

    if penalty_values:
        min_p = min(penalty_values) if min(penalty_values) < 0 else -1.0
        ax_p.set_xlim(min_p * 1.35, 0)
        text_offset_p = abs(min_p) * 0.03
        for bar, val in zip(bars_p, penalty_values):
            ax_p.text(
                bar.get_width() - text_offset_p,
                bar.get_y() + bar.get_height() / 2,
                _fmt(val),
                va="center",
                ha="right",
                fontsize=8,
                color=_C_DARK,
            )

    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.14, top=0.84, wspace=0.42)

    finalise(fig, output, tight=False,
             title=f"Converged Reward Breakdown — {variant_label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run directory name (e.g. 2026-03-13_21-01-58_ppo_torch)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=False,
        default=None,
        choices=VARIANT_NAMES,
        help="Required for single-run mode; ignored in --seeds / --baselines modes.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Workspace root directory. Figure and report paths are resolved relative to this.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Override log directory for the variant (absolute or relative). "
        "If not set, defaults to <root>/logs/skrl/reach/<variant>.",
    )
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
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Seed-aggregation mode: list of *_ppo_torch run directories "
            "(absolute paths or paths relative to --root). Variants are inferred "
            "from the parent directory name. Plots mean ± std bands across seeds, "
            "one curve per variant. Mutually exclusive with --baselines."
        ),
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Baseline-comparison mode: list of evaluate.py JSON files. "
            "Plots a grouped bar chart of success_rate per variant for the "
            "{zero, random, checkpoint} agents, plus a dashed reference line "
            "at the PD heuristic's mean success rate. Mutually exclusive with --seeds."
        ),
    )
    args = parser.parse_args()

    if args.seeds is not None and args.baselines is not None:
        parser.error("--seeds and --baselines are mutually exclusive")

    if args.seeds is not None:
        _main_seeds(args)
        return
    if args.baselines is not None:
        _main_baselines(args)
        return

    if args.variant is None:
        parser.error("--variant is required in single-run mode")

    _main_single(args)


def _main_single(args) -> None:
    root = Path(args.root).resolve()
    if args.logs_dir:
        logs_root = Path(args.logs_dir).resolve()
    else:
        logs_root = root / _LOG_SUBDIR / args.variant
    figures_base = root / _FIGURES_SUBDIR
    reports_base = root / _REPORTS_SUBDIR

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

    figures_dir = figures_base / args.variant
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = reports_base / args.variant
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {figures_dir}/")

    plot_total_reward(accumulator, figures_dir / "01_total_reward.png", curriculum_step, variant_label)
    plot_task_success(accumulator, figures_dir / "02_task_success.png", curriculum_step, variant_label)
    plot_reward_decomposition(accumulator, figures_dir / "03_reward_decomposition.png", curriculum_step, variant_label)
    plot_policy_diagnostics(accumulator, figures_dir / "04_policy_diagnostics.png", curriculum_step, variant_label)
    plot_converged_summary(accumulator, figures_dir / "05_converged_breakdown.png", variant_label)
    plot_episode_length(accumulator, figures_dir / "06_episode_length.png", curriculum_step, variant_label)

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

    print("Done — 6 figures + stats + report generated.")


# ---------------------------------------------------------------------------
# Seed-aggregation mode (CS-4 part A)
# ---------------------------------------------------------------------------

def _infer_variant_from_run_path(run_path: Path) -> str:
    """Variant = name of the parent directory under logs/skrl/reach/."""
    name = run_path.parent.name
    if name in VARIANT_NAMES:
        return name
    raise ValueError(
        f"Cannot infer variant from {run_path}: parent directory '{name}' "
        f"is not one of {VARIANT_NAMES}"
    )


def _common_step_grid(step_arrays: list[np.ndarray], n: int = 400) -> np.ndarray:
    lo = max(float(s[0]) for s in step_arrays if len(s))
    hi = min(float(s[-1]) for s in step_arrays if len(s))
    if hi <= lo:
        raise ValueError("Seed runs do not overlap in step range")
    return np.linspace(lo, hi, n)


def _interp_seeds(step_arrays: list[np.ndarray], value_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = _common_step_grid(step_arrays)
    interped = np.vstack([
        np.interp(grid, s, v) for s, v in zip(step_arrays, value_arrays)
    ])
    return grid, interped.mean(axis=0), interped.std(axis=0)


_AGG_TAG_TOTAL_REWARD = ("Reward/total", "Reward / Total reward (mean)")
# skrl prefixes scalar tags with "Info / "; the reward term names are
# position_reached / orientation_reached / pose_reached (see reach_env_cfg.py).
_AGG_TAG_SUCCESS = (
    "Info / Episode_Reward/position_reached",
    "Info / Episode_Reward/pose_reached",
    "Info / Episode_Reward/orientation_reached",
    "Episode_Reward/position_reached",
    "Episode_Reward/pose_reached",
    "Episode_Reward/orientation_reached",
)


def _main_seeds(args) -> None:
    root = Path(args.root).resolve()
    figures_dir = root / _FIGURES_SUBDIR / "aggregate"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Group input run paths by inferred variant.
    by_variant: dict[str, list[Path]] = {}
    for raw in args.seeds:
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        by_variant.setdefault(_infer_variant_from_run_path(p), []).append(p)

    print(f"Seed aggregation over {sum(len(v) for v in by_variant.values())} runs "
          f"across {len(by_variant)} variants")
    for variant, runs in by_variant.items():
        print(f"  {variant}: {len(runs)} seed(s)")

    palette = [_C_BLUE, _C_GREEN, _C_AMBER, _C_RED, _C_PURPLE, _C_TEAL]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

    # Panel A: total reward
    for (variant, runs), colour in zip(by_variant.items(), palette):
        steps_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []
        for run in runs:
            acc = load_event_accumulator(run)
            steps, vals = load_scalars(acc, _AGG_TAG_TOTAL_REWARD[0])
            if not len(steps):
                steps, vals = load_scalars(acc, _AGG_TAG_TOTAL_REWARD[1])
            if len(steps):
                steps_list.append(steps)
                vals_list.append(smooth(vals))
        if not steps_list:
            print(f"  [warn] no total-reward tag for variant {variant}")
            continue
        grid, mean, std = _interp_seeds(steps_list, vals_list)
        axes[0].plot(steps_to_k(grid), mean, color=colour, label=VARIANT_LABELS[variant])
        axes[0].fill_between(steps_to_k(grid), mean - std, mean + std, alpha=0.20, color=colour)

    axes[0].set_xlabel("Steps [k]")
    axes[0].set_ylabel("Total reward (smoothed)")
    axes[0].set_title("Mean ± std across seeds")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    add_curriculum_marker(axes[0], args.curriculum_step)

    # Panel B: success rate (Episode_Reward / SUCCESS_WEIGHT → fraction)
    for (variant, runs), colour in zip(by_variant.items(), palette):
        steps_list, vals_list = [], []
        for run in runs:
            acc = load_event_accumulator(run)
            steps, vals = np.array([]), np.array([])
            for cand in _AGG_TAG_SUCCESS:
                steps, vals = load_scalars(acc, cand)
                if len(steps):
                    break
            if len(steps):
                steps_list.append(steps)
                vals_list.append(smooth(vals / SUCCESS_WEIGHT))
        if not steps_list:
            continue
        grid, mean, std = _interp_seeds(steps_list, vals_list)
        axes[1].plot(steps_to_k(grid), mean, color=colour, label=VARIANT_LABELS[variant])
        axes[1].fill_between(steps_to_k(grid), mean - std, mean + std, alpha=0.20, color=colour)

    axes[1].set_xlabel("Steps [k]")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Mean ± std across seeds")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)
    add_curriculum_marker(axes[1], args.curriculum_step)

    out = figures_dir / "07_seed_aggregated.png"
    finalise(fig, out, title="PPO learning curves (seed aggregation)")
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Baseline-comparison mode (CS-4 part B)
# ---------------------------------------------------------------------------

_AGENT_LABEL = {
    "zero": "Zero",
    "random": "Random",
    "checkpoint": "PPO",
    "heuristic": "DLS-IK (PD only)",
}
_AGENT_ORDER = ["zero", "random", "checkpoint"]


def _main_baselines(args) -> None:
    root = Path(args.root).resolve()
    figures_dir = root / _FIGURES_SUBDIR / "aggregate"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # records[(variant, agent)] = list of success_rate means (one per seed)
    records: dict[tuple[str, str], list[float]] = {}

    for raw in args.baselines:
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()
        payload = json.loads(p.read_text())
        variant = payload["variant"]
        agent = payload["agent"]
        sr_mean = payload["metrics"]["success_rate"]["mean"]
        records.setdefault((variant, agent), []).append(float(sr_mean))

    print(f"Loaded {len(args.baselines)} baseline JSON files into "
          f"{len(records)} (variant, agent) groups")
    for (variant, agent), vals in sorted(records.items()):
        print(f"  {variant:32s} {agent:11s}  n={len(vals)}  mean={np.mean(vals):.3f}")

    variants_present = sorted({v for (v, _) in records.keys()})

    # Verification gate: expect 5 seeds per (variant, agent in zero/random/checkpoint)
    for variant in variants_present:
        for agent in _AGENT_ORDER:
            n = len(records.get((variant, agent), []))
            if n and n != 5:
                print(f"  [warn] expected 5 seeds for ({variant}, {agent}); got {n}")

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(variants_present)), 4.5))

    n_groups = len(variants_present)
    n_bars = len(_AGENT_ORDER)
    bar_w = 0.8 / n_bars
    x = np.arange(n_groups)
    colours = {"zero": _C_GREY, "random": _C_AMBER, "checkpoint": _C_BLUE}

    for i, agent in enumerate(_AGENT_ORDER):
        means = []
        stds = []
        for variant in variants_present:
            vals = records.get((variant, agent), [])
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals)) if vals else 0.0)
        offset = (i - (n_bars - 1) / 2) * bar_w
        ax.bar(
            x + offset, means, bar_w, yerr=stds, capsize=3,
            color=colours[agent], label=_AGENT_LABEL[agent], edgecolor=_C_DARK, linewidth=0.5,
        )

    # Heuristic reference line (PD variant only).
    heur_vals = records.get(("tensegrity", "heuristic"), [])
    if heur_vals:
        h_mean = float(np.mean(heur_vals))
        ax.axhline(
            h_mean, color=_C_RED, linestyle="--", linewidth=1.2,
            label=f"{_AGENT_LABEL['heuristic']} mean = {h_mean:.2f}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants_present], rotation=15, ha="right")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    out = figures_dir / "08_baseline_comparison.png"
    finalise(fig, out, title="Reach baselines vs. learned policy (mean ± std over seeds)")
    print(f"  ✓ {out}")


if __name__ == "__main__":
    main()

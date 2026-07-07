#!/usr/bin/env python3
"""Generate training result plots for the Shirt Place task README.

Reads TensorBoard event files from a training run and produces
publication-quality figures saved into the task's ``figures/`` directory.

Unlike the cube_place/reach plotting scripts this one does NOT depend on the
``tensorboard`` package (not installed in ``env_isaaclab``) — it uses the
pure-Python reader in ``scripts/model_validation/read_tfevents.py``.

Usage
-----
    cd src/tensegrity_pick
    /home/robot/Isaac/IsaacLab/_isaac_sim/python.sh \
        scripts/plot_shirt_training_results.py [--run DIR_NAME] [--label NAME]

If ``--run`` is omitted the script auto-selects the latest ``*_ppo_torch*``
directory under ``logs/skrl/shirt_place/``.  ``--label`` selects the output
sub-directory under ``figures/`` (default: ``tensegrity``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
# Pure-python tfevents reader (no tensorboard dependency)
sys.path.insert(0, str(_HERE.parent / "model_validation"))
import read_tfevents as tfr  # noqa: E402

# Shared FAPS plot style (repo_root/.config/plot_config.py)
sys.path.insert(0, str(_HERE.parents[3] / ".config"))
import plot_config as pcfg  # noqa: E402
pcfg.apply_style()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGS_ROOT: Final = Path("logs/skrl/shirt_place")
FIGURES_BASE: Final = Path(
    "source/tensegrity_pick/tensegrity_pick/tasks/"
    "manager_based/shirt_place/figures"
)

# Regularisation ramp (action_rate / joint_vel curriculum) for the 2026-07-02
# from-scratch run; fine-tune runs use 2 000.
CURRICULUM_REGULARISATION_STEP: Final = 8_000

SMOOTHING_WEIGHT: Final = 0.90  # EMA smoothing factor for noisy curves
MAX_EPISODE_STEPS: Final = 300  # 5 s @ 60 Hz control

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
# tfevents loading (pure python)
# ---------------------------------------------------------------------------

def load_run_series(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load every scalar tag of *run_dir* as ``tag -> (steps, values)``."""
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard events in {run_dir}")
    series: dict[str, list[tuple[int, float]]] = {}
    for f in event_files:
        for payload in tfr._read_records(str(f)):
            step, kvs = tfr._parse_event(payload)
            for tag, val in kvs:
                series.setdefault(tag, []).append((step, val))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag, pts in series.items():
        pts.sort(key=lambda p: p[0])
        out[tag] = (
            np.array([p[0] for p in pts], dtype=float),
            np.array([p[1] for p in pts], dtype=float),
        )
    return out


def get(series: dict, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) for *tag*, tolerating an ``Info / `` prefix."""
    for candidate in (tag, f"Info / {tag}"):
        if candidate in series:
            return series[candidate]
    return np.array([]), np.array([])


def smooth(values: np.ndarray, weight: float = SMOOTHING_WEIGHT) -> np.ndarray:
    """Exponential moving average with *weight* ∈ (0, 1)."""
    if len(values) == 0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = weight * out[i - 1] + (1 - weight) * values[i]
    return out


def steps_to_k(steps: np.ndarray) -> np.ndarray:
    return steps / 1000.0


def add_curriculum_markers(axis: plt.Axes) -> None:
    axis.axvline(
        CURRICULUM_REGULARISATION_STEP / 1000.0,
        color=_C_GREY, linestyle="--", linewidth=0.9, alpha=0.7,
        label="_nolegend_",
    )


def annotate_curriculum(axis: plt.Axes, y_frac: float = 0.95) -> None:
    axis.text(
        CURRICULUM_REGULARISATION_STEP / 1000.0, y_frac, " Reg. ramp",
        transform=axis.get_xaxis_transform(),
        fontsize=7.5, color=_C_GREY, va="top", ha="left", alpha=0.85,
    )


def finalise(figure: plt.Figure, path: Path, tight: bool = True, *,
             title: str | None = None) -> None:
    pcfg.finalize(figure, path, title=title, tight=tight)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_total_reward(series: dict, output: Path) -> None:
    """Fig 1 — Total episode reward with min/max envelope."""
    steps, mean = get(series, "Reward / Total reward (mean)")
    _, vmax = get(series, "Reward / Total reward (max)")
    _, vmin = get(series, "Reward / Total reward (min)")
    if len(steps) == 0:
        print("  (skipped 01 — no total reward data)")
        return

    x = steps_to_k(steps)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if len(vmin) == len(mean) and len(vmax) == len(mean):
        ax.fill_between(x, smooth(vmin, 0.85), smooth(vmax, 0.85),
                        alpha=0.15, color=_C_BLUE, linewidth=0)
        ax.plot(x, smooth(vmax, 0.85), color=_C_BLUE, linewidth=0.6,
                alpha=0.5, linestyle="--", label="Max")
        ax.plot(x, smooth(vmin, 0.85), color=_C_BLUE, linewidth=0.6,
                alpha=0.5, linestyle=":", label="Min")
    ax.plot(x, smooth(mean), color=_C_BLUE, linewidth=1.8, label="Mean")

    cutoff = int(len(mean) * 0.9)
    plateau = float(np.mean(mean[cutoff:]))
    ax.axhline(plateau, color=_C_GREEN, linestyle="--", linewidth=1.1,
               alpha=0.8, label=f"Plateau ≈ {plateau:.0f}")

    add_curriculum_markers(ax)
    annotate_curriculum(ax)
    ax.set_xlabel("Training Steps (×1 000)")
    ax.set_ylabel("Episode Return")
    ax.set_title("Total Episode Reward")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(left=0)
    finalise(fig, output)


def plot_task_success(series: dict, output: Path) -> None:
    """Fig 2 — Grasp rate, place success rate, and shirt-in-drum fraction."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for tag, label, colour in (
        ("Metrics/grasp_rate", "Grasp Rate (% episodes)", _C_GREEN),
        ("Metrics/place_success_rate", "Place Success Rate (% episodes)", _C_ORANGE),
        ("Metrics/shirt_in_drum_fraction", "Shirt-in-Drum Fraction (% of cloth)", _C_BLUE),
    ):
        steps, vals = get(series, tag)
        if len(steps) == 0:
            continue
        ax.plot(steps_to_k(steps), smooth(vals * 100), color=colour,
                linewidth=1.8, label=label)

    add_curriculum_markers(ax)
    annotate_curriculum(ax)
    ax.set_xlabel("Training Steps (×1 000)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Task Success Metrics")
    ax.set_ylim(-2, 105)
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xlim(left=0)
    finalise(fig, output)


def plot_reward_decomposition(series: dict, output: Path) -> None:
    """Fig 3 — Sequential skill acquisition through reward components."""
    reward_terms: list[tuple[str, str, str]] = [
        ("Episode_Reward/reaching_shirt", "Reach (coarse)", _C_TEAL),
        ("Episode_Reward/reaching_shirt_fine", "Reach (fine)", _C_DARK),
        ("Episode_Reward/grasping", "Grasp", _C_AMBER),
        ("Episode_Reward/lifting_shirt", "Lift", _C_ORANGE),
        ("Episode_Reward/height_bonus", "Height", _C_RED),
        ("Episode_Reward/goal_tracking", "Transport", _C_BLUE),
        ("Episode_Reward/clearance_over_drum", "Clearance", _C_GREY),
        ("Episode_Reward/release_event", "Release Event", _C_PURPLE),
        ("Episode_Reward/shirt_in_target", "Success", _C_GREEN),
        ("Episode_Reward/return_to_neutral", "Return Home", _C_BLUE),
    ]

    fig, axes = plt.subplots(
        len(reward_terms), 1,
        figsize=(11, 1.35 * len(reward_terms) + 1.2),
        sharex=True,
    )
    for ax, (tag, label, colour) in zip(axes, reward_terms):
        steps, values = get(series, tag)
        if len(steps) == 0:
            ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=55, ha="right")
            continue
        x = steps_to_k(steps)
        ax.fill_between(x, 0, smooth(values), alpha=0.25, color=colour, linewidth=0)
        ax.plot(x, smooth(values), color=colour, linewidth=1.4)
        add_curriculum_markers(ax)
        ax.set_ylabel(label, fontsize=9, fontweight="bold", rotation=0,
                      labelpad=55, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(left=0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    axes[-1].set_xlabel("Training Steps (×1 000)")
    fig.subplots_adjust(hspace=0.15)
    finalise(fig, output, tight=False,
             title="Reward Decomposition — Sequential Skill Acquisition")


def plot_penalties(series: dict, output: Path) -> None:
    """Fig 4 — Penalty and regularisation terms."""
    penalty_terms: list[tuple[str, str, str]] = [
        ("Episode_Reward/base_velocity", "Base Velocity", _C_BLUE),
        ("Episode_Reward/shirt_off_conveyor", "Shirt Dropped", _C_RED),
        ("Episode_Reward/carry_time", "Carry Time", _C_ORANGE),
        ("Episode_Reward/belt_contact", "Belt Contact", _C_PURPLE),
        ("Episode_Reward/action_rate", "Action Rate", _C_TEAL),
        ("Episode_Reward/joint_vel", "Joint Velocity", _C_GREY),
        ("Episode_Reward/joint_torque", "Joint Torque", _C_AMBER),
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for tag, label, colour in penalty_terms:
        steps, values = get(series, tag)
        if len(steps) == 0:
            continue
        ax.plot(steps_to_k(steps), smooth(values), color=colour,
                linewidth=1.3, label=label)

    add_curriculum_markers(ax)
    annotate_curriculum(ax, y_frac=0.08)
    ax.set_xlabel("Training Steps (×1 000)")
    ax.set_ylabel("Episode Penalty")
    ax.set_title("Penalties & Regularisation")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)
    ax.set_xlim(left=0)
    finalise(fig, output)


def plot_policy_diagnostics(series: dict, output: Path) -> None:
    """Fig 5 — Policy std, losses, and learning rate."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    panels = [
        (axes[0, 0], "Policy / Standard deviation", "Policy Std. Deviation", "σ", _C_BLUE),
        (axes[0, 1], "Loss / Policy loss", "Policy (Surrogate) Loss", None, _C_GREEN),
        (axes[1, 0], "Loss / Value loss", "Value Function Loss", "MSE", _C_ORANGE),
        (axes[1, 1], "Learning / Learning rate", "Learning Rate (KL-adaptive)", None, _C_PURPLE),
    ]
    for ax, tag, title, ylabel, colour in panels:
        steps, values = get(series, tag)
        if len(steps) > 0:
            vals = values if "Learning rate" in tag else smooth(values)
            ax.plot(steps_to_k(steps), vals, color=colour, linewidth=1.4)
        add_curriculum_markers(ax)
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
    axes[1, 0].set_xlabel("Steps (×1 000)")
    axes[1, 1].set_xlabel("Steps (×1 000)")
    finalise(fig, output, title="Policy & Training Diagnostics")


def plot_converged_summary(series: dict, output: Path) -> None:
    """Fig 6 — Final converged performance bar chart (last 10 % of steps)."""
    reward_tags = [
        ("Episode_Reward/shirt_in_target", "Success"),
        ("Episode_Reward/return_to_neutral", "Return Home"),
        ("Episode_Reward/goal_tracking", "Transport"),
        ("Episode_Reward/goal_tracking_fine", "Fine Pos."),
        ("Episode_Reward/reaching_shirt", "Reach (coarse)"),
        ("Episode_Reward/reaching_shirt_fine", "Reach (fine)"),
        ("Episode_Reward/grasping", "Grasp"),
        ("Episode_Reward/lifting_shirt", "Lift"),
        ("Episode_Reward/height_bonus", "Height"),
        ("Episode_Reward/release", "Release Hint"),
        ("Episode_Reward/release_event", "Release Event"),
        ("Episode_Reward/clearance_over_drum", "Clearance"),
        ("Episode_Reward/arm_utilization", "Arm Use"),
    ]
    penalty_tags = [
        ("Episode_Reward/shirt_off_conveyor", "Shirt Dropped"),
        ("Episode_Reward/base_velocity", "Base Vel."),
        ("Episode_Reward/carry_time", "Carry Time"),
        ("Episode_Reward/belt_contact", "Belt"),
        ("Episode_Reward/joint_torque", "Torque"),
        ("Episode_Reward/action_rate", "Act. Rate"),
        ("Episode_Reward/joint_vel", "Jnt. Vel"),
    ]

    def final_mean(tag: str) -> float:
        _, values = get(series, tag)
        if len(values) == 0:
            return 0.0
        return float(np.mean(values[int(len(values) * 0.9):]))

    r_labels = [l for _, l in reward_tags]
    r_values = [final_mean(t) for t, _ in reward_tags]
    p_labels = [l for _, l in penalty_tags]
    p_values = [final_mean(t) for t, _ in penalty_tags]

    fig, (ax_r, ax_p) = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]},
    )
    y = np.arange(len(r_labels))
    bars = ax_r.barh(y, r_values, color=_C_GREEN, alpha=0.8, height=0.65)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels(r_labels)
    ax_r.set_xlabel("Mean Episode Reward (last 10%)")
    ax_r.set_title("Reward Components")
    ax_r.grid(True, alpha=0.3, axis="x")
    ax_r.invert_yaxis()
    for bar, val in zip(bars, r_values):
        ax_r.text(bar.get_width() + max(r_values) * 0.01,
                  bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                  va="center", fontsize=8, color=_C_DARK)

    y = np.arange(len(p_labels))
    bars = ax_p.barh(y, p_values, color=_C_RED, alpha=0.8, height=0.65)
    ax_p.set_yticks(y)
    ax_p.set_yticklabels(p_labels)
    ax_p.set_xlabel("Mean Episode Penalty (last 10%)")
    ax_p.set_title("Penalty Components")
    ax_p.grid(True, alpha=0.3, axis="x")
    ax_p.invert_yaxis()
    lo = min(p_values) if p_values else 0.0
    for bar, val in zip(bars, p_values):
        offset = lo * 0.02
        ax_p.text(bar.get_width() + (offset if val >= 0 else -abs(offset)),
                  bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                  va="center", ha="left" if val >= 0 else "right",
                  fontsize=8, color=_C_DARK)

    finalise(fig, output, title="Converged Reward Breakdown")


def plot_episode_length(series: dict, output: Path) -> None:
    """Fig 7 — Mean episode length over training."""
    steps, values = get(series, "Metrics/mean_episode_length")
    if len(steps) == 0:
        print("  (skipped 07 — no mean_episode_length data)")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(steps_to_k(steps), smooth(values), color=_C_BLUE, linewidth=1.8,
            label="Mean episode length")
    ax.axhline(MAX_EPISODE_STEPS, color=_C_GREY, linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Maximum ({MAX_EPISODE_STEPS} steps)")
    converged = float(np.mean(values[int(len(values) * 0.9):]))
    ax.axhline(converged, color=_C_GREEN, linestyle="--", linewidth=1.1,
               alpha=0.8, label=f"Converged mean ≈ {converged:.0f} steps")

    add_curriculum_markers(ax)
    annotate_curriculum(ax)
    ax.set_xlabel("Training Steps (×1 000)")
    ax.set_ylabel("Episode Length (control steps)")
    ax.set_title("Average Episode Length")
    ax.set_ylim(0, MAX_EPISODE_STEPS + 20)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(left=0)
    finalise(fig, output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_latest_run(root: Path) -> Path:
    candidates = sorted(p for p in root.glob("*_ppo_torch*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No PPO runs found under {root}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=str, default=None,
        help="Run directory name (e.g. 2026-07-02_00-09-06_ppo_torch_seed1)",
    )
    parser.add_argument(
        "--label", type=str, default="tensegrity",
        help="Figures sub-directory under figures/ (default: tensegrity)",
    )
    args = parser.parse_args()

    run_dir = LOGS_ROOT / args.run if args.run else resolve_latest_run(LOGS_ROOT)
    print(f"Loading run: {run_dir.name}")
    series = load_run_series(run_dir)
    print(f"  {len(series)} scalar tags")

    figures_dir = FIGURES_BASE / args.label
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {figures_dir}/")

    plot_total_reward(series, figures_dir / "01_total_reward.png")
    plot_task_success(series, figures_dir / "02_task_success.png")
    plot_reward_decomposition(series, figures_dir / "03_reward_decomposition.png")
    plot_penalties(series, figures_dir / "04_penalties.png")
    plot_policy_diagnostics(series, figures_dir / "05_policy_diagnostics.png")
    plot_converged_summary(series, figures_dir / "06_converged_breakdown.png")
    plot_episode_length(series, figures_dir / "07_episode_length.png")

    print(f"Done — figures generated for '{args.label}'.")


if __name__ == "__main__":
    main()

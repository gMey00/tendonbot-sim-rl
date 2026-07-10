#!/usr/bin/env python3
"""Generate training result plots for the Shirt Distribute task README.

Reads TensorBoard event files from a training run and produces figures in the
task's ``figures/`` directory.  Tensorboard-free (uses the pure-Python reader
``scripts/model_validation/read_tfevents.py``) — same approach as
``plot_shirt_pick_training_results.py``.

Usage
-----
    cd src/tensegrity_pick
    conda activate env_isaaclab
    python scripts/plot_shirt_distribute_training_results.py \
        [--run DIR_NAME] [--label NAME]

If ``--run`` is omitted the latest ``*_ppo_torch*`` run under
``logs/skrl/shirt_distribute/`` is used.  ``--label`` selects the output
sub-directory under ``figures/`` (default: ``ur5e_f140``).
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
sys.path.insert(0, str(_HERE.parent / "model_validation"))
import read_tfevents as tfr  # noqa: E402

sys.path.insert(0, str(_HERE.parents[3] / ".config"))
import plot_config as pcfg  # noqa: E402
pcfg.apply_style()

LOGS_ROOT: Final = Path("logs/skrl/shirt_distribute")
FIGURES_BASE: Final = Path(
    "source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_distribute/figures"
)
CURRICULUM_REGULARISATION_STEP: Final = 2_000
SMOOTHING_WEIGHT: Final = 0.90
MAX_EPISODE_STEPS: Final = 480  # 8 s @ 60 Hz

_C_BLUE, _C_GREEN, _C_AMBER = pcfg.FAPS_BLUE, pcfg.FAPS_GREEN, pcfg.AMBER
_C_RED, _C_PURPLE, _C_TEAL = pcfg.MUTED_RED, pcfg.PURPLE, pcfg.TEAL
_C_GREY, _C_ORANGE = pcfg.FAPS_DARK_GREY, pcfg.DARK_ORANGE
FIGSIZE_WIDE: Final = (11, 4.5)


def load_run_series(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no tfevents file in {run_dir}")
    raw: dict[str, list[tuple[int, float]]] = {}
    for rec in tfr._read_records(str(event_files[0])):
        ev = tfr._parse_event(rec)
        if not ev:
            continue
        step, vals = ev
        for tag, value in vals:
            raw.setdefault(tag, []).append((step, value))
    return {
        tag: (np.array([s for s, _ in pts]), np.array([v for _, v in pts]))
        for tag, pts in raw.items()
    }


def get(series: dict, tag: str) -> tuple[np.ndarray, np.ndarray]:
    if tag in series:
        return series[tag]
    for k_ in series:  # tolerate "Info / " style prefixes
        if k_.endswith(tag) or k_.replace(" ", "") == tag.replace(" ", ""):
            return series[k_]
    return np.array([]), np.array([])


def smooth(values: np.ndarray, weight: float = SMOOTHING_WEIGHT) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=float)
    acc = values[0]
    for i, v in enumerate(values):
        acc = weight * acc + (1.0 - weight) * v
        out[i] = acc
    return out


def k(steps: np.ndarray) -> np.ndarray:
    return steps / 1000.0


def mark_curriculum(ax: plt.Axes) -> None:
    ax.axvline(CURRICULUM_REGULARISATION_STEP / 1000.0, color=_C_GREY,
               linestyle=":", linewidth=1.2, alpha=0.8)


def finalise(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_total_reward(series: dict, out: Path) -> None:
    s, v = get(series, "Reward / Total reward (mean)")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(k(s), v, color=_C_BLUE, alpha=0.25)
    ax.plot(k(s), smooth(v), color=_C_BLUE, label="total reward (mean)")
    mark_curriculum(ax)
    ax.set_xlabel("trainer steps (k)"); ax.set_ylabel("episode reward")
    ax.set_title("Shirt Distribute — Total Episode Reward"); ax.legend()
    finalise(fig, out / "01_total_reward.png")


def plot_task_metrics(series: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for tag, label, colour in [
        ("Metrics/distribute_success_rate", "distribute success (correct bin)", _C_GREEN),
        ("Metrics/release_rate", "release rate", _C_BLUE),
        ("Metrics/max_target_fraction", "max cloth fraction in commanded drum", _C_TEAL),
        ("Metrics/term_belt_collision", "belt-collision terminations", _C_RED),
    ]:
        s, v = get(series, tag)
        if s.size:
            ax.plot(k(s), v, color=colour, alpha=0.2)
            ax.plot(k(s), smooth(v), color=colour, label=label)
    mark_curriculum(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("trainer steps (k)"); ax.set_ylabel("rate")
    ax.set_title("Shirt Distribute — Task Metrics (stochastic rollouts; select on deterministic eval)")
    ax.legend()
    finalise(fig, out / "02_task_metrics.png")


def plot_per_bin_success(series: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for tag, label, colour in [
        ("Metrics/distribute_success_bin0", "bin 0 — reusable (0.15, 1.0)", _C_BLUE),
        ("Metrics/distribute_success_bin1", "bin 1 — recyclable (1.35, 1.0)", _C_ORANGE),
        ("Metrics/distribute_success_bin2", "bin 2 — trash (0.75, 1.6)", _C_PURPLE),
    ]:
        s, v = get(series, tag)
        if s.size:
            ax.plot(k(s), v, color=colour, alpha=0.15)
            ax.plot(k(s), smooth(v), color=colour, label=label)
    mark_curriculum(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("trainer steps (k)"); ax.set_ylabel("success rate")
    ax.set_title("Shirt Distribute — Per-Bin Success (goal-conditioned, stochastic)")
    ax.legend()
    finalise(fig, out / "03_per_bin_success.png")


def plot_reward_decomposition(series: dict, out: Path) -> None:
    terms = [
        ("Episode_Reward/approach_bin", "carry to commanded bin", _C_BLUE),
        ("Episode_Reward/clearance_over_bin", "whole shirt cleared", _C_TEAL),
        ("Episode_Reward/release_event", "release event (graded one-shot)", _C_GREEN),
        ("Episode_Reward/bad_release", "bad release (one-shot penalty)", _C_RED),
        ("Episode_Reward/in_target_bin", "cloth in commanded drum", _C_ORANGE),
        ("Episode_Reward/in_wrong_bin", "cloth in wrong drum", _C_PURPLE),
        ("Episode_Reward/settle_after_success", "post-drop settling", _C_GREY),
    ]
    fig, axes = plt.subplots(len(terms), 1, figsize=(11, 1.35 * len(terms) + 1.2),
                             sharex=True)
    for ax, (tag, label, colour) in zip(axes, terms):
        s, v = get(series, tag)
        if s.size:
            ax.plot(k(s), v, color=colour, alpha=0.2)
            ax.plot(k(s), smooth(v), color=colour)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        mark_curriculum(ax)
    axes[-1].set_xlabel("trainer steps (k)")
    axes[0].set_title("Shirt Distribute — Reward Decomposition")
    finalise(fig, out / "04_reward_decomposition.png")


def plot_penalties(series: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for tag, label, colour in [
        ("Episode_Reward/action_l2", "action magnitude", _C_TEAL),
        ("Episode_Reward/action_rate", "action rate", _C_AMBER),
        ("Episode_Reward/joint_vel", "joint velocity", _C_PURPLE),
        ("Episode_Reward/belt_contact", "belt contact", _C_RED),
        ("Episode_Reward/carry_time", "carry-time bleed", _C_GREY),
    ]:
        s, v = get(series, tag)
        if s.size:
            ax.plot(k(s), smooth(v), color=colour, label=label)
    mark_curriculum(ax)
    ax.set_xlabel("trainer steps (k)"); ax.set_ylabel("episode reward")
    ax.set_title("Shirt Distribute — Penalties & Regularisation"); ax.legend()
    finalise(fig, out / "05_penalties.png")


def plot_episode_length(series: dict, out: Path) -> None:
    s, v = get(series, "Episode / Total timesteps (mean)")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if s.size:
        ax.plot(k(s), v, color=_C_BLUE, alpha=0.25)
        ax.plot(k(s), smooth(v), color=_C_BLUE)
    ax.axhline(MAX_EPISODE_STEPS, color=_C_GREY, linestyle="--", linewidth=1.0,
               label=f"timeout ({MAX_EPISODE_STEPS} steps = 8 s)")
    mark_curriculum(ax)
    ax.set_xlabel("trainer steps (k)"); ax.set_ylabel("episode length (steps)")
    ax.set_title("Shirt Distribute — Episode Length"); ax.legend()
    finalise(fig, out / "06_episode_length.png")


def resolve_latest_run(root: Path) -> Path:
    runs = sorted(d for d in root.glob("*_ppo_torch*") if d.is_dir())
    if not runs:
        raise FileNotFoundError(f"no runs under {root}")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default=None,
                        help="run directory name under logs/skrl/shirt_distribute/")
    parser.add_argument("--label", type=str, default="ur5e_f140",
                        help="output sub-directory under figures/")
    args = parser.parse_args()

    run_dir = LOGS_ROOT / args.run if args.run else resolve_latest_run(LOGS_ROOT)
    out = FIGURES_BASE / args.label
    out.mkdir(parents=True, exist_ok=True)
    print(f"run: {run_dir}\nout: {out}")

    series = load_run_series(run_dir)
    plot_total_reward(series, out)
    plot_task_metrics(series, out)
    plot_per_bin_success(series, out)
    plot_reward_decomposition(series, out)
    plot_penalties(series, out)
    plot_episode_length(series, out)


if __name__ == "__main__":
    main()

"""Shared helpers for project-thesis plot scripts.

This module centralises the boilerplate that every plotting script in
``doc/Theses/project_thesis/scripts/`` needs:

* path constants (data, figures)
* the FAPS matplotlib style (loaded via :mod:`plot_config`)
* CSV / JSON loaders
* exponential-moving-average smoothing
* curriculum-marker helpers (vertical dashed lines + labels)
* a "missing data" placeholder figure for not-yet-recorded experiments

All scripts must call :func:`init` exactly once at module import time.

Layout assumed by the loaders::

    project_thesis/
        assets/
            data/
                step_response/                  # CSVs + validation_metrics.json
            figures/
                results/                        # rendered figures used in §5
                appendix/                       # rendered figures used in A.4–A.7
        scripts/
            common.py                           # this file
            plot_*.py

RL training curves are read directly from the skrl TensorBoard event files
under ``src/tensegrity_pick/logs/skrl/theses_logs/<task>/<run>/``. The
selected run per (task, variant) pair is declared in :data:`RUNS` below.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Final, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR: Final = Path(__file__).resolve().parent
THESIS_DIR: Final = SCRIPTS_DIR.parent
ASSETS_DIR: Final = THESIS_DIR / "assets"
DATA_DIR: Final = ASSETS_DIR / "data"
FIGURES_DIR: Final = ASSETS_DIR / "figures"

STEP_DATA_DIR: Final = DATA_DIR / "step_response"

RESULTS_FIG_DIR: Final = FIGURES_DIR / "results"
APPENDIX_FIG_DIR: Final = FIGURES_DIR / "appendix"

# Repo-wide shared formatting (faps_thesis.mplstyle, palette).
_REPO_ROOT: Final = THESIS_DIR.parents[2]
_CONFIG_DIR: Final = _REPO_ROOT / ".config"
sys.path.insert(0, str(_CONFIG_DIR))

import plot_config as pcfg  # noqa: E402  – needs sys.path injection above

# Re-export for convenience so scripts only need ``from common import pcfg``.
pcfg.apply_style()

# ── NPZ source directory (primary truth for model-validation data) ───────────
NPZ_DATA_ROOT: Final = _REPO_ROOT / "src" / "tensegrity_pick" / "outputs" / "model_validation"

# Klein (2023) Tabelle 4.2 – Gazebo reference values embedded as ground truth.
# ``None`` means the value was not reported in Klein's table.
KLEIN_TABLE_4_2: Final[dict[tuple[str, int], dict[str, float | None]]] = {
    ("wrist_x_joint", 10): {"nrmse": 25.5, "rise_ms": None,  "overshoot": None},
    ("wrist_x_joint", 20): {"nrmse":  9.9, "rise_ms": None,  "overshoot": 23.4},
    ("wrist_x_joint", 30): {"nrmse": 11.2, "rise_ms": 125.0, "overshoot": 31.5},
    ("wrist_y_joint", 10): {"nrmse": 19.2, "rise_ms": None,  "overshoot": None},
    ("wrist_y_joint", 20): {"nrmse": 11.6, "rise_ms": None,  "overshoot": 33.4},
    ("wrist_y_joint", 30): {"nrmse": 12.5, "rise_ms":  84.5, "overshoot": 40.6},
    ("elbow_joint",   20): {"nrmse": 39.1, "rise_ms": None,  "overshoot": None},
    ("elbow_joint",   30): {"nrmse": 11.7, "rise_ms": None,  "overshoot": None},
    ("elbow_joint",   40): {"nrmse": 12.1, "rise_ms": 367.5, "overshoot":  7.8},
}

# Canonical amplitude sweeps per joint (matching the NPZ filenames).
JOINT_AMPS: Final[dict[str, list[int]]] = {
    "elbow_joint":   [20, 30, 40],
    "wrist_x_joint": [10, 20, 30],
    "wrist_y_joint": [10, 20, 30],
}

# Internal mapping from public variant key to subdirectory name.
_NPZ_VARIANT_DIR: Final[dict[str, str]] = {
    "pd":       "pd",
    "tendon":   "tendon",
    "physical": "tendon_physical",
}


def load_npz_trial(variant: str, joint: str, amplitude: int):
    """Load one step-response NPZ trial directly from the model-validation output.

    Parameters
    ----------
    variant:
        One of ``'pd'``, ``'tendon'``, or ``'physical'``.
    joint:
        Canonical joint name: ``'elbow_joint'``, ``'wrist_x_joint'``, or
        ``'wrist_y_joint'``.
    amplitude:
        Integer step amplitude in degrees.

    Returns
    -------
    An open :class:`numpy.lib.npyio.NpzFile`, or ``None`` when the file does
    not exist.  The archive always contains ``time_s``, ``actual_deg``,
    ``setpoint_deg``, ``rise_time_ms``, ``settling_time_ms``,
    ``overshoot_pct``, and ``nrmse_pct``.
    """
    var_dir = _NPZ_VARIANT_DIR[variant]
    # The physical variant stores the elbow under a distinct name.
    npz_joint = (
        "elbow_physical"
        if joint == "elbow_joint" and variant == "physical"
        else joint
    )
    path = NPZ_DATA_ROOT / var_dir / "data" / f"{npz_joint}__{amplitude}deg.npz"
    if not path.exists():
        return None
    return np.load(path)


# Canonical amplitude sweeps for the prismatic base joints (in mm).
BASE_JOINT_AMPS: Final[dict[str, list[int]]] = {
    "base_y_joint": [50, 100, 200],
    "base_z_joint": [30,  60, 100],
}


def load_npz_base_trial(joint: str, amplitude: int):
    """Load one base-joint step-response NPZ from the model-validation output.

    Parameters
    ----------
    joint:
        ``'base_y_joint'`` or ``'base_z_joint'``.
    amplitude:
        Integer step amplitude in mm (stored in the filename as ``{amp}deg``
        by convention from the recording script).

    Returns
    -------
    An open :class:`numpy.lib.npyio.NpzFile`, or ``None`` when the file does
    not exist.  The archive contains ``time_s``, ``actual_deg`` (mm),
    ``setpoint_deg`` (mm), ``rise_time_ms``, ``settling_time_ms``,
    ``nrmse_pct``, and ``steady_state_error_deg`` (mm).
    """
    path = NPZ_DATA_ROOT / "base" / "data" / f"{joint}__{amplitude}deg.npz"
    if not path.exists():
        return None
    return np.load(path)


# ── Output helpers ───────────────────────────────────────────────────────────
def init() -> None:
    """Ensure output directories exist. Idempotent — safe to call repeatedly."""
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, path: Path | str, *, tight: bool = True) -> None:
    """Save *fig* and close it. Mirrors :func:`pcfg.finalize` but without title."""
    pcfg.finalize(fig, path, title=None, tight=tight)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_step_csv(name: str) -> pd.DataFrame:
    """Load a step-response CSV from ``assets/data/step_response/``.

    Returns a DataFrame with columns ``time_s, actual_deg, setpoint_deg``.
    """
    path = STEP_DATA_DIR / name
    return pd.read_csv(path)


def load_validation_metrics() -> list[dict]:
    """Load the merged validation-metrics JSON (Klein, PD, tendon)."""
    return json.loads((STEP_DATA_DIR / "validation_metrics.json").read_text())


# ── RL training runs (TensorBoard event files) ───────────────────────────────
LOGS_ROOT: Final = (
    _REPO_ROOT / "src" / "tensegrity_pick" / "logs" / "skrl" / "theses_logs"
)

# Selected run per (task, variant). ``None`` means the events.out.tfevents.*
# file is not available in the workspace; loaders return empty arrays so the
# scripts skip the variant gracefully.
RUNS: Final[dict[tuple[str, str], str | None]] = {
    ("reach", "tensegrity_pd"):
        "reach/tensegrity/2026-04-06_17-53-45_ppo_torch",
    ("reach", "tensegrity_tendon"):
        "reach/tensegrity_tendon/2026-04-06_18-19-18_ppo_torch",
    ("reach", "tensegrity_physical"):
        "reach/tensegrity_physical_tendon/2026-04-08_19-30-57_ppo_torch",
    ("cube_place", "tensegrity_pd"):
        "cube_place/tensegrity/2026-04-05_10-54-14_ppo_torch",
    ("cube_place", "tensegrity_tendon"):
        "cube_place/tensegrity_tendon/2026-04-05_15-32-10_ppo_torch",
    ("cube_place", "tensegrity_physical"):
        "cube_place/tensegrity_physical_tendon/2026-04-05_22-43-20_ppo_torch",
}


def run_dir(task: str, variant: str) -> Path | None:
    rel = RUNS.get((task, variant))
    return None if rel is None else LOGS_ROOT / rel


@lru_cache(maxsize=None)
def _accumulator(run_path: str):
    """Cached TensorBoard EventAccumulator for *run_path* (or ``None``)."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    acc = EventAccumulator(run_path, size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def load_rl(task: str, variant: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one TensorBoard scalar series.

    Returns ``(steps, values)``. Missing runs or missing tags yield empty
    arrays so callers can degrade gracefully.
    """
    path = run_dir(task, variant)
    if path is None or not path.exists():
        return np.array([]), np.array([])
    acc = _accumulator(str(path))
    if tag not in acc.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = acc.Scalars(tag)
    return (
        np.array([e.step for e in events]),
        np.array([e.value for e in events]),
    )


def final_mean(task: str, variant: str, tag: str,
               *, tail_frac: float = 0.1, x_max: int | None = None) -> float | None:
    """Mean of the last ``tail_frac`` of *tag* (after optional x cutoff)."""
    steps, values = load_rl(task, variant, tag)
    if len(values) == 0:
        return None
    if x_max is not None:
        mask = steps <= x_max
        values = values[mask]
        if len(values) == 0:
            return None
    cutoff = max(1, int(len(values) * (1.0 - tail_frac)))
    return float(np.mean(values[cutoff:]))


# ── Smoothing ────────────────────────────────────────────────────────────────
def smooth(values: np.ndarray, weight: float = 0.92) -> np.ndarray:
    """Exponential moving average with *weight* in (0, 1)."""
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = weight * out[i - 1] + (1 - weight) * values[i]
    return out


def steps_to_k(steps: np.ndarray | float) -> np.ndarray | float:
    """Convert raw timesteps to thousands for cleaner x-axis labels."""
    return np.asarray(steps) / 1000.0


# ── Curriculum-marker helpers ────────────────────────────────────────────────
def add_curriculum_marker(
    ax: plt.Axes,
    step: int,
    label: str | None = None,
    *,
    color: str | None = None,
    y_frac: float = 0.95,
) -> None:
    """Add a vertical dashed line at *step* (in raw steps, not k-steps)."""
    color = color or pcfg.FAPS_DARK_GREY
    ax.axvline(
        steps_to_k(step), color=color, linestyle="--", linewidth=0.8, alpha=0.7,
    )
    if label is not None:
        transform = ax.get_xaxis_transform()
        ax.text(
            steps_to_k(step), y_frac, " " + label,
            transform=transform, fontsize=7.5,
            color=color, va="top", ha="left", alpha=0.85,
        )


# ── Variant cosmetics (consistent across all RL plots) ──────────────────────
VARIANT_LABEL: Final[dict[str, str]] = {
    "tensegrity_pd": "PD",
    "tensegrity_tendon": "Tendon",
    "tensegrity_physical": "Physical",
}
VARIANT_COLOR: Final[dict[str, str]] = {
    "tensegrity_pd": pcfg.FAPS_BLUE,
    "tensegrity_tendon": pcfg.FAPS_GREEN,
    "tensegrity_physical": pcfg.PURPLE,
}
VARIANTS: Final[tuple[str, ...]] = tuple(VARIANT_LABEL)


def variant_legend(ax: plt.Axes, *, loc: str = "best", title: str | None = None) -> None:
    """Add a consistent variant legend to *ax* (uses any line already on it)."""
    ax.legend(loc=loc, title=title)


# ── Missing-data placeholder ────────────────────────────────────────────────
def missing_panel(ax: plt.Axes, message: str) -> None:
    """Render a grey placeholder panel for not-yet-recorded experiments."""
    ax.set_facecolor("#F8F8F8")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B0B0B0")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.text(
        0.5, 0.5, "Data not recorded\n" + message,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=8, color="#666666",
        style="italic",
    )


# ── Plot helpers (tasks shared across multiple scripts) ─────────────────────
def plot_variants_curve(
    ax: plt.Axes,
    task: str,
    metric: str,
    *,
    smoothing: float = 0.92,
    x_max: int | None = None,
    y_label: str | None = None,
    title: str | None = None,
    show_legend: bool = False,
    variants: Iterable[str] = VARIANTS,
    draw_raw: bool = False,
    value_scale: float = 1.0,
) -> bool:
    """Plot one metric for the given variants on *ax* with consistent styling.

    Only the smoothed curve is drawn by default. Set ``draw_raw=True`` to
    additionally render the faint raw line in the background.
    """
    has_data = False
    for variant in variants:
        steps, values = load_rl(task, variant, metric)
        if len(steps) == 0:
            continue
        if x_max is not None:
            mask = steps <= x_max
            steps, values = steps[mask], values[mask]
        if len(values) == 0:
            continue
        values = values * value_scale
        x = steps_to_k(steps)
        color = VARIANT_COLOR[variant]
        label = VARIANT_LABEL[variant]
        if draw_raw:
            ax.plot(x, values, color=color, alpha=0.18, linewidth=0.7)
        ax.plot(x, smooth(values, smoothing), color=color, linewidth=1.4, label=label)
        has_data = True

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    if y_label:
        ax.set_ylabel(y_label)
    ax.set_xlabel("Timesteps / k")
    if x_max is not None:
        ax.set_xlim(0, steps_to_k(x_max))
    if show_legend and has_data:
        ax.legend(loc="best", fontsize=8)
    return has_data


def plot_envelope(
    ax: plt.Axes,
    task: str,
    mean_tag: str,
    min_tag: str,
    max_tag: str,
    *,
    x_max: int | None = None,
    smoothing: float = 0.92,
    variants: Iterable[str] = VARIANTS,
) -> bool:
    """Plot per-variant min/max envelope + smoothed mean (no ghost line)."""
    has_data = False
    for variant in variants:
        s_mean, v_mean = load_rl(task, variant, mean_tag)
        if len(s_mean) == 0:
            continue
        if x_max is not None:
            mask = s_mean <= x_max
            s_mean, v_mean = s_mean[mask], v_mean[mask]
        if len(v_mean) == 0:
            continue
        _, v_max = load_rl(task, variant, max_tag)
        _, v_min = load_rl(task, variant, min_tag)
        v_max = v_max[: len(v_mean)]
        v_min = v_min[: len(v_mean)]
        x = steps_to_k(s_mean)
        color = VARIANT_COLOR[variant]
        if len(v_min) == len(v_mean) and len(v_max) == len(v_mean):
            ax.fill_between(x, smooth(v_min, 0.85), smooth(v_max, 0.85),
                            color=color, alpha=0.15, linewidth=0)
        ax.plot(x, smooth(v_mean, smoothing), color=color, linewidth=1.6,
                label=VARIANT_LABEL[variant])
        has_data = True
    ax.set_xlabel("Timesteps / k")
    if x_max is not None:
        ax.set_xlim(0, steps_to_k(x_max))
    return has_data


def figure_bottom_legend(fig: plt.Figure, axes, *, ncol: int = 3,
                         y: float = 0.0, fontsize: int = 9) -> None:
    """Add a single legend below all *axes* using their handles (dedup by label).

    Removes any per-axes legend first so only the shared figure-level legend
    remains. Caller must use ``constrained_layout=False`` and allocate the
    space (e.g. ``fig.subplots_adjust(bottom=...)``) when applying.
    """
    seen: dict[str, object] = {}
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label and label not in seen and not label.startswith("_"):
                seen[label] = handle
    if not seen:
        return
    fig.legend(list(seen.values()), list(seen),
               loc="lower center", ncol=ncol, fontsize=fontsize,
               frameon=False, bbox_to_anchor=(0.5, y))


# ── Cube-place curriculum step constants (from place_env_cfg.py) ────────────
PLACE_RED_CUBE_STEP: Final = 125_000   # activate_red_cube_curriculum num_steps
PLACE_REG_RAMP_STEP: Final = 150_000   # action_rate / joint_vel ramp num_steps


def hatched_missing_bar(ax: plt.Axes, x_pos: float, height: float = 1.0,
                        width: float = 0.8) -> None:
    """Draw a slim grey hatched 'n/a' bar at *x_pos* (for missing-data slots)."""
    ax.bar(x_pos, height, width=width,
           color="#EEEEEE", edgecolor="#999999",
           hatch="//", linewidth=0.5)


# ── Multi-seed reach evaluation (JSONs produced by evaluate.py) ─────────────
REACH_EVAL_DIR: Final = (
    _REPO_ROOT / "src" / "tensegrity_pick" / "logs" / "skrl"
    / "theses_logs" / "reach" / "eval"
)

# Variant key used in this thesis ↔ variant key written in the eval JSONs.
REACH_EVAL_VARIANT: Final[dict[str, str]] = {
    "tensegrity_pd":       "tensegrity",
    "tensegrity_tendon":   "tensegrity_tendon",
    "tensegrity_physical": "tensegrity_physical_tendon",
}

# Display order and cosmetics for the four evaluation agents.
AGENTS: Final[tuple[str, ...]] = ("zero", "random", "heuristic", "checkpoint")
AGENT_LABEL: Final[dict[str, str]] = {
    "zero":       "Zero",
    "random":     "Random",
    "heuristic":  "DLS-IK",
    "checkpoint": "PPO",
}
AGENT_COLOR: Final[dict[str, str]] = {
    "zero":       "#999999",
    "random":     "#BBBBBB",
    "heuristic":  pcfg.FAPS_DARK_GREY,
    "checkpoint": pcfg.FAPS_BLUE,
}

# Reach-evaluation success threshold (matches command_term.cfg.success_threshold).
REACH_SUCCESS_THRESHOLD_M: Final = 0.05


@lru_cache(maxsize=None)
def _load_reach_eval_json(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text())


def load_reach_eval(agent: str, variant: str, seed: int) -> dict | None:
    """Load one (agent, variant, seed) evaluation JSON.

    Returns the parsed dict (with a ``metrics`` sub-dict containing
    ``success_rate``, ``position_error``, ``orientation_error``, and
    ``reach_time_mean`` — each a ``{mean, std, n}`` block), or ``None``
    when the JSON does not exist.

    Variant uses this thesis' canonical key (``tensegrity_pd`` /
    ``tensegrity_tendon`` / ``tensegrity_physical``); it is translated
    to the eval-JSON variant key via :data:`REACH_EVAL_VARIANT`.
    """
    eval_variant = REACH_EVAL_VARIANT[variant]
    fname = f"{agent}_{eval_variant}_seed{seed}.json"
    path = REACH_EVAL_DIR / fname
    if not path.exists():
        return None
    return _load_reach_eval_json(str(path))


def reach_eval_seeds(agent: str, variant: str, metric: str,
                     *, max_seed: int = 5) -> np.ndarray:
    """Return the per-seed mean values of *metric* for (agent, variant).

    *metric* is one of the keys under the JSON ``metrics`` block. Missing
    seed files are silently skipped. Returns an empty array when no JSONs
    are found.
    """
    out: list[float] = []
    for seed in range(max_seed):
        d = load_reach_eval(agent, variant, seed)
        if d is None or metric not in d.get("metrics", {}):
            continue
        out.append(float(d["metrics"][metric]["mean"]))
    return np.array(out)

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
                rl_training/<task>/<variant>/   # one directory per training run
            figures/
                results/                        # rendered figures used in §5
                appendix/                       # rendered figures used in A.4–A.7
        scripts/
            common.py                           # this file
            plot_*.py
"""

from __future__ import annotations

import json
import sys
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
RL_DATA_DIR: Final = DATA_DIR / "rl_training"

RESULTS_FIG_DIR: Final = FIGURES_DIR / "results"
APPENDIX_FIG_DIR: Final = FIGURES_DIR / "appendix"

# Repo-wide shared formatting (faps_thesis.mplstyle, palette).
_REPO_ROOT: Final = THESIS_DIR.parents[2]
_CONFIG_DIR: Final = _REPO_ROOT / ".config"
sys.path.insert(0, str(_CONFIG_DIR))

import plot_config as pcfg  # noqa: E402  – needs sys.path injection above

# Re-export for convenience so scripts only need ``from common import pcfg``.
pcfg.apply_style()


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


def load_rl_csv(task: str, variant: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one TensorBoard scalar series exported as CSV.

    Returns ``(steps, values)``. Missing files yield empty arrays so callers
    can degrade gracefully.
    """
    path = RL_DATA_DIR / task / variant / f"{metric}.csv"
    if not path.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(path)
    return df["step"].to_numpy(), df["value"].to_numpy()


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
) -> None:
    """Plot one metric for the given variants on *ax* with consistent styling.

    Each variant's raw curve is drawn faint in the background and the
    smoothed curve is drawn on top.
    """
    has_data = False
    for variant in variants:
        steps, values = load_rl_csv(task, variant, metric)
        if len(steps) == 0:
            continue
        if x_max is not None:
            mask = steps <= x_max
            steps, values = steps[mask], values[mask]
        x = steps_to_k(steps)
        color = VARIANT_COLOR[variant]
        label = VARIANT_LABEL[variant]
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


def hatched_missing_bar(ax: plt.Axes, x_pos: float, height: float = 1.0,
                        width: float = 0.8) -> None:
    """Draw a slim grey hatched 'n/a' bar at *x_pos* (for missing-data slots)."""
    ax.bar(x_pos, height, width=width,
           color="#EEEEEE", edgecolor="#999999",
           hatch="//", linewidth=0.5)

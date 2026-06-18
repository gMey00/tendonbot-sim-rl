"""Centralised plot configuration for the FAPS thesis.

Usage in every plotting script::

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[<N>] / ".config"))
    import plot_config as pcfg

    pcfg.apply_style()               # call once at top-level
    ...
    pcfg.finalize(fig, path, title="My Title")

Toggle ``SHOW_TITLES`` to ``False`` before generating the final
camera-ready figures that will be captioned inside the LaTeX document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# ── Re-export colour palette for convenience ─────────────────────────────────
from faps_colors import (  # noqa: F401 – re-exported
    AMBER,
    CYCLE_COLORS,
    DARK_ORANGE,
    DIVERGING_CMAP,
    FAPS_BLUE,
    FAPS_DARK_GREY,
    FAPS_GREEN,
    FAPS_GREEN_DARK,
    FAPS_GREEN_LIGHT,
    FAPS_LIGHT_GREY,
    FAPS_MID_GREY,
    FAU_BLAU,
    FAU_BLAU_DUNKEL,
    KLEIN_COLOR,
    MUTED_RED,
    PD_COLOR,
    PHYSICAL_COLOR,
    PURPLE,
    REWARD_COLOR,
    SEQUENTIAL_CMAP,
    SUCCESS_COLOR,
    PENALTY_COLOR,
    TEAL,
    TEAL_DARK,
    TF_METALLIC,
    TENDON_COLOR,
)

# ── Global toggle ────────────────────────────────────────────────────────────
SHOW_TITLES: bool = False
"""Set to ``False`` to suppress all figure suptitles / axes titles.
Useful when the LaTeX caption already provides the title."""


# ── Thesis geometry (inches) ─────────────────────────────────────────────────
_TEXTWIDTH_CM = 16.0
_TEXTWIDTH_IN = _TEXTWIDTH_CM / 2.54  # 6.299 in

_GOLDEN = (1 + 5**0.5) / 2  # 1.618


def scaled(orig_w: float, orig_h: float, *, scale: float = 1.0) -> tuple[float, float]:
    """Scale an original ``(width, height)`` to fit the thesis textwidth.

    The original aspect ratio is preserved exactly.  Use *scale* < 1 for
    side-by-side sub-figures (e.g. ``scale=0.48``).
    """
    w = _TEXTWIDTH_IN * scale
    h = orig_h * (w / orig_w)
    return (w, h)


# Keep the old helper alive for any remaining callers.
def figsize(
    aspect: Literal["golden", "wide", "square", "tall", "3d"] = "golden",
    *,
    scale: float = 1.0,
) -> tuple[float, float]:
    """Return ``(width, height)`` in inches for the thesis textwidth.

    Prefer :func:`scaled` for new code — it preserves the *original*
    aspect ratio instead of forcing a preset.
    """
    w = _TEXTWIDTH_IN * scale
    ratios = {
        "golden": _GOLDEN,
        "wide": 2.2,
        "square": 1.0,
        "tall": 1 / 1.1,
        "3d": _TEXTWIDTH_IN / 8.0,
    }
    return (w, w / ratios[aspect])


# ── Style application ────────────────────────────────────────────────────────

_STYLE_PATH = Path(__file__).with_name("faps_thesis.mplstyle")


def apply_style() -> None:
    """Load the FAPS thesis matplotlib style sheet.

    Call this once at the top of every plotting script, *after*
    ``matplotlib.use("Agg")`` if running headless.
    """
    plt.style.use(str(_STYLE_PATH))


# ── Figure finalisation ──────────────────────────────────────────────────────


def finalize(
    fig: plt.Figure,
    path: Path | str,
    *,
    title: str | None = None,
    tight: bool = True,
) -> None:
    """Save *fig* as PDF (or whatever the style sheet sets), then close it.

    Parameters
    ----------
    title
        If given **and** ``SHOW_TITLES`` is ``True``, the string is set
        as the figure suptitle.
    tight
        Whether to call ``tight_layout()`` before saving.
    """
    if title and SHOW_TITLES:
        fig.suptitle(title, fontsize=11, fontweight="bold")
        if tight:
            fig.tight_layout(rect=[0, 0, 1, 0.94])
    elif tight:
        fig.tight_layout()

    # DIN 461 / FAPS: arrow on axis labels + unit at penultimate tick
    apply_din461_style(fig)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}")


# ── DIN 461 / FAPS axis formatting ──────────────────────────────────────────


def apply_din461_style(fig: plt.Figure) -> None:
    """Apply DIN 461 axis formatting to every axes in *fig*.

    Per FAPS diagram rules:

    1. **Arrow on axis labels** – append ' →' to every numerical axis label
       to indicate the direction of increasing values.  Categorical axes
       (e.g. bar-chart x-axes with string tick labels) are excluded per the
       DIN 461 note "nicht bei Versuchsnummern".

    2. **Unit at penultimate tick** – when the label contains ' / ' (e.g.
       "Time / s"), the unit string (here "s") replaces the second-to-last
       visible tick label, matching the DIN 461 / FAPS convention shown in
       the style guide.

    This function is called automatically by :func:`finalize` so individual
    scripts do not need to call it explicitly.
    """
    # Render the figure once so tick positions are finalised.
    fig.canvas.draw()
    for ax in fig.get_axes():
        _apply_din461_axes(ax)


def _is_numeric_str(s: str) -> bool:
    """Return ``True`` if *s* looks like a formatted number (int or float)."""
    try:
        # Matplotlib renders negative signs as U+2212 '−', normalise first.
        float(s.replace("\u2212", "-").replace(",", "."))
        return True
    except ValueError:
        return False


def _apply_din461_axes(ax: plt.Axes) -> None:
    """Apply DIN 461 label arrow + unit-at-penultimate-tick to *ax*."""
    import matplotlib.ticker as mticker

    for axis_name in ("x", "y"):
        axis_obj: matplotlib.axis.Axis = (
            ax.xaxis if axis_name == "x" else ax.yaxis
        )
        label_text: str = axis_obj.get_label().get_text()

        # Nothing to do for empty labels.
        if not label_text:
            continue

        # Already processed (idempotent guard).
        if label_text.endswith(r" $\rightarrow$"):
            continue

        # Skip categorical axes: FixedLocator whose tick labels are non-numeric
        # strings (e.g. "Elbow", "Total reward\nreached", …).
        if isinstance(axis_obj.get_major_locator(), mticker.FixedLocator):
            rendered = [t.get_text() for t in axis_obj.get_ticklabels()]
            non_empty = [t for t in rendered if t.strip()]
            if non_empty and not _is_numeric_str(non_empty[0]):
                continue

        # Parse "Name / unit" → name, unit.
        if " / " in label_text:
            name, unit = label_text.split(" / ", 1)
        else:
            name, unit = label_text, ""

        # 1. Add direction arrow to the axis label (mathtext for font compat.).
        axis_obj.set_label_text(fr"{name} $\rightarrow$")

        # 2. Place the unit string at the penultimate visible tick.
        if unit:
            _place_unit_at_penultimate(ax, axis_name, unit)


def _place_unit_at_penultimate(ax: plt.Axes, axis_name: str, unit: str) -> None:
    """Replace the penultimate rendered tick label with *unit*.

    Uses :class:`matplotlib.ticker.FixedFormatter` so the replacement
    persists through the subsequent ``fig.savefig()`` call.
    """
    import matplotlib.ticker as mticker

    axis_obj: matplotlib.axis.Axis = (
        ax.xaxis if axis_name == "x" else ax.yaxis
    )

    # Collect the currently rendered tick label strings (after canvas.draw).
    all_labels: list[str] = [t.get_text() for t in axis_obj.get_ticklabels()]

    # Identify indices of non-empty (visible) labels.
    visible_idx = [i for i, t in enumerate(all_labels) if t.strip()]

    if len(visible_idx) < 2:
        return  # Too few ticks to apply the rule meaningfully.

    # The penultimate visible tick gets the unit string.
    penultimate_i = visible_idx[-2]
    new_labels = list(all_labels)
    new_labels[penultimate_i] = unit

    # FixedFormatter must be paired with FixedLocator to avoid a UserWarning.
    tick_locs = axis_obj.get_ticklocs()
    axis_obj.set_major_locator(mticker.FixedLocator(tick_locs))
    axis_obj.set_major_formatter(mticker.FixedFormatter(new_labels))


# ── Directional "better" annotation ─────────────────────────────────────────


def annotate_direction(
    ax: plt.Axes,
    text: str = "lower is better",
    direction: Literal["up", "down"] = "down",
    *,
    loc: Literal["left", "right"] = "right",
) -> None:
    """Add a small arrow + label indicating which direction is 'better'.

    Placed unobtrusively to the right (or left) of the axis, using
    axes-coordinate transforms so it works regardless of data range.

    Parameters
    ----------
    ax
        Target axes (typically a bar chart).
    text
        Short label like ``"lower is better"`` or ``"higher is better"``.
    direction
        ``"down"`` draws a downward arrow; ``"up"`` draws an upward arrow.
    loc
        ``"right"`` places the annotation at the right edge of the axes;
        ``"left"`` at the left edge.
    """
    arrow = "↓" if direction == "down" else "↑"
    label = f"{arrow} {text}"

    x = 0.98 if loc == "right" else 0.02
    ha = "right" if loc == "right" else "left"

    ax.text(
        x, 0.97, label,
        transform=ax.transAxes,
        fontsize=7.5,
        color=FAPS_DARK_GREY,
        ha=ha, va="top",
        fontstyle="italic",
    )

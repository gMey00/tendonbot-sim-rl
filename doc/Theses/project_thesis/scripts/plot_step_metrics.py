"""Step-response performance metric overview (Results §5.1.1).

Renders a 4-panel grouped-bar chart comparing the Klein (2023) Gazebo
reference against the three Isaac Sim arm variants (PD, constant-tension
tendon, physical antiparallelogram tendon) along the four canonical
performance indicators (rise time, settling time, overshoot, NRMSE).

The Klein record only contains NRMSE; rise/settling/overshoot for that
column are rendered as hatched "n/a" bars to keep the visual structure
intact.  The physical variant's per-joint means are taken from the
aggregates that are otherwise hard-coded in ``5_0_Results.typ``.

Run::

    python doc/Theses/project_thesis/scripts/plot_step_metrics.py
"""

from __future__ import annotations

from collections import defaultdict
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

# ── Constants ────────────────────────────────────────────────────────────────
# Per-joint means for the *physical* variant are not stored as raw CSVs.
# They are duplicated from the inline annotations in 5_0_Results.typ to keep
# this script the single source of truth for the bar chart.
PHYSICAL_MEANS: Final[dict[str, dict[str, float]]] = {
    "elbow_joint":   {"rise_ms": 105.7, "settle_ms": 161.0, "nrmse_pct": 13.3},
    "wrist_x_joint": {"rise_ms": 105.3, "settle_ms": 233.0, "nrmse_pct": 12.5},
    "wrist_y_joint": {"rise_ms": 108.3, "settle_ms": 183.0, "nrmse_pct": 12.3},
}

JOINTS: Final = ("elbow_joint", "wrist_x_joint", "wrist_y_joint")
JOINT_LABELS: Final = ("Elbow", "Wrist X", "Wrist Y")
VARIANTS: Final = ("Klein (2023)", "PD", "Tendon", "Physical")
VARIANT_COLORS: Final = (
    c.pcfg.AMBER,        # Klein reference
    c.pcfg.FAPS_BLUE,    # PD
    c.pcfg.FAPS_GREEN,   # Tendon
    c.pcfg.PURPLE,       # Physical
)


def _aggregate_joint_means(records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Return ``means[joint][source][metric] = value`` (mean over amplitudes)."""
    accum: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        j = r["joint"]
        for src in ("klein", "pd", "tendon"):
            for metric in ("nrmse", "rise_ms", "settling_ms"):
                key = f"{src}_{metric}"
                if key in r and r[key] is not None:
                    accum[j][src][metric].append(float(r[key]))

    means: dict = {j: {} for j in JOINTS}
    for j in JOINTS:
        for src, metrics in accum[j].items():
            means[j][src] = {m: float(np.mean(vs)) for m, vs in metrics.items()}
    # Patch in the physical variant from the constant table.
    for j in JOINTS:
        phys = PHYSICAL_MEANS[j]
        means[j]["physical"] = {
            "rise_ms":     phys["rise_ms"],
            "settling_ms": phys["settle_ms"],
            "nrmse":       phys["nrmse_pct"],
        }
    return means


def _bar_panel(ax: plt.Axes, title: str, ylabel: str,
               values: list[list[float | None]]) -> None:
    """Grouped bar chart with a hatched fallback for ``None`` entries."""
    n_groups = len(JOINTS)
    n_bars = len(VARIANTS)
    bar_w = 0.78 / n_bars
    base = np.arange(n_groups)

    for idx, variant in enumerate(VARIANTS):
        offset = (idx - (n_bars - 1) / 2) * bar_w
        xs = base + offset
        for x, v in zip(xs, values[idx]):
            if v is None or v == 0:
                ax.bar(x, 1.0, width=bar_w * 0.9,
                       color="#F0F0F0", edgecolor="#A0A0A0",
                       hatch="///", linewidth=0.4, zorder=2)
                ax.text(x, 0.5, "n/a", ha="center", va="center",
                        rotation=90, fontsize=6.5, color="#777777", zorder=3)
            else:
                ax.bar(x, v, width=bar_w * 0.9,
                       color=VARIANT_COLORS[idx], edgecolor="white",
                       linewidth=0.4, zorder=2,
                       label=variant if x == xs[0] else None)
                ax.text(x, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                        ha="center", va="bottom",
                        fontsize=7, color="#333333", zorder=3)

    ax.set_xticks(base)
    ax.set_xticklabels(JOINT_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.set_axisbelow(True)
    # leave headroom for value labels
    cur_top = ax.get_ylim()[1]
    ax.set_ylim(0, cur_top * 1.15)


def main() -> None:
    c.init()
    records = c.load_validation_metrics()
    means = _aggregate_joint_means(records)

    # ── extract per-metric series ───────────────────────────────────────────
    def per_metric(metric: str) -> list[list[float | None]]:
        out: list[list[float | None]] = []
        for src in ("klein", "pd", "tendon", "physical"):
            row = []
            for j in JOINTS:
                row.append(means[j].get(src, {}).get(metric))
            out.append(row)
        return out

    rise   = per_metric("rise_ms")
    settle = per_metric("settling_ms")
    nrmse  = per_metric("nrmse")
    # Overshoot is currently not measured anywhere — full hatched panel.
    overshoot: list[list[float | None]] = [[None] * 3 for _ in VARIANTS]

    fig, axes = plt.subplots(
        2, 2,
        figsize=c.pcfg.scaled(11, 7.5, scale=1.0),
        constrained_layout=True,
    )
    _bar_panel(axes[0, 0], "(a) Rise time (10–90 %)",  "Rise time / ms",      rise)
    _bar_panel(axes[0, 1], "(b) Settling time (±2 %)", "Settling time / ms",  settle)
    _bar_panel(axes[1, 0], "(c) Peak overshoot",        "Overshoot / %",       overshoot)
    _bar_panel(axes[1, 1], "(d) Normalised RMSE",       "NRMSE / %",           nrmse)

    # Single shared legend across the figure
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=col, edgecolor="white")
        for col in VARIANT_COLORS
    ]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, facecolor="#F0F0F0",
                      edgecolor="#A0A0A0", hatch="///", linewidth=0.4)
    )
    fig.legend(
        handles, list(VARIANTS) + ["n/a (not recorded)"],
        loc="lower center", ncol=5, frameon=False,
        fontsize=8, bbox_to_anchor=(0.5, -0.04),
    )

    c.save(fig, c.RESULTS_FIG_DIR / "step_metrics_overview.png", tight=False)


if __name__ == "__main__":
    main()

"""Step-response performance metric overview (Results §5.1.1).

Renders a 4-panel grouped-bar chart comparing the Klein (2023) Gazebo
reference against the three Isaac Sim arm variants (PD, constant-tension
tendon, physical antiparallelogram tendon) along the four canonical
performance indicators (rise time, settling time, overshoot, NRMSE).

Bars show the mean across the per-joint amplitude sweep and error
bars show ±1σ across the same sweep.  Metrics are read directly from
the precomputed NPZ archives in
``src/tensegrity_pick/outputs/model_validation/``.  For the simulation
variants a NaN overshoot value indicates that the recorded trace never
exceeded the setpoint, so it contributes 0\u202f% to the overshoot panel.
Klein entries that were not reported in Tabelle\u202f4.2 (rise / overshoot for
several amplitudes, all settling times) remain hatched ``n/a``
placeholders.

Run::

    python doc/Theses/project_thesis/scripts/plot_step_metrics.py
"""

from __future__ import annotations

from collections import defaultdict
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

# ── Constants ────────────────────────────────────────────────────────────────────────────
JOINTS: Final = ("elbow_joint", "wrist_x_joint", "wrist_y_joint")
JOINT_LABELS: Final = ("Elbow", "Wrist X", "Wrist Y")
VARIANTS: Final = ("Klein (2023)", "PD", "Tendon", "Physical")
VARIANT_KEYS: Final = ("klein", "pd", "tendon", "physical")
VARIANT_COLORS: Final = (
    c.pcfg.AMBER,        # Klein reference
    c.pcfg.FAPS_BLUE,    # PD
    c.pcfg.FAPS_GREEN,   # Tendon
    c.pcfg.PURPLE,       # Physical
)


def _aggregate_joint_stats() -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """Return ``stats[joint][source][metric] = (mean, std)`` over amplitudes.

    Metrics are read directly from the precomputed NPZ archives.  For the
    simulation variants (``pd``, ``tendon``, ``physical``) a NaN
    ``overshoot_pct`` value indicates that the recorded trace never exceeded
    the setpoint and contributes ``0.0`` to the aggregate.  For the Klein
    reference, ``None`` indicates that the metric was not reported in
    Tabelle\u202f4.2, so those entries are omitted (renders as an n/a hatched bar).
    """
    accum: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for joint in JOINTS:
        for amp in c.JOINT_AMPS[joint]:
            # ─ Simulation variants from NPZ ───────────────────────────────────
            for src in ("pd", "tendon", "physical"):
                data = c.load_npz_trial(src, joint, amp)
                if data is None:
                    continue
                rise = float(data["rise_time_ms"])
                settle = float(data["settling_time_ms"])
                ov = float(data["overshoot_pct"])
                nrmse = float(data["nrmse_pct"])
                if not np.isnan(rise):
                    accum[joint][src]["rise_ms"].append(rise)
                if not np.isnan(settle):
                    accum[joint][src]["settling_ms"].append(settle)
                # NaN overshoot ⇒ trace never exceeded setpoint ⇒ 0 %.
                accum[joint][src]["overshoot"].append(0.0 if np.isnan(ov) else ov)
                accum[joint][src]["nrmse"].append(nrmse)

            # ─ Klein (2023) Gazebo reference ─────────────────────────────────
            klein = c.KLEIN_TABLE_4_2.get((joint, amp), {})
            if klein.get("nrmse") is not None:
                accum[joint]["klein"]["nrmse"].append(klein["nrmse"])
            if klein.get("rise_ms") is not None:
                accum[joint]["klein"]["rise_ms"].append(klein["rise_ms"])
            if klein.get("overshoot") is not None:
                accum[joint]["klein"]["overshoot"].append(klein["overshoot"])
            # Klein settling_ms: not provided in Tabelle 4.2 → renders as n/a.

    stats: dict = {j: {} for j in JOINTS}
    for j in JOINTS:
        for src, metrics in accum[j].items():
            stats[j][src] = {
                m: (float(np.mean(vs)), float(np.std(vs, ddof=0)))
                for m, vs in metrics.items()
            }
    return stats


def _bar_panel(ax: plt.Axes, title: str, ylabel: str,
               means: list[list[float | None]],
               stds: list[list[float | None]]) -> None:
    """Grouped bar chart with ±1σ error bars and hatched ``n/a`` fallback."""
    n_groups = len(JOINTS)
    n_bars = len(VARIANTS)
    bar_w = 0.78 / n_bars
    base = np.arange(n_groups)

    for idx, variant in enumerate(VARIANTS):
        offset = (idx - (n_bars - 1) / 2) * bar_w
        xs = base + offset
        for x, v, s in zip(xs, means[idx], stds[idx]):
            if v is None:
                ax.bar(x, 1.0, width=bar_w * 0.9,
                       color="#F0F0F0", edgecolor="#A0A0A0",
                       hatch="///", linewidth=0.4, zorder=2)
                ax.text(x, 0.5, "n/a", ha="center", va="center",
                        rotation=90, fontsize=6.5, color="#777777", zorder=3)
            else:
                err = float(s) if s is not None and s > 0 else None
                ax.bar(x, v, width=bar_w * 0.9,
                       color=VARIANT_COLORS[idx], edgecolor="white",
                       linewidth=0.4, zorder=2,
                       yerr=err,
                       error_kw=dict(ecolor="#333333", elinewidth=0.7,
                                     capsize=2.0, capthick=0.6),
                       label=variant if x == xs[0] else None)
                top = v + (err or 0.0)
                ax.text(x, top, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                        ha="center", va="bottom",
                        fontsize=7, color="#333333", zorder=3)

    ax.set_xticks(base)
    ax.set_xticklabels(JOINT_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.set_axisbelow(True)
    # leave headroom for value labels and error caps
    cur_top = ax.get_ylim()[1]
    ax.set_ylim(0, cur_top * 1.18)


def main() -> None:
    c.init()
    stats = _aggregate_joint_stats()

    # ── extract per-metric (mean, std) series ───────────────────────────────
    def per_metric(metric: str) -> tuple[list[list[float | None]], list[list[float | None]]]:
        means: list[list[float | None]] = []
        stds: list[list[float | None]] = []
        for src in VARIANT_KEYS:
            mrow: list[float | None] = []
            srow: list[float | None] = []
            for j in JOINTS:
                ms = stats[j].get(src, {}).get(metric)
                if ms is None:
                    mrow.append(None)
                    srow.append(None)
                else:
                    mrow.append(ms[0])
                    srow.append(ms[1])
            means.append(mrow)
            stds.append(srow)
        return means, stds

    rise_m,  rise_s  = per_metric("rise_ms")
    set_m,   set_s   = per_metric("settling_ms")
    nrm_m,   nrm_s   = per_metric("nrmse")
    over_m,  over_s  = per_metric("overshoot")

    fig, axes = plt.subplots(
        2, 2,
        figsize=c.pcfg.scaled(11, 7.5, scale=1.0),
        constrained_layout=True,
    )
    _bar_panel(axes[0, 0], "(a) Rise time (10–90 %)",  "Rise time / ms",      rise_m, rise_s)
    _bar_panel(axes[0, 1], "(b) Settling time (±2 %)", "Settling time / ms",  set_m,  set_s)
    _bar_panel(axes[1, 0], "(c) Peak overshoot",        "Overshoot / %",       over_m, over_s)
    _bar_panel(axes[1, 1], "(d) Normalised RMSE",       "NRMSE / %",           nrm_m,  nrm_s)

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
        handles, list(VARIANTS) + ["n/a (not reported by Klein)"],
        loc="lower center", ncol=5, frameon=False,
        fontsize=8, bbox_to_anchor=(0.5, -0.04),
    )

    c.save(fig, c.RESULTS_FIG_DIR / "step_metrics_overview.png", tight=False)


if __name__ == "__main__":
    main()

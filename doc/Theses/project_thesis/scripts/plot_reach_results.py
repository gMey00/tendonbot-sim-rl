"""Reach-task figures for Results §5.2.

Sources:

* Training-progress (``reach_convergence.png``) is read from the skrl
  TensorBoard event files declared in :data:`common.RUNS`.
* The seed-aggregated evaluation tables / bar charts / scatter plots
  are read from the per-(agent, variant, seed) JSON files produced by
  ``src/tensegrity_pick/scripts/skrl/evaluate.py`` and stored under
  :data:`common.REACH_EVAL_DIR`.

Produces:

* ``reach_convergence.png``      — 4-panel training-progress overview.
* ``reach_results_aggregate.png`` — seed-aggregated PPO bars (mean ± σ
  across 5 seeds) with per-seed dots overlaid, for the four reported
  evaluation metrics.
* ``reach_baselines.png``        — grouped success-rate bars covering
  the four agents (zero / random / DLS-IK / PPO) across the three
  robot variants; ``n/a`` hatched bars mark the heuristic on the two
  tendon variants (the DLS-IK controller operates in joint space and
  is undefined for the cable-tension action interface).
* ``reach_per_seed_scatter.png`` — per-seed PPO success rate (linear)
  and final position error (log scale) for all 3 × 5 PPO runs.
"""

from __future__ import annotations

from typing import Callable, Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

REACH_X_MAX: Final = 48_000  # common cutoff (PD variant training length)

# Binary reach-success rewards are logged as Episode_Reward terms with
# weight = 1e-6. Multiply by 1e6 to recover the underlying success rate.
SUCCESS_WEIGHT: Final = 1e-6


def _success_rate_panel(ax: plt.Axes, tag: str, *, title: str) -> bool:
    has = c.plot_variants_curve(
        ax, "reach", tag,
        x_max=REACH_X_MAX, y_label="Rate", title=title,
        value_scale=1.0 / SUCCESS_WEIGHT,
    )
    ax.set_ylim(0, 1.0)
    return has


def _convergence_figure() -> None:
    fig, axes = plt.subplots(
        4, 1,
        figsize=c.pcfg.scaled(11, 14.0, scale=1.0),
        constrained_layout=False,
    )

    # (a) total reward — mean line + min/max envelope per variant
    c.plot_envelope(
        axes[0], "reach",
        "Reward / Total reward (mean)",
        "Reward / Total reward (min)",
        "Reward / Total reward (max)",
        x_max=REACH_X_MAX,
    )
    axes[0].set_title(
        "(a) Total reward (mean ± min/max envelope)",
        fontsize=10, fontweight="bold", loc="left",
    )
    axes[0].set_ylabel("Reward")

    # (b) position-reached success rate
    _success_rate_panel(
        axes[1], "Info / Episode_Reward/position_reached",
        title=r"(b) Position-reached rate ($d_\mathrm{pos} < 5$ cm)",
    )

    # (c) full pose-reached success rate
    _success_rate_panel(
        axes[2], "Info / Episode_Reward/pose_reached",
        title="(c) Full pose-reached rate (position + orientation)",
    )

    # (d) policy std deviation
    c.plot_variants_curve(
        axes[3], "reach", "Policy / Standard deviation",
        x_max=REACH_X_MAX, y_label="σ",
        title="(d) Policy standard deviation",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    c.figure_bottom_legend(fig, axes, ncol=3, y=0.005)
    c.save(fig, c.RESULTS_FIG_DIR / "reach_convergence.png", tight=False)


def _comparison_figure() -> None:
    """Final-performance bar chart — total reward + three success rates.

    Reads from TensorBoard training events (single-run-per-variant) and
    is retained as a quick-look chart against the training plateau. The
    seed-aggregated evaluation results are produced separately by
    :func:`_aggregate_eval_figure`.
    """
    metric_specs: list[tuple[str, str, Callable[[float], float]]] = [
        ("Total reward",  "Reward / Total reward (mean)",                 lambda v: v),
        ("Position\nreached",    "Info / Episode_Reward/position_reached",    lambda v: v / SUCCESS_WEIGHT),
        ("Orientation\nreached", "Info / Episode_Reward/orientation_reached", lambda v: v / SUCCESS_WEIGHT),
        ("Full pose\nreached",   "Info / Episode_Reward/pose_reached",        lambda v: v / SUCCESS_WEIGHT),
    ]
    rows: dict[str, list[float | None]] = {v: [] for v in c.VARIANTS}
    for _, tag, fn in metric_specs:
        for variant in c.VARIANTS:
            mean = c.final_mean("reach", variant, tag, x_max=REACH_X_MAX)
            rows[variant].append(None if mean is None else fn(mean))

    metric_labels = [m[0] for m in metric_specs]
    bar_w = 0.27

    fig, (ax_r, ax_p) = plt.subplots(
        1, 2,
        figsize=c.pcfg.scaled(11, 4.2, scale=1.0),
        gridspec_kw={"width_ratios": (1, 3)},
        constrained_layout=True,
    )

    for ax, slice_ in ((ax_r, slice(0, 1)), (ax_p, slice(1, None))):
        sub_labels = metric_labels[slice_]
        sub_base = np.arange(len(sub_labels))
        for offset, variant in zip((-bar_w, 0.0, bar_w), c.VARIANTS):
            vals = rows[variant][slice_]
            for x_pos, val in zip(sub_base + offset, vals):
                if val is None:
                    c.hatched_missing_bar(ax, x_pos, height=0.05, width=bar_w * 0.95)
                    continue
                ax.bar(x_pos, val, width=bar_w * 0.95,
                       color=c.VARIANT_COLOR[variant], edgecolor="white",
                       linewidth=0.4,
                       label=c.VARIANT_LABEL[variant] if ax is ax_r else None)
                txt = f"{val:+.2f}" if ax is ax_r else f"{100 * val:.1f} %"
                ax.text(x_pos, val + (0.02 if val >= 0 else -0.04), txt,
                        ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=7, color="#333333")
        ax.set_xticks(sub_base); ax.set_xticklabels(sub_labels)
        ax.set_axisbelow(True)

    ax_r.set_ylabel("Total reward")
    ax_r.axhline(0, color=c.pcfg.FAPS_DARK_GREY, linewidth=0.6)
    ax_p.set_ylabel("Success rate"); ax_p.set_ylim(0, 1.10)
    ax_r.set_title("Final reach-task performance — three tensegrity variants",
                   fontsize=10, fontweight="bold", loc="left")
    ax_r.legend(loc="lower left", ncol=3, fontsize=8)
    c.save(fig, c.RESULTS_FIG_DIR / "reach_comparison.png", tight=False)


# ── Seed-aggregated evaluation figures ──────────────────────────────────────
# All three functions below read the per-(agent, variant, seed) JSON files
# under :data:`common.REACH_EVAL_DIR`. Each figure reports five-seed
# aggregates: mean across seeds for bar heights, ±1σ across seeds for
# error bars, and per-seed dots overlaid for distribution.

# Metrics reported by the eval harness, in display order.
# (display label, JSON key, axis-y label, value transform).
_EVAL_METRICS: Final[tuple[tuple[str, str, str, Callable[[float], float]], ...]] = (
    ("Success rate",      "success_rate",      "Success rate",          lambda v: v),
    ("Position error",    "position_error",    "Position error / m",    lambda v: v),
    ("Orientation error", "orientation_error", "Orientation error",     lambda v: v),
    ("Reach time",        "reach_time_mean",   "Reach time / s",        lambda v: v),
)


def _aggregate_eval_figure() -> None:
    """PPO seed-aggregated bars + per-seed dots for all four eval metrics."""
    fig, axes = plt.subplots(
        1, 4,
        figsize=c.pcfg.scaled(11, 3.8, scale=1.0),
        constrained_layout=True,
    )
    x_pos = np.arange(len(c.VARIANTS))
    for ax, (title, json_key, ylabel, _fn) in zip(axes, _EVAL_METRICS):
        for i, variant in enumerate(c.VARIANTS):
            per_seed = c.reach_eval_seeds("checkpoint", variant, json_key)
            if len(per_seed) == 0:
                c.hatched_missing_bar(ax, x_pos[i], height=1.0, width=0.7)
                continue
            mean, std = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))
            color = c.VARIANT_COLOR[variant]
            ax.bar(x_pos[i], mean, width=0.7, color=color,
                   edgecolor="white", linewidth=0.4,
                   yerr=std, capsize=4, ecolor=c.pcfg.FAPS_DARK_GREY,
                   error_kw={"linewidth": 0.8})
            # per-seed dots (small horizontal jitter for visibility)
            jitter = np.linspace(-0.12, 0.12, num=len(per_seed))
            ax.scatter(x_pos[i] + jitter, per_seed,
                       s=14, color="white", edgecolor=color,
                       linewidth=0.9, zorder=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.VARIANT_LABEL[v] for v in c.VARIANTS], fontsize=8,
                           rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.set_axisbelow(True)
    # Mark the 0.05 m success threshold on the position-error panel.
    axes[1].axhline(
        c.REACH_SUCCESS_THRESHOLD_M, color="#C0392B", linestyle="--",
        linewidth=0.8, alpha=0.85, label="5 cm threshold",
    )
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)
    axes[0].set_ylim(0, 1.05)
    c.save(fig, c.RESULTS_FIG_DIR / "reach_results_aggregate.png", tight=False)


def _baselines_figure() -> None:
    """Grouped success-rate bars: 4 agents × 3 variants, seed-aggregated."""
    fig, ax = plt.subplots(
        figsize=c.pcfg.scaled(11, 4.0, scale=1.0),
        constrained_layout=True,
    )
    x_pos = np.arange(len(c.VARIANTS))
    n_agents = len(c.AGENTS)
    bar_w = 0.78 / n_agents
    for i, agent in enumerate(c.AGENTS):
        offset = (i - (n_agents - 1) / 2) * bar_w
        for v_idx, variant in enumerate(c.VARIANTS):
            per_seed = c.reach_eval_seeds(agent, variant, "success_rate")
            x = x_pos[v_idx] + offset
            if len(per_seed) == 0:
                # heuristic on tendon variants — undefined by design.
                c.hatched_missing_bar(ax, x, height=0.04, width=bar_w * 0.92)
                ax.text(x, 0.06, "n/a", ha="center", va="bottom",
                        fontsize=6.5, color="#666666", style="italic")
                continue
            mean, std = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))
            ax.bar(x, mean, width=bar_w * 0.92,
                   color=c.AGENT_COLOR[agent], edgecolor="white",
                   linewidth=0.4,
                   yerr=std, capsize=3, ecolor=c.pcfg.FAPS_DARK_GREY,
                   error_kw={"linewidth": 0.7},
                   label=c.AGENT_LABEL[agent] if v_idx == 0 else None)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c.VARIANT_LABEL[v] for v in c.VARIANTS])
    ax.set_ylabel("Success rate (5-seed mean ± σ)")
    ax.set_ylim(0, 1.10)
    ax.set_title("Reach success rate — agents × robot variants",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    ax.set_axisbelow(True)
    c.save(fig, c.RESULTS_FIG_DIR / "reach_baselines.png", tight=False)


def _per_seed_scatter_figure() -> None:
    """PPO per-seed success rate (linear) and final position error (log)."""
    fig, (ax_s, ax_p) = plt.subplots(
        1, 2,
        figsize=c.pcfg.scaled(11, 3.8, scale=1.0),
        constrained_layout=True,
    )
    x_pos = np.arange(len(c.VARIANTS))
    for i, variant in enumerate(c.VARIANTS):
        s_seeds = c.reach_eval_seeds("checkpoint", variant, "success_rate")
        p_seeds = c.reach_eval_seeds("checkpoint", variant, "position_error")
        color = c.VARIANT_COLOR[variant]
        jitter = np.linspace(-0.12, 0.12, num=max(len(s_seeds), 1))
        ax_s.scatter(x_pos[i] + jitter[: len(s_seeds)], s_seeds,
                     s=36, color=color, edgecolor="white", linewidth=0.6,
                     label=c.VARIANT_LABEL[variant])
        if len(s_seeds) > 0:
            ax_s.hlines(np.mean(s_seeds),
                        x_pos[i] - 0.22, x_pos[i] + 0.22,
                        color=color, linewidth=1.6)
        ax_p.scatter(x_pos[i] + jitter[: len(p_seeds)], p_seeds,
                     s=36, color=color, edgecolor="white", linewidth=0.6)
        if len(p_seeds) > 0:
            ax_p.hlines(np.mean(p_seeds),
                        x_pos[i] - 0.22, x_pos[i] + 0.22,
                        color=color, linewidth=1.6)
    for ax in (ax_s, ax_p):
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.VARIANT_LABEL[v] for v in c.VARIANTS])
        ax.set_axisbelow(True)
    ax_s.set_ylabel("Success rate")
    ax_s.set_ylim(0, 1.05)
    ax_s.set_title("(a) PPO success rate per seed",
                   fontsize=10, fontweight="bold", loc="left")
    ax_s.legend(loc="lower left", fontsize=8, frameon=False)
    ax_p.set_yscale("log")
    ax_p.set_ylabel("Final position error / m")
    ax_p.axhline(
        c.REACH_SUCCESS_THRESHOLD_M, color="#C0392B", linestyle="--",
        linewidth=0.8, alpha=0.85, label="5 cm threshold",
    )
    ax_p.legend(loc="upper left", fontsize=7, frameon=False)
    ax_p.set_title("(b) PPO final position error per seed (log scale)",
                   fontsize=10, fontweight="bold", loc="left")
    c.save(fig, c.RESULTS_FIG_DIR / "reach_per_seed_scatter.png", tight=False)


def main() -> None:
    c.init()
    _convergence_figure()
    _comparison_figure()
    _aggregate_eval_figure()
    _baselines_figure()
    _per_seed_scatter_figure()


if __name__ == "__main__":
    main()

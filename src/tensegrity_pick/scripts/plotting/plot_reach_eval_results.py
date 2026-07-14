#!/usr/bin/env python3
"""Aggregate reach evaluation JSONs into tables and figures 08–12.

Reads the eval JSONs written by ``scripts/skrl/evaluate_reach.py``, aggregates
per (variant, agent) across seeds, prints a markdown summary table, and writes
the aggregate figures referenced by ``doc/reports/task_evaluations/reach_evaluation_findings.md``:

* ``08_baseline_comparison.png``      — success rate bars per (variant, agent)
* ``09_position_error_comparison.png`` — final position error bars + threshold
* ``10_orientation_error_comparison.png`` — final orientation error bars
* ``11_ppo_per_seed.png``             — PPO per-seed scatter (success + pos err)
* ``12_per_seed_distribution.png``    — all cells, per-seed dots + mean ± std

Multiple eval directories can be given; on (variant, agent, seed) collisions
the *first* directory wins.  The default reads the current eval dir first and
falls back to the archived 2026-05 batch for the PD / sim-tendon reference
cells so the comparison covers all variants without re-running their evals.

Usage:
    cd src/tensegrity_pick
    python scripts/plotting/plot_reach_eval_results.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_EVAL_DIRS = [
    "logs/skrl/reach/eval",                # current batch
    "logs/skrl/theses_logs/reach/eval",    # archived 2026-05 batch (PD / tendon reference)
]
FIG_DIR = Path(
    "source/tensegrity_pick/tensegrity_pick/tasks/"
    "manager_based/reach/figures/aggregate"
)
VARIANTS = [
    "tensegrity",
    "tensegrity_tendon",
    "tensegrity_physical_tendon",
    "tensegrity_physical_hier",
]
VARIANT_LABELS = {
    "tensegrity": "PD",
    "tensegrity_tendon": "Sim tendon",
    "tensegrity_physical_tendon": "Physical tendon",
    "tensegrity_physical_hier": "Physical hier.",
}
AGENTS = ["zero", "random", "heuristic", "checkpoint"]
AGENT_LABELS = {"zero": "Zero", "random": "Random", "heuristic": "DLS-IK", "checkpoint": "PPO"}
AGENT_COLORS = {"zero": "#8C8C8C", "random": "#E5A11C", "heuristic": "#C23F38", "checkpoint": "#00519E"}
SUCCESS_THRESHOLD_M = 0.05


def load(eval_dirs: list[str]) -> dict:
    """cells[(variant, agent)] = list of per-seed payloads (dedup by seed, first dir wins)."""
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    for d in eval_dirs:
        for f in sorted(glob.glob(str(Path(d) / "*.json"))):
            p = json.load(open(f))
            key = (p["variant"], p["agent"])
            cells.setdefault(key, {})
            if p["seed"] not in cells[key]:
                cells[key][p["seed"]] = p
    return {k: [v[s] for s in sorted(v)] for k, v in cells.items()}


def seed_stats(payloads: list[dict], metric: str) -> tuple[float, float, list[float]]:
    vals = [p["metrics"][metric]["mean"] for p in payloads
            if metric in p.get("metrics", {}) and np.isfinite(p["metrics"][metric]["mean"])]
    if not vals:
        return float("nan"), float("nan"), []
    return float(np.mean(vals)), float(np.std(vals)), vals


def variants_present(cells: dict) -> list[str]:
    present = {v for (v, _) in cells}
    return [v for v in VARIANTS if v in present] + sorted(present - set(VARIANTS))


def print_table(cells: dict) -> list[str]:
    lines = [
        "| Variant | Agent | Success rate | Success held | Pos err [m] | Ori err | Reach time [s] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for v in variants_present(cells):
        for a in AGENTS:
            payloads = cells.get((v, a))
            if not payloads:
                continue
            sr_m, sr_s, _ = seed_stats(payloads, "success_rate")
            sh_m, _, _ = seed_stats(payloads, "success_held")
            pe_m, _, _ = seed_stats(payloads, "position_error")
            oe_m, _, _ = seed_stats(payloads, "orientation_error")
            rt_m, _, _ = seed_stats(payloads, "reach_time_mean")
            sh_str = f"{sh_m:.3f}" if np.isfinite(sh_m) else "—"
            lines.append(
                f"| {VARIANT_LABELS.get(v, v)} | {AGENT_LABELS.get(a, a)} "
                f"| {sr_m:.3f} ± {sr_s:.3f} | {sh_str} | {pe_m:.3f} | {oe_m:.3f} | {rt_m:.3f} |"
            )
    print("\n".join(lines))
    return lines


def _grouped_bars(cells: dict, metric: str, ylabel: str, title: str, fname: str,
                  hline: float | None = None, hline_label: str | None = None,
                  log_scale: bool = False) -> None:
    vs = variants_present(cells)
    agents = [a for a in AGENTS if any((v, a) in cells for v in vs) and a != "heuristic"]
    x = np.arange(len(vs))
    bar_w = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(max(8, 2.4 * len(vs)), 4.5))
    for i, a in enumerate(agents):
        means, stds = [], []
        for v in vs:
            m, s, _ = seed_stats(cells.get((v, a), []), metric)
            means.append(0.0 if np.isnan(m) else m)
            stds.append(0.0 if np.isnan(s) else s)
        offset = (i - (len(agents) - 1) / 2) * bar_w
        ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
               color=AGENT_COLORS[a], label=AGENT_LABELS[a],
               edgecolor="#333333", linewidth=0.5)

    heur = cells.get(("tensegrity", "heuristic"))
    if heur:
        h_m, _, _ = seed_stats(heur, metric)
        if np.isfinite(h_m):
            ax.axhline(h_m, color=AGENT_COLORS["heuristic"], linestyle=":", linewidth=1.2,
                       label=f"DLS-IK (PD) = {h_m:.2f}")
    if hline is not None:
        ax.axhline(hline, color="#C23F38", linestyle="--", linewidth=1.2, label=hline_label)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in vs])
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / fname}")


def fig_ppo_per_seed(cells: dict, fname: str) -> None:
    vs = [v for v in variants_present(cells) if (v, "checkpoint") in cells]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, (metric, ylabel, log) in enumerate([
        ("success_rate", "Success rate", False),
        ("position_error", "Final position error [m] (log)", True),
    ]):
        ax = axes[j]
        for i, v in enumerate(vs):
            _, _, vals = seed_stats(cells[(v, "checkpoint")], metric)
            ax.scatter([i] * len(vals), vals, s=40, zorder=3, color="#00519E", alpha=0.75)
            if vals:
                ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color="#C23F38", linewidth=2, zorder=4)
        if metric == "position_error":
            ax.axhline(SUCCESS_THRESHOLD_M, color="#C23F38", linestyle="--", linewidth=1,
                       label=f"{SUCCESS_THRESHOLD_M} m threshold")
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in vs])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Trained PPO policies — per-seed spread (dots) and mean (red)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / fname}")


def fig_per_seed_distribution(cells: dict, fname: str) -> None:
    vs = variants_present(cells)
    groups: list[tuple[str, str, list[float]]] = []
    for v in vs:
        for a in AGENTS:
            if (v, a) in cells:
                _, _, vals = seed_stats(cells[(v, a)], "success_rate")
                groups.append((v, a, vals))

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(groups)), 4.8))
    for i, (v, a, vals) in enumerate(groups):
        ax.scatter([i] * len(vals), vals, s=26, zorder=3, color=AGENT_COLORS[a], alpha=0.8)
        if vals:
            m, s = float(np.mean(vals)), float(np.std(vals))
            ax.errorbar([i], [m], yerr=[s], fmt="_", color="#333333",
                        markersize=16, capsize=4, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [f"{VARIANT_LABELS.get(v, v)}\n{AGENT_LABELS.get(a, a)}" for v, a, _ in groups],
        fontsize=7,
    )
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("All evaluation cells — per-seed dots, mean ± std bars")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {FIG_DIR / fname}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_dirs", nargs="+", default=DEFAULT_EVAL_DIRS,
        help="Eval JSON directories, first match per (variant, agent, seed) wins.",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cells = load(args.eval_dirs)
    if not cells:
        raise SystemExit(f"No eval JSONs found in {args.eval_dirs}")

    n = sum(len(v) for v in cells.values())
    print(f"Loaded {n} eval JSONs into {len(cells)} (variant, agent) cells\n")
    print_table(cells)
    print()

    _grouped_bars(cells, "success_rate", "Success rate",
                  "Reach success rate (mean ± std over seeds)",
                  "08_baseline_comparison.png")
    _grouped_bars(cells, "position_error", "Final position error [m] (log)",
                  "Final position error (mean ± std over seeds)",
                  "09_position_error_comparison.png",
                  hline=SUCCESS_THRESHOLD_M, hline_label=f"{SUCCESS_THRESHOLD_M} m success threshold",
                  log_scale=True)
    _grouped_bars(cells, "orientation_error", "Final orientation error [rad-norm]",
                  "Final orientation error (mean ± std over seeds)",
                  "10_orientation_error_comparison.png")
    fig_ppo_per_seed(cells, "11_ppo_per_seed.png")
    fig_per_seed_distribution(cells, "12_per_seed_distribution.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate cube-place evaluation JSONs into tables and figures.

Reads ``logs/skrl/cube_place/eval/*.json`` (written by ``evaluate_place.py``),
aggregates per (variant, agent) across seeds, prints a markdown summary table,
and writes seed-aggregated figures into the cube_place ``figures/aggregate``
tree (analogous to the reach aggregate figures).

Usage:
    cd src/tensegrity_pick
    python scripts/plot_place_eval_results.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths are relative to the intended working directory (src/tensegrity_pick),
# matching the rest of the skrl tooling. FIG_DIR therefore omits the leading
# src/tensegrity_pick/ prefix to avoid writing into a doubled stray tree.
EVAL_DIR = Path("logs/skrl/cube_place/eval")
FIG_DIR = Path(
    "source/tensegrity_pick/tensegrity_pick/tasks/"
    "manager_based/cube_place/figures/aggregate"
)
VARIANTS = ["tensegrity", "tensegrity_tendon", "tensegrity_physical_tendon"]
VARIANT_LABELS = {
    "tensegrity": "PD",
    "tensegrity_tendon": "Sim tendon",
    "tensegrity_physical_tendon": "Physical tendon",
}
AGENTS = ["zero", "random", "checkpoint"]
AGENT_LABELS = {"zero": "zero", "random": "random", "checkpoint": "PPO"}


def load() -> dict:
    """cells[(variant, agent)] = list of per-seed payloads."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for f in sorted(glob.glob(str(EVAL_DIR / "*.json"))):
        p = json.load(open(f))
        cells.setdefault((p["variant"], p["agent"]), []).append(p)
    return cells


def seed_mean_std(payloads: list[dict], metric: str) -> tuple[float, float, list[float]]:
    """Mean of per-seed metric means, std across seeds, and the per-seed list."""
    vals = [p["metrics"][metric]["mean"] for p in payloads
            if metric in p["metrics"] and np.isfinite(p["metrics"][metric]["mean"])]
    if not vals:
        return float("nan"), float("nan"), []
    return float(np.mean(vals)), float(np.std(vals)), vals


def print_table(cells: dict) -> None:
    print("\n| Variant | Agent | Success | Held | Grasp | Red-safe | Place XY err [m] | Place time [s] |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for v in VARIANTS:
        for a in AGENTS:
            ps = cells.get((v, a))
            if not ps:
                continue
            sr_m, sr_s, _ = seed_mean_std(ps, "success_rate")
            hd_m, _, _ = seed_mean_std(ps, "success_held")
            gr_m, _, _ = seed_mean_std(ps, "grasp_rate")
            rs_m, _, _ = seed_mean_std(ps, "red_safe_rate")
            pe_m, _, _ = seed_mean_std(ps, "place_xy_error")
            pt_m, _, _ = seed_mean_std(ps, "place_time_mean")
            print(f"| {VARIANT_LABELS[v]} | {AGENT_LABELS[a]} | {sr_m:.3f} ± {sr_s:.3f} "
                  f"| {hd_m:.3f} | {gr_m:.3f} | {rs_m:.3f} | {pe_m:.3f} | {pt_m:.3f} |")


def fig_per_seed(cells: dict, metric: str, ylabel: str, title: str, fname: str) -> None:
    """PPO per-seed scatter for an arbitrary metric (one column per variant)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(VARIANTS))
    for v_i, v in enumerate(VARIANTS):
        ps = cells.get((v, "checkpoint"))
        if not ps:
            continue
        _, _, vals = seed_mean_std(ps, metric)
        ax.scatter([v_i] * len(vals), vals, s=80, alpha=0.7, zorder=3)
        m = np.mean(vals)
        ax.hlines(m, v_i - 0.2, v_i + 0.2, color="k", lw=2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def fig_baseline(cells: dict, metric: str, ylabel: str, title: str, fname: str) -> None:
    """Grouped bar (zero / random / PPO) for an arbitrary metric, mean ± std."""
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(VARIANTS))
    for a_i, a in enumerate(AGENTS):
        means, stds = [], []
        for v in VARIANTS:
            ps = cells.get((v, a))
            m, s, _ = seed_mean_std(ps, metric) if ps else (np.nan, np.nan, [])
            means.append(m)
            stds.append(s)
        ax.bar(x + (a_i - 1) * width, means, width, yerr=stds, capsize=3,
               label=AGENT_LABELS[a])
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    cells = load()
    if not cells:
        print(f"No eval JSONs found under {EVAL_DIR}")
        return
    print_table(cells)
    # Place success
    fig_per_seed(cells, "success_rate", "Place success rate",
                 "PPO per-seed place success (5 seeds/variant)",
                 "place_ppo_per_seed.png")
    fig_baseline(cells, "success_rate", "Place success rate",
                 "Cube-place success: baselines vs PPO (mean ± std over 5 seeds)",
                 "place_baseline_comparison.png")
    # Grasp success
    fig_per_seed(cells, "grasp_rate", "Grasp success rate",
                 "PPO per-seed grasp success (5 seeds/variant)",
                 "place_ppo_per_seed_grasp.png")
    fig_baseline(cells, "grasp_rate", "Grasp success rate",
                 "Cube-place grasp: baselines vs PPO (mean ± std over 5 seeds)",
                 "place_baseline_comparison_grasp.png")
    print(f"\n[plot] wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()

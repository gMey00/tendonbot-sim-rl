"""Thesis-ready figures from benchmark result artifacts.

Reads the ``results.jsonl`` produced by ``run_matrix.py`` (one run dir per spec)
and renders publication-quality matplotlib figures (PDF for LaTeX + PNG for
preview) into a figures directory. Aggregates process-repeats to median + IQR
per (task/config, N), per the report's statistical protocol (Section 4).

Design follows the dataviz discipline adapted for print:
  * Okabe-Ito colourblind-safe categorical palette (the scientific-publication
    standard; CVD-validated) — hues assigned in fixed order, never cycled.
  * one y-axis per figure (never dual-axis); log-x for env-count sweeps.
  * recessive y-grid, thin spines, direct series labels + legend for >=2 series.
  * magnitude → sequential shading; identity → categorical hues.

Usage
-----
    python scripts/benchmarking/plot_benchmarks.py --auto \
        --out doc/reports/figures/benchmarks
    # or point at explicit run dirs:
    python scripts/benchmarking/plot_benchmarks.py \
        --runs doc/reports/data/benchmarks/rigid_baseline/faps_workstation_* \
        --out doc/reports/figures/benchmarks
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --- Okabe-Ito colourblind-safe palette (fixed order, never cycled) ----------
OKABE = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
CATEGORICAL = [OKABE["blue"], OKABE["vermillion"], OKABE["green"], OKABE["orange"],
               OKABE["skyblue"], OKABE["purple"], OKABE["black"], OKABE["yellow"]]

# __file__ = .../src/tensegrity_pick/scripts/benchmarking/plot_benchmarks.py
# parents: [0]=benchmarking [1]=scripts [2]=tensegrity_pick [3]=src [4]=repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "doc/reports/data/benchmarks"

# Spec names may vary (a quick reduced sweep vs. the full study); pick whichever
# cloth env-count sweep is present so the figures find the data either way.
RIGID_SPEC = "rigid_baseline"
CLOTH_SPEC_CANDIDATES = ("env_count_cloth_quick", "env_count_sweep")


def _cloth_key(runs: dict) -> str | None:
    for k in CLOTH_SPEC_CANDIDATES:
        if k in runs and runs[k]:
            return k
    return None


def set_thesis_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#666666",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "figure.autolayout": True,
    })


# ---------------------------------------------------------------------------
# Loading + aggregation
# ---------------------------------------------------------------------------
def load_run(run_dir: Path) -> list[dict]:
    jl = run_dir / "results.jsonl"
    rows: list[dict] = []
    if jl.exists():
        rows = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    else:  # fall back to per-point files (e.g. a slurm run not yet collected)
        for pf in sorted((run_dir / "points").glob("point_*.json")):
            try:
                rows.append(json.loads(pf.read_text().strip().splitlines()[-1]))
            except Exception:
                pass
    return rows


def discover_latest() -> dict[str, Path]:
    """Newest run dir per spec under DATA_ROOT."""
    latest: dict[str, Path] = {}
    if not DATA_ROOT.exists():
        return latest
    for spec_dir in sorted(DATA_ROOT.iterdir()):
        if not spec_dir.is_dir():
            continue
        runs = sorted([d for d in spec_dir.iterdir() if (d / "results.jsonl").exists()
                       or (d / "points").exists()])
        if runs:
            latest[spec_dir.name] = runs[-1]
    return latest


def valid(r: dict) -> bool:
    return r.get("status") == "ok" and r.get("physics_ok", r.get("sane", False))


def agg(rows: list[dict], key_fn, value_key: str):
    """Group by key_fn → (sorted x, median, lo=q25, hi=q75) using num_envs as x."""
    buckets: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not valid(r) or r.get(value_key) is None:
            continue
        try:
            v = float(r[value_key])
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        buckets[key_fn(r)][int(r["num_envs"])].append(v)
    out = {}
    for k, byN in buckets.items():
        xs = sorted(byN)
        med = [statistics.median(byN[x]) for x in xs]
        lo = [min(byN[x]) for x in xs]
        hi = [max(byN[x]) for x in xs]
        out[k] = (xs, med, lo, hi)
    return out


def _logx(ax, xs_all):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(sorted(set(xs_all)))
    ax.tick_params(axis="x", labelrotation=0)


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_throughput_knee(runs: dict[str, list[dict]], out_dir: Path):
    """env·steps/s vs N — rigid baseline vs cloth on one log-x axis (the knee)."""
    series = []
    if RIGID_SPEC in runs:
        series.append(("Rigid (reach, UR5e joint)", runs[RIGID_SPEC]))
    ck = _cloth_key(runs)
    if ck:
        series.append(("Cloth (shirt-place, PBD ~11k)", runs[ck]))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    xs_all = []
    for i, (label, rows) in enumerate(series):
        a = agg(rows, lambda r: "s", "env_steps_per_s")
        if "s" not in a:
            continue
        xs, med, lo, hi = a["s"]
        xs_all += xs
        c = CATEGORICAL[i]
        ax.plot(xs, med, "-o", color=c, label=label, zorder=3)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)
        # Annotate the knee only if the peak is INTERIOR (throughput then declines)
        # — a still-rising endpoint (rigid) is not a knee.
        pk = med.index(max(med))
        if 0 < pk < len(med) - 1:
            ax.annotate(f"knee  N={xs[pk]}", (xs[pk], med[pk]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7.5, color=c, ha="center", fontweight="bold")
    # log-log: near-linear scaling reads as a straight line (Brax-style); the very
    # different rigid vs cloth magnitudes both stay legible, and the cloth knee
    # shows as a departure/peak.
    ax.set_yscale("log")
    _logx(ax, xs_all)
    ax.set_xlabel("parallel environments $N$")
    ax.set_ylabel("throughput  (env·steps / s)")
    ax.set_title("Aggregate throughput vs. environment count")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out_dir, "throughput_vs_envcount")


def fig_perenv_and_mem(runs: dict[str, list[dict]], out_dir: Path):
    """Two small-multiple panels: per-env steps/s and driver GPU memory vs N."""
    ck = _cloth_key(runs)
    series = [(k, lbl) for k, lbl in ((RIGID_SPEC, "Rigid"), (ck, "Cloth"))
              if k and k in runs]
    if not series:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for panel, (value_key, ylabel, title) in enumerate([
        ("steps_per_s", "per-env step rate (steps/s)", "Per-environment step rate"),
        ("driver_used_gb", "GPU memory, driver-used (GiB)", "GPU memory footprint"),
    ]):
        ax = axes[panel]
        xs_all = []
        for i, (spec, label) in enumerate(series):
            a = agg(runs[spec], lambda r: "s", value_key)
            if "s" not in a:
                continue
            xs, med, lo, hi = a["s"]
            xs_all += xs
            ax.plot(xs, med, "-o", color=CATEGORICAL[i], label=label)
            ax.fill_between(xs, lo, hi, color=CATEGORICAL[i], alpha=0.15, linewidth=0)
        _logx(ax, xs_all)
        ax.set_xlabel("parallel environments $N$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
    _save(fig, out_dir, "perenv_and_memory")


def fig_energy(runs: dict[str, list[dict]], out_dir: Path):
    ck = _cloth_key(runs)
    series = [(k, lbl) for k, lbl in ((RIGID_SPEC, "Rigid"), (ck, "Cloth"))
              if k and k in runs]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs_all = []
    for i, (spec, label) in enumerate(series):
        a = agg(runs[spec], lambda r: "s", "energy_j_per_Mstep")
        if "s" not in a:
            continue
        xs, med, lo, hi = a["s"]
        xs_all += xs
        # kJ per Mstep for readability
        med = [m / 1000 for m in med]; lo = [x / 1000 for x in lo]; hi = [x / 1000 for x in hi]
        ax.plot(xs, med, "-o", color=CATEGORICAL[i], label=label)
        ax.fill_between(xs, lo, hi, color=CATEGORICAL[i], alpha=0.15, linewidth=0)
    _logx(ax, xs_all)
    ax.set_xlabel("parallel environments $N$")
    ax.set_ylabel("energy  (kJ / million env·steps)")
    ax.set_title("Energy efficiency vs. environment count")
    ax.legend(frameon=False)
    _save(fig, out_dir, "energy_vs_envcount")


def fig_action_space(runs: dict[str, list[dict]], out_dir: Path):
    """Marginal per-step controller cost by action space, at a fixed N (grouped
    bars over N). Uses step_ms_p50 (lower = cheaper controller)."""
    rows = runs.get("action_space_cost")
    if not rows:
        return
    order = ["joint", "ik_rel", "ik_abs", "osc"]
    pretty = {"joint": "joint", "ik_rel": "IK-rel", "ik_abs": "IK-abs", "osc": "OSC"}
    a = agg(rows, lambda r: r.get("config", {}).get("action_space", "?"), "step_ms_p50")
    Ns = sorted({x for k in a for x in a[k][0]})
    spaces = [s for s in order if s in a]
    if not spaces or not Ns:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    w = 0.8 / max(len(spaces), 1)
    for i, s in enumerate(spaces):
        xs, med, lo, hi = a[s]
        d = dict(zip(xs, med))
        vals = [d.get(N, 0) for N in Ns]
        positions = [j + (i - (len(spaces) - 1) / 2) * w for j in range(len(Ns))]
        ax.bar(positions, vals, width=w * 0.92, color=CATEGORICAL[i],
               label=pretty.get(s, s), zorder=3)
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("per-step latency, median (ms)")
    ax.set_title("Action-space controller cost (UR5e reach)")
    ax.legend(frameon=False, title="action space")
    _save(fig, out_dir, "action_space_cost")


def fig_dof_matrix(runs: dict[str, list[dict]], out_dir: Path):
    """Per-step cost by robot/morphology at a fixed N — reveals the tensegrity-
    wrist and tendon overhead as a delta over the plain arm."""
    rows = runs.get("robot_dof_matrix")
    if not rows:
        return
    # choose the largest N present so the sim cost dominates controller noise
    Ns = sorted({int(r["num_envs"]) for r in rows if valid(r)})
    if not Ns:
        return
    N = Ns[len(Ns) // 2] if len(Ns) > 1 else Ns[0]
    sub = [r for r in rows if int(r["num_envs"]) == N]
    a = agg(sub, lambda r: r.get("config", {}).get("robot", "?"), "step_ms_p50")
    labels = sorted(a, key=lambda k: a[k][1][0])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    vals = [a[k][1][0] for k in labels]
    los = [a[k][2][0] for k in labels]
    his = [a[k][3][0] for k in labels]
    err = [[v - l for v, l in zip(vals, los)], [h - v for h, v in zip(his, vals)]]
    # colour: rigid arms blue, +wrist vermillion, tensegrity green
    def col(k):
        if "tensegrity" in k:
            return OKABE["green"]
        if "wrist" in k:
            return OKABE["vermillion"]
        return OKABE["blue"]
    # ascending sort, NOT inverted → cheapest (short bars) at the bottom, so the
    # bottom-right legend sits in empty space and never overlaps a bar.
    ax.barh(range(len(labels)), vals, xerr=err, color=[col(k) for k in labels],
            zorder=3, error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("per-step latency, median (ms)")
    ax.set_title(f"Robot morphology / DOF cost (reach, N={N})")
    ax.margins(x=0.16)  # headroom so value labels don't clip
    for i, (v, h) in enumerate(zip(vals, his)):
        ax.text(h + max(vals) * 0.015, i, f"{v:.1f}", va="center", fontsize=7,
                color="#333333")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=OKABE["blue"], label="rigid arm"),
                       Patch(color=OKABE["vermillion"], label="+ tensegrity wrist"),
                       Patch(color=OKABE["green"], label="pure tensegrity")],
              frameon=False, loc="lower right")
    _save(fig, out_dir, "robot_dof_cost")


def fig_optimization_levers(runs: dict[str, list[dict]], out_dir: Path):
    rows = runs.get("optimization_levers")
    if not rows:
        return
    # throughput vs solver iterations, one line per decimation
    def dec(r):
        return r.get("config", {}).get("decimation", r.get("applied_overrides", {}).get("decimation"))
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not valid(r):
            continue
        it = r.get("config", {}).get("cloth_solver_pos_iters")
        if it is None:
            it = r.get("applied_cloth_overrides", {}).get("pbd_params.solver_position_iterations")
        if it is None or r.get("env_steps_per_s") is None:
            continue
        buckets[dec(r)][int(it)].append(float(r["env_steps_per_s"]))
    if not buckets:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, d in enumerate(sorted(buckets)):
        its = sorted(buckets[d])
        med = [statistics.median(buckets[d][it]) for it in its]
        ax.plot(its, med, "-o", color=CATEGORICAL[i], label=f"decimation={d}")
    ax.set_xlabel("PBD solver position iterations")
    ax.set_ylabel("throughput (env·steps / s)")
    ax.set_title("Optimisation lever: solver iterations × decimation")
    ax.legend(frameon=False)
    _save(fig, out_dir, "optimization_levers")


# ---------------------------------------------------------------------------
# Cross-GPU overlay (§4.7): A6000 vs Alex on one axis, WITHOUT touching the
# single-machine §4.x figures. Run dir names are "{machine}_{date}_{time}".
# ---------------------------------------------------------------------------
MACHINE_LABEL = {"pve-robotik-humble": "RTX A6000", "alex": "RTX PRO 6000"}


def _machine_of(run_dir: Path) -> str:
    return run_dir.name.rsplit("_", 2)[0]


def discover_by_machine(spec: str) -> dict[str, Path]:
    """{machine: newest run_dir} for one spec (newest wins per machine)."""
    out: dict[str, Path] = {}
    sd = DATA_ROOT / spec
    if not sd.is_dir():
        return out
    runs = sorted([d for d in sd.iterdir()
                   if (d / "results.jsonl").exists() or (d / "points").exists()])
    for d in runs:
        out[_machine_of(d)] = d
    return out


def fig_crossgpu_throughput(out_dir: Path,
                            machines=("pve-robotik-humble", "alex")) -> None:
    """Rigid + cloth throughput vs N for both GPUs on one log-log axis.
    Colour encodes the task (rigid/cloth); line style / marker encodes the GPU."""
    cloth_spec = next((c for c in CLOTH_SPEC_CANDIDATES if (DATA_ROOT / c).is_dir()), None)
    specs = [(RIGID_SPEC, "Rigid", OKABE["blue"]), (cloth_spec, "Cloth", OKABE["vermillion"])]
    style = {"pve-robotik-humble": dict(marker="o", linestyle="--"),
             "alex": dict(marker="s", linestyle="-")}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xs_all: list[int] = []
    for spec, task_label, c in specs:
        if not spec:
            continue
        bym = discover_by_machine(spec)
        for m in machines:
            if m not in bym:
                continue
            a = agg(load_run(bym[m]), lambda r: "s", "env_steps_per_s")
            if "s" not in a:
                continue
            xs, med, lo, hi = a["s"]
            xs_all += xs
            st = style.get(m, dict(marker="o", linestyle="-"))
            ax.plot(xs, med, color=c, zorder=3,
                    label=f"{task_label} · {MACHINE_LABEL.get(m, m)}", **st)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.10, linewidth=0)
    if not xs_all:
        print("  crossgpu: no data found for either machine")
        plt.close(fig)
        return
    ax.set_yscale("log")
    _logx(ax, xs_all)
    ax.set_xlabel("parallel environments $N$")
    ax.set_ylabel("throughput  (env·steps / s)")
    ax.set_title("Cross-GPU throughput: RTX A6000 vs. RTX PRO 6000 (Blackwell)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _save(fig, out_dir, "crossgpu_throughput")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render thesis figures from benchmark results.")
    ap.add_argument("--auto", action="store_true", help="Discover latest run per spec under DATA_ROOT.")
    ap.add_argument("--crossgpu", action="store_true",
                    help="Render ONLY the cross-GPU overlay (A6000 vs Alex); leaves §4.x figures untouched.")
    ap.add_argument("--runs", nargs="*", default=[], help="Explicit run dirs (glob ok).")
    ap.add_argument("--out", default=str(REPO_ROOT / "doc/reports/figures/benchmarks"))
    args = ap.parse_args()

    if args.crossgpu:
        set_thesis_style()
        out_dir = Path(args.out)
        print(f"Rendering cross-GPU overlay → {out_dir}")
        fig_crossgpu_throughput(out_dir)
        print("done.")
        return

    set_thesis_style()
    runs: dict[str, list[dict]] = {}
    if args.auto:
        for spec, d in discover_latest().items():
            runs[spec] = load_run(d)
            print(f"loaded {spec}: {len(runs[spec])} rows from {d.name}")
    for pat in args.runs:
        for d in glob.glob(pat):
            d = Path(d)
            spec = d.parent.name
            runs.setdefault(spec, []).extend(load_run(d))
            print(f"loaded {spec}: +{len(load_run(d))} rows from {d}")

    if not runs:
        print("No results found. Run a spec first (run_matrix.py --dispatch local).")
        return

    out_dir = Path(args.out)
    print(f"\nRendering figures → {out_dir}")
    fig_throughput_knee(runs, out_dir)
    fig_perenv_and_mem(runs, out_dir)
    fig_energy(runs, out_dir)
    fig_action_space(runs, out_dir)
    fig_dof_matrix(runs, out_dir)
    fig_optimization_levers(runs, out_dir)
    print("done.")


if __name__ == "__main__":
    main()

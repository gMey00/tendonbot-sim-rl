"""Standalone plotting and analysis for tensegrity model validation results.

Reads NPZ data produced by ``run_step_response_pd.py`` and
``run_step_response_tendon.py`` and generates the following plots:

Per joint (PD model):
  - step_response_pd_{joint}.png     — angle time series with setpoint overlay
  - metrics_table_{joint}.png        — combined PD / Tendon / Klein metrics table

Per joint (Tendon model):
  - step_response_tendon_{joint}.png — angle time series + tendon tension profiles

Combined:
  - nrmse_comparison.png             — 3-panel bar chart: Klein | PD | Tendon
  - damping_sweep_elbow.png          — settling time vs. D (from gain-sweep data)
  - settling_time_comparison.png     — grouped bar chart: PD vs Tendon settling

No IsaacLab imports — can be run without a simulator.

Usage
-----
.. code-block:: bash

    python scripts/model_validation/plot_validation.py [options]

    # Options:
    #   --pd_dir PATH       override PD data directory
    #   --tendon_dir PATH    override tendon data directory
    #   --output_dir PATH    override plot output directory
    #   --no_pd             skip PD plots
    #   --no_tendon         skip tendon plots
    #   --no_comparison     skip cross-model comparison plots
    #   --no_sweep          skip damping-sweep plot
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common


# ── Style constants ────────────────────────────────────────────────────────────
THESIS_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue / orange / green
KLEIN_COLOR    = "#ff7f0e"
PD_COLOR       = "#1f77b4"
TENDON_COLOR   = "#2ca02c"
SWEEP_CMAP     = "viridis"

TENDON_LABELS  = [
    "T0 (elbow +)", "T1 (elbow −)",
    "T2 (wrist 0°)", "T3 (wrist 120°)", "T4 (wrist 240°)",
]


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _finalise(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Tighten layout, save, close, and print confirmation."""
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  {path.name}")


def _group_by_joint(trials: list[dict]) -> dict[str, list[dict]]:
    """Group trial dicts by joint name, sorted by amplitude."""
    out: dict[str, list[dict]] = {}
    for t in trials:
        out.setdefault(t["joint_name"], []).append(t)
    for v in out.values():
        v.sort(key=lambda t: t["amplitude_deg"])
    return out


# ── Step response plots (per joint, per model) ────────────────────────────────

def plot_step_responses(
    trials: list[dict],
    model_label: str,
    output_path: Path,
    show_tensions: bool = True,
) -> None:
    """Angle time series (+ optional tendon tensions) for one joint.

    Top panel:   actual angle vs. time, one colour per step amplitude.
                 Setpoint shown as dashed line in matching colour.
                 Metric annotations (NRMSE, rise time, settling time) inset.
    Bottom panel (if has_tensions): all 5 tendon tensions, one line per
                 tendon per amplitude.
    """
    if not trials:
        return

    joint_name = trials[0]["joint_name"]
    has_tensions = show_tensions and np.any(np.array([t["tensions_n"] for t in trials]) != 0)

    nrows = 2 if has_tensions else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 7 if has_tensions else 4.5), sharex=False)
    if nrows == 1:
        axes = [axes]
    fig.suptitle(f"Step Response — {joint_name.replace('_', ' ').title()}  ({model_label})",
                 fontsize=13)

    ax_angle = axes[0]
    for idx, trial in enumerate(trials):
        color = THESIS_COLORS[idx % len(THESIS_COLORS)]
        amp   = trial["amplitude_deg"]
        t     = trial["time_s"]
        y     = trial["actual_deg"]
        sp    = trial["setpoint_deg"]
        nrmse = trial["nrmse_pct"]
        rise  = trial["rise_time_ms"]
        settle = trial["settling_time_ms"]

        ax_angle.plot(t, y,  color=color, linewidth=1.5, label=f"{amp:.0f}° actual")
        ax_angle.plot(t, sp, color=color, linestyle="--", alpha=0.55, linewidth=1.2,
                      label=f"{amp:.0f}° setpoint")

        # Annotate with key metrics
        rise_str   = f"{rise:.0f} ms"   if rise   is not None else "—"
        settle_str = f"{settle:.0f} ms" if settle is not None else "—"
        ax_angle.annotate(
            f"NRMSE={nrmse:.1f}%\nrise={rise_str}\nsettle={settle_str}",
            xy=(common.STEP_HOLD_S * 0.55 + idx * 0.12, amp * (0.72 - 0.05 * idx)),
            fontsize=7, color=color, alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.5, linewidth=0),
        )

    # Klein reference lines (NRMSE targets)
    klein_vals = common.KLEIN_NRMSE.get(joint_name, [])
    ax_angle.set_ylabel("Angle [°]")
    ax_angle.set_xlabel("Time [s]")
    ax_angle.legend(fontsize=8, ncol=2, loc="lower right")
    ax_angle.grid(True, alpha=0.3)
    ax_angle.set_xlim(0, common.STEP_HOLD_S + common.RETURN_HOLD_S)
    ax_angle.axvline(common.STEP_HOLD_S, color="gray", linestyle=":", alpha=0.5,
                     label="return")

    if has_tensions:
        ax_t = axes[1]
        for idx, trial in enumerate(trials):
            amp  = trial["amplitude_deg"]
            t    = trial["time_s"]
            tens = trial["tensions_n"]  # (T, 5)
            for ti in range(5):
                alpha = 0.30 + 0.15 * idx
                lbl   = TENDON_LABELS[ti] if idx == 0 else None
                ax_t.plot(t, tens[:, ti], alpha=alpha, linewidth=0.9, label=lbl)

        ax_t.set_ylabel("Tension [N]")
        ax_t.set_xlabel("Time [s]")
        ax_t.legend(fontsize=6.5, ncol=3, loc="upper right")
        ax_t.grid(True, alpha=0.3)
        ax_t.set_xlim(0, common.STEP_HOLD_S + common.RETURN_HOLD_S)
        ax_t.axvline(common.STEP_HOLD_S, color="gray", linestyle=":", alpha=0.5)

    _finalise(fig, output_path)


# ── Combined metrics table ────────────────────────────────────────────────────

def plot_metrics_table(
    trials_pd: list[dict],
    trials_tendon: list[dict],
    joint_name: str,
    output_path: Path,
) -> None:
    """Matplotlib table comparing PD metrics, Tendon metrics, and Klein NRMSE."""
    klein_vals = common.KLEIN_NRMSE.get(joint_name, [])

    # Gather all unique amplitudes
    all_amps = sorted(set(
        [t["amplitude_deg"] for t in trials_pd] +
        [t["amplitude_deg"] for t in trials_tendon]
    ))
    pd_by_amp  = {t["amplitude_deg"]: t for t in trials_pd}
    ten_by_amp = {t["amplitude_deg"]: t for t in trials_tendon}

    rows = []
    for i, amp in enumerate(all_amps):
        t_pd  = pd_by_amp.get(amp)
        t_ten = ten_by_amp.get(amp)
        k_nrmse = f"{klein_vals[i]:.1f}%" if i < len(klein_vals) else "—"

        def _fmt(t: Optional[dict]) -> tuple[str, str, str, str, str]:
            if t is None:
                return "—", "—", "—", "—", "—"
            rise   = f"{t['rise_time_ms']:.0f}"    if t["rise_time_ms"]    is not None else "—"
            over   = f"{t['overshoot_pct']:.1f}%"  if t["overshoot_pct"]  is not None else "—"
            settle = f"{t['settling_time_ms']:.0f}" if t["settling_time_ms"] is not None else "—"
            nrmse  = f"{t['nrmse_pct']:.1f}%"
            ss     = f"{t['steady_state_error_deg']:.2f}°"
            return rise, over, settle, nrmse, ss

        pd_r, pd_o, pd_s, pd_n, pd_ss   = _fmt(t_pd)
        te_r, te_o, te_s, te_n, te_ss   = _fmt(t_ten)

        rows.append([
            f"{amp:.0f}°",
            pd_r, pd_o, pd_s, pd_n,
            te_r, te_o, te_s, te_n,
            k_nrmse,
        ])

    col_labels = [
        "Step",
        "PD Rise\n[ms]", "PD Over", "PD Settle\n[ms]", "PD NRMSE",
        "Ten Rise\n[ms]", "Ten Over", "Ten Settle\n[ms]", "Ten NRMSE",
        "Klein\nNRMSE",
    ]

    fig, ax = plt.subplots(figsize=(15, max(2.5, 1.1 + 0.7 * len(rows))))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    # Colour-code header cells
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#dce6f1")
        table[0, j].set_text_props(weight="bold")

    fig.suptitle(
        f"Step-Response Metrics — {joint_name.replace('_', ' ').title()}"
        f"  (cf. Klein 2023 Table 4.2)",
        fontsize=11,
    )
    _finalise(fig, output_path)


# ── NRMSE comparison bar chart ────────────────────────────────────────────────

def plot_nrmse_comparison(
    pd_data: dict[str, list[dict]],
    tendon_data: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """3-panel bar chart: NRMSE for Klein / PD / Tendon per joint.

    Each panel corresponds to one arm joint.
    Groups on the x-axis represent step amplitudes; bars within each group
    show the three model/reference values.
    """
    joint_order = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("NRMSE Comparison: Klein (2023) Gazebo vs. Isaac Sim Models", fontsize=13)

    for ax, joint_name in zip(axes, joint_order):
        klein_vals  = common.KLEIN_NRMSE.get(joint_name, [])
        pd_trials   = pd_data.get(joint_name, [])
        ten_trials  = tendon_data.get(joint_name, [])

        pd_nrmse  = [t["nrmse_pct"] for t in pd_trials]
        ten_nrmse = [t["nrmse_pct"] for t in ten_trials]
        amps_deg  = [t["amplitude_deg"] for t in (pd_trials or ten_trials or [])]
        n = max(len(amps_deg), len(klein_vals))

        if n == 0:
            ax.set_title(joint_name.replace("_", " ").title() + "\n(no data)")
            continue

        x = np.arange(n)
        width = 0.26

        if klein_vals:
            ax.bar(x - width, klein_vals[:n], width,
                   label="Klein 2023 (Gazebo)", color=KLEIN_COLOR, alpha=0.85)
        if pd_nrmse:
            ax.bar(x,          pd_nrmse[:n],   width,
                   label="PD  (Isaac, D=20)",   color=PD_COLOR,    alpha=0.85)
        if ten_nrmse:
            ax.bar(x + width, ten_nrmse[:n],   width,
                   label="Tendon (Isaac)",       color=TENDON_COLOR, alpha=0.85)

        ax.set_xlabel("Step Amplitude")
        ax.set_ylabel("NRMSE [%]")
        ax.set_title(joint_name.replace("_", " ").title())
        ax.set_xticks(x[:len(amps_deg)])
        ax.set_xticklabels([f"{a:.0f}°" for a in amps_deg[:n]])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    _finalise(fig, output_path)


# ── Settling time comparison ─────────────────────────────────────────────────

def plot_settling_comparison(
    pd_data: dict[str, list[dict]],
    tendon_data: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Grouped bar chart: PD vs. Tendon settling time per joint per amplitude."""
    joint_order = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Settling Time Comparison: PD vs. Tendon", fontsize=13)

    for ax, joint_name in zip(axes, joint_order):
        pd_trials  = pd_data.get(joint_name, [])
        ten_trials = tendon_data.get(joint_name, [])

        all_amps = sorted(set(
            [t["amplitude_deg"] for t in pd_trials] +
            [t["amplitude_deg"] for t in ten_trials]
        ))
        if not all_amps:
            ax.set_title(joint_name.replace("_", " ").title() + "\n(no data)")
            continue

        pd_by_amp  = {t["amplitude_deg"]: t for t in pd_trials}
        ten_by_amp = {t["amplitude_deg"]: t for t in ten_trials}

        x = np.arange(len(all_amps))
        width = 0.35

        pd_settle  = [pd_by_amp.get(a, {}).get("settling_time_ms") or np.nan for a in all_amps]
        ten_settle = [ten_by_amp.get(a, {}).get("settling_time_ms") or np.nan for a in all_amps]

        ax.bar(x - width / 2, pd_settle,  width, label="PD (D=20)", color=PD_COLOR,    alpha=0.85)
        ax.bar(x + width / 2, ten_settle, width, label="Tendon",    color=TENDON_COLOR, alpha=0.85)

        # Klein target band
        ax.axhline(common.TARGET_SETTLING_S[0] * 1000.0, color="green", linestyle=":",  alpha=0.6, linewidth=1.2)
        ax.axhline(common.TARGET_SETTLING_S[1] * 1000.0, color="green", linestyle="--", alpha=0.8, linewidth=1.5,
                   label=f"Klein target ≤ {common.TARGET_SETTLING_S[1]*1000:.0f} ms")

        ax.set_xlabel("Step Amplitude")
        ax.set_ylabel("Settling Time [ms]")
        ax.set_title(joint_name.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.0f}°" for a in all_amps])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    _finalise(fig, output_path)


# ── Damping sweep ─────────────────────────────────────────────────────────────

def plot_damping_sweep(
    sweep_by_d: dict[float, list[dict]],
    output_path: Path,
) -> None:
    """Two-panel figure for elbow damping sweep.

    Top row (one panel per amplitude):
        Overlaid step responses for all damping candidates.
        The validated D=20 line is drawn thicker and highlighted.

    Bottom row (one panel per amplitude):
        Bar chart of settling time vs. D.
        Target band annotated.  Best candidate highlighted in red.
    """
    if not sweep_by_d:
        return

    d_vals    = sorted(sweep_by_d.keys())
    first_d   = d_vals[0]
    amps      = sorted({t["amplitude_deg"] for t in sweep_by_d[first_d]})
    n_amps    = len(amps)

    if n_amps == 0:
        return

    colors_d = plt.cm.get_cmap(SWEEP_CMAP)(np.linspace(0.15, 0.90, len(d_vals)))

    fig, axes = plt.subplots(2, n_amps, figsize=(5.5 * n_amps, 10))
    if n_amps == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        "Arm Damping Sweep — Elbow Joint  (K = 400 N·m/rad fixed)\n"
        "Confirming D = 20 as near-critical optimum",
        fontsize=13,
    )

    # Precompute best D per amplitude
    best_d_per_amp: dict[float, float] = {}
    for amp in amps:
        best_d_per_amp[amp] = min(
            d_vals,
            key=lambda d: next(
                (t["settling_time_ms"] or 9999.0
                 for t in sweep_by_d[d] if abs(t["amplitude_deg"] - amp) < 0.5),
                9999.0,
            ),
        )

    for col, amp in enumerate(amps):
        # ── Top: overlaid step responses ─────────────────────────────
        ax_top = axes[0, col]
        ax_top.set_title(f"Step {amp:.0f}°", fontsize=11)

        for d_idx, d_val in enumerate(d_vals):
            trial = next(
                (t for t in sweep_by_d[d_val] if abs(t["amplitude_deg"] - amp) < 0.5),
                None,
            )
            if trial is None:
                continue
            is_validated = abs(d_val - common.VALIDATED_DAMPING) < 0.5
            lw    = 2.5  if is_validated else 1.0
            zorder = 3   if is_validated else 2
            ax_top.plot(
                trial["time_s"], trial["actual_deg"],
                color=colors_d[d_idx], linewidth=lw, zorder=zorder,
                label=f"D={d_val:.0f}" + (" ◄" if is_validated else ""),
            )

        ax_top.axhline(amp, color="k", linestyle=":", alpha=0.4, linewidth=1.0)
        ax_top.axhline(amp * 1.02, color="g", linestyle="--", alpha=0.3, linewidth=0.8)
        ax_top.axhline(amp * 0.98, color="g", linestyle="--", alpha=0.3, linewidth=0.8)
        ax_top.set_xlim(0, common.STEP_HOLD_S)
        ax_top.set_xlabel("Time [s]")
        ax_top.set_ylabel("Angle [°]")
        ax_top.legend(fontsize=7, loc="lower right")
        ax_top.grid(True, alpha=0.3)

        # ── Bottom: settling time bar chart ───────────────────────────
        ax_bot = axes[1, col]

        settle_vals = []
        for d_val in d_vals:
            trial = next(
                (t for t in sweep_by_d[d_val] if abs(t["amplitude_deg"] - amp) < 0.5),
                None,
            )
            settle_vals.append(trial["settling_time_ms"] if trial and trial["settling_time_ms"] is not None else np.nan)

        bar_colors = [
            "#e74c3c" if abs(d - common.VALIDATED_DAMPING) < 0.5 else "#3498db"
            for d in d_vals
        ]
        x_pos = np.arange(len(d_vals))
        bars = ax_bot.bar(x_pos, settle_vals, color=bar_colors, alpha=0.85, zorder=2)

        # Target band
        ax_bot.axhline(
            common.TARGET_SETTLING_S[1] * 1000.0,
            color="green", linestyle="--", linewidth=1.5, alpha=0.8, zorder=3,
            label=f"Target ≤ {common.TARGET_SETTLING_S[1]*1000:.0f} ms",
        )

        # Annotate best value
        best_d = best_d_per_amp[amp]
        best_idx = d_vals.index(best_d)
        best_settle = settle_vals[best_idx]
        if not math.isnan(best_settle):
            ax_bot.annotate(
                f"Best\n{best_settle:.0f} ms",
                xy=(best_idx, best_settle),
                xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=7.5, color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
            )

        ax_bot.set_xticks(x_pos)
        ax_bot.set_xticklabels([f"D={d:.0f}" for d in d_vals], fontsize=8)
        ax_bot.set_xlabel("Damping D [N·m·s/rad]")
        ax_bot.set_ylabel("Settling Time [ms]")
        ax_bot.set_title(f"Settling Time @ {amp:.0f}°", fontsize=11)
        ax_bot.legend(fontsize=8)
        ax_bot.grid(axis="y", alpha=0.3)

    _finalise(fig, output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tensegrity model validation results")
    parser.add_argument("--pd_dir",      type=str, default=str(common.PD_DATA_DIR))
    parser.add_argument("--tendon_dir",  type=str, default=str(common.TENDON_DATA_DIR))
    parser.add_argument("--output_dir",  type=str, default=str(common.PLOTS_DIR))
    parser.add_argument("--no_pd",       action="store_true", help="Skip PD plots")
    parser.add_argument("--no_tendon",   action="store_true", help="Skip tendon plots")
    parser.add_argument("--no_comparison", action="store_true", help="Skip comparison plots")
    parser.add_argument("--no_sweep",    action="store_true", help="Skip damping-sweep plot")
    args = parser.parse_args()

    pd_dir     = Path(args.pd_dir)
    tendon_dir = Path(args.tendon_dir)
    plots_dir  = Path(args.output_dir)

    # ── Load data ─────────────────────────────────────────────────────────
    all_pd: list[dict] = []
    all_tendon: list[dict] = []

    if not args.no_pd and pd_dir.exists():
        all_pd = common.load_trials(pd_dir, no_sweep=True)
    if not args.no_tendon and tendon_dir.exists():
        all_tendon = common.load_trials(tendon_dir, no_sweep=True)

    pd_by_joint     = _group_by_joint(all_pd)
    tendon_by_joint = _group_by_joint(all_tendon)

    n_pd     = sum(len(v) for v in pd_by_joint.values())
    n_tendon = sum(len(v) for v in tendon_by_joint.values())
    print(f"Loaded: {n_pd} PD trials,  {n_tendon} tendon trials")
    print(f"Output: {plots_dir}\n")

    if n_pd == 0 and n_tendon == 0:
        print("No data found — run the data-generation scripts first.")
        return

    generated: list[str] = []

    # ── Per-joint step response plots ─────────────────────────────────────
    for joint_name in common.ARM_JOINT_NAMES:
        if not args.no_pd and joint_name in pd_by_joint:
            fname = f"step_response_pd_{joint_name}.png"
            plot_step_responses(
                pd_by_joint[joint_name],
                model_label="PD  (K=400, D=20  ImplicitActuator)",
                output_path=plots_dir / fname,
                show_tensions=True,
            )
            generated.append(fname)

        if not args.no_tendon and joint_name in tendon_by_joint:
            fname = f"step_response_tendon_{joint_name}.png"
            plot_step_responses(
                tendon_by_joint[joint_name],
                model_label="Tendon  (PID + Jacobian transpose)",
                output_path=plots_dir / fname,
                show_tensions=True,
            )
            generated.append(fname)

        # Combined metrics table (PD + Tendon + Klein reference)
        if joint_name in pd_by_joint or joint_name in tendon_by_joint:
            fname = f"metrics_table_{joint_name}.png"
            plot_metrics_table(
                pd_by_joint.get(joint_name, []),
                tendon_by_joint.get(joint_name, []),
                joint_name,
                plots_dir / fname,
            )
            generated.append(fname)

    # ── Cross-model comparison plots ──────────────────────────────────────
    if not args.no_comparison and (pd_by_joint or tendon_by_joint):
        fname = "nrmse_comparison.png"
        plot_nrmse_comparison(pd_by_joint, tendon_by_joint, plots_dir / fname)
        generated.append(fname)

        if pd_by_joint and tendon_by_joint:
            fname = "settling_time_comparison.png"
            plot_settling_comparison(pd_by_joint, tendon_by_joint, plots_dir / fname)
            generated.append(fname)

    # ── Damping sweep ─────────────────────────────────────────────────────
    if not args.no_sweep and pd_dir.exists():
        sweep_trials = common.load_trials(pd_dir, joint_name="elbow_joint", sweep_only=True)
        if sweep_trials:
            sweep_by_d: dict[float, list[dict]] = {}
            for t in sweep_trials:
                d = t["damping_value"]
                if d is not None:
                    sweep_by_d.setdefault(d, []).append(t)
            if sweep_by_d:
                fname = "damping_sweep_elbow.png"
                plot_damping_sweep(sweep_by_d, plots_dir / fname)
                generated.append(fname)
        else:
            print("  (No gain-sweep data found — re-run with --damping_sweep flag)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{len(generated)} plots written to {plots_dir}/")

    # Print NRMSE comparison table to console
    if (pd_by_joint or tendon_by_joint):
        print()
        print("  NRMSE Summary (% — lower is closer to real robot)")
        print(f"  {'Joint':<22}  {'Amp':>5}  {'Klein':>7}  {'PD':>7}  {'Tendon':>8}")
        print("  " + "─" * 56)
        for joint_name in common.ARM_JOINT_NAMES:
            klein_vals = common.KLEIN_NRMSE.get(joint_name, [])
            pd_trials  = pd_by_joint.get(joint_name, [])
            ten_trials = tendon_by_joint.get(joint_name, [])
            pd_by_amp  = {t["amplitude_deg"]: t for t in pd_trials}
            ten_by_amp = {t["amplitude_deg"]: t for t in ten_trials}
            all_amps   = sorted(set(
                [t["amplitude_deg"] for t in pd_trials] +
                [t["amplitude_deg"] for t in ten_trials]
            ))
            for i, amp in enumerate(all_amps):
                k   = f"{klein_vals[i]:.1f}%" if i < len(klein_vals) else "—"
                pd  = f"{pd_by_amp[amp]['nrmse_pct']:.1f}%"  if amp in pd_by_amp  else "—"
                ten = f"{ten_by_amp[amp]['nrmse_pct']:.1f}%" if amp in ten_by_amp else "—"
                print(f"  {joint_name:<22}  {amp:>4.0f}°  {k:>7}  {pd:>7}  {ten:>8}")


if __name__ == "__main__":
    main()

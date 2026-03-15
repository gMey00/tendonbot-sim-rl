"""Shared utilities for tensegrity model validation scripts.

No IsaacLab dependencies — safe to import in standalone plotting scripts.
Contains all constants, data-class definitions, control math, metric
computation, and NPZ serialisation helpers used by the data-generation
and plotting scripts.

References
----------
* Klein, M. (2023). *Arbeitsraumanalyse, Simulation und Bewegungsplanung
  eines seilgetriebenen robotischen Manipulators*. Master's thesis, FAU.
  §3.2.3 (PID), §3.2.4 (tension distribution), §3.5 (test methodology),
  §4.2–4.3 (step-response results and NRMSE).
* Mukherjee et al. — gravity compensation (thesis [85], Eq. 3.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ── Output paths ──────────────────────────────────────────────────────────────
_PROJ_ROOT = Path("/home/robot/studentische-arbeiten/src/tensegrity_pick")
DEFAULT_OUTPUT_ROOT = _PROJ_ROOT / "outputs" / "model_validation"
PD_DATA_DIR = DEFAULT_OUTPUT_ROOT / "pd" / "data"
TENDON_DATA_DIR = DEFAULT_OUTPUT_ROOT / "tendon" / "data"
PLOTS_DIR = DEFAULT_OUTPUT_ROOT / "plots"

# ── Joint identity ─────────────────────────────────────────────────────────────
ARM_JOINT_NAMES: list[str] = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]
JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(ARM_JOINT_NAMES)}

# ── Klein (2023) NRMSE reference — Gazebo simulation vs. real robot ───────────
# Source: Klein (2023) Table 4.2 — Simulation accuracy (Gazebo vs. real)
KLEIN_NRMSE: dict[str, list[float]] = {
    "wrist_x_joint": [25.5, 9.9, 11.2],   # 10°, 20°, 30°
    "wrist_y_joint": [19.2, 11.6, 12.5],   # 10°, 20°, 30°
    "elbow_joint":   [39.1, 11.7, 12.1],   # 20°, 30°, 40°
}

# ── Klein (2023) performance targets (§4.1 / tune_pd_gains.py reference) ──────
TARGET_SETTLING_S = (0.10, 0.30)      # near-critical settling window
TARGET_OVERSHOOT_MAX_PCT = 5.0        # max acceptable overshoot
TARGET_ZETA_RANGE = (1.0, 1.5)       # near-critical damping ratio

# ── Jacobian transpose (3 joints × 5 tendons) ─────────────────────────────────
# Row order matches ARM_JOINT_NAMES: [elbow, wrist_y, wrist_x]
# Derived from xacro Force1–5 attachment coordinates (threedof_manipulator.urdf.xacro).
JACOBIAN_T = np.array([
    [+0.0725, -0.0725,  0.0,       0.0,      0.0],       # elbow
    [ 0.0,     0.0,    -0.013856,  0.0,     +0.013856],   # wrist_y
    [ 0.0,     0.0,    +0.008,    -0.016,   +0.008],      # wrist_x
], dtype=np.float64)

_JACOBIAN_T_PINV: Optional[np.ndarray] = None


def get_jacobian_pinv() -> np.ndarray:
    """Moore–Penrose pseudo-inverse of J^T (cached singleton)."""
    global _JACOBIAN_T_PINV
    if _JACOBIAN_T_PINV is None:
        _JACOBIAN_T_PINV = np.linalg.pinv(JACOBIAN_T)
    return _JACOBIAN_T_PINV


# ── PD gain sweep candidates (from tune_pd_gains.py / pd_tuning_results.md) ───
GAIN_SWEEP_DAMPING_VALUES: list[float] = [120.0, 60.0, 30.0, 25.0, 20.0, 15.0]
GAIN_SWEEP_K_FIXED: float = 400.0
VALIDATED_DAMPING: float = 20.0      # confirmed optimal by tune_pd_gains.py

# ── Simulation timing ──────────────────────────────────────────────────────────
SIM_DT: float = 1.0 / 120.0   # 120 Hz physics (matches step_response_test.py)
RENDER_INTERVAL: int = 2
STEP_HOLD_S: float = 2.5       # duration of step-up phase
RETURN_HOLD_S: float = 1.0     # duration of return-to-zero phase
WARMUP_S: float = 0.5          # pre-trial settle time


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float


@dataclass(frozen=True)
class JointTestSpec:
    """Per-joint test specification from Klein (2023) §3.5."""
    name: str
    joint_index: int             # index in ARM_JOINT_NAMES
    pid: PIDGains
    step_amplitudes_deg: tuple[float, ...]
    min_tension_n: float
    saturation_n: float
    gravity_arm_m: float         # lc — lever arm for gravity compensation
    mass_kg: float               # me — moving link mass


ELBOW_SPEC = JointTestSpec(
    name="elbow_joint",
    joint_index=0,
    pid=PIDGains(kp=0.3, ki=0.03, kd=0.02),
    step_amplitudes_deg=(20.0, 30.0, 40.0),
    min_tension_n=6.0,
    saturation_n=160.0,
    gravity_arm_m=0.15,   # approximate forearm CoM distance
    mass_kg=0.80,
)

WRIST_Y_SPEC = JointTestSpec(
    name="wrist_y_joint",
    joint_index=1,
    pid=PIDGains(kp=0.2, ki=0.03, kd=0.03),
    step_amplitudes_deg=(10.0, 20.0, 30.0),
    min_tension_n=5.0,
    saturation_n=80.0,
    gravity_arm_m=0.06,   # half end-effector length (0.12 m)
    mass_kg=0.32,
)

WRIST_X_SPEC = JointTestSpec(
    name="wrist_x_joint",
    joint_index=2,
    pid=PIDGains(kp=0.2, ki=0.03, kd=0.03),
    step_amplitudes_deg=(10.0, 20.0, 30.0),
    min_tension_n=5.0,
    saturation_n=80.0,
    gravity_arm_m=0.06,
    mass_kg=0.32,
)

ALL_JOINT_SPECS: list[JointTestSpec] = [ELBOW_SPEC, WRIST_Y_SPEC, WRIST_X_SPEC]


@dataclass
class PIDState:
    """Mutable PID integrator state (Python floats — no torch dependency)."""
    integral: float = 0.0
    prev_error: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0


@dataclass
class StepMetrics:
    """Scalar quality metrics for one step-response trial."""
    joint_name: str
    amplitude_deg: float
    rise_time_ms: float | None        # 10% → 90% crossing; None if not reached
    overshoot_pct: float | None       # % above setpoint; None if no overshoot
    settling_time_ms: float | None    # time to enter 2% band and stay; None if never
    rmse_deg: float                   # over step-hold window
    nrmse_pct: float                  # RMSE / range × 100 (thesis Eq. 3.11–3.12)
    steady_state_error_deg: float     # mean last 20% vs setpoint

    def passes(self) -> bool:
        """True if settling time ≤ 0.3 s and overshoot ≤ 5 %."""
        settle_ok = (
            self.settling_time_ms is not None
            and self.settling_time_ms <= TARGET_SETTLING_S[1] * 1000.0
        )
        over_ok = self.overshoot_pct is None or self.overshoot_pct <= TARGET_OVERSHOOT_MAX_PCT
        return settle_ok and over_ok


# ── Control math ───────────────────────────────────────────────────────────────

def compute_pid_torque(
    setpoint_rad: float,
    current_rad: float,
    dt: float,
    gains: PIDGains,
    state: PIDState,
    integral_clamp: float = 50.0,
) -> float:
    """Classic discrete PID with clamped integrator (anti-windup)."""
    error = setpoint_rad - current_rad
    state.integral += error * dt
    state.integral = max(-integral_clamp, min(integral_clamp, state.integral))
    deriv = (error - state.prev_error) / dt if dt > 1e-9 else 0.0
    state.prev_error = error
    return gains.kp * error + gains.ki * state.integral + gains.kd * deriv


def gravity_compensation_torque(
    current_rad: float,
    mass_kg: float,
    gravity_arm_m: float,
) -> float:
    """τ_a = m·g·l_c·sin(θ) — Mukherjee et al. (thesis [85], Eq. 3.2)."""
    return mass_kg * 9.81 * gravity_arm_m * math.sin(current_rad)


def torque_to_tensions(
    desired_torque: float,
    joint_index: int,
    min_tension: float,
    saturation: float,
) -> np.ndarray:
    """Map a single-joint torque to 5 tendon tensions via pinv(J^T).

    Builds the full 3-joint torque vector (zeros except at joint_index),
    multiplies by the pseudo-inverse, and clamps to [min_tension, saturation].
    Returns shape (5,).
    """
    full_torque = np.zeros(3, dtype=np.float64)
    full_torque[joint_index] = desired_torque
    tensions = get_jacobian_pinv() @ full_torque   # (5,)
    return np.clip(tensions, min_tension, saturation)


def tensions_to_torques(tensions: np.ndarray) -> np.ndarray:
    """τ = J^T · T.  Input (5,), output (3,)."""
    return JACOBIAN_T @ tensions


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_step_metrics(
    time_s: np.ndarray,
    actual_deg: np.ndarray,
    setpoint_deg: float,
    step_start_s: float,
    step_end_s: float,
    joint_name: str,
    settling_band_pct: float = 2.0,
) -> StepMetrics:
    """Compute step-response quality metrics from a recorded time series.

    All time-based return values are relative to *step_start_s*.
    The RMSE/NRMSE window is [step_start_s, step_end_s].

    Parameters
    ----------
    time_s, actual_deg:
        Full recorded time series (may include return phase).
    setpoint_deg:
        The commanded step amplitude in degrees.
    step_start_s, step_end_s:
        Analysis window (the step-up phase only).
    settling_band_pct:
        Settling band as a percentage of setpoint (default 2 %).
    """
    mask = (time_s >= step_start_s) & (time_s <= step_end_s)
    t = time_s[mask]
    y = actual_deg[mask]

    if len(y) < 2:
        return StepMetrics(joint_name, setpoint_deg, None, None, None, 0.0, 0.0, 0.0)

    # Rise time (10 % → 90 % of setpoint) — thesis §3.5
    rise_ms: float | None = None
    if abs(setpoint_deg) > 1e-6:
        thr10 = 0.1 * setpoint_deg
        thr90 = 0.9 * setpoint_deg
        i10 = np.where(y >= thr10)[0]
        i90 = np.where(y >= thr90)[0]
        if len(i10) > 0 and len(i90) > 0:
            dt_rise = t[i90[0]] - t[i10[0]]
            if dt_rise > 0:
                rise_ms = float(dt_rise * 1000.0)

    # Overshoot (peak exceedance above setpoint)
    os_pct: float | None = None
    if setpoint_deg > 1e-6:
        peak = float(np.max(y))
        if peak > setpoint_deg:
            os_pct = (peak - setpoint_deg) / setpoint_deg * 100.0

    # Settling time (last sample outside the 2 % band, relative to step_start_s)
    settle_ms: float | None = None
    band = settling_band_pct / 100.0 * abs(setpoint_deg)
    if band > 1e-9:
        outside = np.where(np.abs(y - setpoint_deg) > band)[0]
        if len(outside) > 0:
            last_out = outside[-1]
            if last_out + 1 < len(t):
                settle_ms = float((t[last_out + 1] - step_start_s) * 1000.0)
            # else: exited band at the very last sample → not settled within window
        else:
            settle_ms = 0.0   # always within band → already settled at t=0

    # Steady-state error: mean of last 20 % of window vs. setpoint
    tail = max(1, len(y) // 5)
    ss_error = float(abs(np.mean(y[-tail:]) - setpoint_deg))

    # RMSE / NRMSE (thesis Eq. 3.11–3.12)
    ref = np.full_like(y, setpoint_deg)
    rmse = float(np.sqrt(np.mean((y - ref) ** 2)))
    rng = float(np.max(y) - np.min(y))
    nrmse = rmse / rng * 100.0 if rng > 1e-6 else 0.0

    return StepMetrics(
        joint_name=joint_name,
        amplitude_deg=setpoint_deg,
        rise_time_ms=rise_ms,
        overshoot_pct=os_pct,
        settling_time_ms=settle_ms,
        rmse_deg=float(rmse),
        nrmse_pct=float(nrmse),
        steady_state_error_deg=float(ss_error),
    )


# ── NPZ I/O ────────────────────────────────────────────────────────────────────

def trial_path(
    data_dir: Path,
    joint_name: str,
    amplitude_deg: float,
    damping_value: float | None = None,
) -> Path:
    """Return the canonical NPZ path for a given trial.

    Gain-sweep trials include the D value in the filename so they can be
    distinguished from standard trials without opening the file.
    """
    if damping_value is not None:
        name = f"{joint_name}__D{damping_value:.0f}__{amplitude_deg:.0f}deg.npz"
    else:
        name = f"{joint_name}__{amplitude_deg:.0f}deg.npz"
    return data_dir / name


def save_trial(
    path: Path,
    joint_name: str,
    amplitude_deg: float,
    time_s: np.ndarray,
    actual_deg: np.ndarray,
    setpoint_deg_arr: np.ndarray,
    tensions_n: np.ndarray,
    metrics: StepMetrics,
    model_type: str,
    damping_value: float | None = None,
) -> None:
    """Save one (joint, amplitude) trial to a compressed NPZ file.

    NPZ schema
    ----------
    joint_name      : str   — e.g. "elbow_joint"
    amplitude_deg   : f64   — target step amplitude in degrees
    model_type      : str   — "pd" or "tendon"
    sim_dt          : f64   — physics timestep
    step_hold_s     : f64   — duration of step-up phase
    return_hold_s   : f64   — duration of return-to-zero phase
    damping_value   : f64   — arm damping D (NaN for tendon or non-sweep)
    time_s          : f32 (T,)    — starts at 0.0 (step onset)
    actual_deg      : f32 (T,)    — recorded joint angle
    setpoint_deg    : f32 (T,)    — commanded step signal
    tensions_n      : f32 (T, 5) — tendon tensions in N (0 for PD)
    rise_time_ms    : f64   — NaN if not reached
    overshoot_pct   : f64   — NaN if no overshoot
    settling_time_ms: f64   — NaN if not settled within window
    rmse_deg        : f64
    nrmse_pct       : f64
    steady_state_error_deg : f64
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    d: dict[str, np.ndarray] = {
        "joint_name":      np.array(joint_name),
        "amplitude_deg":   np.float64(amplitude_deg),
        "model_type":      np.array(model_type),
        "sim_dt":          np.float64(SIM_DT),
        "step_hold_s":     np.float64(STEP_HOLD_S),
        "return_hold_s":   np.float64(RETURN_HOLD_S),
        "damping_value":   np.float64(damping_value if damping_value is not None else np.nan),
        "time_s":          time_s.astype(np.float32),
        "actual_deg":      actual_deg.astype(np.float32),
        "setpoint_deg":    setpoint_deg_arr.astype(np.float32),
        "tensions_n":      tensions_n.astype(np.float32),
        "rise_time_ms":    np.float64(metrics.rise_time_ms if metrics.rise_time_ms is not None else np.nan),
        "overshoot_pct":   np.float64(metrics.overshoot_pct if metrics.overshoot_pct is not None else np.nan),
        "settling_time_ms": np.float64(metrics.settling_time_ms if metrics.settling_time_ms is not None else np.nan),
        "rmse_deg":        np.float64(metrics.rmse_deg),
        "nrmse_pct":       np.float64(metrics.nrmse_pct),
        "steady_state_error_deg": np.float64(metrics.steady_state_error_deg),
    }
    np.savez_compressed(path, **d)


def load_trial(path: Path) -> dict:
    """Load a trial NPZ file and return a plain Python dict.

    0-d object arrays are converted to strings; 0-d numeric arrays to floats.
    NaN-encoded optionals (rise_time_ms, overshoot_pct, settling_time_ms,
    damping_value) are converted to None.
    """
    npz = np.load(path, allow_pickle=True)
    d = dict(npz)

    # String fields
    d["joint_name"] = str(d["joint_name"])
    d["model_type"] = str(d["model_type"])

    # Scalar float fields
    for key in (
        "amplitude_deg", "sim_dt", "step_hold_s", "return_hold_s",
        "damping_value", "rise_time_ms", "overshoot_pct", "settling_time_ms",
        "rmse_deg", "nrmse_pct", "steady_state_error_deg",
    ):
        d[key] = float(d[key])

    # NaN → None for optional metrics
    for key in ("rise_time_ms", "overshoot_pct", "settling_time_ms", "damping_value"):
        if math.isnan(d[key]):
            d[key] = None

    return d


def load_trials(
    data_dir: Path,
    joint_name: str | None = None,
    sweep_only: bool = False,
    no_sweep: bool = False,
) -> list[dict]:
    """Load all matching *.npz trial files from *data_dir*.

    Parameters
    ----------
    joint_name : if given, only load trials for this joint.
    sweep_only : if True, load only gain-sweep files (filename contains "__D").
    no_sweep   : if True, skip gain-sweep files.

    Returns list sorted by (joint_name, damping_value or 0, amplitude_deg).
    """
    if not data_dir.exists():
        return []

    results = []
    for f in sorted(data_dir.glob("*.npz")):
        is_sweep = "__D" in f.name
        if sweep_only and not is_sweep:
            continue
        if no_sweep and is_sweep:
            continue
        trial = load_trial(f)
        if joint_name is not None and trial["joint_name"] != joint_name:
            continue
        results.append(trial)

    results.sort(key=lambda t: (
        t["joint_name"],
        t["damping_value"] if t["damping_value"] is not None else 0.0,
        t["amplitude_deg"],
    ))
    return results


# ── Log file writing ───────────────────────────────────────────────────────────

def write_log(
    output_dir: Path,
    mode: str,
    robot_cfg_summary: dict[str, str],
    all_metrics: dict[str, list[StepMetrics]],
    gain_sweep_results: dict[float, list[StepMetrics]] | None = None,
) -> None:
    """Write a Markdown log file with parameters, metrics, and key findings.

    Parameters
    ----------
    output_dir : parent directory of the data/ folder (e.g. outputs/model_validation/pd/)
    mode       : "pd" or "tendon"
    robot_cfg_summary : key–value pairs describing the robot configuration
    all_metrics : {joint_name: [StepMetrics, ...]}
    gain_sweep_results : {damping_value: [StepMetrics, ...]} — PD only
    """
    import datetime
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.md"

    def _or(val: float | None, fmt: str = ".1f") -> str:
        return format(val, fmt) if val is not None else "—"

    lines: list[str] = [
        f"# Model Validation Log — {mode.upper()} Mode",
        "",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Isaac Sim version:** 5.1.0  |  **Physics:** PhysX GPU",
        "",
        "## Simulation Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Physics dt | {SIM_DT:.5f} s  ({1/SIM_DT:.0f} Hz) |",
        f"| Render interval | {RENDER_INTERVAL} steps |",
        f"| Step hold | {STEP_HOLD_S} s |",
        f"| Return hold | {RETURN_HOLD_S} s |",
        f"| Warm-up | {WARMUP_S} s |",
        "",
        "## Robot Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in robot_cfg_summary.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Klein (2023) Target Criteria",
        "",
        "| Criterion | Target |",
        "|-----------|--------|",
        f"| Damping ratio ζ | {TARGET_ZETA_RANGE[0]}–{TARGET_ZETA_RANGE[1]} (near-critical) |",
        f"| Settling time | {TARGET_SETTLING_S[0]}–{TARGET_SETTLING_S[1]} s |",
        f"| Overshoot | < {TARGET_OVERSHOOT_MAX_PCT} % |",
        "",
        "## Step Response Metrics",
        "",
        "| Joint | Step [°] | Rise [ms] | Overshoot | Settle [ms] | SS Err [°] | RMSE [°] | NRMSE [%] | Pass |",
        "|-------|---------|-----------|-----------|-------------|-----------|---------|-----------|------|",
    ]
    for joint_name, mlist in all_metrics.items():
        for m in mlist:
            lines.append(
                f"| {joint_name} | {m.amplitude_deg:.0f} | {_or(m.rise_time_ms)} | "
                f"{_or(m.overshoot_pct)}% | {_or(m.settling_time_ms)} | "
                f"{m.steady_state_error_deg:.2f} | {m.rmse_deg:.2f} | "
                f"{m.nrmse_pct:.1f} | {'✓' if m.passes() else '✗'} |"
            )

    if mode == "tendon":
        lines += [
            "",
            "## NRMSE vs. Klein (2023) Gazebo Reference",
            "",
            "| Joint | Step [°] | Klein NRMSE [%] | Isaac Sim NRMSE [%] | Δ |",
            "|-------|---------|----------------|---------------------|---|",
        ]
        for joint_name, mlist in all_metrics.items():
            klein_vals = KLEIN_NRMSE.get(joint_name, [])
            for i, m in enumerate(mlist):
                k_val = klein_vals[i] if i < len(klein_vals) else None
                k_str = f"{k_val:.1f}" if k_val is not None else "—"
                delta = f"{m.nrmse_pct - k_val:+.1f}" if k_val is not None else "—"
                lines.append(
                    f"| {joint_name} | {m.amplitude_deg:.0f} | {k_str} | {m.nrmse_pct:.1f} | {delta} |"
                )

    if gain_sweep_results:
        best_d = min(
            gain_sweep_results,
            key=lambda d: (
                sum(m.settling_time_ms or 9999.0 for m in gain_sweep_results[d])
                / max(1, len(gain_sweep_results[d]))
            ),
        )
        lines += [
            "",
            "## PD Gain Sweep — Elbow Joint",
            "",
            "| D [N·m·s/rad] | Avg Settle [ms] | Best |",
            "|--------------|----------------|------|",
        ]
        for d_val, mlist in sorted(gain_sweep_results.items()):
            valid_settle = [m.settling_time_ms for m in mlist if m.settling_time_ms is not None]
            avg = sum(valid_settle) / len(valid_settle) if valid_settle else float("nan")
            marker = " ◄ BEST" if abs(d_val - best_d) < 0.1 else ""
            lines.append(f"| {d_val:.0f} | {avg:.1f}{marker} | {'✓' if marker else ''} |")

    lines += ["", "## Key Findings", ""]
    for joint_name, mlist in all_metrics.items():
        if not mlist:
            continue
        avg_nrmse = sum(m.nrmse_pct for m in mlist) / len(mlist)
        valid_settle = [m.settling_time_ms for m in mlist if m.settling_time_ms is not None]
        avg_settle = sum(valid_settle) / len(valid_settle) if valid_settle else float("nan")
        n_pass = sum(1 for m in mlist if m.passes())
        settle_str = f"{avg_settle:.0f} ms" if not math.isnan(avg_settle) else "—"
        lines.append(
            f"- **{joint_name}**: avg NRMSE = {avg_nrmse:.1f} %, "
            f"avg settle = {settle_str}, "
            f"{n_pass}/{len(mlist)} trials passing Klein criteria"
        )

    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Log: {log_path}")

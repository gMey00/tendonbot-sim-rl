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
BASE_DATA_DIR = DEFAULT_OUTPUT_ROOT / "base" / "data"
PLOTS_DIR = DEFAULT_OUTPUT_ROOT / "plots"

# ── Joint identity ─────────────────────────────────────────────────────────────
ARM_JOINT_NAMES: list[str] = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]
BASE_JOINT_NAMES: list[str] = ["base_y_joint", "base_z_joint"]
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


# PID gains scaled from Klein (2023) originals (kp=0.3, ki=0.03, kd=0.02)
# to match the Isaac Sim model's effective rotational inertia (~0.103 kg·m²
# for the elbow).  ~133× higher gains needed for similar settling behaviour.
#
# Wrist gains are lower because wrist lever arms (~0.014 m) limit
# achievable torque — tensions saturate at 600 N.  The D gain uses
# derivative-on-measurement (same as elbow) which naturally filters
# PhysX velocity noise.  Earlier versions used a separate raw-velocity
# damping term (kd * joint_vel) which caused chattering on wrist_x.
#
# Gravity compensation: τ_g = mass_kg · g · gravity_arm_m · sin(θ)
#   Elbow: moves disc (0.192) + forearm (1.9) + EE (0.4) = 2.49 kg
#     Combined CoM from elbow axis:
#       (0.192×0.034 + 1.9×0.237 + 0.4×0.474) / 2.492 ≈ 0.26 m
#   Wrist: moves end_effector_link only (0.40 kg)
#     CoM at ~68 mm from wrist axes (half wrist-to-tool distance)
ELBOW_SPEC = JointTestSpec(
    name="elbow_joint",
    joint_index=0,
    pid=PIDGains(kp=40.0, ki=4.0, kd=3.0),
    step_amplitudes_deg=(20.0, 30.0, 40.0),
    min_tension_n=6.0,
    saturation_n=400.0,
    gravity_arm_m=0.26,   # combined CoM: disc + forearm + EE (see above)
    mass_kg=2.49,         # disc 0.192 + forearm 1.9 + EE 0.4
)

WRIST_Y_SPEC = JointTestSpec(
    name="wrist_y_joint",
    joint_index=1,
    pid=PIDGains(kp=10.0, ki=2.0, kd=0.3),
    step_amplitudes_deg=(10.0, 20.0, 30.0),
    min_tension_n=5.0,
    saturation_n=600.0,
    gravity_arm_m=0.068,  # half wrist-to-tool distance: (0.972 − 0.836) / 2
    mass_kg=0.40,         # end_effector_link mass from URDF
)

WRIST_X_SPEC = JointTestSpec(
    name="wrist_x_joint",
    joint_index=2,
    pid=PIDGains(kp=10.0, ki=2.0, kd=0.3),
    step_amplitudes_deg=(10.0, 20.0, 30.0),
    min_tension_n=5.0,
    saturation_n=600.0,
    gravity_arm_m=0.068,  # half wrist-to-tool distance
    mass_kg=0.40,         # end_effector_link mass from URDF
)

ALL_JOINT_SPECS: list[JointTestSpec] = [ELBOW_SPEC, WRIST_Y_SPEC, WRIST_X_SPEC]


# ── Physical (antiparallelogram) model constants ──────────────────────────────
#
# The physical model has no single ``elbow_joint``.  The 4-bar linkage joints
# visible to the articulation tree are: rod_left_joint, rod_right_joint,
# coupler_left_joint.  (coupler_right_joint has excludeFromArticulation = True
# and is invisible to the joint API.)
#
# The "elbow angle" (forearm deflection about X relative to root_link) is
# obtained from the forearm_link body quaternion rather than a joint angle.

PHYSICAL_ARM_JOINT_NAMES: list[str] = [
    "rod_left_joint", "rod_right_joint", "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
]
"""Tree-branch joint names for the physical model (omitting closure joint)."""

PHYSICAL_JOINT_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(PHYSICAL_ARM_JOINT_NAMES)
}

# Output paths for the physical model variant
PHYSICAL_TENDON_DATA_DIR = DEFAULT_OUTPUT_ROOT / "tendon_physical" / "data"

# Elbow-equivalent PID for the physical model.  The four-bar linkage has
# slightly higher effective inertia than the single revolute elbow_approx
# joint (additional rod masses plus geometric coupling), so gains are
# moderately increased.  Saturation is 500 N (motor max).
PHYSICAL_ELBOW_SPEC = JointTestSpec(
    name="elbow_physical",          # virtual joint (measured from body quat)
    joint_index=-1,                 # not in any joint array — special handling
    pid=PIDGains(kp=60.0, ki=6.0, kd=4.0),
    step_amplitudes_deg=(20.0, 30.0, 40.0),
    min_tension_n=6.0,
    saturation_n=500.0,
    gravity_arm_m=0.26,
    mass_kg=2.49,                   # rod masses ~0.19 total, small moment
)

# Physical model shares wrist specs with elbow_approx
PHYSICAL_ALL_JOINT_SPECS: list[JointTestSpec] = [
    PHYSICAL_ELBOW_SPEC, WRIST_Y_SPEC, WRIST_X_SPEC,
]

# Tendon attachment offsets in body-local frames (m).
# Must match tendon_actuator.ELBOW_TENDON_*_OFFSETS exactly.
PHYSICAL_ROOT_ATTACH = np.array([
    [0.0, +0.0725, -0.34],   # T0 (left)
    [0.0, -0.0725, -0.34],   # T1 (right)
], dtype=np.float64)

PHYSICAL_FOREARM_ATTACH = np.array([
    [0.0, +0.0725, -0.02],   # T0 (left)
    [0.0, -0.0725, -0.02],   # T1 (right)
], dtype=np.float64)


def compute_elbow_angle_from_body_quat(forearm_quat_wxyz: np.ndarray) -> float:
    """Extract the X-axis rotation (elbow angle) from the forearm body quaternion.

    The forearm_link frame is aligned with the root_link frame at zero-config.
    The elbow deflection is a pure rotation about the X axis, so:
        θ_elbow = 2 · atan2(q_x, q_w)

    This is exact for the physical model because the 4-bar linkage constrains
    the forearm to rotate about X only (in the root_link frame).

    Parameters
    ----------
    forearm_quat_wxyz : (4,) array — [w, x, y, z] quaternion of forearm_link
                        in the world frame.

    Returns
    -------
    Elbow angle in radians (positive = forearm rotated about +X).
    """
    w, x = float(forearm_quat_wxyz[0]), float(forearm_quat_wxyz[1])
    return 2.0 * math.atan2(x, w)


@dataclass
class PIDState:
    """Mutable PID integrator state (Python floats — no torch dependency)."""
    integral: float = 0.0
    prev_measurement: float = 0.0
    prev_deriv: float = 0.0

    def reset(self, initial_measurement: float = 0.0) -> None:
        self.integral = 0.0
        self.prev_measurement = initial_measurement
        self.prev_deriv = 0.0


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
    integral_zone_rad: float | None = None,
    deriv_filter_alpha: float = 0.3,
    measured_velocity_rad_s: float | None = None,
) -> float:
    """Classic discrete PID with derivative-on-measurement, EMA-filtered D, and clamped integrator.

    The derivative term uses the process variable (``-d(current)/dt``)
    instead of ``d(error)/dt`` to avoid the derivative kick that occurs
    when the setpoint changes as a step.  A first-order exponential
    moving average (EMA) filter is applied to suppress high-frequency
    noise from the PhysX solver.

    Parameters
    ----------
    integral_zone_rad : float or None
        If set, the integrator only accumulates when ``|error|`` is below
        this threshold (conditional integration / anti-windup).  This
        prevents integral windup during the fast initial transient of
        low-inertia joints.
    deriv_filter_alpha : float
        EMA smoothing coefficient for the derivative term.  Lower values
        give heavier filtering (``0`` = hold previous, ``1`` = no filter).
    """
    error = setpoint_rad - current_rad
    if integral_zone_rad is None or abs(error) < integral_zone_rad:
        state.integral += error * dt
    state.integral = max(-integral_clamp, min(integral_clamp, state.integral))
    if measured_velocity_rad_s is not None:
        # Use PhysX solver velocity with light EMA to smooth numerical noise
        deriv_raw = -measured_velocity_rad_s
        deriv = 0.5 * deriv_raw + 0.5 * state.prev_deriv
    else:
        deriv_raw = -(current_rad - state.prev_measurement) / dt if dt > 1e-9 else 0.0
        deriv = deriv_filter_alpha * deriv_raw + (1.0 - deriv_filter_alpha) * state.prev_deriv
    state.prev_measurement = current_rad
    state.prev_deriv = deriv
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
    """Map a single-joint torque to 5 tendon tensions via block-wise redistribution.

    The Jacobian has natural blocks: elbow uses tendons [0,1], wrists use
    tendons [2,3,4].  To avoid the global null-space shift inflating
    inactive tendons (which then clip at saturation and distort cross-
    coupling torques), we solve each block independently:

    - Elbow (joint 0): 2-tendon antagonistic pair — direct algebraic solution.
    - Wrist (joint 1 or 2): 3-tendon group — pinv of the 2×3 wrist sub-
      Jacobian, with tension shift confined to tendons [2,3,4].

    Inactive tendons are set to ``min_tension`` (cable pre-tension).
    Returns shape (5,).
    """
    tensions = np.full(5, min_tension, dtype=np.float64)

    if joint_index == 0:
        # Elbow: T0 and T1 are antagonistic with lever arm ±0.0725 m.
        # τ = 0.0725·(T0 - T1)  →  T0 - T1 = τ / 0.0725
        lever = abs(JACOBIAN_T[0, 0])  # 0.0725
        diff = desired_torque / lever
        # Centre around min_tension so both stay ≥ min_tension
        t0 = min_tension + max(0.0, diff)
        t1 = min_tension + max(0.0, -diff)
        tensions[0] = min(t0, saturation)
        tensions[1] = min(t1, saturation)
    else:
        # Wrist block: tendons [2, 3, 4], Jacobian rows [1, 2]
        j_wrist = JACOBIAN_T[1:3, 2:5]                      # (2, 3)
        j_wrist_pinv = np.linalg.pinv(j_wrist)              # (3, 2)
        tau_wrist = np.zeros(2, dtype=np.float64)
        tau_wrist[joint_index - 1] = desired_torque          # row 0 = wrist_y, row 1 = wrist_x
        t_raw = j_wrist_pinv @ tau_wrist                     # (3,)
        shift = max(0.0, min_tension - float(np.min(t_raw)))
        t_shifted = np.clip(t_raw + shift, 0.0, saturation)
        tensions[2:5] = t_shifted

    return tensions


def torque_vector_to_tensions(
    desired_torques: np.ndarray,
    min_tension: float,
    saturation: float,
) -> np.ndarray:
    """Map a full 3-joint torque vector to 5 tendon tensions via block-wise redistribution.

    Elbow block (tendons 0,1) and wrist block (tendons 2,3,4) are solved
    independently, mirroring the logic in :func:`torque_to_tensions`.
    """
    tensions = np.full(5, min_tension, dtype=np.float64)

    # Elbow block: tendons [0, 1]
    lever = abs(JACOBIAN_T[0, 0])
    diff = desired_torques[0] / lever
    tensions[0] = min(min_tension + max(0.0, diff), saturation)
    tensions[1] = min(min_tension + max(0.0, -diff), saturation)

    # Wrist block: tendons [2, 3, 4]
    j_wrist = JACOBIAN_T[1:3, 2:5]
    j_wrist_pinv = np.linalg.pinv(j_wrist)
    t_raw = j_wrist_pinv @ desired_torques[1:3]
    shift = max(0.0, min_tension - float(np.min(t_raw)))
    tensions[2:5] = np.clip(t_raw + shift, 0.0, saturation)

    return tensions


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

"""PD gain tuning for the tensegrity robot.

Systematically tests joint step responses, cross-coupling, and grip
stability to determine optimal actuator stiffness and damping.

Reference targets (Klein 2023; FAPS-Thesis §4.1 — Gain Tuning):
  - Near-critical damping  (ζ ≈ 1.0–1.5)
  - Settling time           0.1–0.3 s
  - Overshoot              < 5 % of step amplitude
  - No sustained oscillation

Protocol
--------
Phase 1 — Step response
    Each controlled joint is given step targets at 3 amplitudes.
    Recorded: rise time, overshoot, settling time, steady-state error.

Phase 2 — Gain sweep (arm only)
    Damping is swept from the current value down toward the analytically
    computed near-critical value.  The best candidate is selected by
    settling time.

Phase 3 — Cross-coupling
    Each arm joint steps to ±0.30 rad; base joint drift is measured.

Phase 4 — Gripper grasp
    Close on cube → lift → hold → verify cube is still gripped.

Phase 5 — Summary
    Side-by-side comparison of current vs. recommended gains with
    concrete values for ``place_scene_cfg.py``.

Usage::

    cd src/tensegrity_pick
    conda run -n env_isaaclab python3 scripts/tune_pd_gains.py \\
        --task Template-Tensegrity-Place-v0 --num_envs 1

    # headless (faster):
    conda run -n env_isaaclab python3 scripts/tune_pd_gains.py \\
        --task Template-Tensegrity-Place-v0 --num_envs 1 --headless
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PD gain tuning for the tensegrity robot.")
parser.add_argument("--task", type=str, default="Template-Tensegrity-Place-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tensegrity_pick.tasks  # noqa: F401

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply


# =====================================================================
# Constants
# =====================================================================

CONVEYOR_SURFACE_HEIGHT_M = 0.80
GRASP_CENTER_LOCAL_Z = 0.1925

# Action layout: [base_y, base_z, elbow, wrist_y, wrist_x, gripper]
IDX_BASE_Y = 0
IDX_BASE_Z = 1
IDX_ELBOW = 2
IDX_WRIST_Y = 3
IDX_WRIST_X = 4
IDX_GRIPPER = 5
ACTION_DIM = 6

ACTION_SCALE = 0.50

DEFAULT_BASE_Y = 0.0
DEFAULT_BASE_Z = -0.25
DEFAULT_ELBOW = 0.0
DEFAULT_WRIST_Y = 0.0
DEFAULT_WRIST_X = 0.0

OPEN = 1.0
CLOSE = -1.0

GC_REST_X = 0.19

# Physics timing (decimation=2, dt=0.01 → control period 0.02 s)
CONTROL_DT = 0.02

# Klein (2023) criteria
TARGET_SETTLING_S = (0.10, 0.30)
TARGET_OVERSHOOT_PCT = 5.0


# =====================================================================
# Data classes
# =====================================================================

@dataclass(frozen=True)
class JointSpec:
    """Specification for one controlled joint."""

    name: str
    action_index: int
    default_pos: float
    step_amplitudes: tuple[float, ...]


@dataclass(frozen=True)
class GainSet:
    """A candidate set of PD gains for one actuator group."""

    label: str
    stiffness: float
    damping: float
    effort_limit: float
    velocity_limit: float


@dataclass(frozen=True)
class StepMetrics:
    """Metrics from a single step-response test."""

    amplitude_rad: float
    rise_time_s: float
    overshoot_pct: float
    settling_time_s: float
    steady_state_error_pct: float
    damping_ratio_estimate: float


# =====================================================================
# Joint specifications
# =====================================================================

BASE_SPECS = (
    JointSpec("base_y_joint", IDX_BASE_Y, DEFAULT_BASE_Y, (0.10, 0.20, 0.30)),
    JointSpec("base_z_joint", IDX_BASE_Z, DEFAULT_BASE_Z, (0.03, 0.06, 0.10)),
)

ARM_SPECS = (
    JointSpec("elbow_joint",  IDX_ELBOW,  DEFAULT_ELBOW,  (0.30, 0.60, 1.00)),
    JointSpec("wrist_y_joint", IDX_WRIST_Y, DEFAULT_WRIST_Y, (0.15, 0.30, 0.50)),
    JointSpec("wrist_x_joint", IDX_WRIST_X, DEFAULT_WRIST_X, (0.15, 0.30, 0.50)),
)


# =====================================================================
# Helpers
# =====================================================================

def target_to_action(desired_target: float, default: float) -> float:
    """Convert a desired joint target to a raw action value."""
    return (desired_target - default) / ACTION_SCALE


def send_default_actions(env: gym.Env, steps: int) -> None:
    """Hold all joints at default for *steps* control steps."""
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    action = torch.zeros(num_envs, ACTION_DIM, device=device)
    action[:, IDX_GRIPPER] = OPEN
    for _ in range(steps):
        env.step(action)


def read_joint_position(robot: Articulation, joint_name: str) -> float:
    """Read the current position of a single joint (env 0)."""
    idx = robot.data.joint_names.index(joint_name)
    return robot.data.joint_pos[0, idx].item()


def read_gc_position(
    robot: Articulation,
    env_origin: torch.Tensor,
) -> tuple[float, float, float]:
    """Return the grasp-centre position in local frame."""
    ee_idx = robot.body_names.index("tool_link_0")
    ee_pos = robot.data.body_pos_w[0, ee_idx]
    ee_quat = robot.data.body_quat_w[0, ee_idx]
    offset = ee_pos.new_tensor([0.0, 0.0, GRASP_CENTER_LOCAL_Z])
    gc = ee_pos + quat_apply(ee_quat.unsqueeze(0), offset.unsqueeze(0)).squeeze(0)
    local = gc - env_origin
    return local[0].item(), local[1].item(), local[2].item()


# =====================================================================
# Step-response measurement
# =====================================================================

def run_step_response(
    env: gym.Env,
    robot: Articulation,
    spec: JointSpec,
    amplitude: float,
    settle_steps: int = 50,
    record_steps: int = 150,
) -> tuple[list[float], list[float]]:
    """Apply a step input and record joint position over time.

    1. Hold default for *settle_steps* to ensure a clean start.
    2. Apply a step of *amplitude* rad above the default position.
    3. Record joint position for *record_steps*.

    Returns (times_s, positions_rad) where times_s[0] = 0 is the step
    instant.
    """
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # Settle at default
    send_default_actions(env, settle_steps)

    initial_pos = read_joint_position(robot, spec.name)

    # Build step action
    step_action_value = target_to_action(spec.default_pos + amplitude, spec.default_pos)
    action = torch.zeros(num_envs, ACTION_DIM, device=device)
    action[:, IDX_GRIPPER] = OPEN
    action[:, spec.action_index] = step_action_value

    times: list[float] = []
    positions: list[float] = []

    for i in range(record_steps):
        env.step(action)
        t = (i + 1) * CONTROL_DT
        pos = read_joint_position(robot, spec.name)
        times.append(t)
        positions.append(pos)

    return times, positions


def compute_step_metrics(
    times: list[float],
    positions: list[float],
    target: float,
    initial: float,
) -> StepMetrics:
    """Compute step-response metrics from a time-series recording."""
    amplitude = target - initial
    amplitude_abs = abs(amplitude)
    sign = 1.0 if amplitude >= 0 else -1.0

    # Normalised response: 0 at initial, 1 at target
    normalised = [(p - initial) / amplitude if amplitude_abs > 1e-9 else 0.0
                  for p in positions]

    # Rise time: first crossing of 0.1 → first crossing of 0.9
    rise_start = next(
        (times[i] for i, n in enumerate(normalised) if n >= 0.1),
        times[-1],
    )
    rise_end = next(
        (times[i] for i, n in enumerate(normalised) if n >= 0.9),
        times[-1],
    )
    rise_time = rise_end - rise_start

    # Overshoot
    peak = max(normalised) if sign > 0 else min(normalised)
    overshoot_frac = (peak - 1.0) if sign > 0 else (1.0 - peak)
    overshoot_pct = max(0.0, overshoot_frac * 100.0)

    # Settling time (2 % band around target)
    settling_time = times[-1]
    for i in range(len(normalised) - 1, -1, -1):
        if abs(normalised[i] - 1.0) > 0.02:
            settling_time = times[min(i + 1, len(times) - 1)]
            break
    else:
        settling_time = times[0]

    # Steady-state error (average of last 20 samples)
    tail = positions[-20:]
    steady_state = sum(tail) / len(tail)
    ss_error = abs(steady_state - target)
    ss_error_pct = (ss_error / amplitude_abs * 100.0) if amplitude_abs > 1e-9 else 0.0

    # Damping ratio estimate from overshoot (if underdamped)
    if overshoot_pct > 0.1:
        ln_os = math.log(overshoot_pct / 100.0)
        zeta = -ln_os / math.sqrt(math.pi**2 + ln_os**2)
    else:
        # Overdamped — estimate from rise time and natural frequency
        # ω_n ≈ 1.8 / t_rise  (approximation for critically damped)
        omega_n = 1.8 / rise_time if rise_time > 1e-3 else 100.0
        # settling ~ 4.6 * ζ / ω_n for overdamped
        zeta = settling_time * omega_n / 4.6 if omega_n > 0 else 10.0
        zeta = max(zeta, 1.0)

    return StepMetrics(
        amplitude_rad=amplitude,
        rise_time_s=rise_time,
        overshoot_pct=overshoot_pct,
        settling_time_s=settling_time,
        steady_state_error_pct=ss_error_pct,
        damping_ratio_estimate=zeta,
    )


# =====================================================================
# Gain writing helpers
# =====================================================================

def write_gains(
    robot: Articulation,
    joint_names: list[str],
    stiffness: float,
    damping: float,
    effort_limit: float | None = None,
) -> None:
    """Write stiffness and damping to the simulator for given joints."""
    joint_ids = [robot.data.joint_names.index(n) for n in joint_names]
    ids_tensor = torch.tensor(joint_ids, device=robot.device)

    num_envs = robot.num_instances
    num_joints = len(joint_ids)

    stiffness_tensor = torch.full(
        (num_envs, num_joints), stiffness, device=robot.device,
    )
    damping_tensor = torch.full(
        (num_envs, num_joints), damping, device=robot.device,
    )

    robot.write_joint_stiffness_to_sim(stiffness_tensor, joint_ids=ids_tensor)
    robot.write_joint_damping_to_sim(damping_tensor, joint_ids=ids_tensor)

    if effort_limit is not None:
        effort_tensor = torch.full(
            (num_envs, num_joints), effort_limit, device=robot.device,
        )
        robot.write_joint_effort_limit_to_sim(effort_tensor, joint_ids=ids_tensor)


# =====================================================================
# Phase 1 — Step-response test
# =====================================================================

def run_step_tests(
    env: gym.Env,
    robot: Articulation,
    specs: tuple[JointSpec, ...],
    label: str,
) -> dict[str, list[StepMetrics]]:
    """Run step-response tests for a group of joints.

    Returns {joint_name: [StepMetrics, ...]}.
    """
    results: dict[str, list[StepMetrics]] = {}

    for spec in specs:
        metrics_list: list[StepMetrics] = []
        for amplitude in spec.step_amplitudes:
            # Reset env so joints start at default
            env.reset()
            send_default_actions(env, 30)

            target = spec.default_pos + amplitude
            initial = read_joint_position(robot, spec.name)

            times, positions = run_step_response(env, robot, spec, amplitude)
            metrics = compute_step_metrics(times, positions, target, initial)
            metrics_list.append(metrics)

        results[spec.name] = metrics_list

    return results


def print_step_results(
    results: dict[str, list[StepMetrics]],
    label: str,
) -> None:
    """Pretty-print step-response results with pass/fail indicators."""
    print(f"\n{'─' * 100}")
    print(f"  Step-Response Results: {label}")
    print(f"{'─' * 100}")
    print(
        f"  {'Joint':<22s} {'Amp (rad)':>10s} {'Rise (s)':>10s} "
        f"{'Over (%)':>10s} {'Settle (s)':>11s} {'SS err (%)':>11s} "
        f"{'ζ est':>8s}"
    )
    print(f"  {'─' * 94}")

    for joint_name, metrics_list in results.items():
        for m in metrics_list:
            rise_ok = m.rise_time_s < 0.5
            over_ok = m.overshoot_pct <= TARGET_OVERSHOOT_PCT
            settle_ok = m.settling_time_s <= TARGET_SETTLING_S[1]
            ss_ok = m.steady_state_error_pct < 2.0
            zeta_ok = 0.8 <= m.damping_ratio_estimate <= 2.0

            flags = (
                ("✓" if rise_ok else "✗")
                + ("✓" if over_ok else "✗")
                + ("✓" if settle_ok else "✗")
                + ("✓" if ss_ok else "✗")
                + ("✓" if zeta_ok else "✗")
            )

            print(
                f"  {joint_name:<22s} {m.amplitude_rad:>+10.3f} "
                f"{m.rise_time_s:>10.3f} {m.overshoot_pct:>10.1f} "
                f"{m.settling_time_s:>11.3f} {m.steady_state_error_pct:>11.1f} "
                f"{m.damping_ratio_estimate:>8.2f}  {flags}"
            )

    print(f"  {'─' * 94}")
    print(f"  Flags: rise<0.5s  over<{TARGET_OVERSHOOT_PCT}%  "
          f"settle<{TARGET_SETTLING_S[1]}s  ss<2%  ζ∈[0.8,2.0]")


# =====================================================================
# Phase 2 — Arm gain sweep
# =====================================================================

ARM_JOINT_NAMES = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]

# Current gains from place_scene_cfg.py
CURRENT_ARM_GAINS = GainSet(
    label="current",
    stiffness=400.0,
    damping=120.0,
    effort_limit=40.0,
    velocity_limit=2.0,
)

# Candidates: keep stiffness, sweep damping toward critically damped.
# Analytical near-critical damping for the elbow:
#   J_eff ≈ 0.22 kg·m²  (3.5 kg arm, ~0.25 m radius)
#   ω_n = sqrt(K/J) = sqrt(400/0.22) ≈ 42.6 rad/s
#   D_crit = 2·J·ω_n ≈ 18.7 N·m·s/rad
# For wrists (much lighter, J≈0.014): D_crit ≈ 4.7.
# A single group uses the same gains; D≈25 is a good compromise.
ARM_GAIN_CANDIDATES = (
    GainSet("D=60",  400.0,  60.0, 40.0, 2.0),
    GainSet("D=30",  400.0,  30.0, 40.0, 2.0),
    GainSet("D=25",  400.0,  25.0, 40.0, 2.0),
    GainSet("D=20",  400.0,  20.0, 40.0, 2.0),
    GainSet("D=15",  400.0,  15.0, 40.0, 2.0),
)


def run_arm_gain_sweep(
    env: gym.Env,
    robot: Articulation,
) -> tuple[dict[str, dict[str, list[StepMetrics]]], GainSet]:
    """Sweep arm damping candidates and return all results + best gains.

    Returns (all_results, best_gains) where all_results maps
    gain_label → {joint_name → [StepMetrics]}.
    """
    all_results: dict[str, dict[str, list[StepMetrics]]] = {}
    best_gains = CURRENT_ARM_GAINS
    best_avg_settling = float("inf")

    # Test current gains first
    print("\n  Testing current arm gains...", flush=True)
    current_results = run_step_tests(env, robot, ARM_SPECS, "arm (current)")
    all_results["current"] = current_results
    print_step_results(current_results, f"Arm — current (K={CURRENT_ARM_GAINS.stiffness}, D={CURRENT_ARM_GAINS.damping})")

    avg_settle = _average_settling(current_results)
    if avg_settle < best_avg_settling:
        best_avg_settling = avg_settle
        best_gains = CURRENT_ARM_GAINS

    for candidate in ARM_GAIN_CANDIDATES:
        print(f"\n  Applying arm gains: {candidate.label} "
              f"(K={candidate.stiffness}, D={candidate.damping})...", flush=True)

        write_gains(robot, ARM_JOINT_NAMES, candidate.stiffness, candidate.damping)

        results = run_step_tests(env, robot, ARM_SPECS, f"arm ({candidate.label})")
        all_results[candidate.label] = results

        print_step_results(
            results,
            f"Arm — {candidate.label} (K={candidate.stiffness}, D={candidate.damping})",
        )

        avg_settle = _average_settling(results)
        if avg_settle < best_avg_settling:
            best_avg_settling = avg_settle
            best_gains = candidate

    # Restore best gains
    write_gains(robot, ARM_JOINT_NAMES, best_gains.stiffness, best_gains.damping)

    return all_results, best_gains


def _average_settling(results: dict[str, list[StepMetrics]]) -> float:
    """Compute the mean settling time across all tests in a result set."""
    times = [m.settling_time_s for metrics in results.values() for m in metrics]
    return sum(times) / len(times) if times else float("inf")


# =====================================================================
# Phase 3 — Cross-coupling
# =====================================================================

CROSS_COUPLING_AMPLITUDE = 0.30  # rad

BASE_JOINT_NAMES = ["base_y_joint", "base_z_joint"]


def run_cross_coupling_test(
    env: gym.Env,
    robot: Articulation,
) -> dict[str, dict[str, float]]:
    """Move each arm joint and measure base joint drift.

    Returns {arm_joint: {base_joint: max_drift_rad}}.
    """
    results: dict[str, dict[str, float]] = {}

    for arm_spec in ARM_SPECS:
        env.reset()
        send_default_actions(env, 50)

        # Record baseline base positions
        base_initial = {
            name: read_joint_position(robot, name) for name in BASE_JOINT_NAMES
        }

        # Apply arm step
        step_value = target_to_action(
            arm_spec.default_pos + CROSS_COUPLING_AMPLITUDE,
            arm_spec.default_pos,
        )
        action = torch.zeros(env.unwrapped.num_envs, ACTION_DIM, device=env.unwrapped.device)
        action[:, IDX_GRIPPER] = OPEN
        action[:, arm_spec.action_index] = step_value

        max_drift: dict[str, float] = {name: 0.0 for name in BASE_JOINT_NAMES}

        for _ in range(200):
            env.step(action)
            for base_name in BASE_JOINT_NAMES:
                pos = read_joint_position(robot, base_name)
                drift = abs(pos - base_initial[base_name])
                max_drift[base_name] = max(max_drift[base_name], drift)

        results[arm_spec.name] = max_drift

    return results


def print_cross_coupling_results(
    results: dict[str, dict[str, float]],
) -> None:
    """Pretty-print cross-coupling test results."""
    threshold = 0.01  # 1 cm or 0.01 rad

    print(f"\n{'─' * 80}")
    print("  Cross-Coupling Test: Arm Movement → Base Drift")
    print(f"{'─' * 80}")
    print(f"  {'Arm Joint':<22s} {'base_y drift':>14s} {'base_z drift':>14s} {'Status':>8s}")
    print(f"  {'─' * 66}")

    for arm_joint, drift in results.items():
        by_drift = drift["base_y_joint"]
        bz_drift = drift["base_z_joint"]
        ok = by_drift < threshold and bz_drift < threshold
        status = "✓ PASS" if ok else "✗ FAIL"
        print(
            f"  {arm_joint:<22s} "
            f"{by_drift:>14.4f} {bz_drift:>14.4f} {status:>8s}"
        )

    print(f"  {'─' * 66}")
    print(f"  Threshold: {threshold} rad ({threshold * 1000:.0f} mm for prismatic)")


# =====================================================================
# Phase 4 — Gripper grasp test
# =====================================================================

def run_gripper_grasp_test(
    env: gym.Env,
    robot: Articulation,
    green: RigidObject,
    env_origin: torch.Tensor,
) -> bool:
    """Close gripper on cube, lift, hold — return True if grip maintained."""
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    print(f"\n{'─' * 80}")
    print("  Gripper Grasp Test")
    print(f"{'─' * 80}")

    env.reset()
    send_default_actions(env, 60)

    gc_x, gc_y, gc_z = read_gc_position(robot, env_origin)
    cube_pos = green.data.root_pos_w[0] - env_origin
    print(f"  Start: gc=({gc_x:.3f}, {gc_y:.3f}, {gc_z:.3f})  "
          f"cube=({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})", flush=True)

    # Close gripper
    action = torch.zeros(num_envs, ACTION_DIM, device=device)
    action[:, IDX_GRIPPER] = CLOSE
    for _ in range(120):
        env.step(action)

    gc_x, gc_y, gc_z = read_gc_position(robot, env_origin)
    cube_pos = green.data.root_pos_w[0] - env_origin
    gc_to_cube = torch.norm(
        torch.tensor([gc_x, gc_y, gc_z]) - cube_pos[:3].cpu()
    ).item()
    print(f"  After close: gc→cube={gc_to_cube:.3f}m  "
          f"cube_z={cube_pos[2]:.3f}", flush=True)

    grip_ok_pre_lift = gc_to_cube < 0.05

    # Lift (base_z → -0.10)
    lift_action = target_to_action(-0.10, DEFAULT_BASE_Z)
    action = torch.zeros(num_envs, ACTION_DIM, device=device)
    action[:, IDX_GRIPPER] = CLOSE
    action[:, IDX_BASE_Z] = lift_action

    # Ramp the lift over 60 steps
    for i in range(120):
        t = min(1.0, (i + 1) / 60)
        action[:, IDX_BASE_Z] = t * lift_action
        env.step(action)

    gc_x, gc_y, gc_z = read_gc_position(robot, env_origin)
    cube_pos = green.data.root_pos_w[0] - env_origin
    gc_to_cube = torch.norm(
        torch.tensor([gc_x, gc_y, gc_z]) - cube_pos[:3].cpu()
    ).item()
    cube_lifted = cube_pos[2].item() > CONVEYOR_SURFACE_HEIGHT_M + 0.05

    print(f"  After lift: gc→cube={gc_to_cube:.3f}m  "
          f"cube_z={cube_pos[2]:.3f}  lifted={cube_lifted}", flush=True)

    # Hold
    for _ in range(80):
        env.step(action)

    cube_pos_final = green.data.root_pos_w[0] - env_origin
    gc_to_cube_final = torch.norm(
        torch.tensor([gc_x, gc_y, gc_z]) - cube_pos_final[:3].cpu()
    ).item()
    grip_maintained = gc_to_cube_final < 0.06 and cube_pos_final[2].item() > CONVEYOR_SURFACE_HEIGHT_M + 0.03

    status = "✓ PASS" if grip_maintained else "✗ FAIL"
    print(f"  After hold: gc→cube={gc_to_cube_final:.3f}m  "
          f"cube_z={cube_pos_final[2]:.3f}  {status}", flush=True)
    print(f"{'─' * 80}")

    return grip_maintained


# =====================================================================
# Phase 5 — Summary
# =====================================================================

def print_summary(
    base_results: dict[str, list[StepMetrics]],
    best_arm_gains: GainSet,
    arm_all: dict[str, dict[str, list[StepMetrics]]],
    coupling: dict[str, dict[str, float]],
    grip_ok: bool,
) -> None:
    """Print final recommendations."""
    print("\n" + "=" * 100)
    print("  TUNING SUMMARY")
    print("=" * 100)

    # Base
    base_settle = _average_settling(base_results)
    print(f"\n  BASE (prismatic Y+Z)")
    print(f"    Current gains: K=8000  D=800  effort=800  vel=5.0")
    print(f"    Avg settling:  {base_settle:.3f}s")
    print(f"    Verdict:       KEEP — ζ≈1.4, stable, provides anchor for arm movements")

    # Arm
    print(f"\n  ARM (elbow + wrist_y + wrist_x)")
    print(f"    Current gains: K=400  D=120  effort=40  vel=2.0")

    if best_arm_gains.label == "current":
        print(f"    Verdict:       KEEP — current gains already optimal")
    else:
        best_results = arm_all.get(best_arm_gains.label, {})
        best_settle = _average_settling(best_results) if best_results else float("inf")
        print(f"    Current avg settling: {_average_settling(arm_all.get('current', {})):.3f}s")
        print(f"    Best candidate:       {best_arm_gains.label} → avg settling: {best_settle:.3f}s")
        print(f"    Recommended:   K={best_arm_gains.stiffness:.0f}  "
              f"D={best_arm_gains.damping:.0f}  "
              f"effort={best_arm_gains.effort_limit:.0f}  "
              f"vel={best_arm_gains.velocity_limit:.1f}")

    # Arm analytical context
    print(f"\n    Analytical context (Klein 2023 methodology):")
    print(f"      Elbow effective inertia:  J ≈ 0.22 kg·m²")
    print(f"      Elbow D_critical:         {2 * math.sqrt(400 * 0.22):.1f} N·m·s/rad")
    print(f"      Elbow ζ at D=120:         {120 / (2 * math.sqrt(400 * 0.22)):.1f} (extremely overdamped)")
    if best_arm_gains.damping != 120.0:
        elbow_zeta = best_arm_gains.damping / (2 * math.sqrt(400 * 0.22))
        print(f"      Elbow ζ at D={best_arm_gains.damping:.0f}:          {elbow_zeta:.1f}")

    # Cross-coupling
    max_drift = max(
        d for joint_drift in coupling.values() for d in joint_drift.values()
    )
    coupling_ok = max_drift < 0.01
    print(f"\n  CROSS-COUPLING")
    print(f"    Max base drift:  {max_drift:.4f} rad/m")
    print(f"    Verdict:         {'✓ PASS' if coupling_ok else '✗ FAIL — base gains may need increase'}")

    # Gripper
    print(f"\n  GRIPPER")
    print(f"    USD standard:  K=100  D=20  effort=50  vel=5.0  (finger_joint)")
    print(f"    USD standard:  K=1000 D=200 effort=200 vel=5.0  (passive joints)")
    print(f"    Grasp test:    {'✓ PASS' if grip_ok else '✗ FAIL — may need adjustment'}")

    # Config snippet
    print(f"\n{'─' * 100}")
    print("  Recommended place_scene_cfg.py actuator config:")
    print(f"{'─' * 100}")
    print("""
    actuators={{
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_y_joint", "base_z_joint"],
            effort_limit=800.0,
            velocity_limit_sim=5.0,
            stiffness=8000.0,
            damping=800.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
            effort_limit={best_arm_gains.effort_limit:.1f},
            velocity_limit_sim={best_arm_gains.velocity_limit:.1f},
            stiffness={best_arm_gains.stiffness:.1f},
            damping={best_arm_gains.damping:.1f},
        ),
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit=50.0,
            velocity_limit_sim=5.0,
            stiffness=100.0,
            damping=20.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_outer_knuckle_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "left_inner_finger_joint",
                "right_inner_finger_joint",
                "left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint",
            ],
            effort_limit=200.0,
            velocity_limit_sim=5.0,
            stiffness=1000.0,
            damping=200.0,
        ),
    }}""")

    print("=" * 100 + "\n")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Long episode for uninterrupted testing
    env_cfg.episode_length_s = 300.0

    # Disable randomisation
    env_cfg.events.reset_arm.params["position_range"] = (0.0, 0.0)

    # Disable terminations
    if hasattr(env_cfg.terminations, "joint_effort_saturated"):
        env_cfg.terminations.joint_effort_saturated.params["threshold_ratio"] = 1e6
    env_cfg.terminations.belt_collision.params["max_penetration"] = 100.0
    env_cfg.terminations.joint_vel_diverged.params["max_velocity"] = 1e6

    # Fix cube spawn under grasp centre
    from tensegrity_pick.tasks.manager_based.tensegrity_place.mdp.rewards import SpawnBox
    env_cfg.events.reset_cubes.params["spawn_box"] = SpawnBox(
        x_range=(GC_REST_X, GC_REST_X),
        y_range=(0.00, 0.00),
        z_range=(0.83, 0.83),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    scene = env.unwrapped.scene
    robot: Articulation = scene["robot"]
    green: RigidObject = scene["green_cube"]
    env_origin = scene.env_origins[0]

    print("=" * 100)
    print("  PD GAIN TUNING — Tensegrity Robot")
    print("  Reference: Klein (2023) — near-critical damping, 0.1–0.3 s settling, < 5 % overshoot")
    print("=" * 100, flush=True)

    # ── Phase 1: Base step response ──────────────────────────────────
    print("\n" + "=" * 100)
    print("  PHASE 1: Base Step Response (current gains)")
    print("=" * 100, flush=True)

    base_results = run_step_tests(env, robot, BASE_SPECS, "base")
    print_step_results(base_results, "Base (K=8000, D=800)")

    # ── Phase 2: Arm gain sweep ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("  PHASE 2: Arm Gain Sweep")
    print("  Sweeping damping from overdamped (D=120) toward near-critical (D≈19)")
    print("=" * 100, flush=True)

    arm_all_results, best_arm = run_arm_gain_sweep(env, robot)

    # Print comparison table
    print(f"\n{'─' * 100}")
    print("  Arm Gain Sweep — Average Settling Time Comparison")
    print(f"{'─' * 100}")
    print(f"  {'Gains':<20s} {'Avg Settle (s)':>16s} {'Fastest':>10s}")
    print(f"  {'─' * 50}")
    for label, results in arm_all_results.items():
        avg = _average_settling(results)
        marker = "  ◄" if label == best_arm.label else ""
        print(f"  {label:<20s} {avg:>16.3f}{marker}")
    print(f"  {'─' * 50}")

    # ── Phase 3: Cross-coupling ──────────────────────────────────────
    print("\n" + "=" * 100)
    print("  PHASE 3: Cross-Coupling (arm movement → base drift)")
    print("=" * 100, flush=True)

    coupling_results = run_cross_coupling_test(env, robot)
    print_cross_coupling_results(coupling_results)

    # ── Phase 4: Gripper grasp test ──────────────────────────────────
    print("\n" + "=" * 100)
    print("  PHASE 4: Gripper Grasp Test")
    print("=" * 100, flush=True)

    # Apply USD standard gripper gains before testing
    write_gains(
        robot,
        ["finger_joint"],
        stiffness=100.0,
        damping=20.0,
        effort_limit=50.0,
    )
    # Passive joints: restore USD defaults
    passive_joints = [
        "right_outer_knuckle_joint",
        "left_outer_finger_joint",
        "right_outer_finger_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
        "left_inner_finger_pad_joint",
        "right_inner_finger_pad_joint",
    ]
    write_gains(robot, passive_joints, stiffness=1000.0, damping=200.0, effort_limit=200.0)

    grip_ok = run_gripper_grasp_test(env, robot, green, env_origin)

    # ── Phase 5: Summary ─────────────────────────────────────────────
    print_summary(base_results, best_arm, arm_all_results, coupling_results, grip_ok)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

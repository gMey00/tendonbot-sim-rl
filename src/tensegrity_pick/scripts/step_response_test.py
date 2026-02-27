"""Step-response test for the tendon-driven tensegrity arm in Isaac Sim.

Replicates the experimental methodology of Klein (2023), §4.2–§4.3:

1. A PID position controller converts angle set-points to desired torques.
2. Gravity compensation is added per Mukherjee et al. (thesis [85]).
3. The tendon tension distribution (Jacobian transpose) maps torques to
   cable tensions, which are clamped to the physical saturation limits.
4. Joint angles are recorded via the physics engine and compared against
   the set-point curves.

Reported metrics (per joint, per step amplitude):
  - Rise time  (10 %→90 % of target)
  - Overshoot  (% above target)
  - RMSE / NRMSE between reference and actual response

Plots:
  - Joint-angle time series with set-point overlay  (thesis Fig. 4.4 / 4.5)
  - Per-tendon tension profiles                      (thesis Fig. 4.4 e–j)

Usage
-----
.. code-block:: bash

    cd /home/robot/Isaac/IsaacLab
    ./isaaclab.sh -p /home/robot/studentische-arbeiten/src/tensegrity_pick/scripts/step_response_test.py \\
        [--headless] [--num-envs 1] [--output-dir ./step_response_results]

References
----------
* Klein, M. (2023). *Arbeitsraumanalyse, Simulation und Bewegungsplanung
  eines seilgetriebenen robotischen Manipulators*. Master's thesis, FAU.
  §3.2.3 (PID control), §3.2.4 (tension distribution), §3.5 (test method),
  §4.2 (step response results), §4.3 (simulation accuracy / NRMSE).
* Nemoto, T. et al. — tendon-driven orientation control (thesis [84]).
* Mukherjee et al. — gravity compensation (thesis [85]).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

# ── Robot config ──────────────────────────────────────────────────────────

PROJ_ASSETS_PATH = "/home/robot/studentische-arbeiten/res"

ARM_JOINT_NAMES = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]

ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJ_ASSETS_PATH}/Tensegrity/threedof_manipulator/threedof_manipulator.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={"elbow_joint": 0.0, "wrist_y_joint": 0.0, "wrist_x_joint": 0.0},
    ),
    actuators={
        "arm": IdealPDActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit=200.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class StepResponseSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + 3-DOF tensegrity arm."""

    ground = sim_utils.AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ── Thesis PID parameters (Klein 2023, §3.5) ─────────────────────────────

@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float


@dataclass(frozen=True)
class JointTestSpec:
    """Per-joint test specification from the thesis."""
    name: str
    pid: PIDGains
    step_amplitudes_deg: list[float]
    min_tension_n: float
    max_tension_n: float
    saturation_n: float
    gravity_arm_m: float  # lc — distance from joint to CoM along gravity lever
    mass_kg: float        # me — mass of the moving link


# Thesis §3.5 / Table 4.2 — controller settings
WRIST_Y_SPEC = JointTestSpec(
    name="wrist_y_joint",
    pid=PIDGains(kp=0.2, ki=0.03, kd=0.03),
    step_amplitudes_deg=[10.0, 20.0, 30.0],
    min_tension_n=5.0,
    max_tension_n=15.0,
    saturation_n=80.0,
    gravity_arm_m=0.06,   # half of end-effector length (0.12 m)
    mass_kg=0.32,
)

WRIST_X_SPEC = JointTestSpec(
    name="wrist_x_joint",
    pid=PIDGains(kp=0.2, ki=0.03, kd=0.03),
    step_amplitudes_deg=[10.0, 20.0, 30.0],
    min_tension_n=5.0,
    max_tension_n=15.0,
    saturation_n=80.0,
    gravity_arm_m=0.06,
    mass_kg=0.32,
)

ELBOW_SPEC = JointTestSpec(
    name="elbow_joint",
    pid=PIDGains(kp=0.3, ki=0.03, kd=0.02),
    step_amplitudes_deg=[20.0, 30.0, 40.0],
    min_tension_n=6.0,
    max_tension_n=20.0,
    saturation_n=160.0,
    gravity_arm_m=0.15,   # approximate forearm CoM distance
    mass_kg=0.80,
)

ALL_JOINT_SPECS = [WRIST_Y_SPEC, WRIST_X_SPEC, ELBOW_SPEC]


# ── Jacobian transpose (3 joints × 5 tendons) ────────────────────────────
# Row order: [elbow, wrist_y, wrist_x]  — matches ARM_JOINT_NAMES
JACOBIAN_T = np.array([
    [+0.0725, -0.0725,  0.0,       0.0,      0.0],       # elbow
    [ 0.0,     0.0,     -0.013856,  0.0,     +0.013856],  # wrist_y
    [ 0.0,     0.0,     +0.008,    -0.016,   +0.008],     # wrist_x
], dtype=np.float64)

JACOBIAN_T_TENSOR: torch.Tensor | None = None  # lazily initialised on-device


def _jacobian_pinv() -> np.ndarray:
    """Moore–Penrose pseudo-inverse of J^T used for torque→tension mapping."""
    return np.linalg.pinv(JACOBIAN_T)


# ── PID controller state ─────────────────────────────────────────────────

@dataclass
class PIDState:
    integral: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    previous_error: torch.Tensor = field(default_factory=lambda: torch.zeros(1))

    def reset(self, device: str) -> None:
        self.integral = torch.zeros(1, device=device)
        self.previous_error = torch.zeros(1, device=device)


def compute_pid_torque(
    setpoint_rad: float,
    current_rad: torch.Tensor,
    dt: float,
    gains: PIDGains,
    state: PIDState,
) -> torch.Tensor:
    """Classic PID with clamped integral (anti-windup)."""
    error = setpoint_rad - current_rad
    state.integral += error * dt
    state.integral.clamp_(-50.0, 50.0)
    derivative = (error - state.previous_error) / dt if dt > 0 else torch.zeros_like(error)
    state.previous_error = error.clone()
    return gains.kp * error + gains.ki * state.integral + gains.kd * derivative


def gravity_compensation_torque(
    current_rad: torch.Tensor,
    mass_kg: float,
    gravity_arm_m: float,
) -> torch.Tensor:
    """τ_a = m·g·l_c·sin(θ) — Mukherjee et al. (thesis [85], Eq. 3.2)."""
    return mass_kg * 9.81 * gravity_arm_m * torch.sin(current_rad)


def torque_to_tensions(
    desired_torques: torch.Tensor,
    joint_spec: JointTestSpec,
    joint_index: int,
    device: str,
) -> torch.Tensor:
    """Map desired joint torque (scalar per env) to 5 tendon tensions.

    Uses the pseudoinverse of the Jacobian transpose row for this joint,
    then clamps tensions to the physical limits from the thesis.
    """
    global JACOBIAN_T_TENSOR
    if JACOBIAN_T_TENSOR is None:
        JACOBIAN_T_TENSOR = torch.tensor(JACOBIAN_T, dtype=torch.float32, device=device)

    # Build full 3-joint torque vector (only our joint is non-zero)
    full_torque = torch.zeros(desired_torques.shape[0], 3, device=device)
    full_torque[:, joint_index] = desired_torques.squeeze(-1)

    # t = pinv(J^T) · τ  →  (B, 5)
    pinv = torch.tensor(_jacobian_pinv(), dtype=torch.float32, device=device)
    tensions = torch.mm(full_torque, pinv)  # (B, 3) @ (3, 5) → (B, 5)

    # Apply min/max tension bounds (thesis §3.5)
    tensions.clamp_(min=joint_spec.min_tension_n, max=joint_spec.saturation_n)

    return tensions


def tensions_to_torques(tensions: torch.Tensor, device: str) -> torch.Tensor:
    """τ = J^T · T"""
    global JACOBIAN_T_TENSOR
    if JACOBIAN_T_TENSOR is None:
        JACOBIAN_T_TENSOR = torch.tensor(JACOBIAN_T, dtype=torch.float32, device=device)
    return torch.mm(tensions, JACOBIAN_T_TENSOR.T)  # (B, 5) @ (5, 3) → (B, 3)


# ── Metrics ───────────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    amplitude_deg: float
    rise_time_ms: float | None
    overshoot_pct: float | None
    rmse_deg: float
    nrmse_pct: float


def compute_metrics(
    time_s: np.ndarray,
    actual_deg: np.ndarray,
    setpoint_deg: float,
    settle_start_s: float,
    settle_end_s: float,
) -> StepMetrics:
    """Compute rise time, overshoot and NRMSE for one step response.

    Rise time: time from 10% to 90% of setpoint (thesis §3.5).
    Overshoot: max excursion beyond setpoint (thesis §3.5).
    NRMSE: thesis Eq. 3.11–3.12.
    """
    mask = (time_s >= settle_start_s) & (time_s <= settle_end_s)
    segment = actual_deg[mask]
    segment_time = time_s[mask]

    if len(segment) == 0:
        return StepMetrics(setpoint_deg, None, None, 0.0, 0.0)

    # Rise time (10%→90%)
    threshold_10 = 0.1 * setpoint_deg
    threshold_90 = 0.9 * setpoint_deg
    crossed_10 = np.where(segment >= threshold_10)[0]
    crossed_90 = np.where(segment >= threshold_90)[0]

    rise_time_ms: float | None = None
    if len(crossed_10) > 0 and len(crossed_90) > 0:
        t10 = segment_time[crossed_10[0]]
        t90 = segment_time[crossed_90[0]]
        if t90 > t10:
            rise_time_ms = (t90 - t10) * 1000.0

    # Overshoot
    overshoot_pct: float | None = None
    if setpoint_deg > 0:
        peak = np.max(segment)
        if peak > setpoint_deg:
            overshoot_pct = (peak - setpoint_deg) / setpoint_deg * 100.0

    # RMSE / NRMSE (thesis Eq. 3.11–3.12)
    reference = np.full_like(segment, setpoint_deg)
    errors = segment - reference
    rmse = np.sqrt(np.mean(errors ** 2))
    value_range = np.max(segment) - np.min(segment)
    nrmse = (rmse / value_range * 100.0) if value_range > 1e-6 else 0.0

    return StepMetrics(
        amplitude_deg=setpoint_deg,
        rise_time_ms=rise_time_ms,
        overshoot_pct=overshoot_pct,
        rmse_deg=rmse,
        nrmse_pct=nrmse,
    )


# ── Plotting (thesis style) ──────────────────────────────────────────────

THESIS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def plot_step_responses(
    joint_name: str,
    time_per_step: list[np.ndarray],
    actual_per_step: list[np.ndarray],
    setpoint_per_step: list[np.ndarray],
    tension_per_step: list[np.ndarray],
    amplitudes_deg: list[float],
    metrics: list[StepMetrics],
    output_dir: Path,
) -> None:
    """Generate thesis-style plots for a single joint's step responses."""
    num_steps = len(amplitudes_deg)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(f"Step Response — {joint_name}", fontsize=14)

    # ── Top: angle time series ────────────────────────────────────────────
    ax_angle = axes[0]
    for idx in range(num_steps):
        color = THESIS_COLORS[idx % len(THESIS_COLORS)]
        label_actual = f"{amplitudes_deg[idx]:.0f}° actual"
        label_setpoint = f"{amplitudes_deg[idx]:.0f}° setpoint"
        ax_angle.plot(time_per_step[idx], actual_per_step[idx], color=color, label=label_actual)
        ax_angle.plot(
            time_per_step[idx], setpoint_per_step[idx],
            color=color, linestyle="--", alpha=0.6, label=label_setpoint,
        )
    ax_angle.set_ylabel("Angle [°]")
    ax_angle.set_xlabel("Time [s]")
    ax_angle.legend(fontsize=8, ncol=2)
    ax_angle.grid(True, alpha=0.3)

    # ── Bottom: tendon tensions ───────────────────────────────────────────
    ax_tension = axes[1]
    tendon_labels = ["T0 (elbow+)", "T1 (elbow−)", "T2 (wrist 0°)", "T3 (wrist 120°)", "T4 (wrist 240°)"]
    for idx in range(num_steps):
        tensions = tension_per_step[idx]  # (timesteps, 5)
        for tendon_idx in range(tensions.shape[1]):
            alpha = 0.4 + 0.2 * idx
            ax_tension.plot(
                time_per_step[idx], tensions[:, tendon_idx],
                alpha=alpha,
                label=f"{amplitudes_deg[idx]:.0f}° {tendon_labels[tendon_idx]}" if idx == 0 else None,
            )
    ax_tension.set_ylabel("Tendon Tension [N]")
    ax_tension.set_xlabel("Time [s]")
    ax_tension.legend(fontsize=7, ncol=3, loc="upper right")
    ax_tension.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"step_response_{joint_name}.png", dpi=150)
    plt.close(fig)

    # ── Metrics summary plot ──────────────────────────────────────────────
    fig2, ax_table = plt.subplots(figsize=(8, 2.5))
    ax_table.axis("off")
    table_data = []
    for m in metrics:
        table_data.append([
            f"{m.amplitude_deg:.0f}°",
            f"{m.rise_time_ms:.1f}" if m.rise_time_ms is not None else "—",
            f"{m.overshoot_pct:.1f}%" if m.overshoot_pct is not None else "—",
            f"{m.rmse_deg:.2f}°",
            f"{m.nrmse_pct:.1f}%",
        ])
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Step", "Rise Time [ms]", "Overshoot", "RMSE", "NRMSE"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    fig2.suptitle(f"Metrics — {joint_name}  (cf. Klein 2023, Table 4.2)", fontsize=11)
    plt.tight_layout()
    fig2.savefig(output_dir / f"metrics_{joint_name}.png", dpi=150)
    plt.close(fig2)


def plot_comparison_with_thesis(all_metrics: dict[str, list[StepMetrics]], output_dir: Path) -> None:
    """Bar chart comparing Isaac Sim NRMSE vs. Klein (2023) Gazebo NRMSE."""
    # Thesis Table 4.2 simulation NRMSE values (Sim vs. Real)
    thesis_nrmse = {
        "wrist_x_joint": [25.5, 9.9, 11.2],    # 10°, 20°, 30°
        "wrist_y_joint": [19.2, 11.6, 12.5],    # 10°, 20°, 30°
        "elbow_joint":   [39.1, 11.7, 12.1],    # 20°, 30°, 40°
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("NRMSE Comparison: Isaac Sim vs. Klein (2023) Gazebo Simulation", fontsize=12)

    for ax, joint_name in zip(axes, ["wrist_y_joint", "wrist_x_joint", "elbow_joint"]):
        our_metrics = all_metrics.get(joint_name, [])
        our_nrmse = [m.nrmse_pct for m in our_metrics]
        thesis_vals = thesis_nrmse.get(joint_name, [])
        amplitudes = [m.amplitude_deg for m in our_metrics]

        x = np.arange(len(amplitudes))
        width = 0.35

        ax.bar(x - width / 2, thesis_vals[:len(x)], width, label="Klein 2023 (Gazebo)", color="#ff7f0e", alpha=0.8)
        ax.bar(x + width / 2, our_nrmse[:len(x)], width, label="Isaac Sim (ours)", color="#1f77b4", alpha=0.8)

        ax.set_xlabel("Step Amplitude")
        ax.set_ylabel("NRMSE [%]")
        ax.set_title(joint_name.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.0f}°" for a in amplitudes])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "nrmse_comparison.png", dpi=150)
    plt.close(fig)


# ── Main simulation loop ─────────────────────────────────────────────────

def run_step_response_test(
    num_envs: int,
    headless: bool,
    output_dir: Path,
    step_hold_s: float = 2.5,
    return_hold_s: float = 1.0,
) -> None:
    """Execute the full step response test battery."""

    # ── Simulation setup ──────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            enable_stabilization=True,
            solver_type=1,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, 3.0, 3.0), target=(0.0, 0.0, 1.0))

    scene_cfg = StepResponseSceneCfg(num_envs=num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot: Articulation = scene["robot"]
    joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    device = robot.device
    dt = sim_cfg.dt

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, list[StepMetrics]] = {}

    # ── Run tests for each joint ──────────────────────────────────────────
    for spec in ALL_JOINT_SPECS:
        joint_index = ARM_JOINT_NAMES.index(spec.name)
        print(f"\n{'='*60}")
        print(f"Testing {spec.name} — PID(kp={spec.pid.kp}, ki={spec.pid.ki}, kd={spec.pid.kd})")
        print(f"Steps: {spec.step_amplitudes_deg}° | Tension: [{spec.min_tension_n}, {spec.saturation_n}] N")
        print(f"{'='*60}")

        time_per_step: list[np.ndarray] = []
        actual_per_step: list[np.ndarray] = []
        setpoint_per_step: list[np.ndarray] = []
        tension_per_step: list[np.ndarray] = []
        step_metrics: list[StepMetrics] = []

        for amplitude_deg in spec.step_amplitudes_deg:
            amplitude_rad = math.radians(amplitude_deg)

            # Reset robot to zero
            zero_pos = torch.zeros(num_envs, len(joint_ids), device=device)
            zero_vel = torch.zeros(num_envs, len(joint_ids), device=device)
            robot.write_joint_position_to_sim(zero_pos, joint_ids=joint_ids)
            robot.write_joint_velocity_to_sim(zero_vel, joint_ids=joint_ids)
            scene.write_data_to_sim()

            # Warm-up: let the robot settle for 0.5 s
            warmup_steps = int(0.5 / dt)
            for _ in range(warmup_steps):
                robot.set_joint_effort_target(
                    torch.zeros(num_envs, len(joint_ids), device=device),
                    joint_ids=joint_ids,
                )
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)

            # PID state
            pid_state = PIDState()
            pid_state.reset(device)

            total_time = step_hold_s + return_hold_s
            num_steps_total = int(total_time / dt)

            recorded_time = []
            recorded_angle = []
            recorded_setpoint = []
            recorded_tensions = []

            for step_idx in range(num_steps_total):
                current_time = step_idx * dt

                # Set-point: step up for step_hold_s, then back to 0
                setpoint_rad = amplitude_rad if current_time < step_hold_s else 0.0
                setpoint_deg_now = math.degrees(setpoint_rad)

                # Read current joint angles
                joint_pos = robot.data.joint_pos[:, joint_ids]  # (B, 3)
                current_rad = joint_pos[:, joint_index:joint_index + 1]  # (B, 1)
                current_deg = torch.rad2deg(current_rad).cpu().numpy().mean()

                # PID torque
                pid_torque = compute_pid_torque(
                    setpoint_rad, current_rad.mean(), dt, spec.pid, pid_state,
                )

                # Gravity compensation (thesis Eq. 3.2)
                grav_torque = gravity_compensation_torque(
                    current_rad.mean(), spec.mass_kg, spec.gravity_arm_m,
                )

                total_torque = pid_torque + grav_torque

                # Map torque to tensions
                tensions = torque_to_tensions(
                    total_torque.unsqueeze(0), spec, joint_index, device,
                )

                # Map tensions back to joint torques and apply
                joint_torques = tensions_to_torques(tensions, device)
                # Broadcast to all envs
                joint_torques_batch = joint_torques.expand(num_envs, -1)

                robot.set_joint_effort_target(joint_torques_batch, joint_ids=joint_ids)
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)

                recorded_time.append(current_time)
                recorded_angle.append(current_deg)
                recorded_setpoint.append(setpoint_deg_now)
                recorded_tensions.append(tensions[0].cpu().numpy().copy())

            time_arr = np.array(recorded_time)
            angle_arr = np.array(recorded_angle)
            setpoint_arr = np.array(recorded_setpoint)
            tension_arr = np.stack(recorded_tensions)

            time_per_step.append(time_arr)
            actual_per_step.append(angle_arr)
            setpoint_per_step.append(setpoint_arr)
            tension_per_step.append(tension_arr)

            # Compute metrics for the step-up phase
            metrics = compute_metrics(
                time_arr, angle_arr, amplitude_deg,
                settle_start_s=0.0,
                settle_end_s=step_hold_s,
            )
            step_metrics.append(metrics)

            print(f"  {amplitude_deg:5.1f}° → Rise: {metrics.rise_time_ms or '—':>8} ms | "
                  f"Overshoot: {f'{metrics.overshoot_pct:.1f}%' if metrics.overshoot_pct else '—':>8} | "
                  f"RMSE: {metrics.rmse_deg:6.2f}° | NRMSE: {metrics.nrmse_pct:5.1f}%")

        all_metrics[spec.name] = step_metrics

        # Generate plots
        plot_step_responses(
            spec.name, time_per_step, actual_per_step, setpoint_per_step,
            tension_per_step, spec.step_amplitudes_deg, step_metrics, output_dir,
        )

    # ── Comparison plot ───────────────────────────────────────────────────
    plot_comparison_with_thesis(all_metrics, output_dir)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — Step Response Test Results")
    print(f"{'='*60}")
    print(f"{'Joint':<18} {'Step':>6} {'Rise [ms]':>10} {'Overshoot':>10} {'RMSE [°]':>10} {'NRMSE [%]':>10}")
    print("-" * 70)
    for joint_name, metrics_list in all_metrics.items():
        for m in metrics_list:
            rise_str = f"{m.rise_time_ms:.1f}" if m.rise_time_ms is not None else "—"
            overshoot_str = f"{m.overshoot_pct:.1f}%" if m.overshoot_pct is not None else "—"
            print(f"{joint_name:<18} {m.amplitude_deg:>5.0f}° {rise_str:>10} {overshoot_str:>10} "
                  f"{m.rmse_deg:>10.2f} {m.nrmse_pct:>10.1f}")
    print(f"\nPlots saved to: {output_dir.resolve()}")

    # Save numerical results as CSV
    csv_path = output_dir / "step_response_results.csv"
    with open(csv_path, "w") as f:
        f.write("joint,amplitude_deg,rise_time_ms,overshoot_pct,rmse_deg,nrmse_pct\n")
        for joint_name, metrics_list in all_metrics.items():
            for m in metrics_list:
                f.write(f"{joint_name},{m.amplitude_deg},{m.rise_time_ms or ''},{m.overshoot_pct or ''},"
                        f"{m.rmse_deg},{m.nrmse_pct}\n")
    print(f"CSV saved to: {csv_path.resolve()}")

    sim.stop()


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Tendon step-response test (Klein 2023 methodology)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument(
        "--output-dir", type=str,
        default="/home/robot/studentische-arbeiten/src/tensegrity_pick/scripts/step_response_results",
        help="Directory for output plots and CSV",
    )
    parser.add_argument("--step-hold", type=float, default=2.5, help="Hold time per step [s]")
    parser.add_argument("--return-hold", type=float, default=1.0, help="Return-to-zero hold time [s]")

    # IsaacLab passes extra args; use parse_known_args to be tolerant
    args, _ = parser.parse_known_args()

    run_step_response_test(
        num_envs=args.num_envs,
        headless=args.headless,
        output_dir=Path(args.output_dir),
        step_hold_s=args.step_hold,
        return_hold_s=args.return_hold,
    )


if __name__ == "__main__":
    main()

"""Base joint PD step-response validation for the 5-DOF tensegrity robot.

Validates the ImplicitActuatorCfg PD drives on the two prismatic base joints
(base_y_joint, base_z_joint) using ``TENS_5DOF_GRIPPER_CFG``.  The arm hangs
under gravity during these tests — the base must remain stable despite the
suspended load.

Klein (2023) criteria adapted for prismatic joints:
 - Near-critical damping  (ζ ≈ 0.8–2.0)
 - Settling time          ≤ 0.5 s  (relaxed from 0.3 s for loaded base)
 - Overshoot              < 5 %
 - Steady-state error     < 5 %  (gravity offset acceptable for Z axis)

Protocol
--------
Phase 1 — Standard step response
    Both base joints at 3 amplitudes each.  Y axis tests horizontal
    motion (no gravity coupling), Z axis tests vertical motion under
    the full arm + gripper mass.

Phase 2 — Damping sweep  (``--damping_sweep`` flag)
    base_y_joint only; damping swept to find the optimal value.

Usage
-----
.. code-block:: bash

    cd src/tensegrity_pick
    conda run -n env_isaaclab python3 scripts/model_validation/run_step_response_base.py \\
        --headless [--num_envs 1] [--damping_sweep]

References
----------
* Klein (2023) §4.1 — gain-tuning criteria
* doc/pd_tuning_results.md — previously validated gains (K=8000, D=800)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Base PD step-response validation")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Override output directory (default: outputs/model_validation/base/data)",
)
parser.add_argument(
    "--damping_sweep", action="store_true",
    help="Run base_y gain sweep (D = 1600, 1200, 800, 600, 400, 200, K = 8000 fixed)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_JOINT_NAMES = ["base_y_joint", "base_z_joint"]
ALL_CONTROLLED_JOINTS = ["base_y_joint", "base_z_joint", "elbow_joint", "wrist_x_joint", "wrist_y_joint"]

BASE_DATA_DIR = common.DEFAULT_OUTPUT_ROOT / "base" / "data"

# Joint limits from USD: base_y [-0.5, 0.5], base_z [-0.5, 0.0]
# base_z rest at -0.25 m (centre of travel, matches tune_pd_gains.py DEFAULT_BASE_Z)
BASE_Y_REST_M = 0.0
BASE_Z_REST_M = -0.25

# Base step amplitudes in meters (prismatic joints)
# Steps are *added* to the rest position, keeping within joint limits.
BASE_Y_AMPLITUDES_M = (0.05, 0.10, 0.20)
BASE_Z_AMPLITUDES_M = (0.03, 0.06, 0.10)

# Damping sweep candidates for base_y
BASE_DAMPING_SWEEP = [1600.0, 1200.0, 800.0, 600.0, 400.0, 200.0]
BASE_K_FIXED = 8000.0

# Relaxed settling criterion for the loaded base
BASE_SETTLING_LIMIT_MS = 500.0
BASE_OVERSHOOT_LIMIT_PCT = 5.0

# Gravity feedforward for base_z: the arm weight (6.81 kg) creates a
# constant downward force that offsets the PD equilibrium by m·g/K.
# Shifting the position target UP by this amount compensates the offset.
BASE_Z_GRAVITY_OFFSET_M = 6.81 * 9.81 / BASE_K_FIXED  # ≈ 0.00835 m

MOUNT_HEIGHT_M = 2.30  # matches reach env TENSEGRITY_MOUNT_HEIGHT_M


# ── Scene configuration ────────────────────────────────────────────────────────

def _make_robot_cfg(
    base_stiffness: float = 8000.0,
    base_damping: float = 800.0,
) -> ArticulationCfg:
    """Clone TENS_5DOF_GRIPPER_CFG with given base PD gains."""
    return TENS_5DOF_GRIPPER_CFG.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, MOUNT_HEIGHT_M),
            joint_pos={
                "base_y_joint": BASE_Y_REST_M,
                "base_z_joint": BASE_Z_REST_M,
                "elbow_joint": 0.0,
                "wrist_y_joint": 0.0,
                "wrist_x_joint": 0.0,
                "finger_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "left_outer_finger_joint": 0.0,
                "right_outer_finger_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
                "left_inner_finger_pad_joint": 0.0,
                "right_inner_finger_pad_joint": 0.0,
            },
        ),
        actuators={
            **{k: v for k, v in TENS_5DOF_GRIPPER_CFG.actuators.items() if k != "base"},
            "base": ImplicitActuatorCfg(
                joint_names_expr=BASE_JOINT_NAMES,
                effort_limit_sim=800.0,
                velocity_limit_sim=5.0,
                stiffness=base_stiffness,
                damping=base_damping,
            ),
        },
    )


@configclass
class BaseValidationSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + 5-DOF tensegrity robot."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    robot: ArticulationCfg = _make_robot_cfg().replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


# ── Runtime gain writing ───────────────────────────────────────────────────────

def _write_base_gains(
    robot: Articulation,
    stiffness: float,
    damping: float,
) -> None:
    """Update PhysX drive gains for both base joints at runtime."""
    joint_ids_list = [robot.data.joint_names.index(n) for n in BASE_JOINT_NAMES]
    ids_t = torch.tensor(joint_ids_list, device=robot.device)
    n = robot.num_instances
    k_t = torch.full((n, 2), stiffness, device=robot.device, dtype=torch.float32)
    d_t = torch.full((n, 2), damping, device=robot.device, dtype=torch.float32)
    robot.write_joint_stiffness_to_sim(k_t, joint_ids=ids_t)
    robot.write_joint_damping_to_sim(d_t, joint_ids=ids_t)


# ── Robot reset + warm-up ─────────────────────────────────────────────────────

def _reset_and_warmup(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    all_joint_ids: list[int],
    rest_positions: torch.Tensor,
    num_envs: int,
    device: str,
) -> None:
    """Hard-reset joints to rest positions and hold for WARMUP_S."""
    zero_vel = torch.zeros(num_envs, len(all_joint_ids), device=device, dtype=torch.float32)
    robot.write_joint_position_to_sim(rest_positions, joint_ids=all_joint_ids)
    robot.write_joint_velocity_to_sim(zero_vel, joint_ids=all_joint_ids)
    scene.write_data_to_sim()

    warmup_steps = int(common.WARMUP_S / common.SIM_DT)
    for _ in range(warmup_steps):
        robot.set_joint_position_target(rest_positions, joint_ids=all_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(common.SIM_DT)


# ── Single trial ──────────────────────────────────────────────────────────────

def run_one_trial(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    all_joint_ids: list[int],
    rest_positions: torch.Tensor,
    base_joint_idx_in_all: int,
    rest_m: float,
    num_envs: int,
    device: str,
    joint_name: str,
    amplitude_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one position-step trial for a single base joint.

    The step is applied as ``rest + amplitude``.  Displacement is recorded
    relative to rest, then converted to mm for metrics.

    Returns
    -------
    time_s, displacement_mm, setpoint_mm
        Displacement from rest position, in millimetres.
    """
    dt = common.SIM_DT

    _reset_and_warmup(robot, scene, sim, all_joint_ids, rest_positions, num_envs, device)

    step_steps = int(common.STEP_HOLD_S / dt)
    return_steps = int(common.RETURN_HOLD_S / dt)
    total_steps = step_steps + return_steps

    times: list[float] = []
    displacements_mm: list[float] = []
    setpoints_mm: list[float] = []

    for i in range(total_steps):
        t = i * dt
        step_target_m = rest_m + amplitude_m if i < step_steps else rest_m

        # Gravity feedforward for base_z: shift target to compensate m·g/K offset
        if joint_name == "base_z_joint":
            step_target_m += BASE_Z_GRAVITY_OFFSET_M

        pos_all = robot.data.joint_pos[:, all_joint_ids]
        cur_m = float(pos_all[0, base_joint_idx_in_all].item())

        target = rest_positions.clone()
        target[:, base_joint_idx_in_all] = step_target_m
        robot.set_joint_position_target(target, joint_ids=all_joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

        times.append(t)
        displacements_mm.append((cur_m - rest_m) * 1000.0)
        setpoints_mm.append((step_target_m - rest_m) * 1000.0)

    return (
        np.array(times, dtype=np.float32),
        np.array(displacements_mm, dtype=np.float32),
        np.array(setpoints_mm, dtype=np.float32),
    )


# ── Run all trials ────────────────────────────────────────────────────────────

def run_all_trials(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    all_joint_ids: list[int],
    rest_positions: torch.Tensor,
    base_y_idx: int,
    base_z_idx: int,
    num_envs: int,
    device: str,
    output_dir: Path,
    damping: float | None = None,
) -> dict[str, list[common.StepMetrics]]:
    """Run step responses for both base joints; save NPZ per trial."""
    all_metrics: dict[str, list[common.StepMetrics]] = {}
    is_sweep = damping is not None

    joint_configs = [
        ("base_y_joint", base_y_idx, BASE_Y_REST_M, BASE_Y_AMPLITUDES_M),
        ("base_z_joint", base_z_idx, BASE_Z_REST_M, BASE_Z_AMPLITUDES_M),
    ]

    for joint_name, joint_idx, rest_m, amplitudes in joint_configs:
        metrics_list: list[common.StepMetrics] = []
        for amp_m in amplitudes:
            amp_mm = amp_m * 1000.0
            print(f"    {joint_name:22s}  {amp_mm:5.0f} mm  ...", end="", flush=True)

            time_s, displacement_mm, setpoint_mm = run_one_trial(
                robot, scene, sim, all_joint_ids,
                rest_positions, joint_idx, rest_m,
                num_envs, device,
                joint_name, amp_m,
            )

            # compute_step_metrics works generically — "deg" = mm here
            metrics = common.compute_step_metrics(
                time_s, displacement_mm, amp_mm,
                step_start_s=0.0,
                step_end_s=common.STEP_HOLD_S,
                joint_name=joint_name,
            )
            metrics_list.append(metrics)

            # Save NPZ (reuse common.save_trial with dummy tensions)
            tensions_dummy = np.zeros((len(time_s), 5), dtype=np.float32)
            path = common.trial_path(
                output_dir, joint_name, amp_mm,
                damping_value=damping if is_sweep else None,
            )
            common.save_trial(
                path, joint_name, amp_mm,
                time_s, displacement_mm, setpoint_mm, tensions_dummy,
                metrics, model_type="base_pd",
                damping_value=damping if is_sweep else None,
            )

            rise_s = f"{metrics.rise_time_ms:.0f} ms" if metrics.rise_time_ms else "—"
            settle_s = f"{metrics.settling_time_ms:.0f} ms" if metrics.settling_time_ms else "—"
            passed = (
                (metrics.settling_time_ms is not None and metrics.settling_time_ms <= BASE_SETTLING_LIMIT_MS)
                and (metrics.overshoot_pct is None or metrics.overshoot_pct <= BASE_OVERSHOOT_LIMIT_PCT)
            )
            print(
                f"  rise={rise_s:>8}  settle={settle_s:>8}"
                f"  NRMSE={metrics.nrmse_pct:5.1f}%"
                f"  SS_err={metrics.steady_state_error_deg:.2f} mm"
                f"  {'✓' if passed else '✗'}"
            )

        all_metrics[joint_name] = metrics_list

    return all_metrics


# ── Gain sweep ────────────────────────────────────────────────────────────────

def run_gain_sweep(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    all_joint_ids: list[int],
    rest_positions: torch.Tensor,
    base_y_idx: int,
    base_z_idx: int,
    num_envs: int,
    device: str,
    output_dir: Path,
) -> dict[float, dict[str, list[common.StepMetrics]]]:
    """Sweep base damping and test both joints at each setting."""
    sweep_results: dict[float, dict[str, list[common.StepMetrics]]] = {}

    print("\n  ── Gain Sweep (Base Joints, K=8000 fixed) ──")
    for d_val in BASE_DAMPING_SWEEP:
        print(f"\n    D = {d_val:.0f}")
        _write_base_gains(robot, BASE_K_FIXED, d_val)

        metrics = run_all_trials(
            robot, scene, sim, all_joint_ids,
            rest_positions,
            base_y_idx, base_z_idx,
            num_envs, device, output_dir,
            damping=d_val,
        )
        sweep_results[d_val] = metrics

    # Restore original gains
    _write_base_gains(robot, BASE_K_FIXED, 800.0)
    return sweep_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else BASE_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  Base PD Step-Response Validation")
    print(f"  Robot:  TENS_5DOF_GRIPPER_CFG  (K={BASE_K_FIXED:.0f}, D=800)")
    print(f"  Mount:  z = {MOUNT_HEIGHT_M:.2f} m")
    print(f"  Sim dt: {common.SIM_DT:.6f} s ({1.0/common.SIM_DT:.0f} Hz)")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    # Build scene
    sim_cfg = sim_utils.SimulationCfg(dt=common.SIM_DT, render_interval=common.RENDER_INTERVAL)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.0, 0.0, 3.0], target=[0.0, 0.0, MOUNT_HEIGHT_M])

    scene = InteractiveScene(BaseValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0))
    sim.reset()
    scene.update(common.SIM_DT)

    robot: Articulation = scene["robot"]
    num_envs = args_cli.num_envs
    device = sim.device

    # Resolve joint indices for ALL controlled joints
    all_joint_ids = [robot.data.joint_names.index(n) for n in ALL_CONTROLLED_JOINTS]
    base_y_idx_in_all = ALL_CONTROLLED_JOINTS.index("base_y_joint")
    base_z_idx_in_all = ALL_CONTROLLED_JOINTS.index("base_z_joint")

    # Build rest-position tensor (base at rest, arm at zero)
    rest_positions = torch.zeros(num_envs, len(all_joint_ids), device=device, dtype=torch.float32)
    rest_positions[:, base_y_idx_in_all] = BASE_Y_REST_M
    rest_positions[:, base_z_idx_in_all] = BASE_Z_REST_M

    print(f"\n  Joint names: {robot.data.joint_names}")
    print(f"  All controlled IDs: {all_joint_ids}")
    print(f"  base_y idx={base_y_idx_in_all} rest={BASE_Y_REST_M:.2f} m, "
          f"base_z idx={base_z_idx_in_all} rest={BASE_Z_REST_M:.2f} m")

    # Phase 1: Standard step response
    print("\n  ── Phase 1: Standard Step Response (K=8000, D=800) ──\n")
    standard_metrics = run_all_trials(
        robot, scene, sim, all_joint_ids,
        rest_positions,
        base_y_idx_in_all, base_z_idx_in_all,
        num_envs, device, output_dir,
    )

    # Summary
    print("\n  ── Summary ──")
    total = 0
    passed = 0
    for joint_name, metrics_list in standard_metrics.items():
        for m in metrics_list:
            total += 1
            settle_ok = m.settling_time_ms is not None and m.settling_time_ms <= BASE_SETTLING_LIMIT_MS
            over_ok = m.overshoot_pct is None or m.overshoot_pct <= BASE_OVERSHOOT_LIMIT_PCT
            if settle_ok and over_ok:
                passed += 1
    print(f"  Passed: {passed}/{total}")

    # Phase 2: Damping sweep (optional)
    if args_cli.damping_sweep:
        sweep_results = run_gain_sweep(
            robot, scene, sim, all_joint_ids,
            rest_positions,
            base_y_idx_in_all, base_z_idx_in_all,
            num_envs, device, output_dir,
        )

        print("\n  ── Sweep Summary (base_y_joint, 100 mm step) ──")
        print(f"  {'D':>8s}  {'Settle (ms)':>12s}  {'Overshoot (%)':>14s}  {'NRMSE (%)':>10s}")
        for d_val, metrics_dict in sweep_results.items():
            if "base_y_joint" in metrics_dict:
                y_metrics = metrics_dict["base_y_joint"]
                # Use the 100 mm (0.10 m) trial for comparison
                for m in y_metrics:
                    if abs(m.amplitude_deg - 100.0) < 1.0:
                        settle_str = f"{m.settling_time_ms:.0f}" if m.settling_time_ms else "—"
                        over_str = f"{m.overshoot_pct:.1f}" if m.overshoot_pct else "0.0"
                        print(f"  {d_val:>8.0f}  {settle_str:>12s}  {over_str:>14s}  {m.nrmse_pct:>10.1f}")

    # simulation_app.close() holds the GIL in C++ and hangs indefinitely.
    # All data has been written; let the OS clean up GPU/memory on exit.
    import os
    os._exit(0)


if __name__ == "__main__":
    main()

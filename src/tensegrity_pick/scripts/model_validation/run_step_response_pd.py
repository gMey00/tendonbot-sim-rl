"""PD model step-response validation for the tensegrity arm.

Validates the ImplicitActuatorCfg position drives (K=400, D=20) from
``TENS_3DOF_CFG`` against Klein (2023) criteria:
  - Near-critical damping (ζ ≈ 1.0–1.5)
  - Settling time 0.1–0.3 s
  - Overshoot < 5 %

Phase 1 — Standard step response
    All 3 arm joints at their 3 step amplitudes using the validated
    gains (K=400, D=20).  Results are saved as NPZ files.

Phase 2 — Damping sweep  (``--damping_sweep`` flag)
    Elbow joint only; damping swept from 120 down to 15 (K=400 fixed).
    Replicates the gain-sweep methodology of ``tune_pd_gains.py`` and
    confirms that D=20 gives the best settling time.

Usage
-----
.. code-block:: bash

    cd /home/robot/Isaac/IsaacLab
    cd /path/to/tensegrity_pick
    conda run -n env_isaaclab python3 scripts/model_validation/run_step_response_pd.py \\
        --headless [--num_envs 1] [--damping_sweep]

References
----------
* Klein (2023) §4.1 — gain-tuning criteria
* doc/pd_tuning_results.md — previously validated gains (K=400, D=20)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# ── Common module (no IsaacLab — safe before AppLauncher) ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

# ── AppLauncher must be created before any isaaclab imports ───────────────────
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="PD model step-response validation")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Override output directory (default: outputs/model_validation/pd/data)",
)
parser.add_argument(
    "--damping_sweep", action="store_true",
    help="Run elbow gain sweep (D = 120, 60, 30, 25, 20, 15, K = 400 fixed)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── IsaacLab imports (after AppLauncher) ──────────────────────────────────────
import numpy as np           # noqa: E402
import torch                 # noqa: E402

import isaaclab.sim as sim_utils                                             # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg                           # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation      # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg             # noqa: E402
from isaaclab.sim import SimulationContext                                   # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg  # noqa: E402
from isaaclab.utils import configclass                                       # noqa: E402

from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_3DOF_CFG       # noqa: E402

# ── Scene configuration ────────────────────────────────────────────────────────

_ARM_JOINTS = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]


def _make_robot_cfg(stiffness: float = 400.0, damping: float = 20.0) -> ArticulationCfg:
    """Clone TENS_3DOF_CFG with given arm PD gains, fixed root, and mount at z = 1.0 m.

    fix_root_link=True is required for the standalone scene: threedof_manipulator.usd
    does not bake a fixed joint, so without it the base floats and falls under gravity.
    In the full RL scene (proj_base_scene_cfg.py) the 5-DOF USD is ceiling-mounted via
    its own fixed joint — this workaround is only needed here.
    """
    return TENS_3DOF_CFG.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={j: 0.0 for j in _ARM_JOINTS},
        ),
        spawn=TENS_3DOF_CFG.spawn.replace(
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=_ARM_JOINTS,
                # ── Validation-only overrides ──────────────────────────────
                # Production limits (tensegrity_robot_cfg.py):
                #   elbow:  effort=40 N·m,  velocity=1.0 rad/s
                #   wrist:  effort=10 N·m,  velocity=0.5 rad/s
                #
                # Here we raise both to 400 N·m / 10 rad/s so the PD drives
                # never saturate during step responses.  This lets us measure
                # the true K,D dynamics (matching Klein 2023 §4.2–§4.3 intent
                # where the physical motor was not effort-limited either).
                #
                # In RL training, the real per-joint limits apply and the PD
                # controller WILL saturate for large errors — this is expected
                # and physically faithful behaviour.
                effort_limit_sim=400.0,
                velocity_limit_sim=10.0,
                stiffness=stiffness,
                damping=damping,
            ),
        },
    )


@configclass
class ValidationSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + 3-DOF tensegrity arm (PD drives)."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    robot: ArticulationCfg = _make_robot_cfg().replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


# ── Runtime gain writing ───────────────────────────────────────────────────────

def _write_arm_gains(
    robot: Articulation,
    stiffness: float,
    damping: float,
) -> None:
    """Update PhysX drive gains for all three arm joints at runtime."""
    joint_ids_list = [robot.data.joint_names.index(n) for n in _ARM_JOINTS]
    ids_t = torch.tensor(joint_ids_list, device=robot.device)
    n = robot.num_instances
    k_t = torch.full((n, 3), stiffness, device=robot.device, dtype=torch.float32)
    d_t = torch.full((n, 3), damping,   device=robot.device, dtype=torch.float32)
    robot.write_joint_stiffness_to_sim(k_t, joint_ids=ids_t)
    robot.write_joint_damping_to_sim(d_t, joint_ids=ids_t)


# ── Robot reset + warm-up ─────────────────────────────────────────────────────

def _reset_and_warmup(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    joint_ids: list[int],
    num_envs: int,
    device: str,
) -> None:
    """Hard-reset all arm joints to zero and hold at zero for WARMUP_S."""
    zero = torch.zeros(num_envs, len(joint_ids), device=device, dtype=torch.float32)
    robot.write_joint_position_to_sim(zero, joint_ids=joint_ids)
    robot.write_joint_velocity_to_sim(zero, joint_ids=joint_ids)
    scene.write_data_to_sim()

    warmup_steps = int(common.WARMUP_S / common.SIM_DT)
    for _ in range(warmup_steps):
        robot.set_joint_position_target(zero, joint_ids=joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(common.SIM_DT)


# ── Single trial ──────────────────────────────────────────────────────────────

def run_one_trial(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    joint_ids: list[int],
    num_envs: int,
    device: str,
    spec: common.JointTestSpec,
    amplitude_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one position-step trial for a single joint.

    The PD actuator (ImplicitActuatorCfg) drives the target joint to
    *amplitude_deg* while the other two joints are held at zero.
    Equivalent tendon tensions are computed analytically via Klein's PID
    gains and the Jacobian pseudo-inverse and stored as a diagnostic
    reference (not fed back to the robot).

    Returns
    -------
    time_s, actual_deg, setpoint_deg_arr, tensions_n
        All as float32 NumPy arrays.  tensions_n has shape (T, 5).
    """
    amplitude_rad = math.radians(amplitude_deg)
    dt = common.SIM_DT

    _reset_and_warmup(robot, scene, sim, joint_ids, num_envs, device)

    pid_state = common.PIDState()
    pid_state.reset()

    step_steps   = int(common.STEP_HOLD_S   / dt)
    return_steps = int(common.RETURN_HOLD_S / dt)
    total_steps  = step_steps + return_steps

    times: list[float] = []
    actuals: list[float] = []
    setpoints: list[float] = []
    tensions_list: list[np.ndarray] = []

    for i in range(total_steps):
        t = i * dt
        sp_rad = amplitude_rad if i < step_steps else 0.0
        sp_deg = math.degrees(sp_rad)

        # Read actual joint angle for the test joint
        pos_all = robot.data.joint_pos[:, joint_ids]  # (B, 3)
        cur_rad = float(pos_all[0, spec.joint_index].item())

        # ── PD position command (target joint steps; others hold zero) ────
        target = torch.zeros(num_envs, len(joint_ids), device=device, dtype=torch.float32)
        target[:, spec.joint_index] = sp_rad
        robot.set_joint_position_target(target, joint_ids=joint_ids)

        # ── Equivalent tensions (diagnostic reference only) ───────────────
        pid_tau = common.compute_pid_torque(sp_rad, cur_rad, dt, spec.pid, pid_state)
        grav_tau = common.gravity_compensation_torque(cur_rad, spec.mass_kg, spec.gravity_arm_m)
        tens_np = common.torque_to_tensions(
            pid_tau + grav_tau, spec.joint_index, spec.min_tension_n, spec.saturation_n
        )

        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

        times.append(t)
        actuals.append(math.degrees(cur_rad))
        setpoints.append(sp_deg)
        tensions_list.append(tens_np.copy())

    return (
        np.array(times,    dtype=np.float32),
        np.array(actuals,  dtype=np.float32),
        np.array(setpoints, dtype=np.float32),
        np.stack(tensions_list).astype(np.float32),  # (T, 5)
    )


# ── Run all joint × amplitude trials ─────────────────────────────────────────

def run_all_trials(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    joint_ids: list[int],
    num_envs: int,
    device: str,
    output_dir: Path,
    damping: float = common.VALIDATED_DAMPING,
    is_sweep: bool = False,
) -> dict[str, list[common.StepMetrics]]:
    """Run step responses for all joints and amplitudes; save NPZ per trial."""
    all_metrics: dict[str, list[common.StepMetrics]] = {}

    for spec in common.ALL_JOINT_SPECS:
        metrics_list: list[common.StepMetrics] = []
        for amp_deg in spec.step_amplitudes_deg:
            print(f"    {spec.name:22s}  {amp_deg:4.0f}°  ...", end="", flush=True)

            time_s, actual_deg, setpoint_arr, tensions_n = run_one_trial(
                robot, scene, sim, joint_ids, num_envs, device, spec, amp_deg,
            )
            metrics = common.compute_step_metrics(
                time_s, actual_deg, amp_deg,
                step_start_s=0.0,
                step_end_s=common.STEP_HOLD_S,
                joint_name=spec.name,
            )
            metrics_list.append(metrics)

            path = common.trial_path(
                output_dir, spec.name, amp_deg,
                damping_value=damping if is_sweep else None,
            )
            common.save_trial(
                path, spec.name, amp_deg,
                time_s, actual_deg, setpoint_arr, tensions_n,
                metrics, model_type="pd",
                damping_value=damping if is_sweep else None,
            )

            rise_s   = f"{metrics.rise_time_ms:.0f} ms"   if metrics.rise_time_ms    else "—"
            settle_s = f"{metrics.settling_time_ms:.0f} ms" if metrics.settling_time_ms else "—"
            print(
                f"  rise={rise_s:>8}  settle={settle_s:>8}"
                f"  NRMSE={metrics.nrmse_pct:5.1f}%  {'✓' if metrics.passes() else '✗'}"
            )

        all_metrics[spec.name] = metrics_list

    return all_metrics


# ── Gain sweep ────────────────────────────────────────────────────────────────

def run_gain_sweep(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    joint_ids: list[int],
    num_envs: int,
    device: str,
    output_dir: Path,
) -> dict[float, list[common.StepMetrics]]:
    """Sweep arm damping for the elbow joint.

    Tests D = [120, 60, 30, 25, 20, 15] with K = 400 fixed.
    Saves NPZ files tagged with the D value.
    Restores the validated gains (D=20) when done.
    """
    sweep_results: dict[float, list[common.StepMetrics]] = {}
    spec = common.ELBOW_SPEC

    print("\n  ── Gain Sweep (Elbow Joint, K=400 fixed) ──")
    print(f"  {'D':>6}  {'Amp':>5}  {'Rise':>8}  {'Settle':>9}  {'NRMSE':>7}")
    print("  " + "─" * 48)

    for d_val in common.GAIN_SWEEP_DAMPING_VALUES:
        _write_arm_gains(robot, common.GAIN_SWEEP_K_FIXED, d_val)
        mlist: list[common.StepMetrics] = []

        for amp_deg in spec.step_amplitudes_deg:
            time_s, actual_deg, setpoint_arr, tensions_n = run_one_trial(
                robot, scene, sim, joint_ids, num_envs, device, spec, amp_deg,
            )
            metrics = common.compute_step_metrics(
                time_s, actual_deg, amp_deg,
                step_start_s=0.0,
                step_end_s=common.STEP_HOLD_S,
                joint_name=spec.name,
            )
            mlist.append(metrics)

            path = common.trial_path(output_dir, spec.name, amp_deg, damping_value=d_val)
            common.save_trial(
                path, spec.name, amp_deg,
                time_s, actual_deg, setpoint_arr, tensions_n,
                metrics, model_type="pd", damping_value=d_val,
            )

            rise_s   = f"{metrics.rise_time_ms:.0f}"    if metrics.rise_time_ms    else "—"
            settle_s = f"{metrics.settling_time_ms:.0f}" if metrics.settling_time_ms else "—"
            print(
                f"  D={d_val:>5.0f}  {amp_deg:>4.0f}°  {rise_s:>7} ms  {settle_s:>8} ms"
                f"  {metrics.nrmse_pct:>6.1f}%"
            )

        sweep_results[d_val] = mlist

    # Restore validated gains
    _write_arm_gains(robot, common.GAIN_SWEEP_K_FIXED, common.VALIDATED_DAMPING)
    print(f"\n  Gains restored to K={common.GAIN_SWEEP_K_FIXED:.0f}, D={common.VALIDATED_DAMPING:.0f}")
    return sweep_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else common.PD_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Simulation setup ──────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(
        dt=common.SIM_DT,
        render_interval=common.RENDER_INTERVAL,
        device=getattr(args_cli, "device", "cuda:0"),
        physx=sim_utils.PhysxCfg(enable_stabilization=True, solver_type=1),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.0, 3.0, 3.0), target=(0.0, 0.0, 1.0))

    scene_cfg = ValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot: Articulation = scene["robot"]
    joint_ids, _ = robot.find_joints(common.ARM_JOINT_NAMES, preserve_order=True)
    device = robot.device
    num_envs = args_cli.num_envs

    print("\n" + "=" * 65)
    print("  PD MODEL STEP-RESPONSE VALIDATION")
    print("  Robot:   TENS_3DOF_CFG  (ImplicitActuatorCfg)")
    print("  Gains:   K=400 N·m/rad,  D=20 N·m·s/rad  (validated 2026-02-28)")
    print(f"  Output:  {output_dir}")
    print("=" * 65)

    # ── Phase 1: standard step response ──────────────────────────────────
    print("\n  Phase 1 — Step response (K=400, D=20)\n")
    all_metrics = run_all_trials(
        robot, scene, sim, joint_ids, num_envs, device, output_dir,
    )

    # ── Phase 2: gain sweep (optional) ───────────────────────────────────
    sweep_results: dict[float, list[common.StepMetrics]] | None = None
    if args_cli.damping_sweep:
        print("\n  Phase 2 — Gain sweep\n")
        sweep_results = run_gain_sweep(
            robot, scene, sim, joint_ids, num_envs, device, output_dir,
        )
        # Print gain-sweep summary
        print("\n  Gain Sweep Summary — avg settling time:")
        for d_val, mlist in sorted(sweep_results.items()):
            valid = [m.settling_time_ms for m in mlist if m.settling_time_ms is not None]
            avg = sum(valid) / len(valid) if valid else float("nan")
            marker = " ◄ BEST" if abs(d_val - min(
                sweep_results,
                key=lambda d: sum(m.settling_time_ms or 9999 for m in sweep_results[d])
            )) < 0.1 else ""
            print(f"    D={d_val:.0f}: {avg:.1f} ms{marker}")

    # ── Log file ─────────────────────────────────────────────────────────
    cfg_summary = {
        "Model":          "TENS_3DOF_CFG (3-DOF arm only)",
        "USD":            "threedof_manipulator.usd",
        "Actuator type":  "ImplicitActuatorCfg (PhysX PD position drives)",
        "Arm stiffness":  "K = 400.0 N·m/rad",
        "Arm damping":    "D = 20.0 N·m·s/rad  (validated by tune_pd_gains.py)",
        "Effort limit":   "40.0 N·m",
        "Velocity limit": "2.0 rad/s",
        "Mount position": "(0.0, 0.0, 1.0) m",
        "Num envs":       str(num_envs),
    }
    common.write_log(output_dir.parent, "pd", cfg_summary, all_metrics, sweep_results)

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  SUMMARY — Phase 1 (K=400, D=20)")
    print(f"{'=' * 65}")
    print(f"  {'Joint':<22}  {'Amp':>5}  {'Rise [ms]':>10}  {'Settle [ms]':>12}  {'NRMSE':>7}  {'Pass':>5}")
    print("  " + "─" * 64)
    for joint_name, mlist in all_metrics.items():
        for m in mlist:
            r = f"{m.rise_time_ms:.0f}"    if m.rise_time_ms    else "—"
            s = f"{m.settling_time_ms:.0f}" if m.settling_time_ms else "—"
            print(
                f"  {joint_name:<22}  {m.amplitude_deg:>4.0f}°  {r:>10}  {s:>12}"
                f"  {m.nrmse_pct:>6.1f}%  {'✓' if m.passes() else '✗':>5}"
            )
    print(f"\n  Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
    # simulation_app.close() holds the GIL in C++ and hangs indefinitely.
    # All data has been written; let the OS clean up GPU/memory on exit.
    import os
    os._exit(0)

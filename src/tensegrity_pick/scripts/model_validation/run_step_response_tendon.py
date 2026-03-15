"""Tendon model step-response validation for the tensegrity arm.

Replicates the Klein (2023) §4.2–§4.3 experimental methodology using
the ``TENS_3DOF_TENDON_CFG`` (IdealPDActuator with K=0, D=0):

1. A PID position controller converts angle set-points to desired torques.
2. Gravity compensation is added (Mukherjee et al., thesis [85]).
3. The Jacobian transpose maps torques to 5 tendon tensions.
4. Tensions are converted back to joint torques (τ = J^T · T) and
   applied via ``set_joint_effort_target``.
5. Joint angles and tendon tensions are recorded and compared against
   Klein's Gazebo simulation NRMSE reference values (Table 4.2).

Reported metrics (per joint, per step amplitude):
  - Rise time  (10 % → 90 % of target)
  - Overshoot  (% above target)
  - Settling time  (last exit from 2 % band)
  - RMSE / NRMSE

Usage
-----
.. code-block:: bash

    cd /path/to/tensegrity_pick
    conda run -n env_isaaclab python3 scripts/model_validation/run_step_response_tendon.py \\
        --headless [--num_envs 1]

References
----------
* Klein (2023) §3.2.3 (PID), §3.2.4 (tension distribution), §4.2–4.3
* doc/tendon_simulation.md — Isaac Sim tendon architecture
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

parser = argparse.ArgumentParser(description="Tendon model step-response validation")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Override output directory (default: outputs/model_validation/tendon/data)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── IsaacLab imports (after AppLauncher) ──────────────────────────────────────
import numpy as np           # noqa: E402
import torch                 # noqa: E402

import isaaclab.sim as sim_utils                                             # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation      # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg             # noqa: E402
from isaaclab.sim import SimulationContext                                   # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg  # noqa: E402
from isaaclab.utils import configclass                                       # noqa: E402

from tensegrity_pick.robots.tendon_robot_cfg import TENS_3DOF_TENDON_CFG    # noqa: E402

# ── Scene configuration ────────────────────────────────────────────────────────

_ARM_JOINTS = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]

# Override init_state to mount the arm at z = 1.0 m (above ground plane).
# Uses TENS_3DOF_TENDON_CFG which has IdealPDActuatorCfg(K=0, D=0) —
# i.e. effort passthrough: joint torques are set externally each step.
# fix_root_link=True is required: threedof_manipulator.usd has no baked fixed joint,
# so without it the base floats and falls under gravity in a standalone scene.
_TENDON_ROBOT_CFG = TENS_3DOF_TENDON_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={j: 0.0 for j in _ARM_JOINTS},
    ),
    spawn=TENS_3DOF_TENDON_CFG.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
)


@configclass
class TendonValidationSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + 3-DOF tensegrity arm (tendon passthrough)."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    robot: ArticulationCfg = _TENDON_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


# ── Robot reset + warm-up ─────────────────────────────────────────────────────

def _reset_and_warmup(
    robot: Articulation,
    scene: InteractiveScene,
    sim: SimulationContext,
    joint_ids: list[int],
    num_envs: int,
    device: str,
) -> None:
    """Hard-reset all arm joints to zero and apply zero effort for WARMUP_S."""
    zero_pos = torch.zeros(num_envs, len(joint_ids), device=device, dtype=torch.float32)
    zero_eff = torch.zeros(num_envs, len(joint_ids), device=device, dtype=torch.float32)
    robot.write_joint_position_to_sim(zero_pos, joint_ids=joint_ids)
    robot.write_joint_velocity_to_sim(zero_pos, joint_ids=joint_ids)
    scene.write_data_to_sim()

    warmup_steps = int(common.WARMUP_S / common.SIM_DT)
    for _ in range(warmup_steps):
        robot.set_joint_effort_target(zero_eff, joint_ids=joint_ids)
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
    """Run one PID + Jacobian step-response trial.

    Replicates Klein (2023) §4.2–§4.3 exactly:
      - PID torque for the target joint
      - Gravity compensation
      - Jacobian pseudo-inverse: desired torque → 5 tendon tensions
      - J^T back-multiplication: tensions → full 3-joint torque vector
      - set_joint_effort_target applied to all three arm joints

    Cross-joint coupling (from the Jacobian) is present: moving the elbow
    introduces small wrist torques and vice versa, matching the physical
    cable routing geometry.

    Returns
    -------
    time_s, actual_deg, setpoint_deg_arr, tensions_n
        All float32 NumPy arrays.  tensions_n has shape (T, 5).
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

        # ── PID torque + gravity compensation (Klein §3.2.3) ─────────────
        pid_tau  = common.compute_pid_torque(sp_rad, cur_rad, dt, spec.pid, pid_state)
        grav_tau = common.gravity_compensation_torque(cur_rad, spec.mass_kg, spec.gravity_arm_m)
        total_tau = pid_tau + grav_tau

        # ── Tension distribution via Jacobian pseudo-inverse (§3.2.4) ────
        # Only the test joint contributes to the desired torque vector;
        # the pseudo-inverse still yields physically valid tensions.
        tens_np = common.torque_to_tensions(
            total_tau, spec.joint_index, spec.min_tension_n, spec.saturation_n,
        )  # (5,)

        # ── Back-compute full 3-joint torques via J^T ─────────────────────
        # This includes realistic cross-joint coupling from the cable geometry.
        torques_np = common.tensions_to_torques(tens_np)  # (3,)
        torques_t  = torch.tensor(torques_np, dtype=torch.float32, device=device)
        effort_batch = torques_t.unsqueeze(0).expand(num_envs, -1)  # (B, 3)

        robot.set_joint_effort_target(effort_batch, joint_ids=joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

        times.append(t)
        actuals.append(math.degrees(cur_rad))
        setpoints.append(sp_deg)
        tensions_list.append(tens_np.copy())

    return (
        np.array(times,     dtype=np.float32),
        np.array(actuals,   dtype=np.float32),
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

            path = common.trial_path(output_dir, spec.name, amp_deg)
            common.save_trial(
                path, spec.name, amp_deg,
                time_s, actual_deg, setpoint_arr, tensions_n,
                metrics, model_type="tendon",
            )

            rise_s   = f"{metrics.rise_time_ms:.0f} ms"    if metrics.rise_time_ms    else "—"
            settle_s = f"{metrics.settling_time_ms:.0f} ms" if metrics.settling_time_ms else "—"
            print(
                f"  rise={rise_s:>8}  settle={settle_s:>8}"
                f"  NRMSE={metrics.nrmse_pct:5.1f}%  {'✓' if metrics.passes() else '✗'}"
            )

        all_metrics[spec.name] = metrics_list

    return all_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else common.TENDON_DATA_DIR
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

    scene_cfg = TendonValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot: Articulation = scene["robot"]
    joint_ids, _ = robot.find_joints(common.ARM_JOINT_NAMES, preserve_order=True)
    device = robot.device
    num_envs = args_cli.num_envs

    print("\n" + "=" * 65)
    print("  TENDON MODEL STEP-RESPONSE VALIDATION")
    print("  Robot:   TENS_3DOF_TENDON_CFG  (IdealPDActuatorCfg, K=0, D=0)")
    print("  Control: PID + gravity comp + Jacobian transpose  (Klein 2023 §3.2)")
    print(f"  Output:  {output_dir}")
    print("=" * 65)
    print()
    print("  Klein PID gains:")
    print(f"    Wrist (Y/X): kp={common.WRIST_Y_SPEC.pid.kp}  ki={common.WRIST_Y_SPEC.pid.ki}"
          f"  kd={common.WRIST_Y_SPEC.pid.kd}")
    print(f"    Elbow:       kp={common.ELBOW_SPEC.pid.kp}  ki={common.ELBOW_SPEC.pid.ki}"
          f"  kd={common.ELBOW_SPEC.pid.kd}")
    print()

    all_metrics = run_all_trials(
        robot, scene, sim, joint_ids, num_envs, device, output_dir,
    )

    # ── Log file ─────────────────────────────────────────────────────────
    cfg_summary = {
        "Model":          "TENS_3DOF_TENDON_CFG (3-DOF arm only)",
        "USD":            "threedof_manipulator.usd",
        "Actuator type":  "IdealPDActuatorCfg (K=0, D=0 — effort passthrough)",
        "Control law":    "PID position ctrl + gravity comp + Jacobian transpose",
        "Elbow PID":      f"kp={common.ELBOW_SPEC.pid.kp}, ki={common.ELBOW_SPEC.pid.ki}, kd={common.ELBOW_SPEC.pid.kd}",
        "Wrist PID":      f"kp={common.WRIST_Y_SPEC.pid.kp}, ki={common.WRIST_Y_SPEC.pid.ki}, kd={common.WRIST_Y_SPEC.pid.kd}",
        "Jacobian":       "3×5 constant (zero-config approximation, Klein §3.2.4)",
        "Mount position": "(0.0, 0.0, 1.0) m",
        "Num envs":       str(num_envs),
    }
    common.write_log(output_dir.parent, "tendon", cfg_summary, all_metrics)

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Joint':<22}  {'Amp':>5}  {'Rise [ms]':>10}  {'Settle [ms]':>12}  {'NRMSE':>7}  {'Pass':>5}")
    print("  " + "─" * 64)
    for joint_name, mlist in all_metrics.items():
        klein_vals = common.KLEIN_NRMSE.get(joint_name, [])
        for i, m in enumerate(mlist):
            r = f"{m.rise_time_ms:.0f}"    if m.rise_time_ms    else "—"
            s = f"{m.settling_time_ms:.0f}" if m.settling_time_ms else "—"
            k_ref = f"  Klein={klein_vals[i]:.1f}%" if i < len(klein_vals) else ""
            print(
                f"  {joint_name:<22}  {m.amplitude_deg:>4.0f}°  {r:>10}  {s:>12}"
                f"  {m.nrmse_pct:>6.1f}%  {'✓' if m.passes() else '✗':>5}{k_ref}"
            )

    print(f"\n  Data saved to: {output_dir}")
    sim.stop()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Tendon model step-response validation for the tensegrity arm.

Supports two model variants (``--variant``):

* **elbow_approx** (original) — single revolute elbow joint, Jacobian-
  transpose tension mapping, ``TENS_3DOF_TENDON_CFG``.
* **physical** (default) — antiparallelogram 4-bar linkage, elbow tendons
  applied as body forces at cable attachment points,
  ``TENS_3DOF_PHYSICAL_TENDON_CFG``.

Control loop for both variants:

1. A PID position controller converts angle set-points to desired torques.
2. Gravity compensation is added (Mukherjee et al., thesis [85]).
3. Torques are distributed to tendon tensions.
4. Tensions are applied to the simulation:
   - elbow_approx: J^T → ``set_joint_effort_target``
   - physical: body forces on root_link / forearm_link for elbow,
     wrist J^T → ``set_joint_effort_target`` for wrist
5. Joint angles / tendon tensions are recorded and evaluated.

Usage
-----
.. code-block:: bash

    cd /path/to/tensegrity_pick
    conda run -n env_isaaclab python3 scripts/model_validation/run_step_response_tendon.py \\
        --variant physical --headless [--num_envs 1]

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
    help="Override output directory (default depends on --variant)",
)
parser.add_argument(
    "--variant", type=str, default="physical",
    choices=["physical", "elbow_approx"],
    help="Model variant to validate (default: physical)",
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

from tensegrity_pick.robots.tendon_robot_cfg import (                        # noqa: E402
    TENS_3DOF_TENDON_CFG,
    TENS_3DOF_PHYSICAL_TENDON_CFG,
)
from isaaclab.utils.math import quat_apply                                   # noqa: E402

# ── Variant flag ───────────────────────────────────────────────────────────────
_IS_PHYSICAL = args_cli.variant == "physical"

# ── Scene configuration ────────────────────────────────────────────────────────

_ELBOW_APPROX_ARM_JOINTS = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]
_PHYSICAL_ARM_JOINTS = [
    "rod_left_joint", "rod_right_joint", "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
]
_WRIST_JOINTS = ["wrist_y_joint", "wrist_x_joint"]


def _make_validation_robot_cfg(base_cfg: ArticulationCfg, arm_joints: list[str]) -> ArticulationCfg:
    """Mount the arm at z = 1.0 m with fix_root_link=True for standalone scenes."""
    return base_cfg.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={j: 0.0 for j in arm_joints},
        ),
        spawn=base_cfg.spawn.replace(
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
    )


if _IS_PHYSICAL:
    _ROBOT_CFG = _make_validation_robot_cfg(TENS_3DOF_PHYSICAL_TENDON_CFG, _PHYSICAL_ARM_JOINTS)
    _ARM_JOINTS = _PHYSICAL_ARM_JOINTS
else:
    _ROBOT_CFG = _make_validation_robot_cfg(TENS_3DOF_TENDON_CFG, _ELBOW_APPROX_ARM_JOINTS)
    _ARM_JOINTS = _ELBOW_APPROX_ARM_JOINTS


@configclass
class TendonValidationSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + 3-DOF tensegrity arm (tendon passthrough)."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    robot: ArticulationCfg = _ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


# ── Physical-model body-force helpers ──────────────────────────────────────────

def _setup_physical_bodies(robot: Articulation, device: str):
    """Pre-compute body indices and attachment offsets for body-force tendons."""
    root_idx = robot.find_bodies("root_link")[0][0]
    forearm_idx = robot.find_bodies("forearm_link")[0][0]
    root_offsets = torch.tensor(
        common.PHYSICAL_ROOT_ATTACH.tolist(), dtype=torch.float32, device=device,
    )  # (2, 3)
    forearm_offsets = torch.tensor(
        common.PHYSICAL_FOREARM_ATTACH.tolist(), dtype=torch.float32, device=device,
    )  # (2, 3)
    wrist_ids, _ = robot.find_joints(_WRIST_JOINTS, preserve_order=True)
    linkage_joints = ["rod_left_joint", "rod_right_joint", "coupler_left_joint"]
    linkage_ids, _ = robot.find_joints(linkage_joints, preserve_order=True)
    return root_idx, forearm_idx, root_offsets, forearm_offsets, wrist_ids, linkage_ids


def _apply_physical_forces(
    robot: Articulation,
    elbow_tensions: np.ndarray,
    wrist_torques: np.ndarray,
    root_idx: int,
    forearm_idx: int,
    root_offsets: torch.Tensor,
    forearm_offsets: torch.Tensor,
    wrist_ids: list[int],
    num_envs: int,
    device: str,
) -> None:
    """Apply elbow tensions as body forces and wrist torques as joint efforts."""
    B = num_envs
    # Body transforms
    root_pos_w = robot.data.body_pos_w[:, root_idx]          # (B, 3)
    root_quat_w = robot.data.body_quat_w[:, root_idx]        # (B, 4)
    forearm_pos_w = robot.data.body_pos_w[:, forearm_idx]     # (B, 3)
    forearm_quat_w = robot.data.body_quat_w[:, forearm_idx]   # (B, 4)

    # Rotate local offsets to world frame
    root_q_exp = root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    forearm_q_exp = forearm_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_off_local = root_offsets.unsqueeze(0).expand(B, -1, -1)
    forearm_off_local = forearm_offsets.unsqueeze(0).expand(B, -1, -1)

    root_off_w = quat_apply(root_q_exp, root_off_local)
    forearm_off_w = quat_apply(forearm_q_exp, forearm_off_local)

    root_attach_w = root_pos_w.unsqueeze(1) + root_off_w
    forearm_attach_w = forearm_pos_w.unsqueeze(1) + forearm_off_w

    # Cable directions
    cable_vec_w = root_attach_w - forearm_attach_w
    cable_len = cable_vec_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cable_dir_w = cable_vec_w / cable_len

    # Per-tendon forces (world frame)
    T = torch.tensor(elbow_tensions, dtype=torch.float32, device=device)
    T = T.view(1, 2, 1).expand(B, -1, -1)
    F_forearm_w = T * cable_dir_w
    F_root_w = -F_forearm_w

    # Per-tendon torques (world frame)
    tau_forearm_w = torch.cross(forearm_off_w, F_forearm_w, dim=-1)
    tau_root_w = torch.cross(root_off_w, F_root_w, dim=-1)

    # Sum over tendons
    body_forces = torch.zeros(B, 2, 3, device=device)
    body_torques = torch.zeros(B, 2, 3, device=device)
    body_forces[:, 0] = F_root_w.sum(dim=1)
    body_forces[:, 1] = F_forearm_w.sum(dim=1)
    body_torques[:, 0] = tau_root_w.sum(dim=1)
    body_torques[:, 1] = tau_forearm_w.sum(dim=1)

    # Invalidate the wrench composer's cached link poses so the global→local
    # frame conversion uses up-to-date body orientations (not the stale ones
    # from the first call).
    robot._permanent_wrench_composer._link_poses_updated = False

    robot.set_external_force_and_torque(
        forces=body_forces, torques=body_torques,
        body_ids=[root_idx, forearm_idx], is_global=True,
    )

    # Wrist joint efforts
    wrist_t = torch.tensor(wrist_torques, dtype=torch.float32, device=device)
    robot.set_joint_effort_target(
        wrist_t.unsqueeze(0).expand(B, -1), joint_ids=wrist_ids,
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
    physical_ctx: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one PID + Jacobian step-response trial.

    Parameters
    ----------
    physical_ctx : dict or None
        If not None, the physical model is used.  Must contain keys:
        ``root_idx``, ``forearm_idx``, ``root_offsets``, ``forearm_offsets``,
        ``wrist_ids``, ``forearm_body_name_idx`` (body index of forearm_link).

    Returns
    -------
    time_s, actual_deg, setpoint_deg_arr, tensions_n
        All float32 NumPy arrays.  tensions_n has shape (T, 5).
    """
    amplitude_rad = math.radians(amplitude_deg)
    dt = common.SIM_DT
    is_physical = physical_ctx is not None
    is_elbow = (spec.joint_index == 0) if not is_physical else (spec.joint_index == -1)

    _reset_and_warmup(robot, scene, sim, joint_ids, num_envs, device)

    # PID state for the tested joint
    pid_state = common.PIDState()
    pid_state.reset()

    # PID states + specs for non-tested joints (holding at zero)
    all_specs = common.PHYSICAL_ALL_JOINT_SPECS if is_physical else common.ALL_JOINT_SPECS
    hold_pids: dict[str, tuple[common.PIDState, common.JointTestSpec]] = {}
    for s in all_specs:
        if s.name != spec.name:
            ps = common.PIDState()
            ps.reset()
            hold_pids[s.name] = (ps, s)

    pre_steps    = int(common.PRE_STEP_S    / dt)
    step_steps   = int(common.STEP_HOLD_S   / dt)
    return_steps = int(common.RETURN_HOLD_S / dt)
    total_steps  = pre_steps + step_steps + return_steps

    times: list[float] = []
    actuals: list[float] = []
    setpoints: list[float] = []
    tensions_list: list[np.ndarray] = []
    prev_wrist_tensions: np.ndarray | None = None

    for i in range(total_steps):
        t = i * dt
        if i < pre_steps:
            sp_rad = 0.0  # pre-step baseline
        elif i < pre_steps + step_steps:
            sp_rad = amplitude_rad
        else:
            sp_rad = 0.0  # return to zero
        sp_deg = math.degrees(sp_rad)

        # ── Read all joint angles ─────────────────────────────────────────
        pos_all = robot.data.joint_pos[:, joint_ids]
        vel_all = robot.data.joint_vel[:, joint_ids]

        # Tested joint angle
        if is_elbow and is_physical:
            forearm_quat = robot.data.body_quat_w[0, physical_ctx["forearm_idx"]].cpu().numpy()
            cur_rad = common.compute_elbow_angle_from_body_quat(forearm_quat)
        elif is_elbow:
            cur_rad = float(pos_all[0, spec.joint_index].item())
        else:
            if is_physical:
                wrist_local_idx = _PHYSICAL_ARM_JOINTS.index(spec.name)
            else:
                wrist_local_idx = spec.joint_index
            cur_rad = float(pos_all[0, wrist_local_idx].item())
            cur_vel = float(vel_all[0, wrist_local_idx].item())

        # ── PID torque + gravity for tested joint ─────────────────────────
        izone = None if is_elbow else 0.175
        wrist_vel = cur_vel if not is_elbow else None
        pid_tau = common.compute_pid_torque(
            sp_rad, cur_rad, dt, spec.pid, pid_state,
            integral_zone_rad=izone,
            measured_velocity_rad_s=wrist_vel,
        )
        grav_tau  = common.gravity_compensation_torque(cur_rad, spec.mass_kg, spec.gravity_arm_m)
        total_tau = pid_tau + grav_tau

        # ── Build full 3-joint torque vector (tested + holding) ───────────
        # Order: [elbow, wrist_y, wrist_x]
        torque_vec = np.zeros(3, dtype=np.float64)

        # Place tested joint torque
        if is_elbow:
            torque_vec[0] = total_tau
        else:
            ea_idx = common.JOINT_INDEX[spec.name]
            torque_vec[ea_idx] = total_tau

        # Add holding torques for non-tested joints
        for hold_name, (hold_ps, hold_spec) in hold_pids.items():
            hold_is_elbow = (hold_spec.joint_index == 0) if not is_physical else (hold_spec.joint_index == -1)
            # Read this joint's current angle
            if hold_is_elbow and is_physical:
                fq = robot.data.body_quat_w[0, physical_ctx["forearm_idx"]].cpu().numpy()
                hold_cur = common.compute_elbow_angle_from_body_quat(fq)
            elif hold_is_elbow:
                hold_cur = float(pos_all[0, hold_spec.joint_index].item())
            else:
                if is_physical:
                    h_idx = _PHYSICAL_ARM_JOINTS.index(hold_name)
                else:
                    h_idx = hold_spec.joint_index
                hold_cur = float(pos_all[0, h_idx].item())

            h_izone = None if hold_is_elbow else 0.175
            h_vel = None
            if not hold_is_elbow:
                if is_physical:
                    h_vel_idx = _PHYSICAL_ARM_JOINTS.index(hold_name)
                else:
                    h_vel_idx = hold_spec.joint_index
                h_vel = float(vel_all[0, h_vel_idx].item())

            h_pid_tau = common.compute_pid_torque(
                0.0, hold_cur, dt, hold_spec.pid, hold_ps,
                integral_zone_rad=h_izone,
                measured_velocity_rad_s=h_vel,
            )
            h_grav = common.gravity_compensation_torque(hold_cur, hold_spec.mass_kg, hold_spec.gravity_arm_m)
            h_total = h_pid_tau + h_grav

            if hold_is_elbow:
                torque_vec[0] = h_total
            else:
                ea_idx = common.JOINT_INDEX[hold_name]
                torque_vec[ea_idx] = h_total

        # ── Tension distribution (full 3-joint) ──────────────────────────
        min_t = min(spec.min_tension_n, *(s.min_tension_n for _, s in hold_pids.values()))
        # Per-block saturation: elbow vs wrist may have different limits
        elbow_sat = spec.saturation_n if is_elbow else next(
            s.saturation_n for _, s in hold_pids.values()
            if (s.joint_index == 0 or s.joint_index == -1)
        )
        wrist_sat = spec.saturation_n if not is_elbow else next(
            s.saturation_n for _, s in hold_pids.values()
            if s.joint_index not in (0, -1)
        )
        tens_np = common.torque_vector_to_tensions(
            torque_vec, min_t, saturation=max(elbow_sat, wrist_sat),
            elbow_saturation=elbow_sat, wrist_saturation=wrist_sat,
            prev_wrist_tensions=prev_wrist_tensions,
        )
        prev_wrist_tensions = tens_np[2:5].copy()

        # ── Apply forces ──────────────────────────────────────────────────
        if is_physical:
            # Elbow tensions → body forces, wrist tensions → J^T efforts
            elbow_tens = tens_np[:2]
            # Wrist torques from wrist J^T sub-block
            j_wrist = common.JACOBIAN_T[1:3, 2:5]  # (2, 3)
            wrist_torques = j_wrist @ tens_np[2:5]  # (2,)
            _apply_physical_forces(
                robot, elbow_tens, wrist_torques,
                physical_ctx["root_idx"], physical_ctx["forearm_idx"],
                physical_ctx["root_offsets"], physical_ctx["forearm_offsets"],
                physical_ctx["wrist_ids"], num_envs, device,
            )
            # Zero only linkage joints so passthrough actuators don't fight
            # (must NOT zero wrist joints — their efforts were just set above)
            linkage_ids = physical_ctx["linkage_ids"]
            linkage_zero = torch.zeros(num_envs, len(linkage_ids), device=device)
            robot.set_joint_effort_target(linkage_zero, joint_ids=linkage_ids)
        else:
            torques_np = common.tensions_to_torques(tens_np)  # (3,)
            torques_t  = torch.tensor(torques_np, dtype=torch.float32, device=device)
            effort_batch = torques_t.unsqueeze(0).expand(num_envs, -1)
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
    physical_ctx: dict | None = None,
) -> dict[str, list[common.StepMetrics]]:
    """Run step responses for all joints and amplitudes; save NPZ per trial."""
    all_metrics: dict[str, list[common.StepMetrics]] = {}
    specs = common.PHYSICAL_ALL_JOINT_SPECS if _IS_PHYSICAL else common.ALL_JOINT_SPECS
    model_label = "tendon_physical" if _IS_PHYSICAL else "tendon"

    for spec in specs:
        metrics_list: list[common.StepMetrics] = []

        for amp_deg in spec.step_amplitudes_deg:
            print(f"    {spec.name:22s}  {amp_deg:4.0f}°  ...", end="", flush=True)

            time_s, actual_deg, setpoint_arr, tensions_n = run_one_trial(
                robot, scene, sim, joint_ids, num_envs, device, spec, amp_deg,
                physical_ctx=physical_ctx,
            )
            metrics = common.compute_step_metrics(
                time_s, actual_deg, amp_deg,
                step_start_s=common.PRE_STEP_S,
                step_end_s=common.PRE_STEP_S + common.STEP_HOLD_S,
                joint_name=spec.name,
            )
            metrics_list.append(metrics)

            path = common.trial_path(output_dir, spec.name, amp_deg)
            common.save_trial(
                path, spec.name, amp_deg,
                time_s, actual_deg, setpoint_arr, tensions_n,
                metrics, model_type=model_label,
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
    default_dir = common.PHYSICAL_TENDON_DATA_DIR if _IS_PHYSICAL else common.TENDON_DATA_DIR
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else default_dir
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
    joint_names = _PHYSICAL_ARM_JOINTS if _IS_PHYSICAL else common.ARM_JOINT_NAMES
    joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
    device = robot.device
    num_envs = args_cli.num_envs

    # Setup physical body-force context if needed
    physical_ctx: dict | None = None
    if _IS_PHYSICAL:
        root_idx, forearm_idx, root_off, forearm_off, wrist_ids, linkage_ids = _setup_physical_bodies(robot, device)
        physical_ctx = dict(
            root_idx=root_idx,
            forearm_idx=forearm_idx,
            root_offsets=root_off,
            forearm_offsets=forearm_off,
            wrist_ids=wrist_ids,
            linkage_ids=linkage_ids,
        )

    variant_label = args_cli.variant.upper().replace("_", "-")
    if _IS_PHYSICAL:
        elbow_spec = common.PHYSICAL_ELBOW_SPEC
        robot_label = "TENS_3DOF_PHYSICAL_TENDON_CFG (physical four-bar linkage)"
        usd_label = "tensegrity_threedof_arm_physical.usd"
        control_label = "PID → body forces (elbow) + J^T (wrist)"
    else:
        elbow_spec = common.ELBOW_SPEC
        robot_label = "TENS_3DOF_TENDON_CFG (IdealPDActuatorCfg, K=0, D=0)"
        usd_label = "threedof_manipulator.usd"
        control_label = "PID position ctrl + gravity comp + Jacobian transpose"

    print("\n" + "=" * 65)
    print(f"  TENDON MODEL STEP-RESPONSE VALIDATION  [{variant_label}]")
    print(f"  Robot:   {robot_label}")
    print(f"  Control: {control_label}")
    print(f"  Output:  {output_dir}")
    print("=" * 65)
    print()
    print("  PID gains (scaled from Klein 2023 for simulated inertia):")
    print(f"    Wrist (Y/X): kp={common.WRIST_Y_SPEC.pid.kp}  ki={common.WRIST_Y_SPEC.pid.ki}"
          f"  kd={common.WRIST_Y_SPEC.pid.kd}")
    print(f"    Elbow:       kp={elbow_spec.pid.kp}  ki={elbow_spec.pid.ki}"
          f"  kd={elbow_spec.pid.kd}")
    print()

    all_metrics = run_all_trials(
        robot, scene, sim, joint_ids, num_envs, device, output_dir,
        physical_ctx=physical_ctx,
    )

    # ── Log file ─────────────────────────────────────────────────────────
    model_label = "tendon_physical" if _IS_PHYSICAL else "tendon"
    cfg_summary = {
        "Variant":        args_cli.variant,
        "Model":          robot_label,
        "USD":            usd_label,
        "Actuator type":  "IdealPDActuatorCfg (K=0, D=0 — effort passthrough)",
        "Control law":    control_label,
        "Elbow PID":      f"kp={elbow_spec.pid.kp}, ki={elbow_spec.pid.ki}, kd={elbow_spec.pid.kd}",
        "Wrist PID":      f"kp={common.WRIST_Y_SPEC.pid.kp}, ki={common.WRIST_Y_SPEC.pid.ki}, kd={common.WRIST_Y_SPEC.pid.kd}",
        "Jacobian":       "3×5 constant (zero-config approximation, Klein §3.2.4)",
        "Mount position": "(0.0, 0.0, 1.0) m",
        "Num envs":       str(num_envs),
    }
    common.write_log(output_dir.parent, model_label, cfg_summary, all_metrics)

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


if __name__ == "__main__":
    main()
    # simulation_app.close() holds the GIL in C++ and hangs indefinitely.
    # All data has been written; let the OS clean up GPU/memory on exit.
    import os
    os._exit(0)

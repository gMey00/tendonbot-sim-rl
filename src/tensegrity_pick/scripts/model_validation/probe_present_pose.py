"""Reachability + joint-load probe for the shirt_pick PRESENTATION_POS.

Answers three questions for a candidate presentation pose (default: the
current ``PRESENTATION_POS``), with the shirt actually grasped and hanging
from the gripper:

  1. REACH — can the grasp point get there at all?  Actions run UNCLIPPED
     (the task's action clips are removed in-script), so only the physical
     joint limits and drives constrain the motion.  Servo = numeric-Jacobian
     damped least squares (each arm DOF is nudged once to measure its tip
     sensitivity, then a ≈ a + J⁺·err each step).
  2. LIMIT BINDING — which joints sit at their PHYSICAL position limits at
     the closest approach (kinematic envelope boundary), and which sit at the
     TASK ACTION CLIP values (a config, not hardware, constraint).
  3. STRENGTH — applied torque / effort-limit fraction per joint while
     holding the pose with the shirt.  Saturation (~1.0) means the drive is
     too weak for the load, and only then should effort limits be raised.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/probe_present_pose.py --headless

    # probe an arbitrary candidate pose:
    ... probe_present_pose.py --headless --pose 0.15 0.9 1.6
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Presentation-pose reachability probe.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--pose", type=float, nargs=3, default=None,
                    help="candidate pose (x y z env-local); default PRESENTATION_POS")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (  # noqa: E402
    PRESENTATION_POS,
)

# action layout: [elbow, wrist_y, wrist_x, gripper, base_y, base_z]
IDX_ELBOW, IDX_WY, IDX_WX, IDX_GRIP, IDX_BY, IDX_BZ = range(6)
ARM_DIMS = [IDX_ELBOW, IDX_WY, IDX_WX, IDX_BY, IDX_BZ]
ARM_JOINTS = ["elbow_joint", "wrist_y_joint", "wrist_x_joint", "base_y_joint", "base_z_joint"]
OPEN, CLOSE = 1.0, -1.0

NUDGE = 0.25          # action nudge for the numeric Jacobian
NUDGE_STEPS = 40
SERVO_STEPS = 600     # 10 s of servoing
HOLD_STEPS = 120      # 2 s hold for the torque measurement
DLS_LAMBDA = 0.02     # damped least squares regulariser
GAIN = 0.9


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.episode_length_s = 1.0e6
    # UNCLIP the actions: only physical joint limits should constrain reach.
    cfg.actions.arm_action.clip = None
    cfg.actions.base_delta.clip = None
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    # Flat lay instead of crumpled bank states: the probe answers REACH/LOAD,
    # not grasp robustness — the flat lay makes the scripted grasp reliable
    # (the shirt's highest point sits directly under the gripper).
    u.bank_fraction = 0.0
    env.reset()
    dev = u.device
    n = u.num_envs
    robot = u.scene["robot"]

    pose = tuple(args_cli.pose) if args_cli.pose else PRESENTATION_POS
    target = u.scene.env_origins + torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0)
    print(f"[probe] candidate pose (env-local): {pose}")

    a = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
    a[:, IDX_GRIP] = OPEN

    def step(k: int = 1) -> None:
        with torch.inference_mode():
            for _ in range(k):
                env.step(a)

    def tip() -> torch.Tensor:
        return u._finger_tip_pos().clone()

    # ── 1) grasp the shirt (xy-align on the highest point, descend, close) ──
    # The pre-settled flat lay's highest point sits ~0.19 m off the gripper's
    # default descent line, so alignment servoing is required (same approach
    # as baseline_shirt_pick.py; elbow→tip-x sign auto-calibrated).
    t0x = tip()[:, 0].mean()
    a[:, IDX_ELBOW] = 0.2
    step(15)
    elbow_sign = 1.0 if (tip()[:, 0].mean() - t0x) > 0 else -1.0
    a[:, IDX_ELBOW] = 0.0
    step(15)

    def align_once() -> None:
        err = u.shirt_grasp_point_w - tip()
        a[:, IDX_BY] = torch.clamp(a[:, IDX_BY] + 6.0 * err[:, 1] / 60.0, -2, 2)
        a[:, IDX_ELBOW] = torch.clamp(
            a[:, IDX_ELBOW] + elbow_sign * 3.0 * err[:, 0] / 60.0, -2, 2)

    for _ in range(60):
        align_once()
        step()
    for i in range(150):
        align_once()
        a[:, IDX_BZ] = max(a[0, IDX_BZ].item() - 0.03, -1.2)
        step()
        if i % 30 == 29:
            d = torch.norm(u.shirt_grasp_point_w - tip(), dim=-1)
            print(f"[descend {i+1}] tip→highest d = {[f'{x:.3f}' for x in d.tolist()]}")
        if (torch.norm(u.shirt_grasp_point_w - tip(), dim=-1) < 0.08).all():
            break
    a[:, IDX_GRIP] = CLOSE
    for _ in range(40):
        align_once()
        step()
    a[:, IDX_BZ] = 0.0
    step(80)
    grasped = u.grasp_active.clone()
    print(f"[probe] grasped: {int(grasped.sum())}/{n} (probe continues with grasped envs)")
    if not grasped.any():
        print("RESULT: ABORT — no env grasped the shirt; reachability not probed.")
        env.close()
        return

    # ── 2) numeric Jacobian: tip sensitivity per arm action dim ─────────
    def measure_jacobian() -> torch.Tensor:
        Jm = torch.zeros(n, 3, len(ARM_DIMS), device=dev)
        for j, dim in enumerate(ARM_DIMS):
            t0 = tip()
            a[:, dim] += NUDGE
            step(NUDGE_STEPS)
            Jm[:, :, j] = (tip() - t0) / NUDGE
            a[:, dim] -= NUDGE
            step(NUDGE_STEPS)
        return Jm

    J = measure_jacobian()
    print("[probe] Jacobian magnitudes per dim (m/action):",
          [f"{ARM_JOINTS[j]}:{J[0, :, j].norm():.2f}" for j in range(len(ARM_DIMS))])

    # ── 3) damped least-squares servo toward the pose ───────────────────
    # The Jacobian is configuration-dependent (the elbow axis direction
    # changes as the arm rises), so re-measure it periodically — a single
    # hang-pose estimate stalls the servo far from the target.
    eye = torch.eye(3, device=dev).expand(n, 3, 3)
    min_d = torch.full((n,), 1e9, device=dev)
    for it in range(SERVO_STEPS):
        if it > 0 and it % 200 == 0:
            J = measure_jacobian()
        err = target - u.shirt_grasp_point_w              # [n, 3]
        d = err.norm(dim=-1)
        min_d = torch.minimum(min_d, d)
        JT = J.transpose(1, 2)
        gram = torch.baddbmm(DLS_LAMBDA * eye, J, JT)     # J Jᵀ + λI
        da = torch.bmm(JT, torch.linalg.solve(gram, err.unsqueeze(-1))).squeeze(-1)
        for j, dim in enumerate(ARM_DIMS):
            a[:, dim] = torch.clamp(a[:, dim] + GAIN * da[:, j] * 0.02, -3, 3)
        step()

    # ── 4) hold + measurements ───────────────────────────────────────────
    torque_frac_max = torch.zeros(n, len(ARM_JOINTS), device=dev)
    d_hold = []
    for _ in range(HOLD_STEPS):
        err = target - u.shirt_grasp_point_w
        d_hold.append(err.norm(dim=-1).clone())
        JT = J.transpose(1, 2)
        gram = torch.baddbmm(DLS_LAMBDA * eye, J, JT)
        da = torch.bmm(JT, torch.linalg.solve(gram, err.unsqueeze(-1))).squeeze(-1)
        for j, dim in enumerate(ARM_DIMS):
            a[:, dim] = torch.clamp(a[:, dim] + GAIN * da[:, j] * 0.02, -3, 3)
        step()
        ids = [robot.joint_names.index(jn) for jn in ARM_JOINTS]
        applied = robot.data.applied_torque[:, ids].abs()
        limits = robot.data.joint_effort_limits[:, ids]
        torque_frac_max = torch.maximum(torque_frac_max, applied / (limits + 1e-8))

    d_final = torch.stack(d_hold).mean(dim=0)
    ids = [robot.joint_names.index(jn) for jn in ARM_JOINTS]
    jp = robot.data.joint_pos[:, ids]
    jlim = robot.data.joint_pos_limits[:, ids, :]  # [n, J, 2]
    at_limit = ((jp - jlim[..., 0]).abs() < 0.02) | ((jp - jlim[..., 1]).abs() < 0.02)

    print("\n── per-env results ─────────────────────────────────────────")
    for i in range(n):
        print(f"env {i}: grasped={bool(grasped[i])}  min_d={min_d[i]:.3f}  hold_d={d_final[i]:.3f}")
    # Report joints/torques for the GRASPED env closest to the target —
    # ungrasped envs servo against a belt-bound cloth and rail all joints.
    gi = int(torch.nonzero(grasped).squeeze(-1)[d_final[grasped].argmin()])
    print(f"\n── joint report (env {gi}, grasped, at hold) ────────────────")
    for j, jn in enumerate(ARM_JOINTS):
        print(f"  {jn:15s} pos={jp[gi, j]:+.3f}  limits=[{jlim[gi, j, 0]:+.3f}, {jlim[gi, j, 1]:+.3f}] "
              f"at_limit={bool(at_limit[gi, j])}  max_torque_frac={torque_frac_max[gi, j]:.2f}")
    ok = grasped & (d_final < 0.15)
    reach_d = d_final[grasped].min().item() if grasped.any() else float('nan')
    print(f"\nRESULT: pose {pose} — best hold distance {reach_d:.3f} m "
          f"({'REACHABLE' if reach_d < 0.15 else 'NOT REACHED'}); "
          f"reached in {int(ok.sum())}/{int(grasped.sum())} grasped envs; "
          f"limit-bound joints (env{gi}): {[ARM_JOINTS[j] for j in range(len(ARM_JOINTS)) if at_limit[gi, j]]}; "
          f"peak torque fractions: {[f'{ARM_JOINTS[j]}:{torque_frac_max[gi, j]:.2f}' for j in range(len(ARM_JOINTS))]}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

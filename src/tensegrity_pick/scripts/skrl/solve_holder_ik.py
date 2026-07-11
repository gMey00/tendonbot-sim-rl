# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Solve (and verify) the passive HOLDER robot's IK for the shirt-present task.

The holder tensegrity is scenery representing the retriever gripping the shirt at
the first-grasp / presentation anchor.  Its base is a 2-DOF PRISMATIC rail
(base_y, base_z) plus a 3-DOF arm (elbow, wrist_y, wrist_x) and a Robotiq 2F-140.

The target is the CLOSED gripper's FINGERTIP, not the wrist flange -- the shirt
should hang from where the fingers pinch.  All gripper body FRAMES sit at the
mounting base (robotiq_base_link == tool_link_0), so the real fingertip has no
body to read; it is modelled as ``tool_link_0 + approach_dir * TIP_OFFSET``,
where approach_dir = normalise(tool_link_0 - wrist_link) is the direction the
gripper points.  The solver CLOSES the gripper, then IK-solves the arm/base so
the modelled fingertip sits at ``present_anchor_local`` (two-pass: solve the
flange to a back-shifted target, re-read the approach dir, repeat).

    PYTHONPATH="$PWD/source/tensegrity_pick:$PYTHONPATH" \
    python scripts/skrl/solve_holder_ik.py \
        --task Template-Shirt-Present-UR5e-F140-Play-v0 --headless \
        --anchor_z_sweep 1.60,1.55 --tip_offset 0.14
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Solve the holder robot IK for the anchor.")
parser.add_argument("--task", type=str, default="Template-Shirt-Present-UR5e-F140-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--anchor_z_sweep", type=str, default=None,
    help="Comma-separated anchor z heights to try (keeps anchor x,y). E.g. 1.60,1.55.",
)
parser.add_argument(
    "--tip_offset", type=float, default=0.14,
    help="Fingertip distance (m) beyond tool_link_0 along the gripper approach axis.",
)
parser.add_argument(
    "--verify", action="store_true",
    help="Skip solving: reset with the DEPLOYED cfg and report the held fingertip<->anchor gap over time.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import tensegrity_pick.tasks  # noqa: F401

# Robotiq 2F-140 close command (matches the env's BinaryJointPositionAction).
FINGER_CLOSE = 0.7854


def _settle(env, steps: int = 1):
    for _ in range(steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.sim.get_physics_dt())


def _place_arm(holder: Articulation, ctrl_ids, q_ctrl: torch.Tensor):
    """Teleport + target ONLY the arm/base joints; leave the gripper as-is (closed)."""
    holder.set_joint_position_target(q_ctrl, joint_ids=ctrl_ids)
    holder.write_joint_state_to_sim(q_ctrl, torch.zeros_like(q_ctrl), joint_ids=ctrl_ids)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    holder: Articulation = env.scene["holder_robot"]
    tool_idx = holder.find_bodies("tool_link_0")[0][0]
    wrist_idx = holder.find_bodies("wrist_link")[0][0]
    ctrl_names = ["base_y_joint", "base_z_joint", "elbow_joint", "wrist_y_joint", "wrist_x_joint"]
    # preserve_order=True so ctrl_ids aligns with ctrl_names (the articulation's
    # native order is [.., wrist_x, wrist_y]); the solution vector, the printed
    # labels, and any by-name baking must all use the SAME order.
    ctrl_ids = holder.find_joints(ctrl_names, preserve_order=True)[0]
    finger_id = holder.find_joints("finger_joint")[0]
    base_anchor = torch.tensor(env.present_anchor_local, device=env.device)
    origin = env.scene.env_origins[0]
    L = args_cli.tip_offset

    def approach_dir() -> torch.Tensor:
        d = holder.data.body_pos_w[0, tool_idx] - holder.data.body_pos_w[0, wrist_idx]
        return d / d.norm()

    def fingertip_w() -> torch.Tensor:
        return holder.data.body_pos_w[0, tool_idx] + approach_dir() * L

    if args_cli.verify:
        # Reset with the deployed cfg (baked init_state + PD hold) and watch the
        # modelled fingertip vs the anchor over time -> confirms it grips AND holds.
        env.reset()
        anchor_w = origin + base_anchor
        for name in ("base_y", "base_z", "elbow", "wrist"):
            a = holder.actuators[name]
            print(f"[VERIFY] actuator {name}: stiffness={a.stiffness}, effort_limit_sim={a.effort_limit_sim}")
        for step in range(300):
            env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device))
            if step % 60 == 0 or step == 299:
                gap = fingertip_w() - anchor_w
                fj = holder.data.joint_pos[0, finger_id].item()
                print(f"[VERIFY] step {step:3d}: fingertip-anchor gap xyz="
                      f"{[round(v, 4) for v in gap.tolist()]}  |gap|={gap.norm() * 100:.2f} cm  finger_joint={fj:.3f}")
        qc = holder.data.joint_pos[0, ctrl_ids]
        tgt = holder.data.default_joint_pos[0, ctrl_ids]
        print(f"[VERIFY] held arm joints:  {[round(v, 4) for v in qc.tolist()]}")
        print(f"[VERIFY] baked arm targets:{[round(v, 4) for v in tgt.tolist()]}")
        print(f"[VERIFY] tool_link_0 (local): {[round(v, 4) for v in (holder.data.body_pos_w[0, tool_idx] - origin).tolist()]}")
        print(f"[VERIFY] wrist_link  (local): {[round(v, 4) for v in (holder.data.body_pos_w[0, wrist_idx] - origin).tolist()]}")
        env.close()
        return

    lo = holder.data.joint_pos_limits[0, ctrl_ids, 0]
    hi = holder.data.joint_pos_limits[0, ctrl_ids, 1]
    jbase = 1 if holder.is_fixed_base else 0
    jcols = ctrl_ids if holder.is_fixed_base else [i + 6 for i in ctrl_ids]

    print(f"[IK] controlled joints: {ctrl_names}; tip_offset L={L:.3f} m")
    print(f"[IK] joint limits lo: {[round(v, 3) for v in lo.tolist()]}")
    print(f"[IK] joint limits hi: {[round(v, 3) for v in hi.tolist()]}")

    # Close the gripper and let the four-bar linkage settle.
    holder.set_joint_position_target(
        torch.full((1, 1), FINGER_CLOSE, device=env.device), joint_ids=finger_id
    )
    _settle(env, 90)
    print(f"[IK] gripper closed: finger_joint={holder.data.joint_pos[0, finger_id].item():.3f}")

    def tool_local() -> torch.Tensor:
        return holder.data.body_pos_w[0, tool_idx] - origin

    def fk_tool(q_ctrl: torch.Tensor) -> torch.Tensor:
        _place_arm(holder, ctrl_ids, q_ctrl.unsqueeze(0))
        _settle(env, 1)
        return tool_local()

    def solve_flange(target: torch.Tensor, q0: torch.Tensor):
        """DLS IK putting tool_link_0 at `target` (env-local)."""
        q, lam, it = q0.clone(), 0.02, 0
        p = fk_tool(q)
        for it in range(200):
            err = target - p
            if err.norm() < 5e-4:
                break
            J = holder.root_physx_view.get_jacobians()[0, tool_idx - jbase, 0:3, jcols]
            JJt = J @ J.T + (lam ** 2) * torch.eye(3, device=env.device)
            q = torch.clamp(q + J.T @ torch.linalg.solve(JJt, err), lo, hi)
            p = fk_tool(q)
        return q, p, it + 1

    def solve_tip(anchor: torch.Tensor):
        """Multi-pass: shift the flange target back along the approach axis so the
        modelled fingertip lands on `anchor` (re-read the approach dir each pass)."""
        q = holder.data.default_joint_pos[0, ctrl_ids].clone()
        total = 0
        for _ in range(6):
            appr = approach_dir()
            shifted = anchor - appr * L
            q, _, iters = solve_flange(shifted, q)
            total += iters
        return q, fingertip_w() - origin, total

    zs = ([float(z) for z in args_cli.anchor_z_sweep.split(",")]
          if args_cli.anchor_z_sweep else [float(base_anchor[2])])
    print(f"[IK] anchor x,y fixed at ({base_anchor[0]:.3f}, {base_anchor[1]:.3f}); sweeping z (fingertip target): {zs}")
    for z in zs:
        anchor = base_anchor.clone(); anchor[2] = z
        q, tip, iters = solve_tip(anchor)
        err = anchor - tip
        atlim = [n for n, v, l, h in zip(ctrl_names, q.tolist(), lo.tolist(), hi.tolist())
                 if abs(v - l) < 1e-3 or abs(v - h) < 1e-3]
        print(f"\n[IK] === anchor z={z:.3f} (iters {iters}) ===")
        for name, v in zip(ctrl_names, q.tolist()):
            print(f"        {name:16s}= {v:+.4f}")
        print(f"[IK]   flange tool_link_0: {[round(v, 4) for v in tool_local().tolist()]}")
        print(f"[IK]   modelled fingertip: {[round(v, 4) for v in tip.tolist()]}")
        print(f"[IK]   residual |err|={err.norm() * 100:.2f} cm (z-err {err[2] * 100:+.2f} cm); at-limit joints: {atlim}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

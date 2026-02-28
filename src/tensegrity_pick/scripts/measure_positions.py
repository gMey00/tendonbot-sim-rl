"""Measure actual body positions in the tensegrity place environment.

Runs a configurable number of zero-action steps and then prints the world
positions of all key reference points so that downstream code and
documentation use correct geometry.

Usage::

    conda run -n env_isaaclab python3 scripts/measure_positions.py \
        --task Template-Tensegrity-Place-v0 --num_envs 1 --headless

Re-run this script and update GEOMETRY.md whenever the robot, gripper, or
mount configuration changes.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure body positions.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Template-Tensegrity-Place-v0")
parser.add_argument("--settle_steps", type=int, default=50,
                    help="Number of zero-action steps before measuring.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import tensegrity_pick.tasks  # noqa: F401

from isaaclab.utils.math import quat_apply


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for _ in range(args_cli.settle_steps):
        env.step(actions)

    try:
        scene = env.unwrapped.scene
        robot = scene["robot"]
        green = scene["green_cube"]

        # ── Joint state ──────────────────────────────────────────────
        joint_names = robot.data.joint_names
        joint_pos = robot.data.joint_pos[0]
        print("\n=== JOINT STATE (env 0) ===", flush=True)
        for name, pos in zip(joint_names, joint_pos):
            print(f"  {name:35s} = {pos.item():.6f}", flush=True)

        # ── Body positions ───────────────────────────────────────────
        body_names = robot.data.body_names
        body_pos = robot.data.body_pos_w[0]
        body_quat = robot.data.body_quat_w[0]
        env_origin = scene.env_origins[0]

        print("\n=== BODY POSITIONS (env 0, world frame) ===", flush=True)
        for name, pos in zip(body_names, body_pos):
            local = pos - env_origin
            print(f"  {name:35s}  world=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
                  f"  local=({local[0]:.4f}, {local[1]:.4f}, {local[2]:.4f})",
                  flush=True)

        # ── tool_link_0 with projected offsets ───────────────────────
        tool_idx = (body_names.index("tool_link_0")
                    if "tool_link_0" in body_names else None)
        if tool_idx is not None:
            tool_pos = body_pos[tool_idx]
            tool_quat = body_quat[tool_idx]
            tool_local = tool_pos - env_origin
            print(f"\n=== tool_link_0 ===", flush=True)
            print(f"  world : ({tool_pos[0]:.4f}, {tool_pos[1]:.4f}, {tool_pos[2]:.4f})",
                  flush=True)
            print(f"  local : ({tool_local[0]:.4f}, {tool_local[1]:.4f}, {tool_local[2]:.4f})",
                  flush=True)
            print(f"  quat  : ({tool_quat[0]:.4f}, {tool_quat[1]:.4f},"
                  f" {tool_quat[2]:.4f}, {tool_quat[3]:.4f})", flush=True)

            # Positive offsets → world-downward for ceiling mount
            GRASP_CENTER_LOCAL_Z = 0.1925
            FINGER_TIP_OPEN_Z = 0.215
            FINGER_TIP_CLOSED_Z = 0.235

            print("  --- projected offsets (positive = world-downward) ---",
                  flush=True)
            for label, offset_z in [
                ("grasp_center", GRASP_CENTER_LOCAL_Z),
                ("finger_tip_open", FINGER_TIP_OPEN_Z),
                ("finger_tip_closed", FINGER_TIP_CLOSED_Z),
            ]:
                offset = torch.tensor(
                    [0.0, 0.0, offset_z], device=tool_pos.device)
                world_pos = tool_pos + quat_apply(
                    tool_quat.unsqueeze(0), offset.unsqueeze(0)).squeeze(0)
                local_pos = world_pos - env_origin
                print(f"  {label:25s}  world=({world_pos[0]:.4f},"
                      f" {world_pos[1]:.4f}, {world_pos[2]:.4f})"
                      f"  local=({local_pos[0]:.4f}, {local_pos[1]:.4f},"
                      f" {local_pos[2]:.4f})", flush=True)

        # ── Cube ─────────────────────────────────────────────────────
        green_pos = green.data.root_pos_w[0]
        green_local = green_pos - env_origin
        print(f"\n=== GREEN CUBE ===", flush=True)
        print(f"  world : ({green_pos[0]:.4f}, {green_pos[1]:.4f},"
              f" {green_pos[2]:.4f})", flush=True)
        print(f"  local : ({green_local[0]:.4f}, {green_local[1]:.4f},"
              f" {green_local[2]:.4f})", flush=True)

        # ── Distances ────────────────────────────────────────────────
        if tool_idx is not None:
            gc_offset = torch.tensor(
                [0.0, 0.0, GRASP_CENTER_LOCAL_Z], device=tool_pos.device)
            gc_world = tool_pos + quat_apply(
                tool_quat.unsqueeze(0), gc_offset.unsqueeze(0)).squeeze(0)
            ft_offset = torch.tensor(
                [0.0, 0.0, FINGER_TIP_OPEN_Z], device=tool_pos.device)
            ft_world = tool_pos + quat_apply(
                tool_quat.unsqueeze(0), ft_offset.unsqueeze(0)).squeeze(0)

            dist_gc = torch.norm(gc_world - green_pos).item()
            dist_tl = torch.norm(tool_pos - green_pos).item()
            dist_ft = torch.norm(ft_world - green_pos).item()

            print(f"\n=== DISTANCES (env 0) ===", flush=True)
            print(f"  tool_link_0 → green_cube    : {dist_tl:.4f} m", flush=True)
            print(f"  grasp_center → green_cube   : {dist_gc:.4f} m", flush=True)
            print(f"  finger_tip_open → green_cube: {dist_ft:.4f} m", flush=True)
            gc_gap = (gc_world[2] - green_pos[2]).item()
            ft_gap = (ft_world[2] - green_pos[2]).item()
            print(f"  vertical (gc_z - cube_z)    : {gc_gap:+.4f} m", flush=True)
            print(f"  vertical (ft_z - cube_z)    : {ft_gap:+.4f} m", flush=True)

    except Exception as exc:
        print(f">>> ERROR in measurements: {exc}", flush=True)
        import traceback
        traceback.print_exc()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Verify the Kinova joint-limit PhysX startup fix (headless).

For a given reach task it:
  * (optionally) removes the prestartup USD clamp to reproduce the pre-fix state;
  * captures the PhysX ``setLimitParams()`` errors emitted during the
    articulation bake by redirecting the process stderr fd around env creation;
  * prints the final baked ``joint_pos_limits`` for the arm joints so parity
    with the ``default ± fallback/2`` rule can be checked;
  * steps a few times to confirm no NaN / divergence.

Usage:
    python verify_kinova_limits.py --task <ID> [--disable_prestartup] --headless
"""

import argparse
import os
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify Kinova joint-limit fix.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--disable_fix",
    action="store_true",
    help="Restore the stock spawner (drop the joint-limit clamp) to reproduce the pre-fix PhysX error.",
)
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


def main():
    tag = "PRE-FIX (clamp spawner disabled)" if args_cli.disable_fix else "POST-FIX (clamp spawner enabled)"
    print(f"\n===== VERIFY [{args_cli.task}] {tag} =====")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    spawn_func_name = getattr(env_cfg.scene.robot.spawn.func, "__name__", str(env_cfg.scene.robot.spawn.func))
    print(f"[verify] robot spawn func: {spawn_func_name}")
    if args_cli.disable_fix:
        from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

        env_cfg.scene.robot.spawn.func = spawn_from_usd
        print("[verify] restored stock spawn_from_usd (simulating pre-fix)")

    # Redirect the process stderr fd to a temp file around env creation so the
    # C++ PhysX/carb errors (which go to the process stderr) are captured here.
    err_path = tempfile.mktemp(suffix="_stderr.txt")
    saved_fd = os.dup(2)
    err_fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(err_fd, 2)
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
    finally:
        os.dup2(saved_fd, 2)
        os.close(err_fd)
        os.close(saved_fd)

    with open(err_path, "r", errors="replace") as f:
        captured = f.read()
    os.remove(err_path)

    n_setlimit = captured.count("setLimitParams")
    print(f"[verify] captured stderr bytes: {len(captured)}")
    print(f"[verify] >>> setLimitParams error count: {n_setlimit} <<<")
    if n_setlimit:
        for line in captured.splitlines():
            if "setLimitParams" in line:
                print(f"[verify]   ERR: {line.strip()[:160]}")
                break

    # Dump arm-joint final baked limits (env 0).
    robot = env.unwrapped.scene["robot"]
    names = robot.joint_names
    limits = robot.data.joint_pos_limits[0].cpu()  # (num_joints, 2)
    defaults = robot.data.default_joint_pos[0].cpu()
    print("[verify] arm/continuous joint limits (rad) [name: lo, hi | span | default]:")
    import math
    for i, nm in enumerate(names):
        lo, hi = float(limits[i, 0]), float(limits[i, 1])
        span = hi - lo
        d = float(defaults[i])
        # Only print arm joints (skip the many gripper joints) + any wide span.
        if nm.startswith("joint_") or nm.startswith("wrist_") or span > 3.0:
            print(f"[verify]   {nm:24s}: [{lo:+.4f}, {hi:+.4f}] | span={span:.4f} | default={d:+.4f}")

    # Step a few times to confirm no NaN / divergence.
    n_nan = 0
    for _ in range(20):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
            jp = env.unwrapped.scene["robot"].data.joint_pos
            if torch.isnan(jp).any():
                n_nan += 1
    print(f"[verify] steps with NaN joint_pos over 20 steps: {n_nan}")
    print(f"[verify] RESULT: setLimitParams_errors={n_setlimit} nan_steps={n_nan}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

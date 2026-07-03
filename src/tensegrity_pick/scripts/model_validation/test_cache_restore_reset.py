"""Stage-0 de-risk: deterministic cache-restore cloth reset validation.

Validates the mechanism the cached state banks rely on (crumpled initial
states, task-to-task terminal states): capturing the full particle state
(positions AND velocities) of a settled cloth and writing it back via
``ClothObject.write_nodal_state_to_sim`` must reproduce the same subsequent
trajectory every time.

Procedure (on the registered shirt_pick env, robot held at default pose):
  1. reset → let the cloth settle SETTLE_STEPS zero-action steps
  2. capture cloth pos+vel and robot joint/root state ("cached state")
  3. run T trials: restore the cached state exactly → step K steps →
     snapshot the cloth every SNAP_EVERY steps
  4. compare snapshots across trials (max |Δpos| over all particles)
  5. additionally check post-restore drift: a settled state must stay
     settled (bounded centroid drift), i.e. the restore is *complete* —
     positions-only restores leave stale solver velocities and fail this.

PASS criteria:
  * determinism: max cross-trial deviation < DEVIATION_TOL at every snapshot
  * completeness: cloth centroid drift over K steps < DRIFT_TOL
Also reports the wall-clock cost of a cache-restore vs. settling from scratch.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/test_cache_restore_reset.py --headless

    # more envs / different task variant
    python scripts/model_validation/test_cache_restore_reset.py --headless --num_envs 8
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic cache-restore cloth reset test.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle_steps", type=int, default=300)
parser.add_argument("--horizon", type=int, default=120, help="steps simulated after each restore")
parser.add_argument("--trials", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402

SNAP_EVERY = 30
DEVIATION_TOL = 1e-4   # m — cross-trial particle deviation (GPU PhysX repeat)
DRIFT_TOL = 0.05       # m — centroid drift of a restored *settled* state


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # Deterministic robot reset (the test isolates the CLOTH restore path).
    cfg.events.reset_arm.params["position_range"] = (0.0, 0.0)
    # No mid-test episode timeouts (a manager reset would overwrite the
    # restored cloth state with the flat lay).
    cfg.episode_length_s = 1.0e6
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    cloth = u.cloth
    robot = u.scene["robot"]
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)

    def step_n(n: int) -> None:
        with torch.inference_mode():
            for _ in range(n):
                env.step(zero)

    # 1) settle from the flat lay
    t0 = time.perf_counter()
    step_n(args_cli.settle_steps)
    t_settle = time.perf_counter() - t0
    cloth.update()

    # 2) cache the full state
    cached_pos = cloth._pos_flat.clone()
    cached_vel = cloth._vel_flat.clone()
    cached_jp = robot.data.joint_pos.clone()
    cached_jv = robot.data.joint_vel.clone()
    cached_root = robot.data.root_state_w.clone()
    max_v = cloth.nodal_vel_w.norm(dim=-1).max().item()
    print(f"[cache] settled in {args_cli.settle_steps} steps ({t_settle:.1f}s), "
          f"max particle |v| = {max_v:.4f} m/s")

    all_ids = torch.arange(u.num_envs, device=u.device)

    def restore() -> float:
        """Restore cloth + robot to the cached state; returns wall time (s)."""
        t = time.perf_counter()
        with torch.inference_mode():
            cloth.reset_attachment(all_ids)
            cloth.write_nodal_state_to_sim(cached_pos, cached_vel, all_ids)
            robot.write_joint_state_to_sim(cached_jp, cached_jv)
            robot.write_root_pose_to_sim(cached_root[:, :7])
            robot.write_root_velocity_to_sim(cached_root[:, 7:])
            cloth.update()
        return time.perf_counter() - t

    # 3) trials
    n_snaps = args_cli.horizon // SNAP_EVERY
    trials: list[list[torch.Tensor]] = []
    restore_times = []
    for t_i in range(args_cli.trials):
        restore_times.append(restore())
        snaps = []
        for _ in range(n_snaps):
            step_n(SNAP_EVERY)
            cloth.update()
            snaps.append(cloth.nodal_pos_w.clone())
        trials.append(snaps)
        print(f"[trial {t_i}] restore {restore_times[-1]*1e3:.1f} ms, {n_snaps} snapshots taken")

    # 4) cross-trial determinism
    det_ok = True
    for s in range(n_snaps):
        devs = [
            (trials[t][s] - trials[0][s]).norm(dim=-1).max().item()
            for t in range(1, args_cli.trials)
        ]
        worst = max(devs)
        status = "OK " if worst < DEVIATION_TOL else "FAIL"
        det_ok &= worst < DEVIATION_TOL
        print(f"[determinism] snapshot @ step {(s+1)*SNAP_EVERY:4d}: "
              f"max cross-trial |Δpos| = {worst:.3e} m  {status}")

    # 5) completeness (restored settled state stays settled)
    drift = (trials[0][-1].mean(dim=1) - cached_pos.view(u.num_envs, -1, 3).mean(dim=1))
    drift_xy = drift[:, :2].norm(dim=-1).max().item()
    drift_z = drift[:, 2].abs().max().item()
    drift_ok = drift_xy < DRIFT_TOL and drift_z < DRIFT_TOL
    print(f"[completeness] centroid drift after {args_cli.horizon} steps: "
          f"xy {drift_xy:.4f} m, z {drift_z:.4f} m "
          f"({'OK' if drift_ok else 'FAIL'} @ tol {DRIFT_TOL})")

    mean_restore_ms = sum(restore_times) / len(restore_times) * 1e3
    print(f"[cost] cache-restore {mean_restore_ms:.1f} ms vs settle-from-scratch "
          f"{t_settle:.1f} s ({args_cli.settle_steps} steps) → "
          f"{t_settle * 1e3 / mean_restore_ms:.0f}× cheaper")

    print(f"\nRESULT: {'PASS' if det_ok and drift_ok else 'FAIL'} "
          f"(determinism={'PASS' if det_ok else 'FAIL'}, "
          f"completeness={'PASS' if drift_ok else 'FAIL'})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Scripted baseline for shirt_distribute: carry → release over each drum.

Validates the MDP before any training (research-report / shirt_place lesson):
grasp-carry-release over EACH of the three commanded drums measures

  * reachability of every drum from the sampled holding-pose init
    distribution (catches unreachable bins for free),
  * the success accounting (release-required particle-fraction metric),
  * carry times → sets the episode length.

Controller: damped-least-squares differential IK on the arm's position
jacobian (position only — the hanging cloth doesn't care about wrist
orientation), emitted through the env's own RELATIVE joint-position actions,
so the exact training action path is exercised.  Phases per env batch:

  CARRY   servo the fingertip to (drum_x, drum_y, carry_z)
  LOWER   descend until the cloth's lowest particle nears the drum rim
  RELEASE open the gripper (+1) — cloth falls
  SETTLE  wait; read the env's own ``_was_distributed`` latch + fraction

Run once per commanded bin (the target bin is forced after reset).

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/baseline_shirt_distribute.py \
        --headless --num_envs 8
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted shirt_distribute baseline.")
parser.add_argument("--task", type=str, default="Template-Shirt-Distribute-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=5)
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
    DRUM_NAMES,
    DRUM_POSITIONS,
)
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import (  # noqa: E402
    DRUM_HEIGHT_M,
)

CLOSE, OPEN = -1.0, 1.0        # binary gripper term: action < 0 ⇒ close
ACTION_SCALE = 0.05            # must match RelativeJointPositionActionCfg.scale
# Fingertip height during the carry: all three drums are 0.6 m lateral from
# the base (0.75, 1.0, 0.75) — 1.45 keeps the flange comfortably inside the
# UR5e envelope while the grasp point stays above the 0.88 m drum top for
# the release; long-drape hangs may brush the rims en route (acceptable).
CARRY_Z = 1.45
RIM_TARGET = DRUM_HEIGHT_M + 0.06   # lower until cloth bottom reaches this
CARRY_STEPS = 360              # 6 s cap
LOWER_STEPS = 120              # 2 s cap
SETTLE_STEPS = 150             # 2.5 s post-release
XY_TOL = 0.06                  # carry convergence tolerance (m)
K_P = 4.0                      # servo gain (m/s per m error)
V_MAX = 1.2                    # max commanded tip speed (m/s)
DLS_LAMBDA = 0.05              # damped-least-squares damping


def main() -> None:
    torch.manual_seed(args_cli.seed)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.episode_length_s = 1.0e6  # scripted rollout — no timeout resets
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    n = u.num_envs
    dev = u.device
    robot = u.scene["robot"]
    arm_ids = u._arm_joint_ids
    a_dim = u.action_manager.total_action_dim
    jacobi_body = u._ee_body_idx - 1  # fixed-base articulation
    dt = u.cfg.sim.dt * u.cfg.decimation

    def tip_pos() -> torch.Tensor:
        return u._finger_tip_pos() - u.scene.env_origins

    def servo_action(waypoint: torch.Tensor, grip: float) -> torch.Tensor:
        """One DLS-IK step toward the env-local waypoint [N, 3]."""
        err = waypoint - tip_pos()
        v = K_P * err
        speed = v.norm(dim=-1, keepdim=True)
        v = torch.where(speed > V_MAX, v * (V_MAX / speed), v)
        jac = robot.root_physx_view.get_jacobians()[:, jacobi_body, :3, :][:, :, arm_ids]
        jjt = torch.bmm(jac, jac.transpose(1, 2))
        jjt += (DLS_LAMBDA ** 2) * torch.eye(3, device=dev).unsqueeze(0)
        dq = torch.bmm(jac.transpose(1, 2), torch.linalg.solve(jjt, (v * dt).unsqueeze(-1)))
        act = torch.zeros(n, a_dim, device=dev)
        act[:, : len(arm_ids)] = torch.clamp(dq.squeeze(-1) / ACTION_SCALE, -1.0, 1.0)
        act[:, -1] = grip
        return act

    print(f"\n── scripted baseline: {args_cli.task}, {n} envs ──────────────")
    overall = []
    for b, (name, pos) in enumerate(zip(DRUM_NAMES, DRUM_POSITIONS)):
        with torch.inference_mode():
            env.reset()
            u._target_bin[:] = b
        waypoint = torch.tensor([pos[0], pos[1], CARRY_Z], device=dev).expand(n, 3)

        # CARRY
        t0 = time.perf_counter()
        carry_steps = torch.full((n,), CARRY_STEPS, device=dev)
        with torch.inference_mode():
            for s in range(CARRY_STEPS):
                env.step(servo_action(waypoint, CLOSE))
                d_xy = (tip_pos()[:, :2] - waypoint[:, :2]).norm(dim=-1)
                arrived = d_xy < XY_TOL
                carry_steps = torch.minimum(
                    carry_steps,
                    torch.where(arrived, torch.full_like(carry_steps, s + 1), carry_steps),
                )
                if arrived.all():
                    break
        d_xy = (tip_pos()[:, :2] - waypoint[:, :2]).norm(dim=-1)
        held = u.grasp_active.clone()

        # LOWER until the cloth bottom nears the rim
        with torch.inference_mode():
            for _ in range(LOWER_STEPS):
                bottom = u.shirt_min_z_w - u.scene.env_origins[:, 2]
                done = bottom <= RIM_TARGET
                if done.all():
                    break
                wp = waypoint.clone()
                wp[:, 2] = torch.where(done, tip_pos()[:, 2], tip_pos()[:, 2] - 0.15)
                env.step(servo_action(wp, CLOSE))

        # RELEASE + SETTLE
        with torch.inference_mode():
            for _ in range(SETTLE_STEPS):
                hold_wp = tip_pos().clone()
                env.step(servo_action(hold_wp, OPEN))

        frac = u._target_fraction_in_bin()
        success = u._was_distributed.clone()
        carry_s = carry_steps.float() * dt
        wall = time.perf_counter() - t0
        overall.append((name, success, frac, carry_s, d_xy, held))
        print(f"[bin {b} {name:16s}] success {int(success.sum())}/{n} | "
              f"fraction mean {frac.mean():.2f} min {frac.min():.2f} | "
              f"carry {carry_s.mean():.2f}s (max {carry_s.max():.2f}s, "
              f"unconverged {int((carry_s >= CARRY_STEPS * dt).sum())}) | "
              f"final xy err {d_xy.max():.3f} m | held@arrival {int(held.sum())}/{n} | "
              f"wall {wall:.0f}s")

    total = torch.cat([s for _, s, _, _, _, _ in overall])
    print(f"\nRESULT: baseline distribute success {int(total.sum())}/{total.numel()} "
          f"({total.float().mean():.2f}) across {len(DRUM_NAMES)} bins x {n} envs")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

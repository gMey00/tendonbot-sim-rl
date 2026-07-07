"""Headless validation of the shirt_place task pipeline (no training).

Builds the registered Tensegrity-Shirt-Place env, runs a short zero-action
rollout, and checks the cloth + reward + metric plumbing:

  1. Env builds and the ParticleClothView is acquired.
  2. Randomized placement: the shirt is laid flat at varied poses per env.
  3. Belt collision: the cloth settles on the box collider (does NOT tunnel
     through to the floor at z≈0).
  4. Rewards and the success metric are finite and computable.
  5. Success path: teleporting the cloth into the drum makes the
     particle-in-drum fraction and the ``shirt_in_target`` reward fire.

Usage::

    conda run -n env_isaaclab python3 scripts/model_validation/validate_shirt_place.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle_steps", type=int, default=200)
parser.add_argument("--record", default=None, help="Directory to write an MP4 + key PNGs.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.record:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import tensegrity_pick.tasks  # noqa: E402,F401  (registers gym envs)
from tensegrity_pick.tasks.manager_based.shirt_place.shirt_place_scene_cfg import (  # noqa: E402
    CONVEYOR_SURFACE_HEIGHT_M,
)

ENV_ID = "Template-Tensegrity-Shirt-Place-Play-v0"


def main() -> None:
    from isaaclab_tasks.utils import parse_env_cfg

    # The record camera is a single prim under env_0, so recording only works
    # with one environment (the camera sensor's reset indexes all env_ids and
    # asserts out of bounds otherwise).  Force a single env when recording.
    num_envs = args_cli.num_envs
    if args_cli.record and num_envs != 1:
        print(f"[record] forcing num_envs=1 (was {num_envs}) for single-camera capture.")
        num_envs = 1
    env_cfg = parse_env_cfg(ENV_ID, num_envs=num_envs)

    recorder = None
    if args_cli.record:
        from _recorder import add_record_camera  # noqa
        add_record_camera(env_cfg.scene, prim_path="/World/envs/env_0/RecordCam")
        env_cfg.scene.dome_light.spawn.intensity = 1500.0  # avoid blown-out renders

    env = gym.make(ENV_ID, cfg=env_cfg).unwrapped

    if args_cli.record:
        from _recorder import FrameRecorder
        cam = env.scene["record_cam"]
        # Close follow-shot of the shirt on the belt (gripper stays above frame).
        # Capture every 2 control steps (control dt = 1/60 s) so 30 fps plays the
        # recording back in REAL TIME (was every=4 @ 20 fps → 1.33× too fast).
        recorder = FrameRecorder(
            cam, every=2,
            tracker=lambda: env.cloth.centroid_pos_w[0],
            eye_offset=(0.60, -0.65, 0.42),
        )

    def _aim_cam_at_shirt() -> None:  # kept for call-site compatibility
        if recorder is not None:
            recorder._reaim()

    n = env.num_envs
    belt = CONVEYOR_SURFACE_HEIGHT_M
    problems: list[str] = []
    print("=" * 70)
    print(f"  Shirt-Place validation  ({n} envs, belt z={belt:.3f})")
    print("=" * 70)

    obs, _ = env.reset()
    cloth = env.cloth
    _aim_cam_at_shirt()
    print(f"[build] cloth particles/env = {cloth.num_particles}")
    print(f"[build] obs dim = {obs['policy'].shape}")
    print(f"[build] action dim = {env.action_manager.total_action_dim}")

    # Per-env placement spread right after reset (randomized lay-flat).
    cloth.update()
    c0 = cloth.centroid_pos_w - env.scene.env_origins
    print(f"[place] post-reset centroid XY per env:\n{c0[:, :2].cpu().numpy()}")
    if n > 1 and torch.allclose(c0[0, :2], c0[1, :2], atol=1e-4):
        problems.append("Randomized placement produced identical poses across envs.")

    # Zero-action rollout — cloth should settle ON the belt box.
    zero = torch.zeros((n, env.action_manager.total_action_dim), device=env.device)
    reward_finite = True
    for i in range(args_cli.settle_steps):
        _, rew, _, _, _ = env.step(zero)
        if recorder is not None:
            recorder.maybe_capture()
        if not torch.isfinite(rew).all():
            reward_finite = False
        if (i + 1) % 50 == 0:
            cloth.update()
            zc = cloth.nodal_pos_w[..., 2] - env.scene.env_origins[:, 2:3]
            minz = zc.min(dim=1).values
            meanz = zc.mean(dim=1)
            print(f"  step {i+1:3d}: cloth meanZ={meanz.mean():.3f} "
                  f"minZ={minz.mean():.3f} reward={rew.mean():.2f}")

    cloth.update()
    minz = (cloth.nodal_pos_w[..., 2] - env.scene.env_origins[:, 2:3]).min(dim=1).values
    on_belt = (minz > belt - 0.06).all().item()      # not tunneled to floor
    not_floor = (minz > 0.20).all().item()
    print(f"[belt] min particle Z per env = {minz.cpu().numpy()}")
    if not reward_finite:
        problems.append("Non-finite reward encountered during rollout.")
    if not not_floor:
        problems.append("Cloth fell through the belt box to the floor (min Z <= 0.20).")
    elif not on_belt:
        problems.append("Cloth settled below the belt surface (collision box too low?).")

    # Success path: teleport the cloth into the drum, confirm metric + reward.
    drum = env.scene["drum_target"]
    from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import get_world_pos
    drum_w = get_world_pos(drum, env=env)
    env_ids = torch.arange(n, device=env.device)
    target = drum_w.clone()
    target[:, 2] = env.scene.env_origins[:, 2] + 0.12  # inside the drum cylinder
    yaws = torch.zeros(n, device=env.device)
    cloth.reset_randomized(env_ids, target, yaws)
    env._was_grasped[:] = True  # success reward is gated on having grasped
    cloth.update()
    env._sync_proxy_to_cloth()
    frac = env._shirt_particle_fraction_in_drum()
    in_drum = env._shirt_in_drum().float()
    place_frac = frac.mean().item()
    print(f"[success] particle-in-drum fraction after teleport = {place_frac:.3f}")
    print(f"[success] proxy-in-drum (shirt_in_target gate) = {in_drum.mean():.2f}")
    if place_frac < 0.5:
        problems.append(f"Teleporting cloth into drum gave low in-drum fraction ({place_frac:.2f}).")
    if in_drum.mean().item() < 0.5:
        problems.append("Proxy not detected in drum after teleport (success gate broken).")

    # Showcase randomized placement: a few resets, each settling on the belt.
    if recorder is not None:
        for _ in range(3):
            env.reset()
            _aim_cam_at_shirt()
            for _ in range(60):
                env.step(zero)
                recorder.maybe_capture()
        recorder.save(args_cli.record, "shirt_place_run", fps=30)

    print("-" * 70)
    if problems:
        print("RESULT: FAIL")
        for p in problems:
            print(f"  - {p}")
    else:
        print("RESULT: PASS — env builds, cloth rests on belt, rewards finite, "
              "success metric works.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

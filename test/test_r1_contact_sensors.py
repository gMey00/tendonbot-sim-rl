"""V1 verification: ContactSensor wiring on the cube_sort gripper pads.

This test guards Iteration R1 of the cube_sort rework. After enabling
``activate_contact_sensors=True`` on the robot and cube spawners and adding
the two per-pad ``ContactSensorCfg`` entries to the scene, this test asserts
that:

* both sensors produce a populated ``force_matrix_w`` tensor (shape
  ``(N_envs, 1, NUM_CUBES_TOTAL, 3)``); and
* the ``net_forces_w`` tensor has the expected per-pad shape.

A ``None`` ``force_matrix_w`` indicates the sensor regex matched zero
filter prims (Isaac Lab Issue #364) — i.e. the wiring is broken.

References:
- doc/reports/tracking/cube_sort_optimization_tracking.md (Iteration R1)
"""

from __future__ import annotations

import gymnasium as gym
import pytest


@pytest.mark.requires_isaac
def test_v1_contact_sensors_wired() -> None:
    """V1: per-pad contact sensors expose ``force_matrix_w`` with the right shape."""

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    try:
        import tensegrity_pick  # noqa: F401  — triggers gym.register calls
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-untyped]

        from tensegrity_pick.tasks.manager_based.cube_sort.cube_sorting_scene_cfg import (
            NUM_CUBES_TOTAL,
        )

        env_id = "Template-Tensegrity-Cube-Sort-v0"
        num_envs = 2

        env_cfg = parse_env_cfg(env_id, num_envs=num_envs)
        env = gym.make(env_id, cfg=env_cfg)
        try:
            env.reset()
            # Step a few times to populate the sensor buffers. Need a torch
            # tensor on the env device because Isaac Lab's action manager calls
            # ``action.to(self.device)``.
            import torch

            action_np = env.action_space.sample()
            action = torch.as_tensor(action_np, device=env.unwrapped.device)
            for _ in range(5):
                env.step(action)

            scene = env.unwrapped.scene  # type: ignore[attr-defined]
            for sensor_name in ("contact_left", "contact_right"):
                sensor = scene[sensor_name]
                fm = sensor.data.force_matrix_w
                nf = sensor.data.net_forces_w
                assert fm is not None, (
                    f"{sensor_name}.force_matrix_w is None — "
                    "filter_prim_paths_expr matched no prims (check pad body name "
                    "or cube prim regex)."
                )
                assert fm.shape == (num_envs, 1, NUM_CUBES_TOTAL, 3), (
                    f"{sensor_name}.force_matrix_w shape {tuple(fm.shape)} "
                    f"!= expected {(num_envs, 1, NUM_CUBES_TOTAL, 3)}"
                )
                assert nf.shape[0] == num_envs and nf.shape[-1] == 3, (
                    f"{sensor_name}.net_forces_w has unexpected shape {tuple(nf.shape)}"
                )
        finally:
            env.close()
    finally:
        launcher.app.close()

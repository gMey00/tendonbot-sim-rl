"""Smoke tests for every registered gymnasium environment.

Each test creates the environment, calls ``reset()`` and ``step()`` a handful
of times, and asserts that observation and action shapes are consistent.

All tests require Isaac Sim and are skipped automatically when the simulator
is not available (see ``conftest.py``).
"""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

# ── Environment registry ─────────────────────────────────────────────────

ALL_ENV_IDS: list[str] = [
    "Template-Tensegrity-Cube-Sort-v0",
    "Template-Tensegrity-Cube-Place-v0",
    "Template-Tensegrity-Cube-Place-Play-v0",
    "Template-Tensegrity-Cube-Place-Tendon-v0",
    "Template-Tensegrity-Cube-Place-Tendon-Play-v0",
    "Template-Tensegrity-Reach-v0",
    "Template-Tensegrity-Reach-Play-v0",
    "Template-Tensegrity-Reach-Tendon-v0",
    "Template-Tensegrity-Reach-Tendon-Play-v0",
]

# Physical tendon envs require the 5-DOF physical USD (Robot Assembler).
# Add them to the smoke test list once the USD is assembled.
PHYSICAL_ENV_IDS: list[str] = [
    "Template-Reach-Tensegrity-Physical-Tendon-v0",
    "Template-Reach-Tensegrity-Physical-Tendon-Play-v0",
    "Template-Tensegrity-Cube-Place-Physical-Tendon-v0",
    "Template-Tensegrity-Cube-Place-Physical-Tendon-Play-v0",
]

NUM_ENVS = 2
NUM_STEPS = 10


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def simulation_app():
    """Start the Isaac Sim application once for all environment tests."""
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    yield launcher.app
    launcher.app.close()


def _make_env(env_id: str, simulation_app: object) -> gym.Env:  # noqa: ARG001
    """Instantiate a single environment with a small number of parallel envs."""
    import tensegrity_pick  # noqa: F401  — triggers gym.register calls

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-untyped]

    env_cfg = parse_env_cfg(env_id, num_envs=NUM_ENVS)
    return gym.make(env_id, cfg=env_cfg)


# ── Parametrised smoke tests ─────────────────────────────────────────────

@pytest.mark.requires_isaac
class TestEnvironmentSmoke:
    """Basic create → reset → step loop for every registered environment."""

    @pytest.fixture(params=ALL_ENV_IDS)
    def env(self, request: pytest.FixtureRequest, simulation_app: object) -> gym.Env:
        environment = _make_env(request.param, simulation_app)
        yield environment
        environment.close()

    def test_that_reset_returns_observations(self, env: gym.Env) -> None:
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)

    def test_that_observations_have_batch_dimension(self, env: gym.Env) -> None:
        obs, _ = env.reset()
        if isinstance(obs, dict):
            for value in obs.values():
                assert value.shape[0] == NUM_ENVS
        else:
            assert obs.shape[0] == NUM_ENVS

    def test_that_step_returns_correct_tuple(self, env: gym.Env) -> None:
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, "Expected (obs, reward, terminated, truncated, info)"

    def test_that_step_loop_does_not_crash(self, env: gym.Env) -> None:
        env.reset()
        for _ in range(NUM_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

    def test_that_observation_shape_is_stable_across_steps(self, env: gym.Env) -> None:
        obs_initial, _ = env.reset()
        initial_shape = _obs_shape(obs_initial)
        for _ in range(NUM_STEPS):
            action = env.action_space.sample()
            obs, *_ = env.step(action)
            assert _obs_shape(obs) == initial_shape

    def test_that_reward_has_batch_dimension(self, env: gym.Env) -> None:
        env.reset()
        action = env.action_space.sample()
        _, reward, *_ = env.step(action)
        if isinstance(reward, torch.Tensor):
            assert reward.shape[0] == NUM_ENVS
        else:
            assert len(reward) == NUM_ENVS


# ── Helpers ───────────────────────────────────────────────────────────────

def _obs_shape(obs: object) -> object:
    """Return a hashable shape descriptor for observations."""
    if isinstance(obs, dict):
        return {k: tuple(v.shape) for k, v in obs.items()}
    return tuple(obs.shape)  # type: ignore[union-attr]

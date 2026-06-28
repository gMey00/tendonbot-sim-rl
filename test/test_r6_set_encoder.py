"""R6 forward-pass test for ``CubeSetPolicy`` and ``CubeSetValue``.

CPU-only, no Isaac Lab needed — verifies the layout/slice math + that
both heads return the expected shapes.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

# ---- import the model module by file path so we don't need the full pkg ----

_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/models/cube_set_policy.py"
)


@pytest.fixture(scope="module")
def cube_set():
    spec = importlib.util.spec_from_file_location("cube_set_policy_under_test", _MODEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_spaces(obs_dim: int, act_dim: int):
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    return obs_space, act_space


def test_layout_slice_math(cube_set):
    layout = cube_set.CubeSetLayout(non_set_dim=34, set_per_cube_dim=15, n_max_cubes=16, privileged_dim=32)
    assert layout.set_start == 34
    assert layout.set_end == 34 + 15 * 16  # 274
    assert layout.mask_start == 274
    assert layout.mask_end == 290
    assert layout.priv_end == 322
    assert layout.total_dim == 322

    x = torch.arange(322 * 4, dtype=torch.float32).view(4, 322)
    non_set, set_3d, mask, priv = layout.split(x)
    assert non_set.shape == (4, 34)
    assert set_3d.shape == (4, 16, 15)
    assert mask.shape == (4, 16)
    assert priv.shape == (4, 32)


def test_policy_forward_shape(cube_set):
    layout = cube_set.CubeSetLayout(non_set_dim=34, set_per_cube_dim=15, n_max_cubes=16, privileged_dim=32)
    obs_space, act_space = _make_spaces(layout.total_dim, act_dim=6)
    policy = cube_set.CubeSetPolicy(
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        layout=layout,
    )
    n_envs = 8
    x = torch.randn(n_envs, layout.total_dim)
    mean, log_std, extras = policy.compute({"states": x}, role="policy")
    assert mean.shape == (n_envs, 6)
    assert log_std.shape == (6,)
    assert extras == {}


def test_value_forward_shape(cube_set):
    layout = cube_set.CubeSetLayout(non_set_dim=34, set_per_cube_dim=15, n_max_cubes=16, privileged_dim=32)
    obs_space, act_space = _make_spaces(layout.total_dim, act_dim=6)
    value = cube_set.CubeSetValue(
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        layout=layout,
    )
    n_envs = 8
    x = torch.randn(n_envs, layout.total_dim)
    v, extras = value.compute({"states": x}, role="value")
    assert v.shape == (n_envs, 1)
    assert extras == {}


def test_encoder_masking_invariance(cube_set):
    """Padded slots (mask=0) must not influence the pooled vector."""
    layout = cube_set.CubeSetLayout(non_set_dim=4, set_per_cube_dim=3, n_max_cubes=5, privileged_dim=0)
    enc = cube_set.CubeSetEncoder(in_dim=3, hidden=(8,), out_dim=4)
    enc.eval()
    set_a = torch.randn(2, 5, 3)
    mask_a = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)

    set_b = set_a.clone()
    set_b[:, 2:] = torch.randn(2, 3, 3) * 99.0  # garbage in padded slots

    with torch.no_grad():
        out_a = enc(set_a, mask_a)
        out_b = enc(set_b, mask_a)
    assert torch.allclose(out_a, out_b, atol=1e-5), "padded slots leak into pooled vector"

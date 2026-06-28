"""V4 verification: per-cube ``is_placed`` predicate (Iteration R3).

Pure-tensor unit tests covering the gentle-release and fly-through cases
called out in :doc:`doc/reports/cube_sort_optimization_tracking.md` (R3),
plus the not-held requirement and the sticky-latch behaviour.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def _placement_module():
    here = pathlib.Path(__file__).resolve().parent.parent
    mod_path = (
        here
        / "src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick"
        / "tasks/manager_based/cube_sort/mdp/placement.py"
    )
    if not mod_path.exists():
        pytest.skip(f"placement module not found at {mod_path}")
    spec = importlib.util.spec_from_file_location("_cube_sort_placement_under_test", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_env_stub(
    *,
    p_cube: torch.Tensor,
    v_lin: torch.Tensor,
    v_ang: torch.Tensor,
    num_envs: int = 1,
) -> SimpleNamespace:
    cubes = SimpleNamespace(
        data=SimpleNamespace(
            object_pos_w=p_cube,
            object_lin_vel_w=v_lin,
            object_ang_vel_w=v_ang,
        )
    )
    return SimpleNamespace(
        scene={"cubes": cubes},
        num_envs=num_envs,
        device=torch.device("cpu"),
    )


def test_v4_gentle_release_latches_after_dwell(_placement_module) -> None:
    """Cube at rest inside drum, not held → latches True after DWELL_STEPS."""
    pm = _placement_module
    drum_center = torch.tensor([[0.0, 0.0, 0.0]])              # (1, 3)
    # cube seated at floor + cube_half: z = 0 + 0.025 + small
    p_cube = torch.tensor([[[0.0, 0.0, pm.CUBE_HALF + 0.005]]])  # (1, 1, 3)
    v_lin = torch.zeros((1, 1, 3))
    v_ang = torch.zeros((1, 1, 3))
    held = torch.tensor([[False]])
    env = _make_env_stub(p_cube=p_cube, v_lin=v_lin, v_ang=v_ang)

    out = None
    for _ in range(pm.DWELL_STEPS + 1):
        out = pm.is_placed(env, drum_center, held)
    assert out is not None and out.shape == (1, 1)
    assert bool(out[0, 0].item()) is True


def test_v4_flythrough_stays_false(_placement_module) -> None:
    """Cube flies horizontally through drum > 1 m/s → never latches True."""
    pm = _placement_module
    drum_center = torch.tensor([[0.0, 0.0, 0.0]])
    p_cube = torch.tensor([[[0.0, 0.0, pm.CUBE_HALF + 0.005]]])
    v_lin = torch.tensor([[[1.5, 0.0, 0.0]]])  # 1.5 m/s along +x → v_ok=False
    v_ang = torch.zeros((1, 1, 3))
    held = torch.tensor([[False]])
    env = _make_env_stub(p_cube=p_cube, v_lin=v_lin, v_ang=v_ang)

    out = None
    for _ in range(20):
        out = pm.is_placed(env, drum_center, held)
    assert out is not None
    assert bool(out[0, 0].item()) is False


def test_v4_held_cube_inside_drum_does_not_latch(_placement_module) -> None:
    """Even if all geometric/velocity conjuncts hold, ``held=True`` blocks the latch."""
    pm = _placement_module
    drum_center = torch.tensor([[0.0, 0.0, 0.0]])
    p_cube = torch.tensor([[[0.0, 0.0, pm.CUBE_HALF + 0.005]]])
    v_lin = torch.zeros((1, 1, 3))
    v_ang = torch.zeros((1, 1, 3))
    held = torch.tensor([[True]])  # ← carried through
    env = _make_env_stub(p_cube=p_cube, v_lin=v_lin, v_ang=v_ang)

    out = None
    for _ in range(pm.DWELL_STEPS + 5):
        out = pm.is_placed(env, drum_center, held)
    assert out is not None
    assert bool(out[0, 0].item()) is False


def test_v4_latch_is_sticky_after_release(_placement_module) -> None:
    """Once latched True, ``is_placed`` stays True even if the cube later moves out."""
    pm = _placement_module
    drum_center = torch.tensor([[0.0, 0.0, 0.0]])
    p_inside = torch.tensor([[[0.0, 0.0, pm.CUBE_HALF + 0.005]]])
    p_outside = torch.tensor([[[1.0, 0.0, 0.5]]])
    v_zero = torch.zeros((1, 1, 3))
    held = torch.tensor([[False]])

    env = _make_env_stub(p_cube=p_inside, v_lin=v_zero, v_ang=v_zero)
    for _ in range(pm.DWELL_STEPS + 1):
        out = pm.is_placed(env, drum_center, held)
    assert bool(out[0, 0].item()) is True

    # Move the cube outside the drum; latch must NOT clear.
    env.scene["cubes"].data.object_pos_w = p_outside
    out = pm.is_placed(env, drum_center, held)
    assert bool(out[0, 0].item()) is True


def test_v4_reset_clears_latch(_placement_module) -> None:
    """``reset_placement_state(env, env_ids)`` clears the latch for the given envs."""
    pm = _placement_module
    drum_center = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # (2, 3)
    p_cube = torch.tensor(
        [
            [[0.0, 0.0, pm.CUBE_HALF + 0.005]],
            [[0.0, 0.0, pm.CUBE_HALF + 0.005]],
        ]
    )  # (2, 1, 3)
    v_zero = torch.zeros((2, 1, 3))
    held = torch.tensor([[False], [False]])
    env = _make_env_stub(p_cube=p_cube, v_lin=v_zero, v_ang=v_zero, num_envs=2)

    for _ in range(pm.DWELL_STEPS + 1):
        out = pm.is_placed(env, drum_center, held)
    assert bool(out[0, 0].item()) is True
    assert bool(out[1, 0].item()) is True

    pm.reset_placement_state(env, torch.tensor([0]))
    out = pm.is_placed(env, drum_center, held)
    # env 0 was reset → must drop back to False on the next call (dwell=1 < threshold).
    assert bool(out[0, 0].item()) is False
    # env 1 was untouched → still latched True.
    assert bool(out[1, 0].item()) is True

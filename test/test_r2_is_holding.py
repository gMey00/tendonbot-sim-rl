"""V2 verification: per-cube ``is_holding`` predicate (Iteration R2).

This test is intentionally **pure-tensor**: it exercises the predicate
logic in :mod:`tensegrity_pick.tasks.manager_based.cube_sort.mdp.grasp`
against three synthetic scenarios (close → True after dwell, open → False,
flythrough → False) without spinning up Isaac Sim.

A scripted in-engine V2 would require a custom action override to hold
the arm still and actuate only the gripper, which is much heavier than
the predicate logic itself warrants.  The predicate is a pure function
of sensor and articulation tensors, so this synthetic check is sufficient
to lock the conjunct semantics in place.

Scenarios:
- **close**: bilateral force above F_MIN, anti-parallel normals,
  co-moving with TCP, above belt, gripper angle in range
  → ``is_holding`` becomes ``True`` after ``DWELL_STEPS`` steps.
- **open**: zero force on both pads → ``is_holding`` is ``False``.
- **flythrough**: bilateral force above F_MIN (a glancing contact) but
  cube velocity is 3 m/s while TCP is stationary → ``co_moving`` fails
  → ``is_holding`` stays ``False`` even after many steps.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def _grasp_module():
    """Load the grasp module by file path so we don't transitively import
    isaaclab.managers (which requires ``omni.timeline``).

    The grasp module itself uses only ``torch`` at module load (Isaac
    types are guarded behind ``TYPE_CHECKING``), so direct loading is safe.
    """
    import importlib.util
    import pathlib

    here = pathlib.Path(__file__).resolve().parent.parent
    grasp_path = (
        here
        / "src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick"
        / "tasks/manager_based/cube_sort/mdp/grasp.py"
    )
    if not grasp_path.exists():
        pytest.skip(f"grasp module not found at {grasp_path}")
    spec = importlib.util.spec_from_file_location("_cube_sort_grasp_under_test", grasp_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_env_stub(
    *,
    fl: torch.Tensor,
    fr: torch.Tensor,
    v_cube: torch.Tensor,
    p_cube: torch.Tensor,
    v_tcp: torch.Tensor,
    g_ang: float,
    num_envs: int = 1,
) -> SimpleNamespace:
    """Build a duck-typed stub that satisfies the parts of ``env`` that
    :func:`is_holding` reads from."""

    sensor_l = SimpleNamespace(data=SimpleNamespace(force_matrix_w=fl))
    sensor_r = SimpleNamespace(data=SimpleNamespace(force_matrix_w=fr))

    robot_data = SimpleNamespace(
        body_names=["tool_link_0"],
        joint_names=["finger_joint"],
        body_lin_vel_w=v_tcp.unsqueeze(1),  # (N, 1, 3) — indexed [:, 0]
        joint_pos=torch.full((num_envs, 1), g_ang),
    )
    robot = SimpleNamespace(data=robot_data)

    cubes_data = SimpleNamespace(object_lin_vel_w=v_cube, object_pos_w=p_cube)
    cubes = SimpleNamespace(data=cubes_data)

    scene = {"robot": robot, "cubes": cubes,
             "contact_left": sensor_l, "contact_right": sensor_r}
    env = SimpleNamespace(scene=scene, num_envs=num_envs, device=torch.device("cpu"))
    return env


def _drive(env_stub, grasp_mod, n_steps: int) -> torch.Tensor:
    """Call is_holding `n_steps` times; return the last output."""
    out = None
    for _ in range(n_steps):
        out = grasp_mod.is_holding(env_stub)
    return out


def test_v2_close_grasp_latches_true_after_dwell(_grasp_module) -> None:
    """Bilateral grip + co-motion → True after DWELL_STEPS steps."""
    grasp_mod = _grasp_module
    # left pad pushes +x with 1 N, right pad pushes -x with 1 N (anti-parallel).
    fl = torch.tensor([[[[1.0, 0.0, 0.0]]]])   # (1, 1, 1, 3)
    fr = torch.tensor([[[[-1.0, 0.0, 0.0]]]])  # (1, 1, 1, 3)
    v_cube = torch.tensor([[[0.0, 0.0, 0.0]]])  # (1, 1, 3)
    p_cube = torch.tensor([[[0.0, 0.0, 0.10]]])  # 10 cm above belt
    v_tcp = torch.tensor([[0.0, 0.0, 0.0]])     # (1, 3)
    env = _make_env_stub(
        fl=fl, fr=fr, v_cube=v_cube, p_cube=p_cube, v_tcp=v_tcp, g_ang=0.5
    )
    out = _drive(env, grasp_mod, n_steps=grasp_mod.DWELL_STEPS + 2)
    assert out.shape == (1, 1)
    assert bool(out[0, 0].item()) is True, (
        "is_holding should latch True after DWELL_STEPS consecutive frames "
        "with all conjuncts satisfied."
    )


def test_v2_open_grasp_stays_false(_grasp_module) -> None:
    """Zero force on both pads → False forever."""
    grasp_mod = _grasp_module
    fl = torch.zeros((1, 1, 1, 3))
    fr = torch.zeros((1, 1, 1, 3))
    v_cube = torch.tensor([[[0.0, 0.0, 0.0]]])
    p_cube = torch.tensor([[[0.0, 0.0, 0.10]]])
    v_tcp = torch.tensor([[0.0, 0.0, 0.0]])
    env = _make_env_stub(
        fl=fl, fr=fr, v_cube=v_cube, p_cube=p_cube, v_tcp=v_tcp, g_ang=0.5
    )
    out = _drive(env, grasp_mod, n_steps=10)
    assert bool(out[0, 0].item()) is False


def test_v2_flythrough_stays_false(_grasp_module) -> None:
    """Cube velocity differs from TCP velocity by 3 m/s → co_moving fails.

    This is the MOST IMPORTANT regression test in the cube_sort rework:
    the previous ``grasp_active`` would flag a fast cube glancing past
    the closed gripper as a successful grasp.
    """
    grasp_mod = _grasp_module
    # bilateral contact present, normals opposing — but cube is flying.
    fl = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    fr = torch.tensor([[[[-1.0, 0.0, 0.0]]]])
    v_cube = torch.tensor([[[3.0, 0.0, 0.0]]])  # 3 m/s along +x
    p_cube = torch.tensor([[[0.0, 0.0, 0.10]]])
    v_tcp = torch.tensor([[0.0, 0.0, 0.0]])     # TCP stationary
    env = _make_env_stub(
        fl=fl, fr=fr, v_cube=v_cube, p_cube=p_cube, v_tcp=v_tcp, g_ang=0.5
    )
    out = _drive(env, grasp_mod, n_steps=10)
    assert bool(out[0, 0].item()) is False, (
        "Flythrough must NOT register as a held cube — the co_moving "
        "conjunct (relative speed < V_CO_MOVING) is the guard against this."
    )


def test_v2_open_clears_after_close(_grasp_module) -> None:
    """A True latch must drop back to False within DWELL_STEPS once contact ends."""
    grasp_mod = _grasp_module
    # close: True for several steps
    fl_close = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    fr_close = torch.tensor([[[[-1.0, 0.0, 0.0]]]])
    v_cube = torch.tensor([[[0.0, 0.0, 0.0]]])
    p_cube = torch.tensor([[[0.0, 0.0, 0.10]]])
    v_tcp = torch.tensor([[0.0, 0.0, 0.0]])
    env = _make_env_stub(
        fl=fl_close, fr=fr_close, v_cube=v_cube, p_cube=p_cube, v_tcp=v_tcp, g_ang=0.5
    )
    _ = _drive(env, grasp_mod, n_steps=grasp_mod.DWELL_STEPS + 2)
    # now open: zero force; reuse the same env so the dwell buffer persists.
    env.scene["contact_left"].data.force_matrix_w = torch.zeros((1, 1, 1, 3))
    env.scene["contact_right"].data.force_matrix_w = torch.zeros((1, 1, 1, 3))
    out = _drive(env, grasp_mod, n_steps=grasp_mod.DWELL_STEPS)
    assert bool(out[0, 0].item()) is False

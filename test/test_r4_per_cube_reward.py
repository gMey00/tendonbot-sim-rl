"""V3 verification: per-cube reward machine (Iteration R4).

Tests the Mealy automaton credit-assignment logic in
``mdp/per_cube_state.py`` with the predicate dependencies
(``is_holding``, ``is_placed``) monkey-patched, so we exercise the
*reward composition* in isolation from Isaac Lab physics.

The single critical regression this test guards against is the
"flythrough place" pseudo-reward: a cube that drops through the drum
without ever being grasped must NOT trigger ``r_place``.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def _per_cube_module(monkeypatch):
    here = pathlib.Path(__file__).resolve().parent.parent
    pkg_root = here / "src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick"
    mdp_dir = pkg_root / "tasks/manager_based/cube_sort/mdp"
    if not (mdp_dir / "per_cube_state.py").exists():
        pytest.skip("per_cube_state.py not yet created")

    # Manually load grasp + placement first under stable names so that
    # per_cube_state's relative imports (`from .grasp import ...`) resolve.
    def _load(name: str, path: pathlib.Path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    import sys
    grasp_mod = _load("_test_per_cube_grasp", mdp_dir / "grasp.py")
    place_mod = _load("_test_per_cube_placement", mdp_dir / "placement.py")
    # Stitch them into a synthetic package so the relative import inside
    # per_cube_state resolves to our copies.
    pkg = type(sys)("_test_per_cube_pkg")
    pkg.__path__ = [str(mdp_dir)]
    sys.modules["_test_per_cube_pkg"] = pkg
    sys.modules["_test_per_cube_pkg.grasp"] = grasp_mod
    sys.modules["_test_per_cube_pkg.placement"] = place_mod

    # Now load per_cube_state under the synthetic package so relative imports work.
    spec = importlib.util.spec_from_file_location(
        "_test_per_cube_pkg.per_cube_state",
        mdp_dir / "per_cube_state.py",
    )
    assert spec is not None and spec.loader is not None
    pcs = importlib.util.module_from_spec(spec)
    sys.modules["_test_per_cube_pkg.per_cube_state"] = pcs
    spec.loader.exec_module(pcs)
    return pcs


def _make_env(num_envs=1, num_cubes=3, labels=(0, 0, 0), tcp=(0.0, 0.0, 1.0),
              cubes_pos=None, drum_pos=(0.0, 0.0, 0.0)):
    if cubes_pos is None:
        cubes_pos = torch.zeros((num_envs, num_cubes, 3))
    cubes = SimpleNamespace(
        data=SimpleNamespace(
            object_pos_w=cubes_pos,
            object_lin_vel_w=torch.zeros_like(cubes_pos),
            object_ang_vel_w=torch.zeros_like(cubes_pos),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["tool_link_0"],
            # R8.10: per_cube_reward now projects a local-Z offset
            # (GRASP_CENTER_LOCAL_Z = -0.1925) from the wrist quaternion to
            # locate the gripper opening. To preserve existing test semantics
            # ("tcp" parameter = where the gripper actually grabs), we set
            # body_pos_w shifted upward by +0.1925 along world +Z and supply
            # an identity body_quat_w; the projection then maps back exactly
            # onto the requested ``tcp`` position.
            body_pos_w=(
                torch.tensor([[list(tcp)]]).expand(num_envs, 1, 3).clone()
                + torch.tensor([0.0, 0.0, 0.1925])
            ),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(num_envs, 1, 4).clone(),
        )
    )
    drum = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([list(drum_pos)]).expand(num_envs, 3).clone(),
        )
    )
    return SimpleNamespace(
        scene={"cubes": cubes, "robot": robot, "drum_target": drum},
        num_envs=num_envs,
        device=torch.device("cpu"),
        cube_labels=torch.tensor(list(labels), dtype=torch.int32),
    )


def test_v3_credit_assignment_held_vs_flythrough(_per_cube_module, monkeypatch):
    """3 green cubes; cube 0 grasped, cube 1 flies through drum, cube 2 grasped+placed.

    Asserts:
      - ``ever_held = [True, False, True]``
      - place_edge fires for cube 2 (was grasped) → r_place > 0
      - place_edge does NOT fire for cube 1 (never grasped) → no false credit
    """
    pcs = _per_cube_module

    # We'll script (held, placed_now) per step.
    held_script = [
        torch.tensor([[True, False, False]]),
        torch.tensor([[False, False, True]]),
        torch.tensor([[False, False, False]]),
    ]
    placed_script = [
        torch.tensor([[False, False, False]]),
        torch.tensor([[False, True, True]]),  # cube 1 flies in, cube 2 placed
        torch.tensor([[False, True, True]]),
    ]
    step = {"i": 0}

    def fake_is_holding(env, **_kw):
        return held_script[step["i"]]

    def fake_is_placed(env, drum, held, **_kw):
        return placed_script[step["i"]]

    monkeypatch.setattr(pcs, "is_holding", fake_is_holding)
    monkeypatch.setattr(pcs, "is_placed", fake_is_placed)

    env = _make_env()
    rewards = []
    for i in range(3):
        step["i"] = i
        rewards.append(pcs.per_cube_reward(env).clone())

    state = env._mdp_state
    # ever_held: cube 0 (step 0), cube 2 (step 1), NOT cube 1.
    assert state["ever_held"][0].tolist() == [True, False, True]
    # placed latch only set when ever_held & placed_now
    assert state["placed"][0].tolist() == [False, False, True]
    # R8.15: K_PLACE one-shot edge bonus removed (was destabilising PPO via
    # +149 single-step spike). Placement reward is now per-step
    # K_PLACE_HOLD while cube is in drum AND was ever held. Test the new
    # invariants:
    #   - cube 1 (flythrough, never held) MUST NOT contribute K_PLACE_HOLD
    #   - cube 2 (placed at step 1) contributes K_PLACE_HOLD/step
    #   - false-credit prevention: r_place_hold only counts cubes with ever_held
    place_hold_signal = pcs.K_PLACE_HOLD  # per cube per step
    # Step 1: cube 2 just transitioned to placed (placed_now=True, ever_held=True).
    # Step 2: cube 2 still placed. Both should pay K_PLACE_HOLD ≈ 15.
    # Cube 1 is in drum but never_held → must NOT pay K_PLACE_HOLD.
    # Use slack tolerance for time/regulariser noise.
    assert rewards[1][0].item() >= 0.5 * place_hold_signal, (
        f"Step 1 reward {rewards[1][0].item()} should include K_PLACE_HOLD "
        f"({place_hold_signal}) for cube 2"
    )
    assert rewards[2][0].item() >= 0.5 * place_hold_signal, (
        f"Step 2 reward {rewards[2][0].item()} should still include "
        f"K_PLACE_HOLD ({place_hold_signal}) for cube 2 (per-step bonus)"
    )
    # No false credit for cube 1 (flythrough): if it counted, reward would
    # be ~2 × K_PLACE_HOLD. Verify single-cube credit ceiling.
    assert rewards[2][0].item() < 1.5 * place_hold_signal, (
        f"Step 2 reward {rewards[2][0].item()} suggests cube 1 (flythrough) "
        f"is being falsely credited with K_PLACE_HOLD"
    )


def test_v3_red_drop_penalty(_per_cube_module, monkeypatch):
    """Holding a red cube and dropping it in the drum → r_red_drop fires."""
    pcs = _per_cube_module
    # Cube 0 is red (label != target_label=0)  -> use labels=(1,) so role=-1.
    held_script = [torch.tensor([[True]]), torch.tensor([[False]])]
    placed_script = [torch.tensor([[False]]), torch.tensor([[True]])]
    step = {"i": 0}

    def fake_is_holding(env, **_kw): return held_script[step["i"]]
    def fake_is_placed(env, drum, held, **_kw): return placed_script[step["i"]]
    monkeypatch.setattr(pcs, "is_holding", fake_is_holding)
    monkeypatch.setattr(pcs, "is_placed", fake_is_placed)

    env = _make_env(num_cubes=1, labels=(1,))
    step["i"] = 0
    r0 = pcs.per_cube_reward(env)
    step["i"] = 1
    r1 = pcs.per_cube_reward(env)

    # r0: held red → -K_RED penalty (plus small extras). Should be < 0.
    assert r0.item() < 0.0
    # r1: red placed in drum (red & place_edge & ever_held) → -K_RED_DROP.
    assert r1.item() < -pcs.K_RED_DROP * 0.5


def test_v3_reset_clears_state(_per_cube_module, monkeypatch):
    """``reset_mdp_state`` zeroes all per-cube buffers for the given env_ids."""
    pcs = _per_cube_module
    monkeypatch.setattr(pcs, "is_holding", lambda env, **_kw: torch.tensor([[True]]))
    monkeypatch.setattr(pcs, "is_placed", lambda env, drum, held, **_kw: torch.tensor([[True]]))

    env = _make_env(num_cubes=1, labels=(0,))
    pcs.per_cube_reward(env)
    pcs.per_cube_reward(env)
    assert env._mdp_state["ever_held"][0, 0].item() is True
    assert env._mdp_state["placed"][0, 0].item() is True

    pcs.reset_mdp_state(env, torch.tensor([0]))
    assert env._mdp_state["ever_held"][0, 0].item() is False
    assert env._mdp_state["placed"][0, 0].item() is False

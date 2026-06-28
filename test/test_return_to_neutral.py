"""Unit tests for the ``return_to_neutral`` reward function.

These tests verify the normalized-distance implementation without Isaac Sim by
building minimal torch-tensor mocks.  No GPU or Isaac Lab import needed.
"""

from __future__ import annotations

import math
import types

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers: build lightweight mocks of the env / robot / asset_cfg objects
# ---------------------------------------------------------------------------

def _make_robot(
    current: list[list[float]],
    default: list[list[float]],
    limits: list[list[list[float]]],
) -> object:
    """Return a mock articulation whose .data attributes are real tensors."""
    device = "cpu"
    cur_t = torch.tensor(current, dtype=torch.float32, device=device)    # (N, J)
    def_t = torch.tensor(default, dtype=torch.float32, device=device)    # (N, J)
    lim_t = torch.tensor(limits, dtype=torch.float32, device=device)     # (N, J, 2)

    data = types.SimpleNamespace(
        joint_pos=cur_t,
        default_joint_pos=def_t,
        soft_joint_pos_limits=lim_t,
    )
    return types.SimpleNamespace(data=data)


def _make_env(robot: object, *, was_placed: bool | list[bool] | None = None) -> object:
    """Return a mock env with .scene[name] → robot and optional .was_placed."""
    scene = {  "robot": robot }
    env = types.SimpleNamespace(scene=scene)
    if was_placed is not None:
        if isinstance(was_placed, bool):
            n = robot.data.joint_pos.shape[0]
            env.was_placed = torch.full((n,), was_placed, dtype=torch.bool)
        else:
            env.was_placed = torch.tensor(was_placed, dtype=torch.bool)
    return env


def _make_asset_cfg(num_joints: int) -> object:
    """Return a minimal SceneEntityCfg-like mock with joint_ids = [0..J-1]."""
    return types.SimpleNamespace(name="robot", joint_ids=list(range(num_joints)))


# ---------------------------------------------------------------------------
# Import the function under test (pure Python / torch, no Isaac Lab)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def return_to_neutral():
    # Importing this module does NOT need Isaac Lab; it only imports torch and
    # standard library.  The isaaclab imports at the top of rewards.py are
    # guarded by TYPE_CHECKING, so they are not executed at runtime.
    #
    # However, the module also imports from
    # tensegrity_pick.tasks.manager_based.shared.gripper_cfg which may pull in
    # isaaclab at import time.  We therefore do a targeted function-only import
    # by executing only the relevant snippet.
    import importlib, sys

    # Try direct import first (works when the full package is installed).
    try:
        from tensegrity_pick.tasks.manager_based.cube_place.mdp.rewards import (
            return_to_neutral as _fn,
        )
        return _fn
    except ImportError:
        pass

    # Fallback: parse only the function from source with exec().
    import ast, pathlib
    src_path = pathlib.Path(__file__).parent.parent / (
        "src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/"
        "tasks/manager_based/cube_place/mdp/rewards.py"
    )
    source = src_path.read_text()
    tree = ast.parse(source)
    # Extract the function definition node + any preceding imports.
    func_src_lines = []
    in_func = False
    indent = 0
    lines = source.splitlines()
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == "return_to_neutral":
            # Grab the source slice for this function.
            end = node.end_lineno
            func_lines = lines[node.lineno - 1: end]
            func_src_lines = func_lines
            break

    func_src = "\n".join(func_src_lines)
    ns: dict = {"torch": torch, "math": math}
    exec(func_src, ns)  # noqa: S102
    return ns["return_to_neutral"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReturnToNeutral:
    """Correctness tests for the normalized return_to_neutral reward."""

    # ── 1. Gate: no reward when was_placed is False ─────────────────────

    def test_zero_when_not_placed(self, return_to_neutral):
        """Result must be exactly 0 for every env where was_placed=False."""
        N, J = 4, 3
        current = [[0.5, -0.3, 0.1]] * N
        default = [[0.0, 0.0, 0.0]] * N
        # limits: [-1, 1] for each joint
        limits = [[[-1.0, 1.0]] * J] * N
        robot = _make_robot(current, default, limits)
        env = _make_env(robot, was_placed=False)
        asset_cfg = _make_asset_cfg(J)

        result = return_to_neutral(env, asset_cfg, std=0.25)
        assert result.shape == (N,)
        assert torch.all(result == 0.0), f"Expected all zeros, got {result}"

    # ── 2. Maximum reward when joints are at default ─────────────────────

    def test_max_reward_at_default_pose(self, return_to_neutral):
        """When current == default the normalized distance is 0 → result is 1.0."""
        N, J = 2, 4
        pose = [[0.1, -0.2, 0.3, -0.4]] * N
        limits = [[[-1.0, 1.0]] * J] * N
        robot = _make_robot(pose, pose, limits)
        env = _make_env(robot, was_placed=True)
        asset_cfg = _make_asset_cfg(J)

        result = return_to_neutral(env, asset_cfg, std=0.25)
        assert torch.allclose(result, torch.ones(N), atol=1e-6), (
            f"Expected 1.0 at default pose, got {result}"
        )

    # ── 3. Reward decreases monotonically with distance ──────────────────

    def test_monotone_decrease_with_distance(self, return_to_neutral):
        """Larger fractional deviation → smaller reward."""
        J = 2
        limits = [[[-1.0, 1.0], [-1.0, 1.0]]]  # (1, 2, 2)
        default = [[0.0, 0.0]]

        deviations = [0.0, 0.1, 0.3, 0.6, 1.0]
        rewards = []
        for d in deviations:
            current = [[d, 0.0]]
            robot = _make_robot(current, default, limits)
            env = _make_env(robot, was_placed=True)
            r = return_to_neutral(env, _make_asset_cfg(J), std=0.25)
            rewards.append(r.item())

        for i in range(len(rewards) - 1):
            assert rewards[i] >= rewards[i + 1], (
                f"Reward not monotone: r[{i}]={rewards[i]:.4f} < r[{i+1}]={rewards[i+1]:.4f}"
            )

    # ── 4. Invariant to prismatic vs revolute given equal fractional dev ──

    def test_invariant_to_joint_type(self, return_to_neutral):
        """
        A prismatic joint (range 2 m) displaced 1 m = 50% and a revolute
        joint (range 2π rad) displaced π rad = 50% must produce the same
        normalized deviation and therefore the same reward.
        """
        # Prismatic: range [0, 2] m, displaced 1.0 m
        lim_prismatic = [[[0.0, 2.0]]]  # (1, 1, 2)
        cur_prismatic  = [[1.0]]
        def_prismatic  = [[0.0]]

        # Revolute: range [-π, π] rad, displaced π rad (same 50%)
        lim_revolute  = [[[-math.pi, math.pi]]]  # (1, 1, 2)
        cur_revolute  = [[math.pi]]
        def_revolute  = [[0.0]]

        rob_p = _make_robot(cur_prismatic, def_prismatic, lim_prismatic)
        env_p = _make_env(rob_p, was_placed=True)

        rob_r = _make_robot(cur_revolute, def_revolute, lim_revolute)
        env_r = _make_env(rob_r, was_placed=True)

        r_p = return_to_neutral(env_p, _make_asset_cfg(1), std=0.25)
        r_r = return_to_neutral(env_r, _make_asset_cfg(1), std=0.25)

        assert torch.isclose(r_p, r_r, atol=1e-5), (
            f"Prismatic reward {r_p.item():.6f} != revolute reward {r_r.item():.6f} "
            "for equal fractional displacement"
        )

    # ── 5. Non-finite / near-zero ranges fall back to scale=1 ────────────

    def test_fallback_for_invalid_limits(self, return_to_neutral):
        """Non-finite or ~zero range joints fall back to scale=1 without NaN."""
        N, J = 1, 3
        # Joint 0: valid range [-1, 1]; Joint 1: zero range (fixed); Joint 2: inf range
        limits = [[[  -1.0, 1.0],
                   [   0.0, 0.0],        # near-zero → fall back to 1.0
                   [-1e38, float("inf")]]]  # non-finite → fall back to 1.0
        current = [[0.5, 0.0, 0.5]]
        default = [[0.0, 0.0, 0.0]]
        robot = _make_robot(current, default, limits)
        env = _make_env(robot, was_placed=True)
        asset_cfg = _make_asset_cfg(J)

        result = return_to_neutral(env, asset_cfg, std=0.25)
        assert torch.isfinite(result).all(), f"Expected finite result, got {result}"
        assert (result >= 0.0).all() and (result <= 1.0).all(), (
            f"Result out of [0,1]: {result}"
        )

    # ── 6. Mixed was_placed per env ──────────────────────────────────────

    def test_per_env_gate(self, return_to_neutral):
        """Envs with was_placed=False get 0, envs with True get non-zero."""
        N, J = 4, 2
        current = [[0.3, 0.3]] * N
        default = [[0.0, 0.0]] * N
        limits = [[[-1.0, 1.0], [-1.0, 1.0]]] * N
        robot = _make_robot(current, default, limits)
        env = _make_env(robot, was_placed=[False, True, False, True])
        asset_cfg = _make_asset_cfg(J)

        result = return_to_neutral(env, asset_cfg, std=0.25)
        assert result[0].item() == pytest.approx(0.0)
        assert result[2].item() == pytest.approx(0.0)
        assert result[1].item() > 0.0
        assert result[3].item() > 0.0

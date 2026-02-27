"""Tests for the ArticulationCfg robot configurations.

These tests validate the *structure* of each config (joint names, actuator
groups, USD paths, limits).  They catch typos and drift between the
URDF/USD assets and the Python config layer.

Importing the robot configs pulls in ``isaaclab`` which depends on Isaac
Sim's ``omni`` stack, so every test class is marked ``requires_isaac``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ── Constants ─────────────────────────────────────────────────────────────

PROJ_ASSETS_PATH = Path("/home/robot/studentische-arbeiten/res")

ARM_JOINT_NAMES = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]
BASE_JOINT_NAMES = ["base_y_joint", "base_z_joint"]
GRIPPER_ACTIVE_JOINT = "finger_joint"
GRIPPER_PASSIVE_JOINTS = [
    "right_outer_knuckle_joint",
    "left_outer_finger_joint",
    "right_outer_finger_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_pad_joint",
    "right_inner_finger_pad_joint",
]

# ── Helpers ───────────────────────────────────────────────────────────────


def _joint_names_in_init_state(cfg: object) -> list[str]:
    """Return the joint names defined in init_state.joint_pos."""
    return list(cfg.init_state.joint_pos.keys())  # type: ignore[attr-defined]


def _actuator_joint_exprs(cfg: object) -> dict[str, list[str]]:
    """Return ``{group_name: joint_names_expr}`` for each actuator group."""
    return {
        name: list(act.joint_names_expr)
        for name, act in cfg.actuators.items()  # type: ignore[attr-defined]
    }


def _usd_path(cfg: object) -> str:
    return cfg.spawn.usd_path  # type: ignore[attr-defined]


# ── Import fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pd_configs() -> tuple[object, object]:
    from tensegrity_pick.robots import TENS_3DOF_CFG, TENS_5DOF_GRIPPER_CFG
    return TENS_3DOF_CFG, TENS_5DOF_GRIPPER_CFG


@pytest.fixture(scope="module")
def tendon_configs() -> tuple[object, object]:
    from tensegrity_pick.robots import TENS_3DOF_TENDON_CFG, TENS_5DOF_GRIPPER_TENDON_CFG
    return TENS_3DOF_TENDON_CFG, TENS_5DOF_GRIPPER_TENDON_CFG


@pytest.fixture(scope="module")
def all_configs(
    pd_configs: tuple[object, object],
    tendon_configs: tuple[object, object],
) -> list[object]:
    return [*pd_configs, *tendon_configs]

pytestmark = pytest.mark.requires_isaac


# ── USD asset existence ───────────────────────────────────────────────────

class TestUsdAssets:

    def test_that_3dof_usd_exists(self, pd_configs: tuple[object, object]) -> None:
        path = _usd_path(pd_configs[0])
        assert os.path.isfile(path), f"Missing USD: {path}"

    def test_that_5dof_gripper_usd_exists(self, pd_configs: tuple[object, object]) -> None:
        path = _usd_path(pd_configs[1])
        assert os.path.isfile(path), f"Missing USD: {path}"

    def test_that_tendon_configs_reference_same_usds(
        self,
        pd_configs: tuple[object, object],
        tendon_configs: tuple[object, object],
    ) -> None:
        assert _usd_path(tendon_configs[0]) == _usd_path(pd_configs[0])
        assert _usd_path(tendon_configs[1]) == _usd_path(pd_configs[1])


# ── 3-DOF config structure ───────────────────────────────────────────────

class TestThreeDofConfig:

    def test_that_init_state_contains_arm_joints(self, pd_configs: tuple[object, object]) -> None:
        joint_names = _joint_names_in_init_state(pd_configs[0])
        for name in ARM_JOINT_NAMES:
            assert name in joint_names

    def test_that_arm_actuator_covers_all_arm_joints(
        self, pd_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(pd_configs[0])
        assert "arm" in groups
        for name in ARM_JOINT_NAMES:
            assert name in groups["arm"]

    def test_that_arm_effort_limit_is_positive(self, pd_configs: tuple[object, object]) -> None:
        arm = pd_configs[0].actuators["arm"]  # type: ignore[attr-defined]
        assert arm.effort_limit > 0

    def test_that_arm_stiffness_is_positive(self, pd_configs: tuple[object, object]) -> None:
        arm = pd_configs[0].actuators["arm"]  # type: ignore[attr-defined]
        assert arm.stiffness > 0

    def test_that_arm_damping_is_positive(self, pd_configs: tuple[object, object]) -> None:
        arm = pd_configs[0].actuators["arm"]  # type: ignore[attr-defined]
        assert arm.damping > 0


# ── 5-DOF gripper config structure ───────────────────────────────────────

class TestFiveDofGripperConfig:

    def test_that_init_state_contains_all_joints(
        self, pd_configs: tuple[object, object],
    ) -> None:
        joint_names = _joint_names_in_init_state(pd_configs[1])
        expected = BASE_JOINT_NAMES + ARM_JOINT_NAMES + [GRIPPER_ACTIVE_JOINT]
        for name in expected:
            assert name in joint_names

    def test_that_all_actuator_groups_are_present(
        self, pd_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(pd_configs[1])
        assert set(groups.keys()) == {"base", "arm", "gripper", "gripper_passive"}

    def test_that_base_actuator_covers_base_joints(
        self, pd_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(pd_configs[1])
        for name in BASE_JOINT_NAMES:
            assert name in groups["base"]

    def test_that_gripper_actuator_covers_finger_joint(
        self, pd_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(pd_configs[1])
        assert GRIPPER_ACTIVE_JOINT in groups["gripper"]

    def test_that_gripper_passive_covers_all_passive_joints(
        self, pd_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(pd_configs[1])
        for name in GRIPPER_PASSIVE_JOINTS:
            assert name in groups["gripper_passive"]

    def test_that_base_stiffness_exceeds_arm_stiffness(
        self, pd_configs: tuple[object, object],
    ) -> None:
        actuators = pd_configs[1].actuators  # type: ignore[attr-defined]
        assert actuators["base"].stiffness > actuators["arm"].stiffness


# ── Tendon config specifics ──────────────────────────────────────────────

class TestTendonConfigs:

    def test_that_3dof_tendon_arm_stiffness_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        arm = tendon_configs[0].actuators["arm"]  # type: ignore[attr-defined]
        assert arm.stiffness == 0.0

    def test_that_3dof_tendon_arm_damping_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        arm = tendon_configs[0].actuators["arm"]  # type: ignore[attr-defined]
        assert arm.damping == 0.0

    def test_that_5dof_tendon_arm_stiffness_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        arm = tendon_configs[1].actuators["arm"]  # type: ignore[attr-defined]
        assert arm.stiffness == 0.0

    def test_that_5dof_tendon_arm_damping_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        arm = tendon_configs[1].actuators["arm"]  # type: ignore[attr-defined]
        assert arm.damping == 0.0

    def test_that_5dof_tendon_base_stiffness_is_unchanged(
        self,
        pd_configs: tuple[object, object],
        tendon_configs: tuple[object, object],
    ) -> None:
        pd_base = pd_configs[1].actuators["base"]  # type: ignore[attr-defined]
        tendon_base = tendon_configs[1].actuators["base"]  # type: ignore[attr-defined]
        assert tendon_base.stiffness == pd_base.stiffness

    def test_that_5dof_tendon_gripper_is_unchanged(
        self,
        pd_configs: tuple[object, object],
        tendon_configs: tuple[object, object],
    ) -> None:
        pd_gripper = pd_configs[1].actuators["gripper"]  # type: ignore[attr-defined]
        tendon_gripper = tendon_configs[1].actuators["gripper"]  # type: ignore[attr-defined]
        assert tendon_gripper.stiffness == pd_gripper.stiffness
        assert tendon_gripper.damping == pd_gripper.damping

    def test_that_tendon_arm_effort_limit_is_positive(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        for cfg in tendon_configs:
            arm = cfg.actuators["arm"]  # type: ignore[attr-defined]
            assert arm.effort_limit > 0

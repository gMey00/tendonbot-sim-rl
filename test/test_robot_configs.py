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
def physical_tendon_configs() -> tuple[object, object]:
    from tensegrity_pick.robots import (
        TENS_3DOF_PHYSICAL_TENDON_CFG,
        TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG,
    )
    return TENS_3DOF_PHYSICAL_TENDON_CFG, TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG


PHYSICAL_ARM_JOINT_NAMES = [
    "rod_left_joint", "rod_right_joint", "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
]


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

    def test_that_arm_actuators_cover_all_arm_joints(
        self, pd_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(pd_configs[0])
        assert "elbow" in groups
        assert "wrist" in groups
        covered = groups["elbow"] + groups["wrist"]
        for name in ARM_JOINT_NAMES:
            assert name in covered

    def test_that_elbow_effort_limit_is_positive(self, pd_configs: tuple[object, object]) -> None:
        elbow = pd_configs[0].actuators["elbow"]  # type: ignore[attr-defined]
        assert elbow.effort_limit_sim > 0

    def test_that_wrist_effort_limit_is_positive(self, pd_configs: tuple[object, object]) -> None:
        wrist = pd_configs[0].actuators["wrist"]  # type: ignore[attr-defined]
        assert wrist.effort_limit_sim > 0

    def test_that_arm_stiffness_is_positive(self, pd_configs: tuple[object, object]) -> None:
        elbow = pd_configs[0].actuators["elbow"]  # type: ignore[attr-defined]
        wrist = pd_configs[0].actuators["wrist"]  # type: ignore[attr-defined]
        assert elbow.stiffness > 0
        assert wrist.stiffness > 0

    def test_that_arm_damping_is_positive(self, pd_configs: tuple[object, object]) -> None:
        elbow = pd_configs[0].actuators["elbow"]  # type: ignore[attr-defined]
        wrist = pd_configs[0].actuators["wrist"]  # type: ignore[attr-defined]
        assert elbow.damping > 0
        assert wrist.damping > 0


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
        assert set(groups.keys()) == {"base_y", "base_z", "elbow", "wrist", "gripper", "gripper_passive"}

    def test_that_base_actuators_cover_base_joints(
        self, pd_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(pd_configs[1])
        covered = groups["base_y"] + groups["base_z"]
        for name in BASE_JOINT_NAMES:
            assert name in covered

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
        assert actuators["base_y"].stiffness > actuators["elbow"].stiffness
        assert actuators["base_z"].stiffness > actuators["wrist"].stiffness


# ── Tendon config specifics ──────────────────────────────────────────────

class TestTendonConfigs:

    def test_that_3dof_tendon_elbow_stiffness_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        elbow = tendon_configs[0].actuators["elbow"]  # type: ignore[attr-defined]
        assert elbow.stiffness == 0.0

    def test_that_3dof_tendon_wrist_stiffness_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        wrist = tendon_configs[0].actuators["wrist"]  # type: ignore[attr-defined]
        assert wrist.stiffness == 0.0

    def test_that_3dof_tendon_elbow_damping_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        elbow = tendon_configs[0].actuators["elbow"]  # type: ignore[attr-defined]
        assert elbow.damping == 0.0

    def test_that_3dof_tendon_wrist_damping_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        wrist = tendon_configs[0].actuators["wrist"]  # type: ignore[attr-defined]
        assert wrist.damping == 0.0

    def test_that_5dof_tendon_elbow_stiffness_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        elbow = tendon_configs[1].actuators["elbow"]  # type: ignore[attr-defined]
        assert elbow.stiffness == 0.0

    def test_that_5dof_tendon_wrist_stiffness_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        wrist = tendon_configs[1].actuators["wrist"]  # type: ignore[attr-defined]
        assert wrist.stiffness == 0.0

    def test_that_5dof_tendon_elbow_damping_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        elbow = tendon_configs[1].actuators["elbow"]  # type: ignore[attr-defined]
        assert elbow.damping == 0.0

    def test_that_5dof_tendon_wrist_damping_is_zero(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        wrist = tendon_configs[1].actuators["wrist"]  # type: ignore[attr-defined]
        assert wrist.damping == 0.0

    def test_that_5dof_tendon_base_stiffness_is_unchanged(
        self,
        pd_configs: tuple[object, object],
        tendon_configs: tuple[object, object],
    ) -> None:
        for key in ("base_y", "base_z"):
            pd_base = pd_configs[1].actuators[key]  # type: ignore[attr-defined]
            tendon_base = tendon_configs[1].actuators[key]  # type: ignore[attr-defined]
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

    def test_that_tendon_arm_effort_limits_are_positive(
        self, tendon_configs: tuple[object, object],
    ) -> None:
        for cfg in tendon_configs:
            elbow = cfg.actuators["elbow"]  # type: ignore[attr-defined]
            wrist = cfg.actuators["wrist"]  # type: ignore[attr-defined]
            assert elbow.effort_limit > 0
            assert wrist.effort_limit > 0


# ── Physical tendon config specifics ─────────────────────────────────────

class TestPhysicalTendonConfigs:

    def test_that_3dof_physical_init_state_contains_linkage_joints(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        joint_names = _joint_names_in_init_state(physical_tendon_configs[0])
        for name in PHYSICAL_ARM_JOINT_NAMES:
            assert name in joint_names

    def test_that_3dof_physical_has_linkage_actuator(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(physical_tendon_configs[0])
        assert "linkage" in groups

    def test_that_3dof_physical_linkage_stiffness_is_zero(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        linkage = physical_tendon_configs[0].actuators["linkage"]  # type: ignore[attr-defined]
        assert linkage.stiffness == 0.0

    def test_that_3dof_physical_linkage_damping_is_zero(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        linkage = physical_tendon_configs[0].actuators["linkage"]  # type: ignore[attr-defined]
        assert linkage.damping == 0.0

    def test_that_3dof_physical_wrist_stiffness_is_zero(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        wrist = physical_tendon_configs[0].actuators["wrist"]  # type: ignore[attr-defined]
        assert wrist.stiffness == 0.0

    def test_that_3dof_physical_wrist_damping_is_zero(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        wrist = physical_tendon_configs[0].actuators["wrist"]  # type: ignore[attr-defined]
        assert wrist.damping == 0.0

    def test_that_3dof_physical_usd_exists(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        path = _usd_path(physical_tendon_configs[0])
        assert os.path.isfile(path), f"Missing USD: {path}"

    def test_that_5dof_physical_has_all_actuator_groups(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        groups = _actuator_joint_exprs(physical_tendon_configs[1])
        assert "linkage" in groups
        assert "wrist" in groups
        assert "base_y" in groups
        assert "base_z" in groups

    def test_that_5dof_physical_init_state_contains_all_joints(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        joint_names = _joint_names_in_init_state(physical_tendon_configs[1])
        expected = BASE_JOINT_NAMES + PHYSICAL_ARM_JOINT_NAMES + [GRIPPER_ACTIVE_JOINT]
        for name in expected:
            assert name in joint_names

    def test_that_physical_arm_effort_limits_are_positive(
        self, physical_tendon_configs: tuple[object, object],
    ) -> None:
        for cfg in physical_tendon_configs:
            linkage = cfg.actuators["linkage"]  # type: ignore[attr-defined]
            wrist = cfg.actuators["wrist"]  # type: ignore[attr-defined]
            assert linkage.effort_limit > 0
            assert wrist.effort_limit > 0

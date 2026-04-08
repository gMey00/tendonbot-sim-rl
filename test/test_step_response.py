"""Regression tests for the step-response PID controller and tension mapping.

These tests validate the *offline* math of the step-response tooling
(PID gains, gravity compensation formula, tension clamping).  They catch
accidental changes to controller parameters that would invalidate the
Klein (2023) comparison.

The step-response module imports ``isaaclab.sim`` at module scope, so all
tests here require Isaac Sim to be importable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.requires_isaac

# ── Constants from Klein (2023) ────────────────────────────────────────────

JACOBIAN_TRANSPOSE = np.array([
    [+0.0725, -0.0725,  0.0,       0.0,      0.0],
    [ 0.0,     0.0,     -0.017321,  0.0,     +0.017321],
    [ 0.0,     0.0,     +0.010,    -0.020,   +0.010],
], dtype=np.float64)


# ── PID gain validation ──────────────────────────────────────────────────

class TestPidGainsMatchThesis:
    """Verify that the coded PID gains match Klein (2023) §3.5 / Table 4.2."""

    WRIST_EXPECTED = {"kp": 0.2, "ki": 0.03, "kd": 0.03}
    ELBOW_EXPECTED = {"kp": 0.3, "ki": 0.03, "kd": 0.02}

    @pytest.fixture(scope="class")
    def specs(self) -> dict[str, object]:
        from tensegrity_pick.scripts.step_response_test import (
            ELBOW_SPEC,
            WRIST_X_SPEC,
            WRIST_Y_SPEC,
        )
        return {"wrist_y": WRIST_Y_SPEC, "wrist_x": WRIST_X_SPEC, "elbow": ELBOW_SPEC}

    def test_that_wrist_y_gains_match(self, specs: dict[str, object]) -> None:
        pid = specs["wrist_y"].pid  # type: ignore[union-attr]
        assert pid.kp == pytest.approx(self.WRIST_EXPECTED["kp"])
        assert pid.ki == pytest.approx(self.WRIST_EXPECTED["ki"])
        assert pid.kd == pytest.approx(self.WRIST_EXPECTED["kd"])

    def test_that_wrist_x_gains_match(self, specs: dict[str, object]) -> None:
        pid = specs["wrist_x"].pid  # type: ignore[union-attr]
        assert pid.kp == pytest.approx(self.WRIST_EXPECTED["kp"])
        assert pid.ki == pytest.approx(self.WRIST_EXPECTED["ki"])
        assert pid.kd == pytest.approx(self.WRIST_EXPECTED["kd"])

    def test_that_elbow_gains_match(self, specs: dict[str, object]) -> None:
        pid = specs["elbow"].pid  # type: ignore[union-attr]
        assert pid.kp == pytest.approx(self.ELBOW_EXPECTED["kp"])
        assert pid.ki == pytest.approx(self.ELBOW_EXPECTED["ki"])
        assert pid.kd == pytest.approx(self.ELBOW_EXPECTED["kd"])


# ── Tension limits ────────────────────────────────────────────────────────

class TestTensionLimits:

    @pytest.fixture(scope="class")
    def specs(self) -> dict[str, object]:
        from tensegrity_pick.scripts.step_response_test import (
            ELBOW_SPEC,
            WRIST_X_SPEC,
            WRIST_Y_SPEC,
        )
        return {"wrist_y": WRIST_Y_SPEC, "wrist_x": WRIST_X_SPEC, "elbow": ELBOW_SPEC}

    def test_that_wrist_min_tension_is_positive(self, specs: dict[str, object]) -> None:
        for name in ("wrist_y", "wrist_x"):
            assert specs[name].min_tension_n > 0  # type: ignore[union-attr]

    def test_that_elbow_min_tension_is_positive(self, specs: dict[str, object]) -> None:
        assert specs["elbow"].min_tension_n > 0  # type: ignore[union-attr]

    def test_that_saturation_exceeds_max_tension(self, specs: dict[str, object]) -> None:
        for name in ("wrist_y", "wrist_x", "elbow"):
            spec = specs[name]
            assert spec.saturation_n > spec.max_tension_n  # type: ignore[union-attr]

    def test_that_wrist_tension_values_match_thesis(self, specs: dict[str, object]) -> None:
        for name in ("wrist_y", "wrist_x"):
            spec = specs[name]
            assert spec.min_tension_n == pytest.approx(5.0)  # type: ignore[union-attr]
            assert spec.max_tension_n == pytest.approx(15.0)  # type: ignore[union-attr]
            assert spec.saturation_n == pytest.approx(80.0)  # type: ignore[union-attr]

    def test_that_elbow_tension_values_match_thesis(self, specs: dict[str, object]) -> None:
        spec = specs["elbow"]
        assert spec.min_tension_n == pytest.approx(6.0)  # type: ignore[union-attr]
        assert spec.max_tension_n == pytest.approx(20.0)  # type: ignore[union-attr]
        assert spec.saturation_n == pytest.approx(160.0)  # type: ignore[union-attr]


# ── PID computation ───────────────────────────────────────────────────────

class TestPidComputation:

    def test_that_zero_error_produces_zero_torque(self) -> None:
        from tensegrity_pick.scripts.step_response_test import PIDGains, PIDState, compute_pid_torque

        state = PIDState()
        state.reset("cpu")
        torque = compute_pid_torque(
            setpoint_rad=0.0,
            current_rad=torch.tensor([0.0]),
            dt=0.01,
            gains=PIDGains(kp=0.2, ki=0.03, kd=0.03),
            state=state,
        )
        assert torque.item() == pytest.approx(0.0, abs=1e-8)

    def test_that_positive_error_produces_positive_torque(self) -> None:
        from tensegrity_pick.scripts.step_response_test import PIDGains, PIDState, compute_pid_torque

        state = PIDState()
        state.reset("cpu")
        torque = compute_pid_torque(
            setpoint_rad=1.0,
            current_rad=torch.tensor([0.0]),
            dt=0.01,
            gains=PIDGains(kp=0.2, ki=0.03, kd=0.03),
            state=state,
        )
        assert torque.item() > 0.0

    def test_that_integral_accumulates_over_steps(self) -> None:
        from tensegrity_pick.scripts.step_response_test import PIDGains, PIDState, compute_pid_torque

        state = PIDState()
        state.reset("cpu")
        gains = PIDGains(kp=0.0, ki=1.0, kd=0.0)

        for _ in range(10):
            compute_pid_torque(
                setpoint_rad=1.0,
                current_rad=torch.tensor([0.0]),
                dt=0.01,
                gains=gains,
                state=state,
            )

        # integral should be 10 * 1.0 * 0.01 = 0.1
        assert state.integral.item() == pytest.approx(0.1, rel=1e-5)


# ── Gravity compensation ─────────────────────────────────────────────────

class TestGravityCompensation:

    def test_that_zero_angle_gives_zero_compensation(self) -> None:
        from tensegrity_pick.scripts.step_response_test import gravity_compensation_torque

        torque = gravity_compensation_torque(
            current_rad=torch.tensor([0.0]),
            mass_kg=1.0,
            gravity_arm_m=0.1,
        )
        assert torque.item() == pytest.approx(0.0, abs=1e-10)

    def test_that_90deg_gives_maximum_compensation(self) -> None:
        from tensegrity_pick.scripts.step_response_test import gravity_compensation_torque

        torque = gravity_compensation_torque(
            current_rad=torch.tensor([math.pi / 2]),
            mass_kg=1.0,
            gravity_arm_m=0.1,
        )
        expected = 1.0 * 9.81 * 0.1  # m·g·lc·sin(π/2) = m·g·lc
        assert torque.item() == pytest.approx(expected, rel=1e-5)


# ── Jacobian pseudo-inverse consistency ───────────────────────────────────

class TestJacobianPinvConsistency:

    def test_that_script_jacobian_matches_actuator_module(self) -> None:
        from tensegrity_pick.robots.tendon_actuator import DEFAULT_JACOBIAN_TRANSPOSE
        from tensegrity_pick.scripts.step_response_test import JACOBIAN_T

        expected = np.array(DEFAULT_JACOBIAN_TRANSPOSE, dtype=np.float64)
        np.testing.assert_allclose(JACOBIAN_T, expected, atol=1e-12)

    def test_that_pinv_round_trips_identity(self) -> None:
        from tensegrity_pick.scripts.step_response_test import JACOBIAN_T

        pinv = np.linalg.pinv(JACOBIAN_T)
        product = JACOBIAN_T @ pinv  # (3, 3) should be close to I_3
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_that_torque_to_tensions_respects_clamping(self) -> None:
        from tensegrity_pick.scripts.step_response_test import WRIST_Y_SPEC, torque_to_tensions

        desired = torch.tensor([[100.0]])  # an unreasonably large torque
        tensions = torque_to_tensions(desired, WRIST_Y_SPEC, joint_index=1, device="cpu")
        assert (tensions >= WRIST_Y_SPEC.min_tension_n).all()
        assert (tensions <= WRIST_Y_SPEC.saturation_n).all()

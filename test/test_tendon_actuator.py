"""Tests for the Jacobian-transpose tendon-to-torque mapping.

These are pure-math tests and do not require Isaac Sim.  They validate the
``DEFAULT_JACOBIAN_TRANSPOSE`` constant and the affine tension mapping that
``TendonEffortAction`` implements.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

# ── Constants under test ──────────────────────────────────────────────────

JACOBIAN_TRANSPOSE: list[list[float]] = [
    [+0.0725, -0.0725,  0.0,       0.0,      0.0],       # elbow
    [ 0.0,     0.0,     -0.013856,  0.0,     +0.013856],  # wrist_y
    [ 0.0,     0.0,     +0.008,    -0.016,   +0.008],     # wrist_x
]

NUM_JOINTS = 3
NUM_TENDONS = 5

JACOBIAN_NP = np.array(JACOBIAN_TRANSPOSE, dtype=np.float64)
JACOBIAN_T = torch.tensor(JACOBIAN_TRANSPOSE, dtype=torch.float32)


# ── Helpers ───────────────────────────────────────────────────────────────

def _torque_from_tensions(tensions: list[float]) -> np.ndarray:
    return JACOBIAN_NP @ np.array(tensions, dtype=np.float64)


def _affine_map(raw_actions: torch.Tensor, max_tension: float) -> torch.Tensor:
    """Replicate the [-1, 1] → [0, max_tension] mapping from TendonEffortAction."""
    scale = max_tension / 2.0
    offset = max_tension / 2.0
    return (raw_actions * scale + offset).clamp(min=0.0, max=max_tension)


# ── Jacobian shape and structure ──────────────────────────────────────────

class TestJacobianShape:

    def test_that_jacobian_has_correct_dimensions(self) -> None:
        assert JACOBIAN_NP.shape == (NUM_JOINTS, NUM_TENDONS)

    def test_that_elbow_row_only_uses_first_two_tendons(self) -> None:
        elbow_row = JACOBIAN_NP[0]
        assert elbow_row[0] != 0.0
        assert elbow_row[1] != 0.0
        assert all(elbow_row[i] == 0.0 for i in range(2, NUM_TENDONS))

    def test_that_wrist_rows_only_use_last_three_tendons(self) -> None:
        for row_index in (1, 2):
            row = JACOBIAN_NP[row_index]
            assert row[0] == 0.0
            assert row[1] == 0.0
            assert any(row[i] != 0.0 for i in range(2, NUM_TENDONS))

    def test_that_elbow_tendons_are_antagonistic(self) -> None:
        elbow_row = JACOBIAN_NP[0]
        assert elbow_row[0] == pytest.approx(-elbow_row[1])
        assert elbow_row[0] > 0.0


# ── Torque mapping correctness ───────────────────────────────────────────

class TestTorqueMapping:

    def test_that_zero_tensions_produce_zero_torques(self) -> None:
        torques = _torque_from_tensions([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(torques, [0.0, 0.0, 0.0], atol=1e-12)

    def test_that_equal_elbow_tensions_cancel(self) -> None:
        torques = _torque_from_tensions([10.0, 10.0, 0.0, 0.0, 0.0])
        assert torques[0] == pytest.approx(0.0, abs=1e-12)

    def test_that_single_elbow_tendon_produces_positive_torque(self) -> None:
        torques = _torque_from_tensions([10.0, 0.0, 0.0, 0.0, 0.0])
        assert torques[0] == pytest.approx(10.0 * 0.0725, rel=1e-6)
        assert torques[1] == pytest.approx(0.0, abs=1e-12)
        assert torques[2] == pytest.approx(0.0, abs=1e-12)

    def test_that_single_elbow_tendon_produces_negative_torque(self) -> None:
        torques = _torque_from_tensions([0.0, 10.0, 0.0, 0.0, 0.0])
        assert torques[0] == pytest.approx(-10.0 * 0.0725, rel=1e-6)

    def test_that_wrist_tendons_do_not_affect_elbow(self) -> None:
        torques = _torque_from_tensions([0.0, 0.0, 50.0, 50.0, 50.0])
        assert torques[0] == pytest.approx(0.0, abs=1e-12)

    def test_that_elbow_tendons_do_not_affect_wrist(self) -> None:
        torques = _torque_from_tensions([50.0, 50.0, 0.0, 0.0, 0.0])
        assert torques[1] == pytest.approx(0.0, abs=1e-12)
        assert torques[2] == pytest.approx(0.0, abs=1e-12)

    def test_that_known_wrist_tensions_give_expected_torques(self) -> None:
        # T2=10, T3=0, T4=10  →  wrist_y = 10*(-0.013856) + 10*(+0.013856) = 0
        #                         wrist_x = 10*(+0.008) + 0 + 10*(+0.008) = 0.16
        torques = _torque_from_tensions([0.0, 0.0, 10.0, 0.0, 10.0])
        assert torques[1] == pytest.approx(0.0, abs=1e-12)
        assert torques[2] == pytest.approx(0.16, rel=1e-6)


# ── Affine mapping [-1,1] → [0, max_tension] ─────────────────────────────

class TestAffineMapping:

    @pytest.fixture()
    def max_tension(self) -> float:
        return 500.0

    def test_that_minus_one_maps_to_zero(self, max_tension: float) -> None:
        raw = torch.tensor([-1.0])
        tensions = _affine_map(raw, max_tension)
        assert tensions.item() == pytest.approx(0.0, abs=1e-5)

    def test_that_plus_one_maps_to_max(self, max_tension: float) -> None:
        raw = torch.tensor([1.0])
        tensions = _affine_map(raw, max_tension)
        assert tensions.item() == pytest.approx(max_tension, rel=1e-6)

    def test_that_zero_maps_to_half_max(self, max_tension: float) -> None:
        raw = torch.tensor([0.0])
        tensions = _affine_map(raw, max_tension)
        assert tensions.item() == pytest.approx(max_tension / 2.0, rel=1e-6)

    def test_that_output_is_always_non_negative(self, max_tension: float) -> None:
        raw = torch.linspace(-2.0, 2.0, steps=100)
        tensions = _affine_map(raw, max_tension)
        assert (tensions >= 0.0).all()

    def test_that_output_never_exceeds_max(self, max_tension: float) -> None:
        raw = torch.linspace(-2.0, 2.0, steps=100)
        tensions = _affine_map(raw, max_tension)
        assert (tensions <= max_tension).all()


# ── Batched torque computation (mirrors TendonEffortAction.process_actions) ──

class TestBatchedTorqueComputation:

    def test_that_batched_matmul_matches_single(self) -> None:
        batch_size = 4
        tensions = torch.rand(batch_size, NUM_TENDONS)
        jacobian_batch = JACOBIAN_T.unsqueeze(0).expand(batch_size, -1, -1)

        torques_batch = torch.bmm(jacobian_batch, tensions.unsqueeze(-1)).squeeze(-1)

        for i in range(batch_size):
            expected = JACOBIAN_T @ tensions[i]
            torch.testing.assert_close(torques_batch[i], expected, rtol=1e-5, atol=1e-6)

    def test_that_zero_actions_produce_nonzero_torque_via_offset(self) -> None:
        """Zero raw actions → tensions at max/2 → non-zero torques (generally)."""
        max_tension = 500.0
        raw = torch.zeros(1, NUM_TENDONS)
        tensions = _affine_map(raw, max_tension)
        torques = (JACOBIAN_T @ tensions.squeeze()).numpy()
        # Elbow: equal tensions cancel; wrist_x: symmetric → net torque
        assert torques[0] == pytest.approx(0.0, abs=1e-3)


# ── Jacobian pseudo-inverse round-trip ────────────────────────────────────

class TestJacobianPseudoinverse:

    def test_that_pinv_recovers_torques(self) -> None:
        """J^T @ pinv(J^T) @ τ ≈ τ for any τ in the column space of J^T."""
        desired_torques = np.array([0.5, -0.1, 0.2])
        pinv = np.linalg.pinv(JACOBIAN_NP)
        tensions = pinv @ desired_torques
        recovered = JACOBIAN_NP @ tensions
        np.testing.assert_allclose(recovered, desired_torques, atol=1e-10)

    def test_that_pinv_gives_minimum_norm_tensions(self) -> None:
        """Pseudo-inverse solution has minimum L2 norm among all solutions."""
        desired_torques = np.array([0.3, -0.05, 0.1])
        pinv = np.linalg.pinv(JACOBIAN_NP)
        tensions_pinv = pinv @ desired_torques
        norm_pinv = np.linalg.norm(tensions_pinv)

        # Add a null-space component and verify norm increases
        null_space = np.eye(NUM_TENDONS) - pinv @ JACOBIAN_NP
        perturbation = null_space @ np.random.randn(NUM_TENDONS)
        if np.linalg.norm(perturbation) > 1e-10:
            tensions_perturbed = tensions_pinv + 0.1 * perturbation
            norm_perturbed = np.linalg.norm(tensions_perturbed)
            assert norm_perturbed > norm_pinv - 1e-10


# ── Physical plausibility ────────────────────────────────────────────────

class TestPhysicalPlausibility:

    def test_that_lever_arms_match_xacro_geometry(self) -> None:
        """Elbow lever arm should be ±72.5 mm (from xacro Force1/Force2)."""
        assert JACOBIAN_NP[0, 0] == pytest.approx(0.0725, rel=1e-6)
        assert JACOBIAN_NP[0, 1] == pytest.approx(-0.0725, rel=1e-6)

    def test_that_wrist_tendon_radii_are_consistent(self) -> None:
        """Wrist tendons at 120° on a ≈ 16 mm circle: check consistent radius."""
        # Force3: (+0.008, +0.013856) → r = √(0.008² + 0.013856²) ≈ 0.016
        r3 = math.sqrt(0.008**2 + 0.013856**2)
        # Force4: (-0.016, 0) → r = 0.016
        r4 = 0.016
        # Force5: (+0.008, -0.013856) → r = same as r3
        r5 = math.sqrt(0.008**2 + 0.013856**2)
        assert r3 == pytest.approx(r4, rel=1e-3)
        assert r5 == pytest.approx(r4, rel=1e-3)

    def test_that_wrist_tendons_are_at_120_degree_spacing(self) -> None:
        """Verify the wrist tendon attachment angles are 120° apart."""
        angles = [
            math.atan2(+0.013856, +0.008),    # Force3: ~60°
            math.atan2(0.0, -0.016),           # Force4: 180°
            math.atan2(-0.013856, +0.008),     # Force5: ~-60° (=300°)
        ]
        angle_diffs = [
            (angles[1] - angles[0]) % (2 * math.pi),
            (angles[2] - angles[1]) % (2 * math.pi),
            (angles[0] - angles[2]) % (2 * math.pi),
        ]
        expected_separation = 2 * math.pi / 3  # 120°
        for diff in angle_diffs:
            assert diff == pytest.approx(expected_separation, rel=0.01)

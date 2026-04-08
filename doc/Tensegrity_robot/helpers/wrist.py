"""Cable-driven 2-DOF wrist mechanism kinematics.

References
----------
Nemoto et al. (2022) — cable topology, IEEE RA-L 7(2), 5727–5734.
Klein (2023), §3.2.4 — tension distribution.
Verhoeven (2004) — structure matrix.
Pott et al. (2009) — closed-form tension distribution.
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Rotation Matrices
# ═══════════════════════════════════════════════════════════════════

def rotation_x(psi: float) -> np.ndarray:
    """Rotation matrix about the X axis by angle ψ [rad]."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(theta: float) -> np.ndarray:
    """Rotation matrix about the Y axis by angle θ [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_xy(psi: float, theta: float) -> np.ndarray:
    """X–Y fixed Euler rotation: R = Rᵧ(θ) · Rₓ(ψ)."""
    return rotation_y(theta) @ rotation_x(psi)


# ═══════════════════════════════════════════════════════════════════
# Cable Geometry
# ═══════════════════════════════════════════════════════════════════

def platform_anchors_world(
    psi: float,
    theta: float,
    anchors_body: np.ndarray,
) -> np.ndarray:
    """Transform body-frame platform anchors to the world frame."""
    return (rotation_xy(psi, theta) @ anchors_body.T).T


def cable_geometry(
    base_points: np.ndarray,
    platform_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cable vectors (base→platform direction), lengths, and unit vectors."""
    vectors = base_points - platform_world
    lengths = np.linalg.norm(vectors, axis=1)
    units = vectors / lengths[:, None]
    return vectors, lengths, units


# ═══════════════════════════════════════════════════════════════════
# Structure Matrix & Tension Distribution
# ═══════════════════════════════════════════════════════════════════

def structure_matrix(
    psi: float,
    theta: float,
    base_anchors: np.ndarray,
    platform_body: np.ndarray,
) -> np.ndarray:
    """2×m structure matrix: A[k,i] = ê_k · (pᵢ × uᵢ)."""
    pw = platform_anchors_world(psi, theta, platform_body)
    _, _, units = cable_geometry(base_anchors, pw)
    m = len(base_anchors)
    A = np.zeros((2, m))
    for i in range(m):
        moment = np.cross(pw[i], units[i])
        A[0, i] = moment[0]
        A[1, i] = moment[1]
    return A


def tension_distribution(
    A: np.ndarray,
    tau_desired: np.ndarray,
    t_ref: np.ndarray,
) -> np.ndarray:
    """Closed-form tension distribution (Pott et al. 2009)."""
    return t_ref + np.linalg.pinv(A) @ (tau_desired - A @ t_ref)


def is_feasible(
    psi: float,
    theta: float,
    base_anchors: np.ndarray,
    platform_body: np.ndarray,
    t_ref_val: float = 1.0,
) -> tuple[bool, np.ndarray]:
    """Check whether all cable tensions are positive at zero torque."""
    A = structure_matrix(psi, theta, base_anchors, platform_body)
    if np.linalg.matrix_rank(A, tol=1e-8) < 2:
        return False, np.zeros(A.shape[1])
    t_ref = np.full(A.shape[1], t_ref_val)
    t_star = tension_distribution(A, np.zeros(2), t_ref)
    return bool(np.all(t_star > 0)), t_star


# ═══════════════════════════════════════════════════════════════════
# 3D Geometry Helpers
# ═══════════════════════════════════════════════════════════════════

def ring_points(
    radius: float,
    z: float,
    n: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circle of n points at height z for drawing rings."""
    angles = np.linspace(0, 2 * np.pi, n + 1)
    return (
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full(n + 1, z),
    )


def compute_anchors(
    R: float,
    r: float,
    h: float,
    h_b: float,
) -> dict[str, np.ndarray]:
    """Compute all anchor positions for the wrist mechanism.

    Returns dict with keys: base_passive, base_active, passive_convergence,
    platform_active_body, passive_base_angles, active_base_angles.
    """
    passive_base_angles = np.radians([0, 120, 240])
    active_base_angles = np.radians([60, 180, 300])

    base_passive = np.array([
        [R * np.cos(a), R * np.sin(a), -h_b] for a in passive_base_angles
    ])
    base_active = np.array([
        [R * np.cos(a), R * np.sin(a), -h_b] for a in active_base_angles
    ])
    passive_convergence = np.array([0.0, 0.0, 0.0])
    platform_active_body = np.array([
        [r * np.cos(a), r * np.sin(a), -h] for a in active_base_angles
    ])

    return dict(
        base_passive=base_passive,
        base_active=base_active,
        passive_convergence=passive_convergence,
        platform_active_body=platform_active_body,
        passive_base_angles=passive_base_angles,
        active_base_angles=active_base_angles,
    )

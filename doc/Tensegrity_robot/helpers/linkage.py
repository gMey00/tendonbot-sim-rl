"""Antiparallelogram four-bar linkage and disc-approximation kinematics.

References
----------
Klein (2023), §3.2.1 — mechanism dimensions and joint limits.
McCarthy & Soh (2011), §1.3 — closure condition.
Uicker et al. (2017), §3.4 — Kennedy's Theorem / ICR.
Dijksman (1976) — rolling centrode ellipses.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


# ═══════════════════════════════════════════════════════════════════
# Core Antiparallelogram Kinematics
# ═══════════════════════════════════════════════════════════════════

def antiparallel_phi(theta: float, k_e: float, l_e: float) -> float:
    """Output rod angle for the antiparallelogram branch.

    McCarthy & Soh (2011), §1.3:
        t = tan((φ−θ)/2) = −k_e cos θ / (l_e − k_e sin θ)
    """
    t = -k_e * np.cos(theta) / (l_e - k_e * np.sin(theta))
    return theta + 2.0 * np.arctan(t)


def linkage_positions(
    delta: float,
    k_e: float,
    l_e: float,
    theta_0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Joint positions at input deviation δ from equilibrium.

    Returns (A, B, D, C, phi, M, theta).
    """
    theta = theta_0 + delta
    phi = antiparallel_phi(theta, k_e, l_e)
    A = np.array([-k_e / 2.0, 0.0])
    B = np.array([k_e / 2.0, 0.0])
    D = A + l_e * np.array([np.sin(theta), -np.cos(theta)])
    C = B + l_e * np.array([np.sin(phi), -np.cos(phi)])
    M = 0.5 * (D + C)
    return A, B, D, C, phi, M, theta


def instantaneous_center(
    delta: float,
    k_e: float,
    l_e: float,
    theta_0: float,
) -> np.ndarray | None:
    """ICR via Kennedy's Theorem (Uicker et al. 2017, §3.4)."""
    theta = theta_0 + delta
    phi = antiparallel_phi(theta, k_e, l_e)
    denom = np.sin(theta - phi)
    if abs(denom) < 1e-9:
        return None
    s = k_e * np.cos(theta) / denom
    return np.array([k_e / 2.0 + s * np.sin(phi), -s * np.cos(phi)])


def elbow_angle(delta: float, k_e: float, l_e: float, theta_0: float) -> float:
    """Signed coupler rotation from equilibrium [rad]."""
    _, _, D, C, _, _, _ = linkage_positions(delta, k_e, l_e, theta_0)
    dc = C - D
    cross = -k_e * dc[1]
    dot_ = -k_e * dc[0]
    return np.arctan2(cross, dot_)


def delta_for_elbow(
    target_deg: float,
    theta_0: float,
    k_e: float,
    l_e: float,
) -> float:
    """Rod-angle deviation δ [rad] that produces a given elbow deflection [deg]."""
    target_rad = np.radians(target_deg)
    if abs(target_rad) < 1e-10:
        return 0.0
    if target_rad > 0:
        return brentq(
            lambda d: elbow_angle(d, k_e, l_e, theta_0) - target_rad,
            1e-4,
            np.pi / 2 - theta_0 - 0.02,
        )
    return brentq(
        lambda d: elbow_angle(d, k_e, l_e, theta_0) - target_rad,
        -(np.pi / 2 + theta_0 - 0.02),
        -1e-4,
    )


# ═══════════════════════════════════════════════════════════════════
# Centrode Ellipses  (Dijksman 1976)
# ═══════════════════════════════════════════════════════════════════

def centrode_params(l_e: float, k_e: float) -> tuple[float, float, float]:
    """Centrode ellipse semi-axes (a, b, c) where 2a = l_e, 2c = k_e."""
    a = l_e / 2.0
    c = k_e / 2.0
    b = np.sqrt(a**2 - c**2)
    return a, b, c


def centrode_ellipse_pts(
    a: float,
    b: float,
    cx: float = 0.0,
    cy: float = 0.0,
    angle: float = 0.0,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Points on an axis-aligned ellipse, rotated and translated."""
    t = np.linspace(0, 2 * np.pi, n)
    ca, sa = np.cos(angle), np.sin(angle)
    ex = a * np.cos(t)
    ey = b * np.sin(t)
    return ca * ex - sa * ey + cx, sa * ex + ca * ey + cy


def moving_centrode_pts(
    D: np.ndarray,
    C: np.ndarray,
    M: np.ndarray,
    a_ell: float,
    b_ell: float,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Moving centrode ellipse in fixed-frame coordinates."""
    dc = C - D
    angle = np.arctan2(dc[1], dc[0])
    return centrode_ellipse_pts(a_ell, b_ell, M[0], M[1], angle, n)


# ═══════════════════════════════════════════════════════════════════
# Tendon Force Directions
# ═══════════════════════════════════════════════════════════════════

ARROW_LEN = 35.0  # tendon arrow length [mm]


def tendon_force_dirs(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    arrow_len: float = ARROW_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tendon force direction vectors (crossing cables).

    T₀: cable at D pulls toward B.
    T₁: cable at C pulls toward A.
    Returns (t0_base, t0_tip, t1_base, t1_tip).
    """
    dir_t0 = B - D
    dir_t0 = dir_t0 / np.linalg.norm(dir_t0)
    dir_t1 = A - C
    dir_t1 = dir_t1 / np.linalg.norm(dir_t1)
    return D, D + arrow_len * dir_t0, C, C + arrow_len * dir_t1


# ═══════════════════════════════════════════════════════════════════
# Animation Sweep Sequence
# ═══════════════════════════════════════════════════════════════════

def elbow_sweep_sequence(elbow_max_deg: float, n_frames: int) -> np.ndarray:
    """Generate 0 → +max → −max → 0 elbow-angle sweep [deg]."""
    quarter = n_frames // 4
    half = n_frames // 2
    rest = n_frames - quarter - half
    return np.concatenate([
        np.linspace(0, elbow_max_deg, quarter, endpoint=False),
        np.linspace(elbow_max_deg, -elbow_max_deg, half, endpoint=False),
        np.linspace(-elbow_max_deg, 0, rest),
    ])


# ═══════════════════════════════════════════════════════════════════
# Disc Approximation Geometry
# ═══════════════════════════════════════════════════════════════════

def disc_platform_center(
    disc_center: np.ndarray,
    link_length: float,
    elbow_angle_rad: float,
) -> np.ndarray:
    """Platform center position in the disc approximation.

    The lower link rotates around disc_center by the elbow angle.
    """
    return disc_center + link_length * np.array([
        np.sin(elbow_angle_rad),
        -np.cos(elbow_angle_rad),
    ])


def disc_platform_bar(
    center: np.ndarray,
    half_width: float,
    elbow_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Platform bar endpoints (perpendicular to lower link direction)."""
    perp = np.array([np.cos(elbow_angle_rad), np.sin(elbow_angle_rad)])
    return center - half_width * perp, center + half_width * perp

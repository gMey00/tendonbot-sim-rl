"""Garment-region definitions on the flat rest shape (offline, pure torch/numpy).

Region assignment for the grasp-pair quality map: trials log grasp particles'
flat-rest coordinates (``present_heuristics_flat_rest.pt`` +
``rest_x_*``/``rest_y_*`` CSV columns), and THIS module bins them into
interpretable garment regions — so region boundaries can be iterated offline
without re-running any simulation.

Coordinate frame (see the region-map figure produced by ``plot_study.py``):
the ClothesNet t-shirt laid flat, x = left↔right (sleeve tips at x ≈ −0.262 /
+0.333 — the mesh is right-shifted; its axis of bilateral symmetry is
``X_CENTER`` = 0.0355, the midpoint of the sleeve tips), y = hem (−0.37) ↔
collar (+0.26).

Also hosts the pure-torch loader used by every offline analysis script.

Usage: ``from regions import assign_regions, REGION_NAMES, load_metrics``.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

# Axis of bilateral symmetry (midpoint of the sleeve tips, x = −0.262 / +0.333).
# The mesh OUTLINE is symmetric about it (median left-vs-mirrored-right particle
# distance ≈ 0.003 m ≈ mesh spacing); the tessellation is ~17 % denser on the
# left, which affects only per-cell particle COUNTS, not the (symmetric) cells.
X_CENTER = 0.0355


def _sym(name: str, dx: float, y: float) -> dict:
    """A left/right landmark pair mirrored exactly about ``X_CENTER``."""
    return {f"{name}_l": (X_CENTER - dx, y), f"{name}_r": (X_CENTER + dx, y)}


# Nearest-landmark (Voronoi) region assignment.  A pure x/y-threshold scheme
# fails on this mesh (the sleeves angle inward and overlap the shoulders in x),
# so regions are the Voronoi cells of hand-placed landmarks read off the
# flat-rest scatter.  The landmarks are placed SYMMETRICALLY about ``X_CENTER``
# (every l/r pair is an exact mirror; the central regions sit on the axis), and
# were tuned so that: the collar owns the neck opening (not the shoulders), the
# sleeve cells own the whole sleeve INCLUDING the outer tip + underside (the
# side cells no longer reach into the sleeves), and the shoulder cells own the
# shoulder seam.  Verified via ``plot_study.py``'s region-map figure.
REGION_LANDMARKS = {
    "collar":     (X_CENTER, 0.235),
    **_sym("shoulder", 0.16, 0.20),
    **_sym("sleeve", 0.26, 0.045),
    "chest":      (X_CENTER, 0.06),
    **_sym("side", 0.18, -0.12),
    "belly":      (X_CENTER, -0.18),
    "hem_l":      (X_CENTER - 0.185, -0.34),
    "hem_c":      (X_CENTER, -0.35),
    "hem_r":      (X_CENTER + 0.185, -0.34),
}
REGION_NAMES = list(REGION_LANDMARKS)


def assign_regions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized region id (index into ``REGION_NAMES``) for flat-rest coords."""
    pts = np.stack([np.asarray(x), np.asarray(y)], axis=-1)      # [..., 2]
    lms = np.array(list(REGION_LANDMARKS.values()))              # [R, 2]
    d2 = ((pts[..., None, :] - lms) ** 2).sum(axis=-1)           # [..., R]
    return d2.argmin(axis=-1)


def mirror_region(reg_id: np.ndarray) -> np.ndarray:
    """Map region ids to their left↔right mirror (for symmetrized aggregation)."""
    pairs = {"shoulder_l": "shoulder_r", "shoulder_r": "shoulder_l",
             "sleeve_l": "sleeve_r", "sleeve_r": "sleeve_l",
             "side_l": "side_r", "side_r": "side_l",
             "hem_l": "hem_r", "hem_r": "hem_l"}  # collar/chest/belly/hem_c self-mirror
    lut = np.array([REGION_NAMES.index(pairs.get(n, n)) for n in REGION_NAMES])
    return lut[reg_id]


def load_metrics():
    """Import ``shared/cloth_metrics.py`` by file path (pure torch, no isaaclab).

    Importing it through the ``tensegrity_pick`` package would pull isaaclab
    via the package ``__init__``; the direct file load works in a plain
    python env — the module is deliberately isaaclab-free.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(
        here, "..", "..", "..", "source", "tensegrity_pick", "tensegrity_pick",
        "tasks", "manager_based", "shared", "cloth_metrics.py"))
    spec = importlib.util.spec_from_file_location("cloth_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

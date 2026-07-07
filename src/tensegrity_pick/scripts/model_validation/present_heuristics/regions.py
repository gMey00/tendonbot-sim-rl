"""Garment-region definitions on the flat rest shape (offline, pure torch/numpy).

Region assignment for the grasp-pair quality map: trials log grasp particles'
flat-rest coordinates (``present_heuristics_flat_rest.pt`` +
``rest_x_*``/``rest_y_*`` CSV columns), and THIS module bins them into
interpretable garment regions — so region boundaries can be iterated offline
without re-running any simulation.

Coordinate frame (see the region-map figure produced by ``plot_study.py``):
the ClothesNet t-shirt laid flat, x = left↔right (sleeve tips at x ≈ −0.26 /
+0.33 — the mesh is slightly right-shifted, ``X_CENTER`` compensates),
y = hem (−0.37) ↔ collar (+0.26).

Also hosts the pure-torch loader used by every offline analysis script.

Usage: ``from regions import assign_regions, REGION_NAMES, load_metrics``.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

# Nearest-landmark (Voronoi) region assignment.  A pure x/y-threshold scheme
# fails on this mesh (the left sleeve angles inward and overlaps the shoulder
# in x), so regions are the Voronoi cells of hand-placed landmarks read off
# the flat-rest scatter — compact, interpretable, and robust to the mesh's
# left/right asymmetry.  Landmark placement validated visually via
# ``plot_study.py``'s region-map figure.
REGION_LANDMARKS = {
    "collar":     (0.03, 0.235),
    "shoulder_l": (-0.135, 0.205),
    "shoulder_r": (0.195, 0.205),
    "sleeve_l":   (-0.235, 0.06),
    "sleeve_r":   (0.305, 0.06),
    "chest":      (0.03, 0.05),
    "belly":      (0.03, -0.18),
    "side_l":     (-0.17, -0.10),
    "side_r":     (0.22, -0.10),
    "hem_l":      (-0.16, -0.34),
    "hem_c":      (0.03, -0.35),
    "hem_r":      (0.21, -0.34),
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

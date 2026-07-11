"""ClothesNet marker access — borderpoints + keypoints on the sim garment.

Thin loader over ``doc/reports/data/present_markers.pt`` (built once by
``build_markers.py``).  Pure numpy/torch, no isaaclab — usable both in the
booted run scripts (stratified sampling, keypoint picks) and in the offline
analysis (``plot_study.py``).

The two marker families and what they give the study:

* **borderpoints** — the garment's OPEN mesh edges (neck opening, the two
  sleeve cuffs, hem), recovered from the sim mesh topology and reconstructed
  complete + symmetric.  ``BORDER_DIST[i]`` is particle ``i``'s distance (m)
  to the nearest borderpoint: the "border nearness" whose effect on
  presentation quality the study measures.
* **keypoints** — the study's 12 SYMMETRIC region-landmark keypoints (one per
  Voronoi region, landmark snapped to the nearest particle,
  ``KEYPOINT_IDX``): candidate perception-friendly grasp targets.  The raw
  ClothesNet keypoints of the source garment (``SOURCE_SHIRT``) are kept only
  as a comparison overlay (``CLOTHESNET_KP_*``).

Helpers: ``region_particles(r)`` (particle ids in a folded region, for
stratified second-grasp sampling), ``bank_states_in_region(r, bank_anchor_idx)``
(bank states whose pinned anchor is in a folded region, for stratified FIRST
grasps), and ``border_dist_of(idx)`` (join border nearness onto logged grasp
particle indices).
"""

from __future__ import annotations

import os

import numpy as np
import torch

_DATA = os.path.join("/home/robot/studentische-arbeiten", "doc", "reports", "data")
_PATH = os.path.join(_DATA, "present_markers.pt")

_M = torch.load(_PATH, weights_only=False)

SOURCE_SHIRT: str = _M["source_shirt"]
REGISTER_CHAMFER: float = float(_M["register_chamfer"])
SYM_CLASSES: list[str] = list(_M["sym_classes"])

BOUNDARY_IDX: np.ndarray = _M["boundary_idx"].numpy()        # on-border particles
BORDER_XY: np.ndarray = _M["border_xy"].numpy()              # [Nb, 2] symmetric
BORDER_DIST: np.ndarray = _M["border_dist"].numpy()          # [P] metres
PARTICLE_REGION: np.ndarray = _M["particle_region"].numpy()  # [P] 0..7

# KEYPOINTS: symmetric, one per garment region, coinciding with the Voronoi
# cells (the region landmark snapped to the nearest particle).  Replaces
# ClothesNet's raw keypoints, which are asymmetric and do not align to the
# regions — those are kept only for the comparison overlay (CLOTHESNET_KP_*).
KEYPOINT_IDX: np.ndarray = _M["keypoint_idx"].numpy()        # [12]
KEYPOINT_NAMES: list[str] = list(_M["keypoint_names"])       # [12] region names
KEYPOINT_XY: np.ndarray = _M["keypoint_xy"].numpy()          # [12, 2]
KEYPOINT_REGION: np.ndarray = _M["keypoint_region"].numpy()  # [12] 0..7
KEYPOINT_BORDER_DIST: np.ndarray = _M["keypoint_border_dist"].numpy()
KEYPOINT_NEAREST_BANK: np.ndarray = _M["keypoint_nearest_bank"].numpy()
KEYPOINT_MIRROR_IDX: np.ndarray = _M["keypoint_mirror_idx"].numpy()  # [12]

# ClothesNet's raw keypoints + border.obj mapped into the flat-rest frame,
# for the reference/validation comparison figures only.
CLOTHESNET_KP_XY: np.ndarray = _M["clothesnet_kp_xy"].numpy()
CLOTHESNET_KP_REGION: np.ndarray = _M["clothesnet_kp_region"].numpy()
CLOTHESNET_BORDER_XY: np.ndarray = _M["clothesnet_border_xy"].numpy()

N_KEYPOINTS = len(KEYPOINT_IDX)

# The symmetric keypoint set (keypoints + mirror partners, deduplicated): the
# pool a keypoint-restricted grasp chooser can pick from on either side.  With
# region-landmark keypoints the mirror of a left keypoint is already the right
# keypoint, so this is effectively the 12 keypoints themselves.
KEYPOINT_IDX_SYM: np.ndarray = np.unique(
    np.concatenate([KEYPOINT_IDX, KEYPOINT_MIRROR_IDX]))


def region_id(name: str) -> int:
    return SYM_CLASSES.index(name)


def region_particles(r: int | str) -> np.ndarray:
    """Particle ids whose folded region == ``r`` (name or id)."""
    if isinstance(r, str):
        r = region_id(r)
    return np.nonzero(PARTICLE_REGION == r)[0]


def bank_states_in_region(r: int | str, bank_anchor_idx: np.ndarray) -> np.ndarray:
    """Bank state ids whose pinned anchor particle is in folded region ``r``.

    Mirror-invariant (the folded class is preserved by the bank's l<->r mirror
    augmentation), so a state selected here stays in ``r`` after augmentation.
    """
    if isinstance(r, str):
        r = region_id(r)
    return np.nonzero(PARTICLE_REGION[np.asarray(bank_anchor_idx)] == r)[0]


def border_dist_of(idx) -> np.ndarray:
    """Border nearness (m) of grasp particle indices (join for CSV analysis)."""
    return BORDER_DIST[np.asarray(idx)]


def keypoint_label(k: int) -> str:
    return KEYPOINT_NAMES[k]

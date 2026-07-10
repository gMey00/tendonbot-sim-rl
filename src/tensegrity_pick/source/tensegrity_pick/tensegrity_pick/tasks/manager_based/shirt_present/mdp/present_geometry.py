# present_geometry.py
#
# Garment-landmark geometry for the hem-to-hem presentation (pipeline task 2).
#
# PURE TORCH — no isaaclab imports, so the same code serves (a) the
# ShirtPresentEnv grasp-target / holder-filter logic and (b) offline unit
# tests / bank inspection without booting Isaac Sim.
#
# The heuristics study (doc/reports/present_heuristics_study.md §5.4) found the
# scripted **hem_corner <-> hem_corner** grasp pair is the presentation winner
# (median camera-plane coverage 0.820 @ ratio 1.05, vs 0.679 for the naive
# lowest-point rule).  Hem corners are identified here by the SAME hand-placed
# landmark scheme that study used
# (scripts/model_validation/present_heuristics/regions.py): Voronoi cells of
# flat-rest landmarks (a pure x/y threshold fails — this mesh's left sleeve
# angles inward and overlaps the shoulder in x).  Ported verbatim so the RL
# task's hem detection matches the study's ground truth exactly.

from __future__ import annotations

import torch

# Flat-rest landmark coordinates (x = left<->right, y = hem<->collar).  Copied
# from scripts/model_validation/present_heuristics/regions.py (validated
# visually via that study's region-map figure) — see it for the rationale.
REGION_LANDMARKS: dict[str, tuple[float, float]] = {
    "collar": (0.03, 0.235),
    "shoulder_l": (-0.135, 0.205),
    "shoulder_r": (0.195, 0.205),
    "sleeve_l": (-0.235, 0.06),
    "sleeve_r": (0.305, 0.06),
    "chest": (0.03, 0.05),
    "belly": (0.03, -0.18),
    "side_l": (-0.17, -0.10),
    "side_r": (0.22, -0.10),
    "hem_l": (-0.16, -0.34),
    "hem_c": (0.03, -0.35),
    "hem_r": (0.21, -0.34),
}
REGION_NAMES: list[str] = list(REGION_LANDMARKS)

# The bottom-edge regions a valid presentation HOLDER may grip: the two hem
# corners plus the centre hem.  side->hem_c and hem<->hem are the top oracle
# cells (study §4); anchoring the holder here reproduces the winning geometry.
HOLDER_REGIONS: tuple[str, ...] = ("hem_l", "hem_c", "hem_r")
# The two hem CORNERS — the hand's second-grasp target set (study §5.4).
HEM_CORNER_REGIONS: tuple[str, ...] = ("hem_l", "hem_r")


def assign_regions(flat_xy: torch.Tensor) -> torch.Tensor:
    """Nearest-landmark region id for flat-rest (x, y) coords ``[..., 2] -> [...]``."""
    lms = torch.tensor(
        list(REGION_LANDMARKS.values()), device=flat_xy.device, dtype=flat_xy.dtype
    )                                                       # [R, 2]
    d2 = ((flat_xy[..., None, :] - lms) ** 2).sum(dim=-1)   # [..., R]
    return d2.argmin(dim=-1)


def _nearest_particle(flat_rest_pos: torch.Tensor, landmark: tuple[float, float]) -> int:
    """Index of the particle closest to a flat-rest landmark (x, y)."""
    lm = torch.tensor(landmark, device=flat_rest_pos.device, dtype=flat_rest_pos.dtype)
    d2 = ((flat_rest_pos[:, :2] - lm) ** 2).sum(dim=-1)
    return int(d2.argmin())


def hem_corner_particle_ids(flat_rest_pos: torch.Tensor) -> torch.Tensor:
    """The two hem-corner particle indices ``[2]`` (hem_l, hem_r) for this mesh.

    Deterministic from the garment's flat-rest shape — the same landmark
    lookup the heuristics study used, so the RL grasp target IS a hem corner.
    """
    return torch.tensor(
        [_nearest_particle(flat_rest_pos, REGION_LANDMARKS[r]) for r in HEM_CORNER_REGIONS],
        device=flat_rest_pos.device, dtype=torch.long,
    )


def holder_region_mask(flat_rest_pos: torch.Tensor, anchor_idx: torch.Tensor) -> torch.Tensor:
    """Bool ``[K]``: which bank-anchor particles sit on the garment bottom edge.

    ``anchor_idx`` = the pinned particle of each hanging-bank state; True where
    that particle falls in a ``HOLDER_REGIONS`` (hem) cell — the states that
    reproduce the study's hem-held presentation when restored.
    """
    reg = assign_regions(flat_rest_pos[anchor_idx][:, :2])   # [K]
    holder_ids = torch.tensor(
        [REGION_NAMES.index(r) for r in HOLDER_REGIONS], device=flat_rest_pos.device
    )
    return torch.isin(reg, holder_ids)

# cloth_metrics.py
#
# Camera-plane visibility and stretch metrics for the hanging garment.
#
# PURE TORCH — deliberately free of isaaclab imports so the same functions
# serve (a) shirt_present observation/reward terms, (b) the robot-free
# presentation-heuristic study (scripts/model_validation/present_heuristics/)
# and (c) offline unit tests without booting Isaac Sim
# (scripts/model_validation/test_cloth_metrics.py).
#
# Geometry convention: the inspection camera looks along −Y
# (``INSPECTION_CAMERA_VIEW_DIR``), so its image plane is the world XZ plane
# → ``view_axis=1``.  A front+back ("both sides") inspection sees the SAME
# occluding silhouette, so one projected-silhouette area serves both
# viewpoints; what distinguishes a good presentation is how much of the
# garment's one-sided flat area that silhouette recovers
# (``silhouette_coverage`` → 1.0 = fully unfolded and camera-parallel; folds,
# wrinkles, and out-of-plane rotation all reduce it).

from __future__ import annotations

import torch

# Rasterization cell for silhouette areas.  The shirt's particle spacing is
# ~5 mm (11 k particles on a ~0.5 m garment), so 2 cm cells are guaranteed
# gap-free on connected cloth while keeping the grids tiny.
DEFAULT_CELL_SIZE = 0.02
_MAX_GRID = 2048  # safety cap (≈ 41 m extent at the default cell size)


def _as_batch(points: torch.Tensor) -> torch.Tensor:
    """Accept ``[P, 3]`` or ``[N, P, 3]``; return ``[N, P, 3]``."""
    return points.unsqueeze(0) if points.dim() == 2 else points


def silhouette_area(
    points: torch.Tensor,
    view_axis: int = 1,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> torch.Tensor:
    """Projected-silhouette area (m²) of a particle cloud per env ``[N]``.

    Projects the particles onto the plane perpendicular to ``view_axis``
    (1 = the −Y inspection-camera image plane, i.e. world XZ), rasterizes
    them into ``cell_size`` bins and returns occupied-cell count × cell area
    — the area of the shape the camera actually sees, robust to folds and
    double layers (overlapping cloth counts once, exactly like a silhouette).

    Args:
        points: ``[N, P, 3]`` (or a single ``[P, 3]``) world/local positions.
    """
    pts = _as_batch(points)
    axes = [a for a in range(3) if a != view_axis]
    uv = pts[..., axes]                                        # [N, P, 2]
    ij = ((uv - uv.min(dim=1, keepdim=True).values) / cell_size).long()
    grid_dim = int(ij.max().item()) + 1 if ij.numel() else 1
    if grid_dim > _MAX_GRID:
        raise ValueError(
            f"silhouette_area: extent {grid_dim * cell_size:.1f} m exceeds the "
            f"{_MAX_GRID}-cell grid cap — wrong units or exploded cloth?"
        )
    flat = ij[..., 0] * grid_dim + ij[..., 1]                  # [N, P]
    grid = torch.zeros(
        pts.shape[0], grid_dim * grid_dim, dtype=torch.bool, device=pts.device
    )
    grid.scatter_(1, flat, True)
    return grid.sum(dim=1).float() * (cell_size ** 2)


def flat_silhouette_area(
    flat_rest_pos: torch.Tensor, cell_size: float = DEFAULT_CELL_SIZE,
) -> float:
    """One-sided flat area (m²) of the garment's rest shape.

    The reference for ``silhouette_coverage``: pass
    ``ClothObject.flat_rest_pos`` (the flat lay lies in the XY plane →
    projected along Z).  Compute once at env init.
    """
    return float(silhouette_area(flat_rest_pos, view_axis=2, cell_size=cell_size)[0])


def silhouette_coverage(
    points: torch.Tensor,
    ref_area: float,
    view_axis: int = 1,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> torch.Tensor:
    """Fraction ``[N]`` of the flat one-sided area visible in the camera plane.

    1.0 ⇔ the garment is fully unfolded AND parallel to the image plane —
    the honest "inspectable" score.  A centre-hung crumpled shirt scores
    ~0.2–0.4; a two-point stretched shirt facing the camera approaches 1.
    """
    return silhouette_area(points, view_axis, cell_size) / max(ref_area, 1e-9)


def two_sided_visible_fraction(
    points: torch.Tensor,
    view_axis: int = 1,
    cell_size: float = DEFAULT_CELL_SIZE,
    layer_gap: float = 0.015,
) -> torch.Tensor:
    """Fraction ``[N]`` of particles visible from the front OR back camera.

    Complements ``silhouette_coverage``: the silhouette counts overlapping
    layers ONCE, so a sleeve draped in front of the torso does not reduce
    coverage even though the torso patch behind it is not inspectable from
    either side.  This metric rasterizes the same image plane and, per cell,
    clusters the particles along the view axis into depth layers (split where
    consecutive depth gaps exceed ``layer_gap`` ≈ 3× the mesh spacing): the
    NEAREST layer is what the front camera (+view_axis, looking along
    −view_axis) sees, the FARTHEST what the back camera sees; middle layers
    are hidden from both.  Returns visible particles ÷ all particles.

    A flat two-panel presentation scores ≈ 1.0 (front camera sees the front
    panel, back camera the back panel); every extra fabric layer folded across
    the silhouette lowers it.  Limitation: layers in CONTACT (< ``layer_gap``
    apart) merge into one cluster and count as visible, so the metric is a
    LOWER bound on occlusion — only clearly separated occluded layers are
    caught.
    """
    pts = _as_batch(points)
    axes = [a for a in range(3) if a != view_axis]
    uv = pts[..., axes]
    depth = pts[..., view_axis]                                # [N, P]
    ij = ((uv - uv.min(dim=1, keepdim=True).values) / cell_size).long()
    grid_dim = int(ij.max().item()) + 1 if ij.numel() else 1
    if grid_dim > _MAX_GRID:
        raise ValueError(
            f"two_sided_visible_fraction: extent {grid_dim * cell_size:.1f} m "
            f"exceeds the {_MAX_GRID}-cell grid cap — wrong units or exploded "
            "cloth?"
        )
    cell = ij[..., 0] * grid_dim + ij[..., 1]                  # [N, P]
    # Group by cell with depth ascending inside each cell (two stable sorts).
    ord1 = depth.argsort(dim=1)
    ord2 = cell.gather(1, ord1).argsort(dim=1, stable=True)
    order = ord1.gather(1, ord2)                               # [N, P]
    c_s = cell.gather(1, order)
    d_s = depth.gather(1, order)
    new_cell = torch.ones_like(c_s, dtype=torch.bool)
    new_cell[:, 1:] = c_s[:, 1:] != c_s[:, :-1]
    gap = torch.zeros_like(d_s)
    gap[:, 1:] = d_s[:, 1:] - d_s[:, :-1]
    clus = (new_cell | (gap > layer_gap)).long().cumsum(dim=1) - 1
    # First / last cluster id of each cell run (clus is non-decreasing, so a
    # forward cummax over cell starts / reversed cummin over cell ends
    # propagates the boundary cluster ids across the run).
    first = torch.where(new_cell, clus, torch.zeros_like(clus)).cummax(dim=1).values
    end = torch.ones_like(new_cell)
    end[:, :-1] = new_cell[:, 1:]
    big = clus.max() + 1 if clus.numel() else 1
    last = torch.where(end, clus, torch.full_like(clus, int(big)))
    last = last.flip(1).cummin(dim=1).values.flip(1)
    vis_sorted = (clus == first) | (clus == last)
    visible = torch.zeros_like(vis_sorted)
    visible.scatter_(1, order, vis_sorted)
    return visible.float().mean(dim=1)


def plane_extent(points: torch.Tensor, view_axis: int = 1) -> torch.Tensor:
    """Bounding-box extents ``[N, 2]`` (u, v) of the projected silhouette.

    Cheaper, coarser cousin of the silhouette area (spread without fill
    information) — useful as a shaping signal where the rasterized area is
    too step-like.
    """
    pts = _as_batch(points)
    axes = [a for a in range(3) if a != view_axis]
    uv = pts[..., axes]
    return uv.max(dim=1).values - uv.min(dim=1).values


def stretch_ratio(
    point_a: torch.Tensor, point_b: torch.Tensor, rest_dist: torch.Tensor | float,
) -> torch.Tensor:
    """Tautness ``[N]`` of the cloth span between two grasp points.

    ``‖a − b‖ / rest distance``: < 1 slack, → 1 taut, > 1 stretched beyond
    rest (two-attachment stretch validated stable through 1.15, Stage-0).
    ``rest_dist`` is the distance between the same two particles in
    ``ClothObject.flat_rest_pos``.
    """
    d = torch.norm(point_a - point_b, dim=-1)
    if not isinstance(rest_dist, torch.Tensor):
        rest_dist = torch.as_tensor(rest_dist, device=d.device, dtype=d.dtype)
    return d / rest_dist.clamp(min=1e-9)


def rest_distance(flat_rest_pos: torch.Tensor, idx_a, idx_b) -> torch.Tensor:
    """Rest-shape distance(s) between particle indices (tensors broadcast)."""
    return torch.norm(flat_rest_pos[idx_a] - flat_rest_pos[idx_b], dim=-1)

"""Workspace visualisation for the 5-DOF tensegrity robot.

Loads pre-computed workspace data (from ``workspace_sample.py``) and generates
publication-quality figures:

1. Yoshikawa manipulability index [1]
2. Inverse condition number (isotropy index) [2]
3. Reachability density [4]

All three figures share an identical layout produced by a single
``create_workspace_figure`` function, ensuring consistent formatting.

References
----------
[1] Yoshikawa, T. (1985). "Manipulability of Robotic Mechanisms."
    The International Journal of Robotics Research, 4(2), 3-9.
[2] Salisbury, J. K., & Craig, J. J. (1982). "Articulated Hands: Force
    Control and Kinematic Issues." IJRR, 1(1), 4-17.
[3] Klein, R. (2023). Master's Thesis -- Tensegrity Robot Pick-and-Place.
[4] MathWorks (2024). "Workspace Analysis for Manipulators."
    https://de.mathworks.com/help/robotics/ug/workspace-analysis-for-manipulators.html
[5] Togai, M. (1986). "An Application of the Singular Value Decomposition
    to Manipulability and Sensitivity of Industrial Robots."

Usage
-----
    cd src/tensegrity_pick
    python3 scripts/workspace_analysis/workspace_visualize.py
    python3 scripts/workspace_analysis/workspace_visualize.py --input_dir outputs/workspace_analysis/
    python3 scripts/workspace_analysis/workspace_visualize.py --voxel_size 0.03
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────

# Desired workspace box in env-local coordinates [Klein 2023, Ref. 3].
DESIRED_WS_MIN = np.array([-0.05, -0.40, 0.80])
DESIRED_WS_MAX = np.array([0.35, 0.95, 1.30])


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeatmapViewConfig:
    """Configuration for a single 2-D projection heatmap."""

    column_a: int
    column_b: int
    xlabel: str
    ylabel: str
    title: str
    rect_x_min: float
    rect_x_max: float
    rect_y_min: float
    rect_y_max: float


TOP_VIEW = HeatmapViewConfig(
    column_a=0, column_b=1,
    xlabel="X (m)", ylabel="Y (m)", title="Top View XY",
    rect_x_min=DESIRED_WS_MIN[0], rect_x_max=DESIRED_WS_MAX[0],
    rect_y_min=DESIRED_WS_MIN[1], rect_y_max=DESIRED_WS_MAX[1],
)

SIDE_VIEW = HeatmapViewConfig(
    column_a=0, column_b=2,
    xlabel="X (m)", ylabel="Z (m)", title="Side View XZ",
    rect_x_min=DESIRED_WS_MIN[0], rect_x_max=DESIRED_WS_MAX[0],
    rect_y_min=DESIRED_WS_MIN[2], rect_y_max=DESIRED_WS_MAX[2],
)

FRONT_VIEW = HeatmapViewConfig(
    column_a=1, column_b=2,
    xlabel="Y (m)", ylabel="Z (m)", title="Front View YZ",
    rect_x_min=DESIRED_WS_MIN[1], rect_x_max=DESIRED_WS_MAX[1],
    rect_y_min=DESIRED_WS_MIN[2], rect_y_max=DESIRED_WS_MAX[2],
)


# ── Voxelisation ──────────────────────────────────────────────────────────

def voxelize(
    positions: np.ndarray,
    values: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin positions into a 3-D voxel grid and compute per-voxel mean.

    Returns ``(voxel_centers, voxel_means, voxel_counts)``.
    """
    grid_indices = np.floor(positions / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices, voxel_counts = np.unique(
        grid_indices, axis=0, return_inverse=True, return_counts=True,
    )

    valid_mask = np.isfinite(values)
    voxel_value_sums = np.zeros(len(unique_voxels), dtype=np.float64)
    voxel_valid_counts = np.zeros(len(unique_voxels), dtype=np.int64)

    np.add.at(voxel_value_sums, inverse_indices[valid_mask], values[valid_mask].astype(np.float64))
    np.add.at(voxel_valid_counts, inverse_indices[valid_mask], 1)

    voxel_means = np.where(
        voxel_valid_counts > 0,
        voxel_value_sums / voxel_valid_counts,
        np.nan,
    ).astype(np.float32)

    voxel_centers = (unique_voxels.astype(np.float64) + 0.5) * voxel_size
    return voxel_centers.astype(np.float32), voxel_means, voxel_counts


def compute_2d_heatmap(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2-D binned-mean heatmap of *values*.

    Returns ``(a_edges, b_edges, grid_mean)`` where ``grid_mean`` has NaN
    for empty bins.
    """
    a_edges = np.arange(coord_a.min(), coord_a.max() + bin_size, bin_size)
    b_edges = np.arange(coord_b.min(), coord_b.max() + bin_size, bin_size)

    a_idx = np.clip(np.digitize(coord_a, a_edges) - 1, 0, len(a_edges) - 2)
    b_idx = np.clip(np.digitize(coord_b, b_edges) - 1, 0, len(b_edges) - 2)

    grid_sum = np.zeros((len(a_edges) - 1, len(b_edges) - 1), dtype=np.float64)
    grid_count = np.zeros_like(grid_sum)

    valid = np.isfinite(values)
    np.add.at(grid_sum, (a_idx[valid], b_idx[valid]), values[valid].astype(np.float64))
    np.add.at(grid_count, (a_idx[valid], b_idx[valid]), 1)

    grid_mean = np.full_like(grid_sum, np.nan, dtype=np.float32)
    occupied = grid_count > 0
    grid_mean[occupied] = (grid_sum[occupied] / grid_count[occupied]).astype(np.float32)

    return a_edges, b_edges, grid_mean


def compute_2d_density(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2-D log-density heatmap (NaN where zero counts).

    Returns ``(a_edges, b_edges, log_histogram)``.
    """
    a_edges = np.arange(coord_a.min(), coord_a.max() + bin_size, bin_size)
    b_edges = np.arange(coord_b.min(), coord_b.max() + bin_size, bin_size)
    histogram, _, _ = np.histogram2d(coord_a, coord_b, bins=[a_edges, b_edges])
    log_histogram = np.log1p(histogram)
    log_histogram[histogram == 0] = np.nan
    return a_edges, b_edges, log_histogram


# ── Drawing helpers ───────────────────────────────────────────────────────

def _draw_box_3d(ax: plt.Axes, ws_min: np.ndarray, ws_max: np.ndarray) -> None:
    """Overlay a translucent red wireframe box onto a 3-D axes."""
    x = [ws_min[0], ws_max[0]]
    y = [ws_min[1], ws_max[1]]
    z = [ws_min[2], ws_max[2]]

    vertices = [
        [x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]],
        [x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]],
    ]
    faces = [
        [vertices[i] for i in face]
        for face in (
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5),
        )
    ]
    ax.add_collection3d(
        Poly3DCollection(faces, alpha=0.10, facecolor="red", edgecolor="red", linewidth=0.8),
    )


def _draw_rect_2d(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Overlay a translucent red rectangle onto a 2-D axes."""
    ax.add_patch(plt.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=True, facecolor="red", edgecolor="red", alpha=0.15, linewidth=1.5,
    ))


def _set_proportional_3d_axes(
    ax: plt.Axes,
    positions: np.ndarray,
    margin_fraction: float = 0.05,
) -> None:
    """Set 3-D axis limits and box aspect so that spatial scaling is uniform."""
    ranges = np.array([
        positions[:, i].max() - positions[:, i].min() for i in range(3)
    ])
    ranges = np.where(ranges < 1e-6, 1e-6, ranges)
    margins = ranges * margin_fraction
    ax.set_xlim(positions[:, 0].min() - margins[0], positions[:, 0].max() + margins[0])
    ax.set_ylim(positions[:, 1].min() - margins[1], positions[:, 1].max() + margins[1])
    ax.set_zlim(positions[:, 2].min() - margins[2], positions[:, 2].max() + margins[2])
    ax.set_box_aspect(ranges.tolist())


def _add_heatmap(
    ax: plt.Axes,
    a_edges: np.ndarray,
    b_edges: np.ndarray,
    grid: np.ndarray,
    colormap: str,
    norm: Normalize,
    view: HeatmapViewConfig,
) -> None:
    """Populate a 2-D axes with pcolormesh, desired-workspace rectangle and labels."""
    ax.pcolormesh(a_edges, b_edges, grid.T, cmap=colormap, norm=norm, shading="flat")
    _draw_rect_2d(ax, view.rect_x_min, view.rect_x_max, view.rect_y_min, view.rect_y_max)
    ax.set_xlabel(view.xlabel)
    ax.set_ylabel(view.ylabel)
    ax.set_title(view.title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


# ── Unified figure creation ──────────────────────────────────────────────

def create_workspace_figure(
    positions: np.ndarray,
    per_sample_values: np.ndarray,
    voxel_size: float,
    output_path: Path,
    figure_title: str,
    colorbar_label: str,
    colormap: str = "plasma",
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    *,
    density_mode: bool = False,
) -> None:
    """Create and save a workspace analysis figure.

    Layout: 3-D scatter spanning the left column, Top XY and Side XZ views
    on the upper-right, Front YZ view spanning the lower-right (same width
    as top + side combined).  A single shared colour bar is placed on the
    right edge.

    Parameters
    ----------
    positions : (N, 3) array of env-local EE positions.
    per_sample_values : (N,) metric values for mean-mode, or ignored in
        density mode.
    voxel_size : bin edge length in metres.
    output_path : file path for the saved PNG.
    figure_title : suptitle text (may contain newlines).
    colorbar_label : label next to the colour bar.
    colormap : matplotlib colourmap name.
    percentile_clip : lower/upper percentile for colour normalisation
        (mean-mode only).
    density_mode : when True, colour encodes log(1 + count) instead of
        per-voxel mean of *per_sample_values*.
    """
    # ── Compute 3-D voxel values ──────────────────────────────────────
    if density_mode:
        ones = np.ones(len(positions), dtype=np.float32)
        voxel_centers, _, voxel_counts = voxelize(positions, ones, voxel_size)
        scatter_values = np.log1p(voxel_counts.astype(np.float32))
        norm = Normalize(vmin=0.0, vmax=float(np.percentile(scatter_values, 98)))
    else:
        voxel_centers, voxel_means, _ = voxelize(positions, per_sample_values, voxel_size)
        scatter_values = voxel_means
        finite = scatter_values[np.isfinite(scatter_values)]
        if len(finite) == 0:
            print(f"  WARNING: No finite values -- skipping {output_path.name}")
            return
        vmin = max(float(np.percentile(finite, percentile_clip[0])), 0.0)
        vmax = max(float(np.percentile(finite, percentile_clip[1])), 1e-8)
        norm = Normalize(vmin=vmin, vmax=vmax)

    # ── Layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(figure_title, fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.3, 0.5, 0.5],
        height_ratios=[1, 1],
        hspace=0.35, wspace=0.35,
    )

    # ── 3-D scatter ───────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    finite_mask = np.isfinite(scatter_values)
    ax3d.scatter(
        voxel_centers[finite_mask, 0],
        voxel_centers[finite_mask, 1],
        voxel_centers[finite_mask, 2],
        c=scatter_values[finite_mask],
        cmap=colormap, norm=norm, s=3, alpha=0.6, edgecolors="none",
    )
    _draw_box_3d(ax3d, DESIRED_WS_MIN, DESIRED_WS_MAX)
    _set_proportional_3d_axes(ax3d, positions)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Workspace (Voxelised)")
    ax3d.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax3d.tick_params(axis="x", labelsize=7, pad=1)

    # ── 2-D heatmaps ─────────────────────────────────────────────────
    upper_views = [(0, 1, TOP_VIEW), (0, 2, SIDE_VIEW)]
    heatmap_axes: list[plt.Axes] = []

    for gs_row, gs_col, view in upper_views:
        ax = fig.add_subplot(gs[gs_row, gs_col])
        if density_mode:
            a_edges, b_edges, grid = compute_2d_density(
                positions[:, view.column_a], positions[:, view.column_b], voxel_size,
            )
        else:
            a_edges, b_edges, grid = compute_2d_heatmap(
                positions[:, view.column_a], positions[:, view.column_b],
                per_sample_values, voxel_size,
            )
        _add_heatmap(ax, a_edges, b_edges, grid, colormap, norm, view)
        heatmap_axes.append(ax)

    # Front view spanning both right columns
    ax_front = fig.add_subplot(gs[1, 1:])
    if density_mode:
        a_edges, b_edges, grid = compute_2d_density(
            positions[:, FRONT_VIEW.column_a], positions[:, FRONT_VIEW.column_b], voxel_size,
        )
    else:
        a_edges, b_edges, grid = compute_2d_heatmap(
            positions[:, FRONT_VIEW.column_a], positions[:, FRONT_VIEW.column_b],
            per_sample_values, voxel_size,
        )
    _add_heatmap(ax_front, a_edges, b_edges, grid, colormap, norm, FRONT_VIEW)
    heatmap_axes.append(ax_front)

    # ── Shared colour bar ─────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm, ax=heatmap_axes, shrink=0.85, pad=0.03,
        label=colorbar_label, location="right",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise pre-computed workspace analysis data for the 5-DOF tensegrity robot",
    )
    parser.add_argument(
        "--input_dir", type=str, default="outputs/workspace_analysis",
        help="Directory containing .npy data files (default: outputs/workspace_analysis/)",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.02,
        help="Voxel edge length in metres for 3-D binning (default: 0.02)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    voxel_size = args.voxel_size

    positions = np.load(input_dir / "ee_positions.npy")
    yoshikawa = np.load(input_dir / "yoshikawa.npy")
    condition = np.load(input_dir / "condition_number.npy")

    print(f"Loaded {len(positions):,} samples from {input_dir}/")

    # Figure 1: Yoshikawa manipulability [Ref. 1]
    create_workspace_figure(
        positions, yoshikawa, voxel_size,
        output_path=input_dir / "workspace_manipulability.png",
        figure_title=(
            "5-DOF Tensegrity Robot -- Workspace Manipulability Analysis\n"
            "Colour = Yoshikawa Manipulability Index  w(q) = sqrt(det(J J^T))  [Yoshikawa 1985]"
        ),
        colorbar_label="Yoshikawa Index",
        colormap="plasma",
    )

    # Figure 2: Inverse condition number [Ref. 2]
    create_workspace_figure(
        positions, condition, voxel_size,
        output_path=input_dir / "workspace_condition.png",
        figure_title=(
            "5-DOF Tensegrity Robot -- Workspace Isotropy Analysis\n"
            "Colour = Inverse Condition Number  sigma_min / sigma_max  [Salisbury & Craig 1982]"
        ),
        colorbar_label="Inv. Condition Number",
        colormap="inferno",
    )

    # Figure 3: Reachability density [Ref. 4]
    create_workspace_figure(
        positions, positions[:, 0], voxel_size,  # per_sample_values ignored in density mode
        output_path=input_dir / "workspace_density.png",
        figure_title=(
            "5-DOF Tensegrity Robot -- Reachability Density Analysis\n"
            "Colour = log(1 + sample count per voxel)  [Monte Carlo FK sampling, Ref. 4]"
        ),
        colorbar_label="log(1 + count)",
        colormap="viridis",
        density_mode=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()

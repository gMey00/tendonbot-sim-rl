"""Workspace visualisation for the 5-DOF tensegrity robot.

Loads pre-computed workspace data (from ``workspace_sample.py``) and generates
publication-quality figures:

1. Yoshikawa manipulability index [1]
2. Inverse condition number (isotropy index) [2]
3. Reachability density [4]

Each figure contains a full 3-D voxelised scatter on the left and three 2-D
cross-section heatmaps on the right.  The cross-sections are thin slices
through the midpoint of the desired workspace along each axis, revealing the
metric distribution *inside* the workspace rather than projecting all data.

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

import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[4] / ".config"))
import plot_config as pcfg  # noqa: E402
pcfg.apply_style()

from workspace_analysis_helper import (  # noqa: E402
    DESIRED_WS_MAX,
    DESIRED_WS_MIN,
    ROBOT_DISPLAY_NAMES,
    compute_2d_density,
    compute_2d_heatmap,
    load_workspace_data,
    voxelize,
)

from workspace_config import DEFAULT_SLICE_THICKNESS, DEFAULT_VOXEL_SIZE  # noqa: E402

# Contrasting box colours per colormap (high visibility against the palette)
BOX_COLORS: dict[str, str] = {
    "plasma":  "cyan",
    "inferno": "cyan",
    "viridis": "magenta",
}
BOX_3D_ALPHA = 0.20
BOX_2D_ALPHA = 0.25


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeatmapViewConfig:
    """Configuration for a single 2-D cross-section heatmap."""

    column_a: int
    column_b: int
    slice_column: int
    xlabel: str
    ylabel: str
    title: str
    rect_x_min: float
    rect_x_max: float
    rect_y_min: float
    rect_y_max: float


DESIRED_WS_MID = (DESIRED_WS_MIN + DESIRED_WS_MAX) / 2
_AXIS_LABELS = ["X", "Y", "Z"]

TOP_VIEW = HeatmapViewConfig(
    column_a=0, column_b=1, slice_column=2,
    xlabel="X (m)", ylabel="Y (m)",
    title=f"Top XY  (Z = {DESIRED_WS_MID[2]:.2f} m)",
    rect_x_min=DESIRED_WS_MIN[0], rect_x_max=DESIRED_WS_MAX[0],
    rect_y_min=DESIRED_WS_MIN[1], rect_y_max=DESIRED_WS_MAX[1],
)

SIDE_VIEW = HeatmapViewConfig(
    column_a=0, column_b=2, slice_column=1,
    xlabel="X (m)", ylabel="Z (m)",
    title=f"Side XZ  (Y = {DESIRED_WS_MID[1]:.2f} m)",
    rect_x_min=DESIRED_WS_MIN[0], rect_x_max=DESIRED_WS_MAX[0],
    rect_y_min=DESIRED_WS_MIN[2], rect_y_max=DESIRED_WS_MAX[2],
)

FRONT_VIEW = HeatmapViewConfig(
    column_a=1, column_b=2, slice_column=0,
    xlabel="Y (m)", ylabel="Z (m)",
    title=f"Front YZ  (X = {DESIRED_WS_MID[0]:.2f} m)",
    rect_x_min=DESIRED_WS_MIN[1], rect_x_max=DESIRED_WS_MAX[1],
    rect_y_min=DESIRED_WS_MIN[2], rect_y_max=DESIRED_WS_MAX[2],
)


# ── Drawing helpers ───────────────────────────────────────────────────────

def _draw_box_3d(
    ax: plt.Axes,
    ws_min: np.ndarray,
    ws_max: np.ndarray,
    color: str = "cyan",
    alpha: float = BOX_3D_ALPHA,
) -> None:
    """Overlay a translucent wireframe box onto a 3-D axes."""
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
        Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.8),
    )


def _draw_rect_2d(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    color: str = "cyan",
    alpha: float = BOX_2D_ALPHA,
) -> None:
    """Overlay a translucent rectangle onto a 2-D axes."""
    ax.add_patch(plt.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=True, facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5,
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
    box_color: str = "cyan",
) -> None:
    """Populate a 2-D axes with pcolormesh, desired-workspace rectangle and labels."""
    ax.pcolormesh(a_edges, b_edges, grid.T, cmap=colormap, norm=norm, shading="flat")
    _draw_rect_2d(ax, view.rect_x_min, view.rect_x_max, view.rect_y_min, view.rect_y_max,
                  color=box_color)
    ax.set_xlabel(view.xlabel, fontsize=12)
    ax.set_ylabel(view.ylabel, fontsize=12)
    ax.set_title(view.title, fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_aspect("equal", adjustable="box")


# ── Unified figure creation ──────────────────────────────────────────────

def _slice_mask(
    positions: np.ndarray,
    view: HeatmapViewConfig,
    half_thickness: float,
) -> np.ndarray:
    """Boolean mask selecting samples within *half_thickness* of the workspace midpoint."""
    centre = float(DESIRED_WS_MID[view.slice_column])
    coord = positions[:, view.slice_column]
    return (coord >= centre - half_thickness) & (coord <= centre + half_thickness)


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
    slice_thickness: float = 0.05,
    higher_is_better: bool = True,
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
    slice_thickness : full thickness (metres) of cross-section slices
        centred on the workspace midpoint (default 0.05 m).
    higher_is_better : when True an upward arrow below the colour bar reads
        "better"; when False the arrow points downward.
    """
    # Resolve box colour for this colormap
    box_color = BOX_COLORS.get(colormap, "cyan")

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
    fig = plt.figure(figsize=(18, 10))
    if pcfg.SHOW_TITLES and figure_title:
        fig.suptitle(figure_title, y=0.98, fontsize=14)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.3, 0.55, 0.55],
        height_ratios=[1, 1],
        hspace=0.38, wspace=0.22,
        left=0.04, right=0.82, top=0.93, bottom=0.07,
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
    _draw_box_3d(ax3d, DESIRED_WS_MIN, DESIRED_WS_MAX, color=box_color)
    _set_proportional_3d_axes(ax3d, positions)
    ax3d.view_init(elev=20, azim=-55)
    ax3d.set_xlabel("X (m)", labelpad=-2)
    ax3d.set_ylabel("Y (m)", labelpad=-2)
    ax3d.set_zlabel("Z (m)", labelpad=-2)
    ax3d.set_title("3D Workspace (Voxelised)", fontsize=14)
    ax3d.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax3d.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax3d.zaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax3d.tick_params(axis="both", labelsize=11, pad=0)

    # ── 2-D cross-section heatmaps ────────────────────────────────────
    half_thickness = slice_thickness / 2.0
    upper_views = [(0, 1, TOP_VIEW), (0, 2, SIDE_VIEW)]
    heatmap_axes: list[plt.Axes] = []

    for gs_row, gs_col, view in upper_views:
        ax = fig.add_subplot(gs[gs_row, gs_col])
        mask = _slice_mask(positions, view, half_thickness)
        sliced_pos = positions[mask]
        sliced_vals = per_sample_values[mask]
        if density_mode:
            a_edges, b_edges, grid = compute_2d_density(
                sliced_pos[:, view.column_a], sliced_pos[:, view.column_b], voxel_size,
            )
        else:
            a_edges, b_edges, grid = compute_2d_heatmap(
                sliced_pos[:, view.column_a], sliced_pos[:, view.column_b],
                sliced_vals, voxel_size,
            )
        _add_heatmap(ax, a_edges, b_edges, grid, colormap, norm, view, box_color=box_color)
        heatmap_axes.append(ax)

    # Front view spanning both right columns
    ax_front = fig.add_subplot(gs[1, 1:])
    front_mask = _slice_mask(positions, FRONT_VIEW, half_thickness)
    sliced_pos = positions[front_mask]
    sliced_vals = per_sample_values[front_mask]
    if density_mode:
        a_edges, b_edges, grid = compute_2d_density(
            sliced_pos[:, FRONT_VIEW.column_a], sliced_pos[:, FRONT_VIEW.column_b], voxel_size,
        )
    else:
        a_edges, b_edges, grid = compute_2d_heatmap(
            sliced_pos[:, FRONT_VIEW.column_a], sliced_pos[:, FRONT_VIEW.column_b],
            sliced_vals, voxel_size,
        )
    _add_heatmap(ax_front, a_edges, b_edges, grid, colormap, norm, FRONT_VIEW,
                 box_color=box_color)
    heatmap_axes.append(ax_front)

    # ── Shared colour bar (fixed position, right of heatmaps) ────────
    cbar_ax = fig.add_axes([0.855, 0.10, 0.018, 0.75])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(colorbar_label, fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    # ── "Better" annotation below the colour bar ──────────────────────
    arrow = "↑" if higher_is_better else "↓"
    cbar.ax.text(
        0.5, -0.03, f"{arrow} better",
        transform=cbar.ax.transAxes,
        ha="center", va="top",
        fontsize=11,
        fontfamily="DejaVu Sans",
        clip_on=False,
    )

    pcfg.finalize(fig, output_path, tight=False)


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
        "--voxel_size", type=float, default=DEFAULT_VOXEL_SIZE,
        help=f"Voxel edge length in metres for 3-D binning (default: {DEFAULT_VOXEL_SIZE})",
    )
    parser.add_argument(
        "--slice_thickness", type=float, default=DEFAULT_SLICE_THICKNESS,
        help=f"Thickness of 2-D cross-section slices in metres (default: {DEFAULT_SLICE_THICKNESS})",
    )
    parser.add_argument(
        "--robot_name", type=str, default=None,
        help="Robot display name for figure titles (auto-detected from statistics.json if present)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    voxel_size = args.voxel_size
    slice_thickness = args.slice_thickness

    # Auto-detect robot name from statistics.json if available
    robot_display_name = args.robot_name
    if robot_display_name is None:
        stats_path = input_dir / "statistics.json"
        if stats_path.exists():
            import json
            with open(stats_path) as f:
                stats = json.load(f)
            robot_key = stats.get("robot", "tensegrity")
            robot_display_name = ROBOT_DISPLAY_NAMES.get(robot_key, robot_key)
        else:
            robot_display_name = "5-DOF Tensegrity Robot"

    positions, yoshikawa, condition = load_workspace_data(input_dir)

    print(f"Loaded {len(positions):,} samples from {input_dir}/")

    # Figure 1: Yoshikawa manipulability [Ref. 1]
    create_workspace_figure(
        positions, yoshikawa, voxel_size,
        output_path=input_dir / "workspace_manipulability.png",
        figure_title=(
            f"{robot_display_name} — Workspace Manipulability Analysis\n"
            "Colour = Yoshikawa Manipulability Index  w(q) = sqrt(det(J J^T))  [Yoshikawa 1985]"
        ),
        colorbar_label="Yoshikawa Index",
        colormap="plasma",
        slice_thickness=slice_thickness,
    )

    # Figure 2: Inverse condition number [Ref. 2]
    create_workspace_figure(
        positions, condition, voxel_size,
        output_path=input_dir / "workspace_condition.png",
        figure_title=(
            f"{robot_display_name} — Workspace Isotropy Analysis\n"
            "Colour = Inverse Condition Number  σ_min / σ_max  [Salisbury & Craig 1982]"
        ),
        colorbar_label="Inv. Condition Number",
        colormap="inferno",
        slice_thickness=slice_thickness,
    )

    # Figure 3: Reachability density [Ref. 4]
    create_workspace_figure(
        positions, positions[:, 0], voxel_size,  # per_sample_values ignored in density mode
        output_path=input_dir / "workspace_density.png",
        figure_title=(
            f"{robot_display_name} — Reachability Density Analysis\n"
            "Colour = log(1 + sample count per voxel)  [Monte Carlo FK sampling, Ref. 4]"
        ),
        colorbar_label="log(1 + count)",
        colormap="viridis",
        density_mode=True,
        slice_thickness=slice_thickness,
    )

    print("Done.")


if __name__ == "__main__":
    main()

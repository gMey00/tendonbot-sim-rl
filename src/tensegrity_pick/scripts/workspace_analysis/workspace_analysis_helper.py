"""Shared utilities for the workspace analysis Jupyter notebook.

Provides data loading, statistics computation, voxelisation, and
interactive / static plotting helpers.  This module intentionally avoids
any Isaac Sim imports so it can be used in a plain Jupyter kernel.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display

from workspace_config import (
    AXIS_LABELS,
    DEFAULT_VOXEL_SIZE,
    DESIRED_WS_MAX,
    DESIRED_WS_MIN,
    ROBOTS,
    VIEW_CONFIGS,
)

# Derived convenience dicts (so callers don't need to unpack RobotConfig).
ROBOT_DISPLAY_NAMES: dict[str, str] = {k: r.display_name for k, r in ROBOTS.items()}
DEFAULT_OUTPUT_DIRS: dict[str, str] = {k: r.output_directory for k, r in ROBOTS.items()}
DEFAULT_MOUNT_DIRECTION: dict[str, str] = {k: r.default_mount_direction for k, r in ROBOTS.items()}


# ── Data I/O ──────────────────────────────────────────────────────────────

def load_workspace_data(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the three ``.npy`` arrays from *directory*."""
    positions = np.load(directory / "ee_positions.npy")
    yoshikawa = np.load(directory / "yoshikawa.npy")
    condition = np.load(directory / "condition_number.npy")
    return positions, yoshikawa, condition


def load_statistics(directory: Path) -> dict | None:
    """Load ``statistics.json`` from *directory*, or ``None``."""
    path = directory / "statistics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Statistics computation ────────────────────────────────────────────────

def compute_desired_workspace_coverage(
    positions: np.ndarray,
    ws_min: np.ndarray = DESIRED_WS_MIN,
    ws_max: np.ndarray = DESIRED_WS_MAX,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> tuple[float, int, int]:
    """Fraction of the desired workspace reached by at least one sample."""
    x_edges = np.arange(ws_min[0], ws_max[0] + voxel_size, voxel_size)
    y_edges = np.arange(ws_min[1], ws_max[1] + voxel_size, voxel_size)
    z_edges = np.arange(ws_min[2], ws_max[2] + voxel_size, voxel_size)
    total_voxels = (len(x_edges) - 1) * (len(y_edges) - 1) * (len(z_edges) - 1)

    inside = np.all((positions >= ws_min) & (positions <= ws_max), axis=1)
    inside_positions = positions[inside]
    if len(inside_positions) == 0:
        return 0.0, 0, total_voxels

    ix = np.clip(np.digitize(inside_positions[:, 0], x_edges) - 1, 0, len(x_edges) - 2)
    iy = np.clip(np.digitize(inside_positions[:, 1], y_edges) - 1, 0, len(y_edges) - 2)
    iz = np.clip(np.digitize(inside_positions[:, 2], z_edges) - 1, 0, len(z_edges) - 2)
    occupied = len(set(zip(ix, iy, iz)))
    return occupied / total_voxels, occupied, total_voxels


def metric_summary(voxel_means: np.ndarray) -> str:
    """One-line summary of *per-voxel* metric values (NaN voxels excluded)."""
    valid = voxel_means[np.isfinite(voxel_means)]
    if len(valid) == 0:
        return "No valid voxels"
    return (
        f"mean={valid.mean():.4f}  median={np.median(valid):.4f}  "
        f"std={valid.std():.4f}  "
        f"min={valid.min():.4f}  max={valid.max():.4f}"
    )


def _voxel_means_inside_ws(
    positions: np.ndarray,
    values: np.ndarray,
    ws_min: np.ndarray,
    ws_max: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Per-voxel means of *values* for samples inside the desired workspace."""
    inside = np.all((positions >= ws_min) & (positions <= ws_max), axis=1)
    _, means, _ = voxelize(positions[inside], values[inside], voxel_size)
    return means


def display_statistics_table(
    positions: np.ndarray,
    yoshikawa: np.ndarray,
    condition: np.ndarray,
    robot_display: str,
    ws_min: np.ndarray = DESIRED_WS_MIN,
    ws_max: np.ndarray = DESIRED_WS_MAX,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> None:
    """Render an HTML statistics table in the notebook.

    Manipulability statistics are computed on *per-voxel means* inside the
    desired workspace so that unfilled voxels do not skew the numbers.
    """
    inside_mask = np.all((positions >= ws_min) & (positions <= ws_max), axis=1)
    coverage_frac, voxels_reached, voxels_total = compute_desired_workspace_coverage(
        positions, ws_min, ws_max, voxel_size,
    )
    yosh_voxel = _voxel_means_inside_ws(positions, yoshikawa, ws_min, ws_max, voxel_size)
    cond_voxel = _voxel_means_inside_ws(positions, condition, ws_min, ws_max, voxel_size)
    display(HTML(f"""
    <h3>{robot_display} — Workspace Coverage</h3>
    <table style="text-align:left">
    <tr><td><b>Total samples</b></td><td>{len(positions):,}</td></tr>
    <tr><td><b>Inside desired WS</b></td>
        <td>{inside_mask.sum():,} ({inside_mask.mean()*100:.1f}%)</td></tr>
    <tr><td><b>Voxels reached</b></td>
        <td>{voxels_reached:,} / {voxels_total:,}</td></tr>
    <tr><td><b>Volume coverage</b></td>
        <td>{coverage_frac*100:.1f}%</td></tr>
    </table>
    <h3>Manipulability (per-voxel means, inside desired WS)</h3>
    <table style="text-align:left">
    <tr><td><b>Yoshikawa</b></td>
        <td>{metric_summary(yosh_voxel)}</td></tr>
    <tr><td><b>Inv. Condition</b></td>
        <td>{metric_summary(cond_voxel)}</td></tr>
    </table>
    """))


# ── Voxelisation ──────────────────────────────────────────────────────────

def voxelize(
    positions: np.ndarray,
    values: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3-D voxel binning returning ``(centres, means, counts)``."""
    grid_indices = np.floor(positions / voxel_size).astype(np.int32)
    unique, inverse, counts = np.unique(
        grid_indices, axis=0, return_inverse=True, return_counts=True,
    )
    valid = np.isfinite(values)
    sums = np.zeros(len(unique), dtype=np.float64)
    valid_counts = np.zeros(len(unique), dtype=np.int64)
    np.add.at(sums, inverse[valid], values[valid].astype(np.float64))
    np.add.at(valid_counts, inverse[valid], 1)
    means = np.where(valid_counts > 0, sums / valid_counts, np.nan).astype(np.float32)
    centres = (unique.astype(np.float64) + 0.5) * voxel_size
    return centres.astype(np.float32), means, counts


# ── 2-D heatmap helpers ──────────────────────────────────────────────────

def compute_2d_heatmap(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned-mean 2-D heatmap returning ``(a_edges, b_edges, grid_mean)``."""
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
    """Log-density 2-D histogram returning ``(a_edges, b_edges, grid)``."""
    a_edges = np.arange(coord_a.min(), coord_a.max() + bin_size, bin_size)
    b_edges = np.arange(coord_b.min(), coord_b.max() + bin_size, bin_size)
    grid, _, _ = np.histogram2d(coord_a, coord_b, bins=[a_edges, b_edges])
    log_grid = np.log1p(grid).astype(np.float32)
    log_grid[grid == 0] = np.nan
    return a_edges, b_edges, log_grid


# ── Interactive 3-D plotting (Plotly) ─────────────────────────────────────

def create_ws_box_mesh(
    ws_min: np.ndarray = DESIRED_WS_MIN,
    ws_max: np.ndarray = DESIRED_WS_MAX,
) -> go.Mesh3d:
    """Translucent box representing the desired workspace."""
    x0, y0, z0 = ws_min
    x1, y1, z1 = ws_max
    vertices_x = [x0, x1, x1, x0, x0, x1, x1, x0]
    vertices_y = [y0, y0, y1, y1, y0, y0, y1, y1]
    vertices_z = [z0, z0, z0, z0, z1, z1, z1, z1]
    tri_i = [0, 0, 0, 0, 4, 4, 0, 0, 1, 1, 0, 0]
    tri_j = [1, 2, 4, 5, 5, 6, 1, 5, 2, 6, 3, 7]
    tri_k = [2, 3, 5, 6, 6, 7, 5, 4, 6, 5, 7, 4]
    return go.Mesh3d(
        x=vertices_x, y=vertices_y, z=vertices_z,
        i=tri_i, j=tri_j, k=tri_k,
        color="lightblue", opacity=0.12,
        name="Desired WS", showlegend=True,
    )


def interactive_3d_scatter(
    positions: np.ndarray,
    values: np.ndarray,
    voxel_size: float,
    title: str,
    colorbar_label: str,
    colorscale: str = "Plasma",
    ws_min: np.ndarray = DESIRED_WS_MIN,
    ws_max: np.ndarray = DESIRED_WS_MAX,
) -> go.FigureWidget:
    """Create an interactive Plotly 3-D voxelised scatter.

    Returns a ``FigureWidget`` so that VS Code renders it interactively.
    """
    centres, means, _ = voxelize(positions, values, voxel_size)
    finite = np.isfinite(means)
    centres, means = centres[finite], means[finite]

    fig = go.FigureWidget()
    fig.add_trace(create_ws_box_mesh(ws_min, ws_max))
    fig.add_trace(go.Scatter3d(
        x=centres[:, 0], y=centres[:, 1], z=centres[:, 2],
        mode="markers",
        marker=dict(
            size=1.8, color=means,
            colorscale=colorscale, opacity=0.6,
            colorbar=dict(title=colorbar_label),
        ),
        hovertemplate=(
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            "<br>value=%{marker.color:.4f}<extra></extra>"
        ),
        name="Voxels",
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=900, height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def interactive_3d_density(
    positions: np.ndarray,
    voxel_size: float,
    title: str,
    ws_min: np.ndarray = DESIRED_WS_MIN,
    ws_max: np.ndarray = DESIRED_WS_MAX,
) -> go.FigureWidget:
    """Create an interactive Plotly 3-D density scatter."""
    centres, _, counts = voxelize(positions, np.ones(len(positions)), voxel_size)
    log_counts = np.log1p(counts.astype(np.float32))

    fig = go.FigureWidget()
    fig.add_trace(create_ws_box_mesh(ws_min, ws_max))
    fig.add_trace(go.Scatter3d(
        x=centres[:, 0], y=centres[:, 1], z=centres[:, 2],
        mode="markers",
        marker=dict(
            size=1.8, color=log_counts,
            colorscale="Viridis", opacity=0.6,
            colorbar=dict(title="log(1+count)"),
        ),
        hovertemplate=(
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            "<br>count=%{marker.color:.1f}<extra></extra>"
        ),
        name="Density",
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=900, height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── Matplotlib 2-D cross-section figures ─────────────────────────────────

def draw_cross_section_figure(
    positions: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    voxel_size: float = 0.02,
    slice_thickness: float = 0.05,
    cmap: str = "plasma",
    ws_min: np.ndarray = DESIRED_WS_MIN,
    ws_max: np.ndarray = DESIRED_WS_MAX,
    density_mode: bool = False,
    higher_is_better: bool = True,
) -> plt.Figure:
    """Three 2-D cross-section heatmaps with a single shared colorbar.

    A small arrow next to the colorbar indicates which direction of the
    scale corresponds to *better* values.
    """
    ws_mid = (ws_min + ws_max) / 2
    half = slice_thickness / 2

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5),
        gridspec_kw={"wspace": 0.30},
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    last_im = None
    for ax, view in zip(axes, VIEW_CONFIGS):
        sa = view["slice_axis"]
        centre = float(ws_mid[sa])
        mask = (positions[:, sa] >= centre - half) & (positions[:, sa] <= centre + half)
        sliced_pos = positions[mask]
        ai, bi = view["a"], view["b"]

        if density_mode:
            a_edges, b_edges, grid = compute_2d_density(
                sliced_pos[:, ai], sliced_pos[:, bi], voxel_size,
            )
        else:
            a_edges, b_edges, grid = compute_2d_heatmap(
                sliced_pos[:, ai], sliced_pos[:, bi], values[mask], voxel_size,
            )

        last_im = ax.pcolormesh(a_edges, b_edges, grid.T, cmap=cmap, shading="flat")
        rect = plt.Rectangle(
            (ws_min[ai], ws_min[bi]),
            ws_max[ai] - ws_min[ai], ws_max[bi] - ws_min[bi],
            linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
        ax.set_xlabel(AXIS_LABELS[ai])
        ax.set_ylabel(AXIS_LABELS[bi])
        ax.set_title(view["name"])
        ax.set_aspect("equal")

    # Single shared colorbar on the right edge
    cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.85, pad=0.03, label=colorbar_label)

    # Arrow indicating which direction is "better"
    arrow_label = "better \u2191" if higher_is_better else "better \u2193"
    cbar_ax = cbar.ax
    cbar_ax.annotate(
        arrow_label,
        xy=(0.5, 1.02 if higher_is_better else -0.02),
        xycoords="axes fraction",
        ha="center", va="bottom" if higher_is_better else "top",
        fontsize=9, fontweight="bold",
    )

    fig.tight_layout()
    return fig


# ── Multi-robot comparison table ──────────────────────────────────────────

def display_comparison_table(package_root: Path) -> None:
    """Load all available ``statistics.json`` and show a comparison table."""
    rows: list[str] = []
    for key, display_name in ROBOT_DISPLAY_NAMES.items():
        stats_file = package_root / DEFAULT_OUTPUT_DIRS[key] / "statistics.json"
        if not stats_file.exists():
            continue
        with open(stats_file) as f:
            s = json.load(f)
        cov = s.get("desired_ws_coverage_fraction", 0) * 100
        vr = s.get("desired_ws_voxels_reached", "?")
        vt = s.get("desired_ws_voxels_total", "?")
        yw = s.get("yoshikawa_inside_ws", {}).get("mean", float("nan"))
        cn = s.get("condition_inside_ws", {}).get("mean", float("nan"))
        ns = s.get("total_samples", "?")
        mh = s.get("mount_height_m", "?")
        rows.append(
            f"<tr><td>{display_name}</td><td>{ns:,}</td><td>{mh}</td>"
            f"<td>{cov:.1f}%</td><td>{vr:,}/{vt:,}</td>"
            f"<td>{yw:.4f}</td><td>{cn:.4f}</td></tr>"
        )

    if rows:
        header = (
            "<tr><th>Robot</th><th>Samples</th><th>Height (m)</th>"
            "<th>Coverage</th><th>Voxels</th>"
            "<th>Yosh. (WS)</th><th>Cond. (WS)</th></tr>"
        )
        display(HTML(
            "<h3>Robot Comparison</h3>"
            f'<table border="1" cellpadding="4" style="border-collapse:collapse">'
            f"{header}{chr(10).join(rows)}</table>"
        ))
    else:
        print("No statistics.json files found in default output directories.")

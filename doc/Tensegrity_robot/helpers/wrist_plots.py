"""Cable-driven wrist visualization — static and animated 3D plots.

Public API
----------
wrist_static_figure   — neutral + ghost poses, base ring, cables, axis arrows
wrist_animated_figure — circular (Ψ, Θ) sweep with tension labels
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .colors import (
    FAPS_GREEN,
    FAPS_BLUE,
    FAPS_GRAY,
    FAPS_LIGHT,
    ACTIVE_COL,
    PASSIVE_COL,
    apply_3d_layout,
    animation_buttons,
    animation_slider,
)
from .wrist import (
    platform_anchors_world,
    cable_geometry,
    structure_matrix,
    tension_distribution,
    ring_points,
    compute_anchors,
)


# ═══════════════════════════════════════════════════════════════════
# 1.  Static 3D Overview
# ═══════════════════════════════════════════════════════════════════

def _build_wrist_traces(
    psi: float,
    theta: float,
    R: float,
    r: float,
    h: float,
    h_b: float,
    base_passive: np.ndarray,
    base_active: np.ndarray,
    passive_convergence: np.ndarray,
    platform_active_body: np.ndarray,
    *,
    opacity: float = 1.0,
    show_legend: bool = True,
    line_scale: float = 1.0,
) -> list[go.Scatter3d]:
    """Build all Plotly traces for one wrist configuration."""
    traces = []
    lw = lambda base: base * line_scale

    p_active = platform_anchors_world(psi, theta, platform_active_body)

    # Base ring
    bx, by, bz = ring_points(R, -h_b)
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines", line=dict(color=FAPS_GRAY, width=lw(5)),
        opacity=opacity, showlegend=show_legend,
        name=f"Base ring (R={R:.0f} mm, z=−{h_b})",
        legendgroup="base"))

    # Platform ring
    body_ring = np.column_stack(ring_points(r, -h))
    world_ring = platform_anchors_world(psi, theta, body_ring)
    traces.append(go.Scatter3d(
        x=world_ring[:, 0], y=world_ring[:, 1], z=world_ring[:, 2],
        mode="lines", line=dict(color=FAPS_GREEN, width=lw(5)),
        opacity=opacity, showlegend=show_legend,
        name=f"Platform (r={r:.0f} mm)", legendgroup="platform"))

    # Passive cables
    for i in range(3):
        traces.append(go.Scatter3d(
            x=[base_passive[i, 0], passive_convergence[0]],
            y=[base_passive[i, 1], passive_convergence[1]],
            z=[base_passive[i, 2], passive_convergence[2]],
            mode="lines",
            line=dict(color=PASSIVE_COL, width=lw(3), dash="dash"),
            opacity=opacity,
            showlegend=(show_legend and i == 0),
            name="Passive cables (→ origin)" if i == 0 else None,
            legendgroup="passive"))

    # Active cables
    for i in range(3):
        traces.append(go.Scatter3d(
            x=[base_active[i, 0], p_active[i, 0]],
            y=[base_active[i, 1], p_active[i, 1]],
            z=[base_active[i, 2], p_active[i, 2]],
            mode="lines", line=dict(color=ACTIVE_COL, width=lw(3)),
            opacity=opacity,
            showlegend=(show_legend and i == 0),
            name="Active cables (→ platform)" if i == 0 else None,
            legendgroup="active"))

    # Anchor markers
    traces.append(go.Scatter3d(
        x=base_passive[:, 0], y=base_passive[:, 1], z=base_passive[:, 2],
        mode="markers+text",
        marker=dict(size=5, color=PASSIVE_COL),
        text=[f"P{i+1}" for i in range(3)],
        textposition="top center",
        textfont=dict(size=9, color=PASSIVE_COL),
        opacity=opacity, showlegend=False))

    traces.append(go.Scatter3d(
        x=base_active[:, 0], y=base_active[:, 1], z=base_active[:, 2],
        mode="markers+text",
        marker=dict(size=5, color=ACTIVE_COL),
        text=[f"A{i+1}" for i in range(3)],
        textposition="top center",
        textfont=dict(size=9, color=ACTIVE_COL),
        opacity=opacity, showlegend=False))

    traces.append(go.Scatter3d(
        x=p_active[:, 0], y=p_active[:, 1], z=p_active[:, 2],
        mode="markers+text",
        marker=dict(size=4, color=ACTIVE_COL, symbol="diamond"),
        text=[f"a{i+1}" for i in range(3)],
        textposition="bottom center",
        textfont=dict(size=8, color=ACTIVE_COL),
        opacity=opacity, showlegend=False))

    # Rotation center
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=7, color="black", symbol="cross"),
        text=["O (rotation center)"], textposition="middle right",
        textfont=dict(size=10, color="black"),
        opacity=opacity, showlegend=show_legend,
        name="Rotation center (origin)", legendgroup="center"))

    return traces


def wrist_static_figure(
    R: float,
    r: float,
    h: float,
    h_b: float,
    *,
    show_caption: bool = True,
) -> go.Figure:
    """Static overview with neutral pose + 4 ghost poses at ±30°."""
    anchors = compute_anchors(R, r, h, h_b)

    ghost_configs = [
        (np.radians(30),  0.0,            "+30° about X"),
        (np.radians(-30), 0.0,            "−30° about X"),
        (0.0,             np.radians(30),  "+30° about Y"),
        (0.0,             np.radians(-30), "−30° about Y"),
    ]

    fig = go.Figure()

    common = dict(
        R=R, r=r, h=h, h_b=h_b,
        base_passive=anchors["base_passive"],
        base_active=anchors["base_active"],
        passive_convergence=anchors["passive_convergence"],
        platform_active_body=anchors["platform_active_body"],
    )

    for psi, theta, _ in ghost_configs:
        for tr in _build_wrist_traces(
                psi, theta, **common,
                opacity=0.2, show_legend=False, line_scale=0.6):
            fig.add_trace(tr)

    for tr in _build_wrist_traces(
            0.0, 0.0, **common,
            opacity=1.0, show_legend=True):
        fig.add_trace(tr)

    # Axis arrows
    arrow_len = R * 0.8
    for axis_vec, color, name in [
        ([arrow_len, 0, 0], "#E74C3C", "X axis (Ψ)"),
        ([0, arrow_len, 0], "#27AE60", "Y axis (Θ)"),
        ([0, 0, arrow_len * 0.5], "#3498DB", "Z axis"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis_vec[0]], y=[0, axis_vec[1]], z=[0, axis_vec[2]],
            mode="lines+text",
            line=dict(color=color, width=3),
            text=["", name], textposition="top center",
            textfont=dict(size=9, color=color),
            showlegend=False))

    # Base disk fill
    bx, by, bz = ring_points(R, -h_b, n=40)
    fig.add_trace(go.Mesh3d(
        x=np.append(bx, [0]),
        y=np.append(by, [0]),
        z=np.append(bz, [-h_b]),
        color=FAPS_LIGHT, opacity=0.15, showlegend=False))

    # Platform cone: semi-translucent cone from origin to platform circle
    n_cone = 40
    cone_angles = np.linspace(0, 2 * np.pi, n_cone, endpoint=False)
    cone_base_x = r * np.cos(cone_angles)
    cone_base_y = r * np.sin(cone_angles)
    cone_base_z = np.full(n_cone, -h)
    cone_x = np.append(cone_base_x, 0.0)
    cone_y = np.append(cone_base_y, 0.0)
    cone_z = np.append(cone_base_z, 0.0)
    apex_idx = n_cone
    i_tri, j_tri, k_tri = [], [], []
    for ci in range(n_cone):
        cj = (ci + 1) % n_cone
        i_tri.append(apex_idx)
        j_tri.append(ci)
        k_tri.append(cj)
    fig.add_trace(go.Mesh3d(
        x=cone_x, y=cone_y, z=cone_z,
        i=i_tri, j=j_tri, k=k_tri,
        color=FAPS_GREEN, opacity=0.12,
        name="Platform cone", showlegend=False))

    title = (
        "Cable-Driven Wrist — Static Overview<br>"
        f"<sup>Nemoto (2022): R={R:.0f}, r={r:.0f}, h={h:.0f}, "
        f"h_b={h_b:.0f} mm  |  Ghost poses at ±30° about X and Y</sup>"
    )
    apply_3d_layout(
        fig, title=title, show_caption=show_caption,
        x_range=[-100, 100], y_range=[-100, 100], z_range=[-100, 20])

    return fig


# ═══════════════════════════════════════════════════════════════════
# 2.  Animated 3D Visualization
# ═══════════════════════════════════════════════════════════════════

def wrist_animated_figure(
    R: float,
    r: float,
    h: float,
    h_b: float,
    *,
    tilt_max_deg: float = 65.0,
    t_ref_val: float = 1.0,
    n_frames: int = 120,
    sweep_amplitude_deg: float = 40.0,
    show_caption: bool = True,
) -> go.Figure:
    """Animated circular sweep through the (Ψ, Θ) workspace."""
    anchors = compute_anchors(R, r, h, h_b)
    base_passive = anchors["base_passive"]
    base_active = anchors["base_active"]
    passive_convergence = anchors["passive_convergence"]
    platform_active_body = anchors["platform_active_body"]

    sweep_rad = np.radians(sweep_amplitude_deg)
    t_param = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    psi_seq = sweep_rad * np.cos(t_param)
    theta_seq = sweep_rad * np.sin(t_param)

    def _frame_data(psi: float, theta: float) -> dict:
        p_active = platform_anchors_world(psi, theta, platform_active_body)
        _, L_active, _ = cable_geometry(base_active, p_active)
        A = structure_matrix(psi, theta, base_active, platform_active_body)
        t_ref = np.full(3, t_ref_val)
        t_star = tension_distribution(A, np.zeros(2), t_ref)
        body_ring = np.column_stack(ring_points(r, -h, n=40))
        world_ring = platform_anchors_world(psi, theta, body_ring)
        return dict(p_active=p_active, L_active=L_active,
                    tensions=t_star, world_ring=world_ring)

    bx, by, bz = ring_points(R, -h_b, n=60)
    ini = _frame_data(psi_seq[0], theta_seq[0])

    # ── Static traces (0–1) + passive cables + base markers ─────
    traces: list[go.BaseTraceType] = []

    # 0: base ring
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines", line=dict(color=FAPS_GRAY, width=5),
        name=f"Base (R={R:.0f})", legendgroup="base"))

    # 1: passive cables (static — base → origin)
    px, py, pz = [], [], []
    for i in range(3):
        px.extend([base_passive[i, 0], passive_convergence[0], None])
        py.extend([base_passive[i, 1], passive_convergence[1], None])
        pz.extend([base_passive[i, 2], passive_convergence[2], None])
    traces.append(go.Scatter3d(
        x=px, y=py, z=pz,
        mode="lines", line=dict(color=PASSIVE_COL, width=3, dash="dash"),
        name="Passive (→ origin)", legendgroup="passive"))

    # 2: platform ring (animated)
    traces.append(go.Scatter3d(
        x=ini["world_ring"][:, 0],
        y=ini["world_ring"][:, 1],
        z=ini["world_ring"][:, 2],
        mode="lines", line=dict(color=FAPS_GREEN, width=5),
        name=f"Platform (r={r:.0f})", legendgroup="platform"))

    # 3–5: active cables (animated)
    for i in range(3):
        t_val = ini["tensions"][i]
        traces.append(go.Scatter3d(
            x=[base_active[i, 0], ini["p_active"][i, 0]],
            y=[base_active[i, 1], ini["p_active"][i, 1]],
            z=[base_active[i, 2], ini["p_active"][i, 2]],
            mode="lines+text",
            line=dict(color=ACTIVE_COL, width=4),
            text=["", f"T{i+2}={t_val:.2f}"],
            textposition="middle center",
            textfont=dict(size=8, color=ACTIVE_COL),
            showlegend=(i == 0),
            name="Active" if i == 0 else None,
            legendgroup="active"))

    # 6: base passive markers (static)
    traces.append(go.Scatter3d(
        x=base_passive[:, 0], y=base_passive[:, 1], z=base_passive[:, 2],
        mode="markers+text",
        marker=dict(size=4, color=PASSIVE_COL),
        text=[f"P{i+1}" for i in range(3)],
        textposition="top center",
        textfont=dict(size=8, color=PASSIVE_COL),
        showlegend=False))

    # 7: base active markers (static)
    traces.append(go.Scatter3d(
        x=base_active[:, 0], y=base_active[:, 1], z=base_active[:, 2],
        mode="markers+text",
        marker=dict(size=4, color=ACTIVE_COL),
        text=[f"A{i+1}" for i in range(3)],
        textposition="top center",
        textfont=dict(size=8, color=ACTIVE_COL),
        showlegend=False))

    # 8: platform active markers (animated)
    traces.append(go.Scatter3d(
        x=ini["p_active"][:, 0],
        y=ini["p_active"][:, 1],
        z=ini["p_active"][:, 2],
        mode="markers",
        marker=dict(size=3, color=ACTIVE_COL, symbol="diamond"),
        showlegend=False))

    # 9: rotation center (static)
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=5, color="black", symbol="cross"),
        name="Rotation center", showlegend=True))

    # 10: platform cone (animated) — tip at origin, base at platform ring
    n_cone = 40
    cone_angles = np.linspace(0, 2 * np.pi, n_cone, endpoint=False)
    cone_body = np.column_stack([
        r * np.cos(cone_angles),
        r * np.sin(cone_angles),
        np.full(n_cone, -h),
    ])
    cone_world_ini = platform_anchors_world(psi_seq[0], theta_seq[0], cone_body)
    cone_x = np.append(cone_world_ini[:, 0], 0.0)
    cone_y = np.append(cone_world_ini[:, 1], 0.0)
    cone_z = np.append(cone_world_ini[:, 2], 0.0)
    apex_idx = n_cone
    i_tri, j_tri, k_tri = [], [], []
    for ci in range(n_cone):
        cj = (ci + 1) % n_cone
        i_tri.append(apex_idx)
        j_tri.append(ci)
        k_tri.append(cj)
    traces.append(go.Mesh3d(
        x=cone_x, y=cone_y, z=cone_z,
        i=i_tri, j=j_tri, k=k_tri,
        color=FAPS_GREEN, opacity=0.12,
        showlegend=False))

    # Dynamic indices: platform ring (2), active cables (3,4,5), markers (8), cone (10)
    dyn_indices = [2, 3, 4, 5, 8, 10]

    # ── Frames ──────────────────────────────────────────────────
    frames = []
    for fi, (psi, theta) in enumerate(zip(psi_seq, theta_seq)):
        fd = _frame_data(psi, theta)

        frame_traces = [
            go.Scatter3d(
                x=fd["world_ring"][:, 0],
                y=fd["world_ring"][:, 1],
                z=fd["world_ring"][:, 2]),
        ]
        for i in range(3):
            t_val = fd["tensions"][i]
            lw = max(2, min(8, 4 * t_val / t_ref_val))
            frame_traces.append(go.Scatter3d(
                x=[base_active[i, 0], fd["p_active"][i, 0]],
                y=[base_active[i, 1], fd["p_active"][i, 1]],
                z=[base_active[i, 2], fd["p_active"][i, 2]],
                line=dict(color=ACTIVE_COL, width=lw),
                text=["", f"T{i+2}={t_val:.2f}"],
                textfont=dict(size=8, color=ACTIVE_COL)))
        frame_traces.append(go.Scatter3d(
            x=fd["p_active"][:, 0],
            y=fd["p_active"][:, 1],
            z=fd["p_active"][:, 2]))

        # Animated cone
        cone_world = platform_anchors_world(psi, theta, cone_body)
        frame_traces.append(go.Mesh3d(
            x=np.append(cone_world[:, 0], 0.0),
            y=np.append(cone_world[:, 1], 0.0),
            z=np.append(cone_world[:, 2], 0.0)))

        psi_d = np.degrees(psi)
        theta_d = np.degrees(theta)
        t = fd["tensions"]

        title_text = ""
        if show_caption:
            title_text = (
                f"Cable-Driven Wrist — Circular Sweep "
                f"(±{sweep_amplitude_deg:.0f}°)<br>"
                f"<sup>Ψ={psi_d:+.1f}°, Θ={theta_d:+.1f}°  |  "
                f"T₂={t[0]:.2f}, T₃={t[1]:.2f}, T₄={t[2]:.2f} N</sup>"
            )

        frames.append(go.Frame(
            data=frame_traces, traces=dyn_indices, name=str(fi),
            layout=go.Layout(title=dict(text=title_text))))

    # ── Slider ──────────────────────────────────────────────────
    slider_labels = [f"{np.degrees(p):+.0f}°" for p in psi_seq]

    base_title = (
        f"Cable-Driven Wrist — Circular Sweep "
        f"(±{sweep_amplitude_deg:.0f}°)<br>"
        "<sup>Nemoto (2022)  |  Press ▶ to animate</sup>"
    )

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=dict(text=base_title if show_caption else "",
                   font=dict(size=14)),
        scene=dict(
            xaxis=dict(title="X [mm]", range=[-100, 100]),
            yaxis=dict(title="Y [mm]", range=[-100, 100]),
            zaxis=dict(title="Z [mm]", range=[-100, 30]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.8))),
        legend=dict(x=1.02, y=0.5, font=dict(size=9)),
        margin=dict(l=0, r=200, t=80 if show_caption else 20, b=80),
        width=950, height=750,
        updatemenus=animation_buttons(play_duration=60),
        sliders=animation_slider(
            slider_labels, prefix="Ψ ≈ ", suffix="°"))

    return fig

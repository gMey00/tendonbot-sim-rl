"""Elbow joint visualization — antiparallelogram and disc approximation plots.

Public API
----------
antiparallelogram_static_figure   — 5-pose static overview
antiparallelogram_animated_figure — animated sweep 0 → +max → −max → 0
disc_static_figure                — 5-pose disc approximation overview
disc_animated_figure              — animated disc approximation
comparison_animated_figure        — both mechanisms overlaid, animated
comparison_static_figure          — trajectory + deviation subplot
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .colors import (
    FAPS_GREEN,
    FAPS_BLUE,
    FAPS_GRAY,
    FAPS_LIGHT,
    ACTIVE_COL,
    ICR_COLOR,
    TENDON_T0_COL,
    TENDON_T1_COL,
    apply_2d_layout,
    base_hatching_shapes,
    animation_buttons,
    animation_slider,
)
from .linkage import (
    ARROW_LEN,
    linkage_positions,
    instantaneous_center,
    elbow_angle,
    centrode_params,
    centrode_ellipse_pts,
    moving_centrode_pts,
    tendon_force_dirs,
    delta_for_elbow,
    elbow_sweep_sequence,
    disc_platform_center,
    disc_platform_bar,
)


# ═══════════════════════════════════════════════════════════════════
# Internal helpers shared across multiple plot functions
# ═══════════════════════════════════════════════════════════════════

def _precompute_common(
    l_e: float,
    k_e: float,
    elbow_max_deg: float,
) -> dict:
    """Derived quantities used by several plot functions."""
    theta_0 = np.arcsin(k_e / l_e)
    a_ell, b_ell, _ = centrode_params(l_e, k_e)
    delta_pos = delta_for_elbow(elbow_max_deg, theta_0, k_e, l_e)
    delta_neg = delta_for_elbow(-elbow_max_deg, theta_0, k_e, l_e)
    delta_range = np.linspace(delta_neg, delta_pos, 600)

    icr_list = [instantaneous_center(d, k_e, l_e, theta_0) for d in delta_range]
    icr_valid = np.array([p for p in icr_list if p is not None])
    icr_eq = instantaneous_center(0.0, k_e, l_e, theta_0)

    fixed_cx, fixed_cy = centrode_ellipse_pts(a_ell, b_ell)
    _, _, D0, C0, _, M0, _ = linkage_positions(0.0, k_e, l_e, theta_0)
    mov_cx_0, mov_cy_0 = moving_centrode_pts(D0, C0, M0, a_ell, b_ell)

    return dict(
        theta_0=theta_0,
        a_ell=a_ell,
        b_ell=b_ell,
        delta_pos=delta_pos,
        delta_neg=delta_neg,
        icr_valid=icr_valid,
        icr_eq=icr_eq,
        fixed_cx=fixed_cx,
        fixed_cy=fixed_cy,
        mov_cx_0=mov_cx_0,
        mov_cy_0=mov_cy_0,
    )


def _ghost_elbows(elbow_max_deg: float) -> list[float]:
    """Five symmetric elbow angles for static overview poses."""
    return [
        -elbow_max_deg,
        -elbow_max_deg / 2,
        0.0,
        elbow_max_deg / 2,
        elbow_max_deg,
    ]


_GHOST_ALPHAS = [0.45, 0.20, 1.0, 0.20, 0.45]


# ═══════════════════════════════════════════════════════════════════
# 1.  Antiparallelogram — Static Overview
# ═══════════════════════════════════════════════════════════════════

def antiparallelogram_static_figure(
    l_e: float,
    k_e: float,
    elbow_max_deg: float,
    *,
    show_caption: bool = True,
) -> go.Figure:
    """Five-pose static overview with centrode ellipses and tendon arrows."""
    pre = _precompute_common(l_e, k_e, elbow_max_deg)
    theta_0 = pre["theta_0"]

    show_elbows = _ghost_elbows(elbow_max_deg)
    show_deltas = [delta_for_elbow(e, theta_0, k_e, l_e) for e in show_elbows]

    fig = go.Figure()

    # ── Centrode ellipses ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.append(pre["fixed_cx"], pre["fixed_cx"][0]),
        y=np.append(pre["fixed_cy"], pre["fixed_cy"][0]),
        mode="lines", line=dict(color=FAPS_BLUE, width=2, dash="dash"),
        name="Fixed centrode (foci A, B)", opacity=0.4))

    fig.add_trace(go.Scatter(
        x=np.append(pre["mov_cx_0"], pre["mov_cx_0"][0]),
        y=np.append(pre["mov_cy_0"], pre["mov_cy_0"][0]),
        mode="lines", line=dict(color=FAPS_GREEN, width=2, dash="dash"),
        name="Moving centrode (foci D, C)", opacity=0.4))

    fig.add_trace(go.Scatter(
        x=pre["icr_valid"][:, 0], y=pre["icr_valid"][:, 1],
        mode="markers", marker=dict(size=2, color=ICR_COLOR, opacity=0.25),
        name="ICR path"))

    # ── Ghost poses with tendon force arrows ────────────────────
    for i, (dd, ea_deg, alpha) in enumerate(
            zip(show_deltas, show_elbows, _GHOST_ALPHAS)):
        A, B, D, C, _, M, _ = linkage_positions(dd, k_e, l_e, theta_0)
        is_neutral = abs(ea_deg) < 1e-6
        lw_f = 6 if is_neutral else 2.5
        lw_r = 4 if is_neutral else 2
        lw_c = 5 if is_neutral else 2.5
        ms = 10 if is_neutral else 6
        first = i == 0

        fig.add_trace(go.Scatter(
            x=[A[0], B[0]], y=[A[1], B[1]],
            mode="lines", line=dict(color=FAPS_GRAY, width=lw_f),
            opacity=alpha, showlegend=False))
        fig.add_trace(go.Scatter(
            x=[A[0], D[0]], y=[A[1], D[1]],
            mode="lines", line=dict(color=FAPS_BLUE, width=lw_r),
            opacity=alpha, showlegend=first,
            name="Rod AD" if first else None))
        fig.add_trace(go.Scatter(
            x=[B[0], C[0]], y=[B[1], C[1]],
            mode="lines", line=dict(color=ACTIVE_COL, width=lw_r),
            opacity=alpha, showlegend=first,
            name="Rod BC" if first else None))
        fig.add_trace(go.Scatter(
            x=[D[0], C[0]], y=[D[1], C[1]],
            mode="lines", line=dict(color=FAPS_GREEN, width=lw_c),
            opacity=alpha, showlegend=first,
            name="Coupler DC (platform)" if first else None))
        fig.add_trace(go.Scatter(
            x=[A[0], B[0], D[0], C[0], M[0]],
            y=[A[1], B[1], D[1], C[1], M[1]],
            mode="markers",
            marker=dict(
                size=[ms] * 2 + [ms] * 2 + [ms - 2],
                color=[FAPS_GRAY] * 2 + [FAPS_GREEN] * 2 + ["#2d5a1e"],
                line=dict(width=1, color="white")),
            opacity=alpha, showlegend=False))

        t0b, t0t, t1b, t1t = tendon_force_dirs(A, B, D, C, ARROW_LEN)
        fig.add_trace(go.Scatter(
            x=[t0b[0], t0t[0]], y=[t0b[1], t0t[1]],
            mode="lines+markers",
            line=dict(color=TENDON_T0_COL, width=2.5),
            marker=dict(size=[0, 9], symbol=["circle", "arrow-up"],
                        color=TENDON_T0_COL, angleref="previous"),
            opacity=alpha, showlegend=first,
            name="T₀ (D→B)" if first else None))
        fig.add_trace(go.Scatter(
            x=[t1b[0], t1t[0]], y=[t1b[1], t1t[1]],
            mode="lines+markers",
            line=dict(color=TENDON_T1_COL, width=2.5),
            marker=dict(size=[0, 9], symbol=["circle", "arrow-up"],
                        color=TENDON_T1_COL, angleref="previous"),
            opacity=alpha, showlegend=first,
            name="T₁ (C→A)" if first else None))

        if not is_neutral:
            fig.add_annotation(
                x=M[0], y=M[1] - 10,
                text=f"{ea_deg:+.0f}°",
                showarrow=False, font=dict(size=9, color=FAPS_GREEN),
                opacity=alpha)

        if is_neutral:
            for pt, name in [(A, "A"), (B, "B"), (D, "D"), (C, "C"), (M, "M")]:
                ox = -12 if pt[0] < 0 else 12
                oy = 10 if pt[1] >= 0 else -12
                fig.add_annotation(
                    x=pt[0], y=pt[1], text=f"<b>{name}</b>",
                    showarrow=False, xshift=ox, yshift=oy,
                    font=dict(size=13))
            fig.add_annotation(
                x=0, y=18, text="Static frame (base)",
                showarrow=False, font=dict(size=10, color=FAPS_GRAY))

    # ── ICR markers ─────────────────────────────────────────────
    icr_eq = pre["icr_eq"]
    fig.add_trace(go.Scatter(
        x=[icr_eq[0]], y=[icr_eq[1]],
        mode="markers+text",
        marker=dict(size=16, color=ICR_COLOR, symbol="star",
                    line=dict(width=1, color="white")),
        text=[f"  ICR at eq. (0, {icr_eq[1]:.1f})"],
        textposition="middle right",
        textfont=dict(size=10, color=ICR_COLOR),
        name="ICR at equilibrium"))

    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=8, color=FAPS_BLUE, symbol="x", line=dict(width=2)),
        text=["  Centrode center"], textposition="middle right",
        textfont=dict(size=9, color=FAPS_BLUE),
        name="Centrode center (0, 0)"))

    for shape in base_hatching_shapes(k_e / 2):
        fig.add_shape(**shape)

    title = (
        f"Antiparallelogram Elbow — l<sub>e</sub>={l_e:.0f} mm, "
        f"k<sub>e</sub>={k_e:.0f} mm, "
        f"elbow=±{elbow_max_deg:.0f}°"
        f"<br><sup>Klein (2023), §3.2.1, Fig. 3.5</sup>"
    )
    apply_2d_layout(fig, title=title, show_caption=show_caption)
    return fig


# ═══════════════════════════════════════════════════════════════════
# 2.  Antiparallelogram — Animation
# ═══════════════════════════════════════════════════════════════════

def antiparallelogram_animated_figure(
    l_e: float,
    k_e: float,
    elbow_max_deg: float,
    *,
    n_frames: int = 120,
    show_caption: bool = True,
) -> go.Figure:
    """Animated antiparallelogram: sweep 0 → +max → −max → 0."""
    pre = _precompute_common(l_e, k_e, elbow_max_deg)
    theta_0 = pre["theta_0"]
    a_ell, b_ell = pre["a_ell"], pre["b_ell"]
    icr_eq = pre["icr_eq"]

    elbow_seq = elbow_sweep_sequence(elbow_max_deg, n_frames)
    delta_seq = np.array([delta_for_elbow(e, theta_0, k_e, l_e) for e in elbow_seq])

    def _frame_data(delta: float) -> dict:
        A, B, D, C, phi, M, theta = linkage_positions(delta, k_e, l_e, theta_0)
        icr = instantaneous_center(delta, k_e, l_e, theta_0)
        mx, my = moving_centrode_pts(D, C, M, a_ell, b_ell, n=150)
        t0b, t0t, t1b, t1t = tendon_force_dirs(A, B, D, C, ARROW_LEN)
        ea = np.degrees(elbow_angle(delta, k_e, l_e, theta_0))
        return dict(A=A, B=B, D=D, C=C, M=M, phi=phi, theta=theta,
                    icr=icr, mx=mx, my=my,
                    t0b=t0b, t0t=t0t, t1b=t1b, t1t=t1t, elbow=ea)

    ini = _frame_data(delta_seq[0])

    # ── Static traces (0–4) ─────────────────────────────────────
    traces: list[go.BaseTraceType] = []

    traces.append(go.Scatter(
        x=np.append(pre["fixed_cx"], pre["fixed_cx"][0]),
        y=np.append(pre["fixed_cy"], pre["fixed_cy"][0]),
        mode="lines", line=dict(color=FAPS_BLUE, width=1.5, dash="dash"),
        opacity=0.35, name="Fixed centrode"))
    traces.append(go.Scatter(
        x=pre["icr_valid"][:, 0], y=pre["icr_valid"][:, 1],
        mode="markers", marker=dict(size=2, color=ICR_COLOR, opacity=0.15),
        name="ICR path"))
    traces.append(go.Scatter(
        x=[-k_e / 2 - 15, k_e / 2 + 15], y=[2, 2],
        mode="lines", line=dict(color=FAPS_GRAY, width=2),
        showlegend=False))
    traces.append(go.Scatter(
        x=[icr_eq[0]], y=[icr_eq[1]],
        mode="markers",
        marker=dict(size=12, color=ICR_COLOR, symbol="star", opacity=0.4,
                    line=dict(width=1, color="white")),
        name=f"ICR at eq. (0, {icr_eq[1]:.1f})"))
    traces.append(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=7, color=FAPS_BLUE, symbol="x", line=dict(width=2)),
        opacity=0.4, name="Centrode center"))

    n_static = len(traces)

    # ── Dynamic traces (5–15) ───────────────────────────────────
    traces.append(go.Scatter(
        x=[ini["A"][0], ini["B"][0]], y=[ini["A"][1], ini["B"][1]],
        mode="lines", line=dict(color=FAPS_GRAY, width=6),
        name=f"Frame AB ({k_e:.0f} mm)"))
    traces.append(go.Scatter(
        x=[ini["A"][0], ini["D"][0]], y=[ini["A"][1], ini["D"][1]],
        mode="lines", line=dict(color=FAPS_BLUE, width=3.5),
        name="Rod AD"))
    traces.append(go.Scatter(
        x=[ini["B"][0], ini["C"][0]], y=[ini["B"][1], ini["C"][1]],
        mode="lines", line=dict(color=ACTIVE_COL, width=3.5),
        name="Rod BC"))
    traces.append(go.Scatter(
        x=[ini["D"][0], ini["C"][0]], y=[ini["D"][1], ini["C"][1]],
        mode="lines", line=dict(color=FAPS_GREEN, width=5),
        name="Coupler DC (platform)"))
    traces.append(go.Scatter(
        x=[ini["A"][0], ini["B"][0]], y=[ini["A"][1], ini["B"][1]],
        mode="markers",
        marker=dict(size=10, color=FAPS_GRAY, line=dict(width=1, color="white")),
        showlegend=False))
    traces.append(go.Scatter(
        x=[ini["D"][0], ini["C"][0]], y=[ini["D"][1], ini["C"][1]],
        mode="markers",
        marker=dict(size=10, color=FAPS_GREEN, line=dict(width=1, color="white")),
        showlegend=False))
    traces.append(go.Scatter(
        x=[ini["M"][0]], y=[ini["M"][1]],
        mode="markers",
        marker=dict(size=7, color="#2d5a1e", symbol="square"),
        showlegend=False))
    ix = [ini["icr"][0]] if ini["icr"] is not None else []
    iy = [ini["icr"][1]] if ini["icr"] is not None else []
    traces.append(go.Scatter(
        x=ix, y=iy,
        mode="markers",
        marker=dict(size=9, color=ICR_COLOR, symbol="circle",
                    line=dict(width=1.5, color="white")),
        name="Current ICR"))
    traces.append(go.Scatter(
        x=np.append(ini["mx"], ini["mx"][0]),
        y=np.append(ini["my"], ini["my"][0]),
        mode="lines", line=dict(color=FAPS_GREEN, width=1.5, dash="dash"),
        opacity=0.5, name="Moving centrode"))
    traces.append(go.Scatter(
        x=[ini["t0b"][0], ini["t0t"][0]],
        y=[ini["t0b"][1], ini["t0t"][1]],
        mode="lines+markers+text",
        line=dict(color=TENDON_T0_COL, width=3),
        marker=dict(size=[0, 10], symbol=["circle", "arrow-up"],
                    color=TENDON_T0_COL, angleref="previous"),
        text=["", "T₀"], textposition="top center",
        textfont=dict(size=11, color=TENDON_T0_COL),
        name="T₀ (D→B)"))
    traces.append(go.Scatter(
        x=[ini["t1b"][0], ini["t1t"][0]],
        y=[ini["t1b"][1], ini["t1t"][1]],
        mode="lines+markers+text",
        line=dict(color=TENDON_T1_COL, width=3),
        marker=dict(size=[0, 10], symbol=["circle", "arrow-up"],
                    color=TENDON_T1_COL, angleref="previous"),
        text=["", "T₁"], textposition="top center",
        textfont=dict(size=11, color=TENDON_T1_COL),
        name="T₁ (C→A)"))

    dyn_indices = list(range(n_static, len(traces)))

    # ── Frames ──────────────────────────────────────────────────
    frames = []
    for fi, (delta, ea_deg) in enumerate(zip(delta_seq, elbow_seq)):
        fd = _frame_data(delta)
        icr_x = [fd["icr"][0]] if fd["icr"] is not None else []
        icr_y = [fd["icr"][1]] if fd["icr"] is not None else []

        frame_data = [
            go.Scatter(x=[fd["A"][0], fd["B"][0]], y=[fd["A"][1], fd["B"][1]]),
            go.Scatter(x=[fd["A"][0], fd["D"][0]], y=[fd["A"][1], fd["D"][1]]),
            go.Scatter(x=[fd["B"][0], fd["C"][0]], y=[fd["B"][1], fd["C"][1]]),
            go.Scatter(x=[fd["D"][0], fd["C"][0]], y=[fd["D"][1], fd["C"][1]]),
            go.Scatter(x=[fd["A"][0], fd["B"][0]], y=[fd["A"][1], fd["B"][1]]),
            go.Scatter(x=[fd["D"][0], fd["C"][0]], y=[fd["D"][1], fd["C"][1]]),
            go.Scatter(x=[fd["M"][0]], y=[fd["M"][1]]),
            go.Scatter(x=icr_x, y=icr_y),
            go.Scatter(x=np.append(fd["mx"], fd["mx"][0]),
                       y=np.append(fd["my"], fd["my"][0])),
            go.Scatter(x=[fd["t0b"][0], fd["t0t"][0]],
                       y=[fd["t0b"][1], fd["t0t"][1]],
                       marker=dict(size=[0, 10], symbol=["circle", "arrow-up"],
                                   color=TENDON_T0_COL, angleref="previous")),
            go.Scatter(x=[fd["t1b"][0], fd["t1t"][0]],
                       y=[fd["t1b"][1], fd["t1t"][1]],
                       marker=dict(size=[0, 10], symbol=["circle", "arrow-up"],
                                   color=TENDON_T1_COL, angleref="previous")),
        ]

        title_text = ""
        if show_caption:
            deg_d = np.degrees(delta)
            deg_t = np.degrees(fd["theta"])
            deg_p = np.degrees(fd["phi"])
            title_text = (
                f"Antiparallelogram Elbow — "
                f"l<sub>e</sub>={l_e:.0f}, k<sub>e</sub>={k_e:.0f} mm<br>"
                f"<sup>elbow={fd['elbow']:+.1f}°  |  "
                f"δ={deg_d:+.1f}°  |  "
                f"θ={deg_t:+.1f}°  |  "
                f"φ={deg_p:+.1f}°</sup>"
            )

        frames.append(go.Frame(
            data=frame_data,
            traces=dyn_indices,
            name=str(fi),
            layout=go.Layout(title=dict(text=title_text))))

    # ── Slider labels ───────────────────────────────────────────
    slider_labels = [f"{e:+.0f}°" for e in elbow_seq]

    # ── Assemble ────────────────────────────────────────────────
    fig = go.Figure(data=traces, frames=frames)

    base_title = (
        f"Antiparallelogram Elbow — "
        f"l<sub>e</sub>={l_e:.0f}, k<sub>e</sub>={k_e:.0f} mm<br>"
        f"<sup>Klein (2023), §3.2.1</sup>"
    )
    fig.update_layout(
        title=dict(
            text=base_title if show_caption else "",
            font=dict(size=14)),
        xaxis=dict(title="x [mm]", scaleanchor="y", scaleratio=1,
                   range=[-180, 180], gridcolor="#eee", zeroline=False),
        yaxis=dict(title="y [mm]",
                   range=[-190, 50], gridcolor="#eee", zeroline=False),
        plot_bgcolor="white",
        legend=dict(x=1.02, y=0.5, font=dict(size=9)),
        margin=dict(l=60, r=220, t=80 if show_caption else 20, b=80),
        width=950, height=750,
        updatemenus=animation_buttons(),
        sliders=animation_slider(slider_labels))

    fig.add_shape(type="rect",
                  x0=-k_e / 2 - 15, x1=k_e / 2 + 15, y0=2, y1=12,
                  fillcolor=FAPS_LIGHT, line=dict(width=0))

    return fig


# ═══════════════════════════════════════════════════════════════════
# 3.  Disc Approximation — Static Overview
# ═══════════════════════════════════════════════════════════════════

def disc_static_figure(
    l_e: float,
    k_e: float,
    elbow_max_deg: float,
    *,
    show_caption: bool = True,
) -> go.Figure:
    """Five-pose static overview of the disc approximation model.

    Dimensions match the antiparallelogram:
      - base width = k_e
      - disc diameter = k_e  (radius = k_e / 2)
      - disc center at antiparallelogram ICR at equilibrium
      - lower link length = b_ell (centrode semi-minor)
    """
    _, b_ell, _ = centrode_params(l_e, k_e)
    disc_center = np.array([0.0, -b_ell])
    disc_radius = k_e / 2.0
    lower_link_len = b_ell
    half_w = k_e / 2.0

    A = np.array([-half_w, 0.0])
    B = np.array([half_w, 0.0])

    show_elbows = _ghost_elbows(elbow_max_deg)

    fig = go.Figure()

    # ── Fixed geometry ──────────────────────────────────────────
    # Base bar
    fig.add_trace(go.Scatter(
        x=[A[0], B[0]], y=[A[1], B[1]],
        mode="lines+markers",
        line=dict(color=FAPS_GRAY, width=6),
        marker=dict(size=10, color=FAPS_GRAY,
                    line=dict(width=1, color="white")),
        name=f"Frame AB ({k_e:.0f} mm)"))

    # Upper structural link
    fig.add_trace(go.Scatter(
        x=[0, disc_center[0]], y=[0, disc_center[1]],
        mode="lines",
        line=dict(color=FAPS_BLUE, width=3.5),
        name="Upper link"))

    # Disc circle
    theta_c = np.linspace(0, 2 * np.pi, 120)
    fig.add_trace(go.Scatter(
        x=disc_center[0] + disc_radius * np.cos(theta_c),
        y=disc_center[1] + disc_radius * np.sin(theta_c),
        mode="lines", line=dict(color=FAPS_BLUE, width=2, dash="dash"),
        name=f"Disc (r={disc_radius:.0f} mm)"))

    # ICR at disc center
    fig.add_trace(go.Scatter(
        x=[disc_center[0]], y=[disc_center[1]],
        mode="markers+text",
        marker=dict(size=16, color=ICR_COLOR, symbol="star",
                    line=dict(width=1, color="white")),
        text=[f"  ICR (0, {disc_center[1]:.1f})"],
        textposition="middle right",
        textfont=dict(size=10, color=ICR_COLOR),
        name="ICR (fixed)"))

    # ── Tendon force arrows (shown once, at neutral) ────────────
    # T₀ attached at left tangent of disc (x = -disc_radius) → pulls up toward A
    # T₁ attached at right tangent of disc (x = +disc_radius) → pulls up toward B
    t0_anchor = disc_center + np.array([-disc_radius, 0.0])
    t1_anchor = disc_center + np.array([disc_radius, 0.0])
    t0_dir = (A - t0_anchor) / np.linalg.norm(A - t0_anchor)  # straight up
    t1_dir = (B - t1_anchor) / np.linalg.norm(B - t1_anchor)  # straight up

    fig.add_trace(go.Scatter(
        x=[t0_anchor[0], t0_anchor[0] + ARROW_LEN * t0_dir[0]],
        y=[t0_anchor[1], t0_anchor[1] + ARROW_LEN * t0_dir[1]],
        mode="lines+markers",
        line=dict(color=TENDON_T0_COL, width=2.5),
        marker=dict(size=[0, 9], symbol=["circle", "arrow-up"],
                    color=TENDON_T0_COL, angleref="previous"),
        name="T₀ (→ A)"))

    fig.add_trace(go.Scatter(
        x=[t1_anchor[0], t1_anchor[0] + ARROW_LEN * t1_dir[0]],
        y=[t1_anchor[1], t1_anchor[1] + ARROW_LEN * t1_dir[1]],
        mode="lines+markers",
        line=dict(color=TENDON_T1_COL, width=2.5),
        marker=dict(size=[0, 9], symbol=["circle", "arrow-up"],
                    color=TENDON_T1_COL, angleref="previous"),
        name="T₁ (→ B)"))

    # ── Ghost poses ─────────────────────────────────────────────
    for i, (ea_deg, alpha) in enumerate(zip(show_elbows, _GHOST_ALPHAS)):
        ea_rad = np.radians(ea_deg)
        is_neutral = abs(ea_deg) < 1e-6
        center = disc_platform_center(disc_center, lower_link_len, ea_rad)
        left, right = disc_platform_bar(center, half_w, ea_rad)

        lw_link = 4 if is_neutral else 2
        lw_plat = 5 if is_neutral else 2.5
        ms = 10 if is_neutral else 6

        # Lower link
        fig.add_trace(go.Scatter(
            x=[disc_center[0], center[0]],
            y=[disc_center[1], center[1]],
            mode="lines",
            line=dict(color=FAPS_GREEN, width=lw_link),
            opacity=alpha, showlegend=(i == 0),
            name="Lower link" if i == 0 else None))

        # Platform bar
        fig.add_trace(go.Scatter(
            x=[left[0], right[0]], y=[left[1], right[1]],
            mode="lines+markers",
            line=dict(color=FAPS_GREEN, width=lw_plat),
            marker=dict(size=ms, color=FAPS_GREEN,
                        line=dict(width=1, color="white")),
            opacity=alpha, showlegend=(i == 0),
            name="Platform (coupler)" if i == 0 else None))

        # Platform center marker
        fig.add_trace(go.Scatter(
            x=[center[0]], y=[center[1]],
            mode="markers",
            marker=dict(size=ms - 2, color="#2d5a1e", symbol="square"),
            opacity=alpha, showlegend=False))

        # Angle annotation
        if not is_neutral:
            fig.add_annotation(
                x=center[0], y=center[1] - 10,
                text=f"{ea_deg:+.0f}°",
                showarrow=False,
                font=dict(size=9, color=FAPS_GREEN),
                opacity=alpha)

        if is_neutral:
            fig.add_annotation(
                x=0, y=18, text="Static frame (base)",
                showarrow=False, font=dict(size=10, color=FAPS_GRAY))
            for pt, name in [(A, "A"), (B, "B")]:
                ox = -12 if pt[0] < 0 else 12
                fig.add_annotation(
                    x=pt[0], y=pt[1], text=f"<b>{name}</b>",
                    showarrow=False, xshift=ox, yshift=10,
                    font=dict(size=13))
            fig.add_annotation(
                x=center[0], y=center[1], text="<b>M</b>",
                showarrow=False, xshift=12, yshift=-12,
                font=dict(size=13))

    for shape in base_hatching_shapes(k_e / 2):
        fig.add_shape(**shape)

    title = (
        f"Disc Approximation — r<sub>disc</sub>={disc_radius:.0f} mm, "
        f"l<sub>lower</sub>={lower_link_len:.1f} mm, "
        f"elbow=±{elbow_max_deg:.0f}°"
        f"<br><sup>Simplified elbow model (constant lever arm)</sup>"
    )
    apply_2d_layout(fig, title=title, show_caption=show_caption)
    return fig


# ═══════════════════════════════════════════════════════════════════
# 4.  Disc Approximation — Animation
# ═══════════════════════════════════════════════════════════════════

def disc_animated_figure(
    l_e: float,
    k_e: float,
    elbow_max_deg: float,
    *,
    n_frames: int = 120,
    show_caption: bool = True,
) -> go.Figure:
    """Animated disc approximation: sweep 0 → +max → −max → 0."""
    _, b_ell, _ = centrode_params(l_e, k_e)
    disc_center = np.array([0.0, -b_ell])
    disc_radius = k_e / 2.0
    lower_link_len = b_ell
    half_w = k_e / 2.0

    A = np.array([-half_w, 0.0])
    B = np.array([half_w, 0.0])

    elbow_seq = elbow_sweep_sequence(elbow_max_deg, n_frames)

    # ── Static traces (0–5) ─────────────────────────────────────
    traces: list[go.BaseTraceType] = []

    # 0: Base bar
    traces.append(go.Scatter(
        x=[A[0], B[0]], y=[A[1], B[1]],
        mode="lines+markers",
        line=dict(color=FAPS_GRAY, width=6),
        marker=dict(size=10, color=FAPS_GRAY,
                    line=dict(width=1, color="white")),
        name=f"Frame AB ({k_e:.0f} mm)"))

    # 1: Upper link
    traces.append(go.Scatter(
        x=[0, disc_center[0]], y=[0, disc_center[1]],
        mode="lines", line=dict(color=FAPS_BLUE, width=3.5),
        name="Upper link"))

    # 2: Disc circle
    theta_c = np.linspace(0, 2 * np.pi, 120)
    traces.append(go.Scatter(
        x=disc_center[0] + disc_radius * np.cos(theta_c),
        y=disc_center[1] + disc_radius * np.sin(theta_c),
        mode="lines", line=dict(color=FAPS_BLUE, width=2, dash="dash"),
        name=f"Disc (r={disc_radius:.0f} mm)"))

    # 3: ICR
    traces.append(go.Scatter(
        x=[disc_center[0]], y=[disc_center[1]],
        mode="markers",
        marker=dict(size=14, color=ICR_COLOR, symbol="star",
                    line=dict(width=1, color="white")),
        name="ICR (fixed)"))

    # 4–5: Tendon arrows (static)
    t0_anchor = disc_center + np.array([-disc_radius, 0])
    t1_anchor = disc_center + np.array([disc_radius, 0])
    t0_dir = A - t0_anchor
    t0_dir = t0_dir / np.linalg.norm(t0_dir)
    t1_dir = B - t1_anchor
    t1_dir = t1_dir / np.linalg.norm(t1_dir)
    traces.append(go.Scatter(
        x=[t0_anchor[0], t0_anchor[0] + ARROW_LEN * t0_dir[0]],
        y=[t0_anchor[1], t0_anchor[1] + ARROW_LEN * t0_dir[1]],
        mode="lines+markers",
        line=dict(color=TENDON_T0_COL, width=3),
        marker=dict(size=[0, 10], symbol=["circle", "arrow-up"],
                    color=TENDON_T0_COL, angleref="previous"),
        name="T₀ (→A)"))
    traces.append(go.Scatter(
        x=[t1_anchor[0], t1_anchor[0] + ARROW_LEN * t1_dir[0]],
        y=[t1_anchor[1], t1_anchor[1] + ARROW_LEN * t1_dir[1]],
        mode="lines+markers",
        line=dict(color=TENDON_T1_COL, width=3),
        marker=dict(size=[0, 10], symbol=["circle", "arrow-up"],
                    color=TENDON_T1_COL, angleref="previous"),
        name="T₁ (→B)"))

    n_static = len(traces)

    # ── Dynamic traces (6–8) at initial pose ────────────────────
    ini_rad = np.radians(elbow_seq[0])
    ini_center = disc_platform_center(disc_center, lower_link_len, ini_rad)
    ini_left, ini_right = disc_platform_bar(ini_center, half_w, ini_rad)

    # 6: Lower link
    traces.append(go.Scatter(
        x=[disc_center[0], ini_center[0]],
        y=[disc_center[1], ini_center[1]],
        mode="lines", line=dict(color=FAPS_GREEN, width=4),
        name="Lower link"))

    # 7: Platform bar
    traces.append(go.Scatter(
        x=[ini_left[0], ini_right[0]], y=[ini_left[1], ini_right[1]],
        mode="lines+markers",
        line=dict(color=FAPS_GREEN, width=5),
        marker=dict(size=10, color=FAPS_GREEN,
                    line=dict(width=1, color="white")),
        name="Platform (coupler)"))

    # 8: Platform center marker
    traces.append(go.Scatter(
        x=[ini_center[0]], y=[ini_center[1]],
        mode="markers",
        marker=dict(size=7, color="#2d5a1e", symbol="square"),
        showlegend=False))

    dyn_indices = list(range(n_static, len(traces)))

    # ── Frames ──────────────────────────────────────────────────
    frames = []
    for fi, ea_deg in enumerate(elbow_seq):
        ea_rad = np.radians(ea_deg)
        center = disc_platform_center(disc_center, lower_link_len, ea_rad)
        left, right = disc_platform_bar(center, half_w, ea_rad)

        frame_data = [
            go.Scatter(x=[disc_center[0], center[0]],
                       y=[disc_center[1], center[1]]),
            go.Scatter(x=[left[0], right[0]], y=[left[1], right[1]]),
            go.Scatter(x=[center[0]], y=[center[1]]),
        ]

        title_text = ""
        if show_caption:
            title_text = (
                f"Disc Approximation — r<sub>disc</sub>={disc_radius:.0f} mm<br>"
                f"<sup>elbow={ea_deg:+.1f}°</sup>"
            )

        frames.append(go.Frame(
            data=frame_data, traces=dyn_indices, name=str(fi),
            layout=go.Layout(title=dict(text=title_text))))

    slider_labels = [f"{e:+.0f}°" for e in elbow_seq]

    fig = go.Figure(data=traces, frames=frames)

    base_title = (
        f"Disc Approximation — "
        f"r<sub>disc</sub>={disc_radius:.0f} mm, "
        f"l<sub>lower</sub>={lower_link_len:.1f} mm<br>"
        f"<sup>Simplified elbow model  |  Press ▶ to animate</sup>"
    )
    fig.update_layout(
        title=dict(text=base_title if show_caption else "",
                   font=dict(size=14)),
        xaxis=dict(title="x [mm]", scaleanchor="y", scaleratio=1,
                   range=[-180, 180], gridcolor="#eee", zeroline=False),
        yaxis=dict(title="y [mm]",
                   range=[-190, 50], gridcolor="#eee", zeroline=False),
        plot_bgcolor="white",
        legend=dict(x=1.02, y=0.5, font=dict(size=9)),
        margin=dict(l=60, r=220, t=80 if show_caption else 20, b=80),
        width=950, height=750,
        updatemenus=animation_buttons(),
        sliders=animation_slider(slider_labels))

    fig.add_shape(type="rect",
                  x0=-k_e / 2 - 15, x1=k_e / 2 + 15, y0=2, y1=12,
                  fillcolor=FAPS_LIGHT, line=dict(width=0))

    return fig


# ═══════════════════════════════════════════════════════════════════
# 5.  Comparison — Animated Overlay
# ═══════════════════════════════════════════════════════════════════

def comparison_animated_figure(
    l_e: float,
    k_e: float,
    elbow_max_deg: float,
    *,
    n_frames: int = 120,
    show_caption: bool = True,
) -> go.Figure:
    """Both mechanisms overlaid, animated through the same sweep."""
    pre = _precompute_common(l_e, k_e, elbow_max_deg)
    theta_0 = pre["theta_0"]
    _, b_ell, _ = centrode_params(l_e, k_e)

    disc_center = np.array([0.0, -b_ell])
    disc_radius = k_e / 2.0
    lower_link_len = b_ell
    half_w = k_e / 2.0

    A_base = np.array([-half_w, 0.0])
    B_base = np.array([half_w, 0.0])

    elbow_seq = elbow_sweep_sequence(elbow_max_deg, n_frames)
    delta_seq = np.array([delta_for_elbow(e, theta_0, k_e, l_e)
                          for e in elbow_seq])

    # Precompute full trajectories for background paths
    n_path = 400
    elbow_path = np.linspace(-elbow_max_deg, elbow_max_deg, n_path)
    antipar_path = np.array([
        linkage_positions(
            delta_for_elbow(e, theta_0, k_e, l_e), k_e, l_e, theta_0
        )[5]  # M
        for e in elbow_path
    ])
    disc_path = np.array([
        disc_platform_center(disc_center, lower_link_len, np.radians(e))
        for e in elbow_path
    ])

    # ── Static traces (0–4) ─────────────────────────────────────
    traces: list[go.BaseTraceType] = []

    # 0: Base hatching line
    traces.append(go.Scatter(
        x=[-k_e / 2 - 15, k_e / 2 + 15], y=[2, 2],
        mode="lines", line=dict(color=FAPS_GRAY, width=2),
        showlegend=False))

    # 1: Disc circle (translucent)
    theta_c = np.linspace(0, 2 * np.pi, 120)
    traces.append(go.Scatter(
        x=disc_center[0] + disc_radius * np.cos(theta_c),
        y=disc_center[1] + disc_radius * np.sin(theta_c),
        mode="lines", line=dict(color=FAPS_BLUE, width=1.5, dash="dash"),
        opacity=0.3, name="Disc", showlegend=True))

    # 2: Antipar background trajectory
    traces.append(go.Scatter(
        x=antipar_path[:, 0], y=antipar_path[:, 1],
        mode="lines", line=dict(color=FAPS_BLUE, width=1.5),
        opacity=0.25, name="Antipar. trajectory"))

    # 3: Disc background trajectory
    traces.append(go.Scatter(
        x=disc_path[:, 0], y=disc_path[:, 1],
        mode="lines", line=dict(color=FAPS_GREEN, width=1.5, dash="dash"),
        opacity=0.25, name="Disc trajectory"))

    # 4: Upper structural link (static)
    traces.append(go.Scatter(
        x=[0, disc_center[0]], y=[0, disc_center[1]],
        mode="lines", line=dict(color=FAPS_BLUE, width=2, dash="dot"),
        opacity=0.3, showlegend=False))

    n_static = len(traces)

    # ── Dynamic traces at initial frame ─────────────────────────
    ini_d = delta_seq[0]
    ini_A, ini_B, ini_D, ini_C, _, ini_M, _ = linkage_positions(
        ini_d, k_e, l_e, theta_0)
    ini_ea_rad = np.radians(elbow_seq[0])
    ini_disc_ctr = disc_platform_center(disc_center, lower_link_len, ini_ea_rad)
    ini_dl, ini_dr = disc_platform_bar(ini_disc_ctr, half_w, ini_ea_rad)

    # 5: Antipar frame AB
    traces.append(go.Scatter(
        x=[ini_A[0], ini_B[0]], y=[ini_A[1], ini_B[1]],
        mode="lines", line=dict(color=FAPS_GRAY, width=5),
        name=f"Frame ({k_e:.0f} mm)"))
    # 6: Rod AD
    traces.append(go.Scatter(
        x=[ini_A[0], ini_D[0]], y=[ini_A[1], ini_D[1]],
        mode="lines", line=dict(color=FAPS_BLUE, width=3),
        name="Rod AD"))
    # 7: Rod BC
    traces.append(go.Scatter(
        x=[ini_B[0], ini_C[0]], y=[ini_B[1], ini_C[1]],
        mode="lines", line=dict(color=ACTIVE_COL, width=3),
        name="Rod BC"))
    # 8: Coupler DC
    traces.append(go.Scatter(
        x=[ini_D[0], ini_C[0]], y=[ini_D[1], ini_C[1]],
        mode="lines", line=dict(color=FAPS_GREEN, width=4),
        name="Antipar. coupler"))
    # 9: Antipar joints
    traces.append(go.Scatter(
        x=[ini_A[0], ini_B[0], ini_D[0], ini_C[0]],
        y=[ini_A[1], ini_B[1], ini_D[1], ini_C[1]],
        mode="markers",
        marker=dict(size=8, color=[FAPS_GRAY]*2 + [FAPS_GREEN]*2,
                    line=dict(width=1, color="white")),
        showlegend=False))
    # 10: Antipar M marker
    traces.append(go.Scatter(
        x=[ini_M[0]], y=[ini_M[1]],
        mode="markers",
        marker=dict(size=12, color=FAPS_BLUE, symbol="circle",
                    line=dict(width=2, color="white")),
        name="M (antipar.)"))
    # 11: Disc lower link
    traces.append(go.Scatter(
        x=[disc_center[0], ini_disc_ctr[0]],
        y=[disc_center[1], ini_disc_ctr[1]],
        mode="lines", line=dict(color=FAPS_GREEN, width=3, dash="dash"),
        name="Disc lower link"))
    # 12: Disc platform bar
    traces.append(go.Scatter(
        x=[ini_dl[0], ini_dr[0]], y=[ini_dl[1], ini_dr[1]],
        mode="lines", line=dict(color=FAPS_GREEN, width=4, dash="dash"),
        name="Disc coupler"))
    # 13: Disc platform center
    traces.append(go.Scatter(
        x=[ini_disc_ctr[0]], y=[ini_disc_ctr[1]],
        mode="markers",
        marker=dict(size=12, color=FAPS_GREEN, symbol="diamond",
                    line=dict(width=2, color="white")),
        name="M (disc)"))
    # 14: Deviation line
    traces.append(go.Scatter(
        x=[ini_M[0], ini_disc_ctr[0]], y=[ini_M[1], ini_disc_ctr[1]],
        mode="lines+text",
        line=dict(color=FAPS_GRAY, width=2, dash="dot"),
        text=["", f"Δ={np.linalg.norm(ini_M - ini_disc_ctr):.1f}"],
        textposition="middle right",
        textfont=dict(size=10, color=FAPS_GRAY),
        name="Deviation"))

    dyn_indices = list(range(n_static, len(traces)))

    # ── Frames ──────────────────────────────────────────────────
    frames = []
    for fi, (delta, ea_deg) in enumerate(zip(delta_seq, elbow_seq)):
        fa, fb, fd_p, fc, _, fm, _ = linkage_positions(
            delta, k_e, l_e, theta_0)
        ea_rad = np.radians(ea_deg)
        d_ctr = disc_platform_center(disc_center, lower_link_len, ea_rad)
        d_left, d_right = disc_platform_bar(d_ctr, half_w, ea_rad)
        dev = np.linalg.norm(fm - d_ctr)

        frame_data = [
            go.Scatter(x=[fa[0], fb[0]], y=[fa[1], fb[1]]),
            go.Scatter(x=[fa[0], fd_p[0]], y=[fa[1], fd_p[1]]),
            go.Scatter(x=[fb[0], fc[0]], y=[fb[1], fc[1]]),
            go.Scatter(x=[fd_p[0], fc[0]], y=[fd_p[1], fc[1]]),
            go.Scatter(x=[fa[0], fb[0], fd_p[0], fc[0]],
                       y=[fa[1], fb[1], fd_p[1], fc[1]]),
            go.Scatter(x=[fm[0]], y=[fm[1]]),
            go.Scatter(x=[disc_center[0], d_ctr[0]],
                       y=[disc_center[1], d_ctr[1]]),
            go.Scatter(x=[d_left[0], d_right[0]],
                       y=[d_left[1], d_right[1]]),
            go.Scatter(x=[d_ctr[0]], y=[d_ctr[1]]),
            go.Scatter(x=[fm[0], d_ctr[0]], y=[fm[1], d_ctr[1]],
                       text=["", f"Δ={dev:.1f}"],
                       textfont=dict(size=10, color=FAPS_GRAY)),
        ]

        title_text = ""
        if show_caption:
            title_text = (
                "Antiparallelogram vs. Disc Approximation<br>"
                f"<sup>elbow={ea_deg:+.1f}°  |  "
                f"deviation Δ = {dev:.1f} mm</sup>"
            )

        frames.append(go.Frame(
            data=frame_data, traces=dyn_indices, name=str(fi),
            layout=go.Layout(title=dict(text=title_text))))

    slider_labels = [f"{e:+.0f}°" for e in elbow_seq]

    fig = go.Figure(data=traces, frames=frames)

    base_title = (
        "Antiparallelogram vs. Disc Approximation<br>"
        f"<sup>l<sub>e</sub>={l_e:.0f}, k<sub>e</sub>={k_e:.0f} mm  |  "
        f"Press ▶ to animate</sup>"
    )
    fig.update_layout(
        title=dict(text=base_title if show_caption else "",
                   font=dict(size=14)),
        xaxis=dict(title="x [mm]", scaleanchor="y", scaleratio=1,
                   range=[-180, 180], gridcolor="#eee", zeroline=False),
        yaxis=dict(title="y [mm]",
                   range=[-190, 50], gridcolor="#eee", zeroline=False),
        plot_bgcolor="white",
        legend=dict(x=1.02, y=0.5, font=dict(size=9)),
        margin=dict(l=60, r=220, t=80 if show_caption else 20, b=80),
        width=950, height=750,
        updatemenus=animation_buttons(),
        sliders=animation_slider(slider_labels))

    fig.add_shape(type="rect",
                  x0=-k_e / 2 - 15, x1=k_e / 2 + 15, y0=2, y1=12,
                  fillcolor=FAPS_LIGHT, line=dict(width=0))

    return fig


# ═══════════════════════════════════════════════════════════════════
# 6.  Comparison — Static (Trajectory + Deviation Subplot)
# ═══════════════════════════════════════════════════════════════════

def comparison_static_figure(
    l_e: float,
    k_e: float,
    elbow_max_deg: float,
    *,
    n_samples: int = 400,
    show_caption: bool = True,
) -> go.Figure:
    """Two-panel static comparison: platform trajectory + deviation."""
    theta_0 = np.arcsin(k_e / l_e)
    _, b_ell, _ = centrode_params(l_e, k_e)
    disc_center = np.array([0.0, -b_ell])
    lower_link_len = b_ell

    elbow_arr = np.linspace(-elbow_max_deg, elbow_max_deg, n_samples)
    antipar_ctr = np.array([
        linkage_positions(
            delta_for_elbow(ea, theta_0, k_e, l_e), k_e, l_e, theta_0
        )[5]
        for ea in elbow_arr
    ])
    disc_ctr = np.array([
        disc_platform_center(disc_center, lower_link_len, np.radians(ea))
        for ea in elbow_arr
    ])

    deviation = np.linalg.norm(antipar_ctr - disc_ctr, axis=1)
    max_dev = float(deviation.max())
    max_dev_idx = int(np.argmax(deviation))

    key_elbows = _ghost_elbows(elbow_max_deg)
    key_idx = [int(np.argmin(np.abs(elbow_arr - e))) for e in key_elbows]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.13,
        subplot_titles=[
            "Platform Centre Trajectory  (x–y space)",
            (f"Positional Deviation  "
             f"(max {max_dev:.1f}\u202fmm at "
             f"±{abs(elbow_arr[max_dev_idx]):.0f}°)"),
        ])

    # ── Panel 1: trajectory ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=antipar_ctr[:, 0], y=antipar_ctr[:, 1],
        mode="lines", line=dict(color=FAPS_BLUE, width=2.5),
        name="Antiparallelogram linkage"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=disc_ctr[:, 0], y=disc_ctr[:, 1],
        mode="lines", line=dict(color=FAPS_GREEN, width=2.5, dash="dash"),
        name="Disc approximation"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=antipar_ctr[key_idx, 0], y=antipar_ctr[key_idx, 1],
        mode="markers",
        marker=dict(size=8, color=FAPS_BLUE, symbol="circle",
                    line=dict(width=1.5, color="white")),
        showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=disc_ctr[key_idx, 0], y=disc_ctr[key_idx, 1],
        mode="markers",
        marker=dict(size=8, color=FAPS_GREEN, symbol="diamond",
                    line=dict(width=1.5, color="white")),
        showlegend=False), row=1, col=1)

    for ki in key_idx:
        fig.add_trace(go.Scatter(
            x=[antipar_ctr[ki, 0], disc_ctr[ki, 0]],
            y=[antipar_ctr[ki, 1], disc_ctr[ki, 1]],
            mode="lines", line=dict(color=FAPS_GRAY, width=0.8, dash="dot"),
            showlegend=False), row=1, col=1)

    for ki, ea in zip(key_idx, key_elbows):
        if abs(ea) < 1e-6 or abs(abs(ea) - elbow_max_deg) < 1e-6:
            pt = antipar_ctr[ki]
            label = "0°" if abs(ea) < 1e-6 else f"{ea:+.0f}°"
            yshift = 12 if abs(ea) < 1e-6 else -16
            fig.add_annotation(
                x=pt[0], y=pt[1], text=label,
                showarrow=False, xshift=0, yshift=yshift,
                font=dict(size=9, color=FAPS_BLUE),
                xref="x", yref="y")

    p_ap = antipar_ctr[max_dev_idx]
    p_di = disc_ctr[max_dev_idx]
    mid_x = (p_ap[0] + p_di[0]) / 2
    mid_y = (p_ap[1] + p_di[1]) / 2
    fig.add_annotation(
        x=mid_x, y=mid_y,
        text=f"Δ<sub>max</sub>\u202f=\u202f{max_dev:.1f}\u202fmm",
        ax=75, ay=-22,
        showarrow=True, arrowhead=2, arrowsize=0.8,
        arrowcolor=FAPS_GRAY,
        font=dict(size=9, color=FAPS_GRAY),
        xref="x", yref="y")

    fig.add_shape(type="rect",
                  x0=-k_e / 2 - 15, x1=k_e / 2 + 15, y0=2, y1=12,
                  fillcolor=FAPS_LIGHT, line=dict(width=0),
                  xref="x", yref="y")
    fig.add_shape(type="line",
                  x0=-k_e / 2 - 15, x1=k_e / 2 + 15, y0=2, y1=2,
                  line=dict(color=FAPS_GRAY, width=2),
                  xref="x", yref="y")

    # ── Panel 2: deviation ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([elbow_arr, elbow_arr[::-1]]),
        y=np.concatenate([deviation, np.zeros(n_samples)]),
        fill="toself", mode="lines", line=dict(width=0),
        fillcolor="rgba(142,68,173,0.12)",
        showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=elbow_arr, y=deviation,
        mode="lines", line=dict(color=ICR_COLOR, width=2),
        name="Deviation |Δ|  [mm]"), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[elbow_arr[max_dev_idx], -elbow_arr[max_dev_idx]],
        y=[max_dev, max_dev],
        mode="markers+text",
        marker=dict(size=8, color=ICR_COLOR, symbol="circle",
                    line=dict(width=1.5, color="white")),
        text=[f"\u202f{max_dev:.1f}\u202fmm", ""],
        textposition="middle right",
        textfont=dict(size=9, color=ICR_COLOR),
        showlegend=False), row=2, col=1)

    fig.add_hline(y=0, line=dict(color=FAPS_GRAY, width=1, dash="dot"),
                  row=2, col=1)

    # ── Axes ────────────────────────────────────────────────────
    fig.update_xaxes(title_text="x  [mm]", gridcolor="#eee", zeroline=False,
                     range=[-115, 115], row=1, col=1)
    fig.update_yaxes(title_text="y  [mm]", gridcolor="#eee", zeroline=False,
                     range=[-155, 20], row=1, col=1)
    fig.update_xaxes(title_text="Elbow angle  [°]",
                     tickvals=list(range(-75, 76, 25)),
                     gridcolor="#eee", zeroline=False, row=2, col=1)
    fig.update_yaxes(title_text="Deviation  [mm]",
                     gridcolor="#eee", zeroline=False, rangemode="tozero",
                     row=2, col=1)

    title = (
        "Antiparallelogram vs. Disc Approximation — "
        "Platform Centre Workpath<br>"
        f"<sup>l<sub>e</sub>\u202f=\u202f{l_e:.0f}\u202fmm,\u2002"
        f"k<sub>e</sub>\u202f=\u202f{k_e:.0f}\u202fmm,\u2002"
        f"±{elbow_max_deg:.0f}°"
        "\u2002·\u2002Klein (2023), §3.2.1</sup>"
    )
    fig.update_layout(
        title=dict(
            text=title if show_caption else "",
            font=dict(size=13)),
        plot_bgcolor="white",
        legend=dict(x=1.02, y=0.72, font=dict(size=10),
                    bordercolor="#ccc", borderwidth=0.5),
        margin=dict(l=65, r=205,
                    t=90 if show_caption else 20, b=65),
        width=820, height=700)

    return fig

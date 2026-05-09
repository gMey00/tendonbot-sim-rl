"""Background concept plots for tendon-driven and tensegrity mechanisms.

Illustrates general principles referenced in the thesis background (§2.2):
- Antagonistic tendon actuation at a revolute joint
- Torque–stiffness decoupling in antagonistic cable drives
- Four-bar linkage topologies (parallelogram vs. antiparallelogram)
- Tensegrity structure classification (class-1 vs. class-2)
- Cable-driven parallel mechanism (CDPM) concept

All figures follow the FAPS colour scheme and Plotly layout conventions
established in ``colors.py``.

Public API
----------
antagonistic_joint_figure     — 3-state antagonistic tendon pair
torque_stiffness_figure       — Feasible (τ, κ) region
four_bar_comparison_figure    — Parallelogram vs. antiparallelogram
tensegrity_classes_figure     — Class-1 prism and class-2 X-joint
cdpm_concept_figure           — Cable-driven parallel platform schematic
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
    PLATFORM_COL,
    BASE_COL,
    PASSIVE_COL,
    apply_2d_layout,
    apply_3d_layout,
    base_hatching_shapes,
)


# ──────────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────────

def _circle_xy(cx: float, cy: float, r: float, n: int = 80):
    """Return (xs, ys) arrays tracing a closed circle."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _arc_xy(cx: float, cy: float, r: float,
            start_deg: float, end_deg: float, n: int = 40):
    """Return (xs, ys) arrays tracing a circular arc."""
    t = np.linspace(np.radians(start_deg), np.radians(end_deg), n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _arrowhead_xy(tip_x, tip_y, dx, dy, size=6):
    """Return 3-point arrowhead polygon (left, tip, right)."""
    length = np.hypot(dx, dy)
    if length < 1e-9:
        return [tip_x], [tip_y]
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular
    base_x = tip_x - size * ux
    base_y = tip_y - size * uy
    return (
        [base_x + size * 0.4 * px, tip_x, base_x - size * 0.4 * px, None],
        [base_y + size * 0.4 * py, tip_y, base_y - size * 0.4 * py, None],
    )


_SCHEMATIC_LAYOUT = dict(
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=50, b=20),
)


def _apply_schematic_layout(fig, *, title="", show_caption=True,
                            width=900, height=550,
                            x_range=None, y_range=None):
    """Clean layout for schematic figures (no axes, equal aspect)."""
    margin_l, margin_r = 20, 180
    margin_t = 80 if show_caption else 10
    margin_b = 40 if show_caption else 10

    # For tight export (no caption), auto-compute height from data aspect ratio
    # so equal-aspect scaling doesn't produce vertical whitespace.
    if not show_caption and x_range is not None and y_range is not None:
        plot_w = width - margin_l - margin_r
        data_aspect = (x_range[1] - x_range[0]) / max(y_range[1] - y_range[0], 1)
        plot_h = plot_w / data_aspect
        height = int(plot_h + margin_t + margin_b + 0.5)

    fig.update_layout(
        title=dict(text=title if show_caption else "", font=dict(size=14)),
        xaxis=dict(
            scaleanchor="y", scaleratio=1,
            range=x_range, showgrid=False, zeroline=False,
            showticklabels=False, title="",
        ),
        yaxis=dict(
            range=y_range, showgrid=False, zeroline=False,
            showticklabels=False, title="",
        ),
        plot_bgcolor="white",
        legend=dict(x=1.02, y=0.5, font=dict(size=10)),
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        width=width, height=height,
    )


# ══════════════════════════════════════════════════════════════════
# 1.  Antagonistic Tendon Joint
# ══════════════════════════════════════════════════════════════════

def antagonistic_joint_figure(*, show_caption: bool = True) -> go.Figure:
    """Three-state schematic of a revolute joint with antagonistic tendons.

    The output link is a rigid bar passing through the pulley along its
    diameter (horizontal at rest); cables attach where the bar meets
    the rim and run straight up to fixed motor anchors above each
    attachment.  Differential tension rotates the bar (and pulley)
    about the joint axis.

    References
    ----------
    - Salisbury & Craig (1982) — antagonistic cable principle
    - Kobayashi et al. (1998) — null-space stiffness modulation
    """
    fig = go.Figure()

    # --- geometry constants (arbitrary mm units) ---
    r_pulley = 28
    load_len = 70           # short distal extension showing the load
    frame_y = 80
    col_spacing = 220

    configs = [
        # (angle_deg, T1_norm, T2_norm, label)
        # Geometry: T₁ wraps the lower-LEFT quadrant, T₂ the lower-RIGHT.
        # Pulling T₁ harder shortens its wrap and rotates the pulley CCW
        # (positive θ in math convention); pulling T₂ harder rotates CW.
        (-20, 0.4, 1.0, "T₁ > T₂  →  τ > 0"),
        (0,   0.7, 0.7, "T₁ = T₂  →  τ = 0"),
        (20,  1.0, 0.4, "T₁ < T₂  →  τ < 0"),
    ]

    for idx, (angle_deg, t1, t2, label) in enumerate(configs):
        ox = (idx - 1) * col_spacing
        θ = np.radians(angle_deg)
        is_center = idx == 1
        opa = 1.0 if is_center else 0.55

        # ── Frame bar with hatching ──
        fig.add_trace(go.Scatter(
            x=[ox - r_pulley - 30, ox + r_pulley + 30],
            y=[frame_y, frame_y],
            mode="lines", line=dict(color=FAPS_GRAY, width=4),
            showlegend=False, opacity=opa,
        ))
        for hx in np.linspace(ox - r_pulley - 25, ox + r_pulley + 25, 9):
            fig.add_trace(go.Scatter(
                x=[hx, hx - 6], y=[frame_y, frame_y + 8],
                mode="lines", line=dict(color=FAPS_GRAY, width=1),
                showlegend=False, opacity=opa * 0.5,
            ))

        # ── Output link (rotates with pulley) ──
        # Vertical bar in the link's body frame, extending from the top
        # rim of the pulley through its centre to the load tip below.
        # Its rim crossing at the top serves as the cable anchor region.
        link_top = np.array([ox - r_pulley * np.sin(θ),
                                  r_pulley * np.cos(θ)])
        load_end = np.array([ox + load_len * np.sin(θ),
                                  -load_len * np.cos(θ)])
        fig.add_trace(go.Scatter(
            x=[link_top[0], load_end[0]],
            y=[link_top[1], load_end[1]],
            mode="lines+markers",
            line=dict(color=FAPS_GREEN, width=5),
            marker=dict(size=[0, 9], color=FAPS_GREEN),
            showlegend=False, opacity=opa,
        ))

        # ── Pulley circle ──
        pcx, pcy = _circle_xy(ox, 0, r_pulley)
        fig.add_trace(go.Scatter(
            x=pcx, y=pcy, mode="lines",
            line=dict(color=FAPS_BLUE, width=2),
            fill="toself",
            fillcolor=f"rgba(41, 97, 147, {0.06 * opa})",
            showlegend=False, opacity=opa,
        ))
        fig.add_trace(go.Scatter(
            x=[ox], y=[0], mode="markers",
            marker=dict(size=5, color=FAPS_BLUE),
            showlegend=False, opacity=opa,
        ))

        # ── Cable attachments at the rim, slightly offset to either
        # side of the link's BOTTOM crossing (where the link emerges
        # toward the load) so the two cables remain distinguishable.
        # Both rotate with the pulley by θ. ──
        alpha = np.radians(8.0)
        ang1 = -np.pi / 2 - alpha + θ       # T₁ side (lower-LEFT of bottom)
        ang2 = -np.pi / 2 + alpha + θ       # T₂ side (lower-RIGHT of bottom)
        att1 = np.array([ox + r_pulley * np.cos(ang1),
                              r_pulley * np.sin(ang1)])
        att2 = np.array([ox + r_pulley * np.cos(ang2),
                              r_pulley * np.sin(ang2)])

        # ── Motor anchors above the leftmost / rightmost points so the
        # free cable segment is a true vertical tangent that descends
        # past the pulley centre to the wrap entry point. ──
        m1x = ox - r_pulley
        m2x = ox + r_pulley
        for mx, col in [(m1x, TENDON_T0_COL), (m2x, TENDON_T1_COL)]:
            fig.add_trace(go.Scatter(
                x=[mx], y=[frame_y],
                mode="markers",
                marker=dict(size=14, color=col, symbol="square"),
                showlegend=False, opacity=opa,
            ))

        # ── Cables: vertical tangent (motor → side of pulley) + ~90°
        # wrap around the LOWER hemisphere to the rim attachments. ──
        lw1 = 1.0 + 3.0 * t1
        lw2 = 1.0 + 3.0 * t2
        n_arc = 36

        # T₁: tangent at angle 180°, wrap clockwise (angle decreasing
        # from π through 3π/2 ≡ −π/2) to attachment ang1.
        a1 = np.linspace(-np.pi, ang1, n_arc)
        arc1_x = ox + r_pulley * np.cos(a1)
        arc1_y =      r_pulley * np.sin(a1)
        fig.add_trace(go.Scatter(
            x=[m1x, m1x, *arc1_x],
            y=[frame_y, 0.0, *arc1_y],
            mode="lines", line=dict(color=TENDON_T0_COL, width=lw1),
            showlegend=False, opacity=opa,
        ))

        # T₂: tangent at angle 0°, wrap counter-clockwise (angle
        # decreasing from 0 to −π/2) through the lower-right quadrant.
        a2 = np.linspace(0.0, ang2, n_arc)
        arc2_x = ox + r_pulley * np.cos(a2)
        arc2_y =      r_pulley * np.sin(a2)
        fig.add_trace(go.Scatter(
            x=[m2x, m2x, *arc2_x],
            y=[frame_y, 0.0, *arc2_y],
            mode="lines", line=dict(color=TENDON_T1_COL, width=lw2),
            showlegend=False, opacity=opa,
        ))

        # Attachment markers at the rim
        fig.add_trace(go.Scatter(
            x=[att1[0], att2[0]], y=[att1[1], att2[1]],
            mode="markers",
            marker=dict(size=8, color=[TENDON_T0_COL, TENDON_T1_COL],
                        line=dict(width=1, color="white")),
            showlegend=False, opacity=opa,
        ))

        # ── Tension direction arrows on the free vertical segment ──
        for cable_x, label_t, col, anchor_x, xanchor in [
            (m1x, "T₁", TENDON_T0_COL, m1x - 6, "right"),
            (m2x, "T₂", TENDON_T1_COL, m2x + 6, "left"),
        ]:
            mid_y = 0.5 * (frame_y + 0.0)
            fig.add_annotation(
                x=cable_x, y=mid_y + 10,
                ax=cable_x, ay=mid_y - 10,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowwidth=2, arrowcolor=col,
                opacity=opa,
            )
            if is_center:
                fig.add_annotation(
                    x=anchor_x, y=mid_y, text=label_t,
                    xanchor=xanchor,
                    showarrow=False, font=dict(size=12, color=col),
                )

        # ── Torque arc — placed in the lower hemisphere on the side
        # opposite the load-arm tilt so it never crosses the cables
        # (upper) or the load arm (lower-centre). ──
        if angle_deg != 0:
            arc_r = r_pulley + 14
            opp_side_deg = -90 - 30 * np.sign(angle_deg)
            half = 28
            if angle_deg > 0:        # link rotated CCW → τ arc CCW
                start_a = np.radians(opp_side_deg - half)
                end_a   = np.radians(opp_side_deg + half)
            else:                     # link rotated CW  → τ arc CW
                start_a = np.radians(opp_side_deg + half)
                end_a   = np.radians(opp_side_deg - half)
            n_tau = 28
            arc_a = np.linspace(start_a, end_a, n_tau)
            ax_arr = ox + arc_r * np.cos(arc_a)
            ay_arr =      arc_r * np.sin(arc_a)
            fig.add_trace(go.Scatter(
                x=ax_arr, y=ay_arr, mode="lines",
                line=dict(color=ICR_COLOR, width=2.5),
                showlegend=False, opacity=0.9,
            ))
            # Arrowhead drawn as a small filled triangle (polygon trace)
            # so it survives the static SVG/PNG export reliably.
            tip_x, tip_y = ax_arr[-1], ay_arr[-1]
            tan_dx = ax_arr[-1] - ax_arr[-3]
            tan_dy = ay_arr[-1] - ay_arr[-3]
            tlen = np.hypot(tan_dx, tan_dy) or 1.0
            ux, uy = tan_dx / tlen, tan_dy / tlen
            px, py = -uy, ux
            head_len = 7.0
            head_w = 4.0
            base_x = tip_x - head_len * ux
            base_y = tip_y - head_len * uy
            tri_x = [tip_x, base_x + head_w * px, base_x - head_w * px, tip_x]
            tri_y = [tip_y, base_y + head_w * py, base_y - head_w * py, tip_y]
            fig.add_trace(go.Scatter(
                x=tri_x, y=tri_y, mode="lines",
                fill="toself", fillcolor=ICR_COLOR,
                line=dict(color=ICR_COLOR, width=0),
                showlegend=False, opacity=0.95,
            ))
            fig.add_annotation(
                x=ox + (arc_r + 10) * np.cos(np.radians(opp_side_deg)),
                y=     (arc_r + 10) * np.sin(np.radians(opp_side_deg)),
                text="τ", showarrow=False,
                font=dict(size=13, color=ICR_COLOR),
            )

        # ── Configuration label below ──
        fig.add_annotation(
            x=ox, y=-load_len - 28,
            text=label, showarrow=False,
            font=dict(size=11, color=FAPS_GRAY), align="center",
        )

        # ── Motor labels (centre config only) ──
        if is_center:
            fig.add_annotation(
                x=m1x, y=frame_y + 14,
                text="M₁", showarrow=False,
                font=dict(size=11, color=TENDON_T0_COL),
            )
            fig.add_annotation(
                x=m2x, y=frame_y + 14,
                text="M₂", showarrow=False,
                font=dict(size=11, color=TENDON_T1_COL),
            )

    # ── Legend (dummy traces) ──
    for name, color, width in [
        ("Cable T₁ (left tendon)",  TENDON_T0_COL, 3),
        ("Cable T₂ (right tendon)", TENDON_T1_COL, 3),
        ("Output link (rotates with pulley)", FAPS_GREEN, 4),
        ("Pulley (radius r)", FAPS_BLUE, 2),
        ("Joint torque τ", ICR_COLOR, 2),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=color, width=width), name=name,
        ))

    _apply_schematic_layout(
        fig,
        title="",
        show_caption=show_caption,
        width=950, height=380,
        x_range=[-340, 340],
        y_range=[-101, 97],
    )
    return fig



# ══════════════════════════════════════════════════════════════════
# 2.  Torque–Stiffness Feasibility Space
# ══════════════════════════════════════════════════════════════════

def torque_stiffness_figure(
    *,
    T_max: float = 1.0,
    r_lever: float = 1.0,
    show_caption: bool = True,
) -> go.Figure:
    """Diamond-shaped feasible region in normalised (τ, κ) space.

    For an antagonistic pair with tensions 0 ≤ Tᵢ ≤ T_max and lever
    arm r, the net torque is τ = r(T₁ − T₂) and the co-contraction
    (apparent stiffness proxy) is κ = r²(T₁ + T₂).

    The feasible region is a rhombus whose vertices reveal the
    fundamental trade-off: maximum torque allows no stiffness
    modulation, while zero torque permits the full stiffness range.

    References
    ----------
    - Kobayashi et al. (1998) — null-space stiffness
    - Salisbury & Craig (1982) — positive-tension constraint
    """
    τ_max = r_lever * T_max
    κ_max = r_lever**2 * 2 * T_max

    # Diamond vertices: bottom → right → top → left → bottom
    τ_verts = np.array([0, τ_max, 0, -τ_max, 0])
    κ_verts = np.array([0, κ_max / 2, κ_max, κ_max / 2, 0])

    # Normalise
    τ_n = τ_verts / τ_max
    κ_n = κ_verts / κ_max

    fig = go.Figure()

    # Feasible region (filled polygon)
    fig.add_trace(go.Scatter(
        x=τ_n, y=κ_n, mode="lines",
        line=dict(color=FAPS_BLUE, width=2),
        fill="toself",
        fillcolor="rgba(41, 97, 147, 0.15)",
        name="Feasible region",
    ))

    # Corner labels
    corners = [
        (0, 0, "T₁=T₂=0", "top center"),
        (1, 0.5, "T₁=T_max, T₂=0<br>(max CW torque)", "middle left"),
        (0, 1, "T₁=T₂=T_max<br>(max stiffness)", "bottom center"),
        (-1, 0.5, "T₁=0, T₂=T_max<br>(max CCW torque)", "middle right"),
    ]
    for tx, ty, txt, anchor in corners:
        fig.add_trace(go.Scatter(
            x=[tx], y=[ty], mode="markers",
            marker=dict(size=8, color=FAPS_BLUE),
            showlegend=False,
        ))
        xshift = -8 if "left" in anchor else (8 if "right" in anchor else 0)
        yshift = -14 if "top" in anchor else (14 if "bottom" in anchor else 0)
        fig.add_annotation(
            x=tx, y=ty, text=txt, showarrow=False,
            font=dict(size=10, color=FAPS_GRAY),
            xanchor="right" if "left" in anchor else (
                "left" if "right" in anchor else "center"
            ),
            yanchor="top" if "top" in anchor else (
                "bottom" if "bottom" in anchor else "middle"
            ),
            xshift=xshift, yshift=yshift,
        )

    # Null-space direction arrow (vertical at τ=0)
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0.05, 0.95],
        mode="lines",
        line=dict(color=ICR_COLOR, width=2, dash="dash"),
        name="Null-space (co-contraction)",
    ))
    fig.add_annotation(
        x=0, y=0.5, text="κ modulation<br>at τ = 0",
        showarrow=False, font=dict(size=10, color=ICR_COLOR),
        xshift=55,
    )

    # Scan lines showing constant-torque stiffness range
    for idx, τ_line in enumerate([0.3, 0.6]):
        κ_lo = τ_line * κ_max / (2 * τ_max)
        κ_hi = κ_max - τ_line * κ_max / (2 * τ_max)
        is_first = (idx == 0)
        fig.add_trace(go.Scatter(
            x=[τ_line / τ_max, τ_line / τ_max],
            y=[κ_lo / κ_max, κ_hi / κ_max],
            mode="lines",
            line=dict(color=FAPS_GREEN, width=2),
            showlegend=is_first,
            name="κ range at fixed τ" if is_first else None,
            legendgroup="scan",
        ))

    fig.update_layout(
        title=dict(text="", font=dict(size=14)),
        xaxis=dict(
            title="Normalised torque  τ / τ_max",
            range=[-1.35, 1.35],
            gridcolor="#eee", zeroline=True,
            zerolinecolor="#ccc",
        ),
        yaxis=dict(
            title="Normalised stiffness  κ / κ_max",
            range=[-0.12, 1.18],
            gridcolor="#eee", zeroline=True,
            zerolinecolor="#ccc",
        ),
        plot_bgcolor="white",
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        margin=dict(l=70, r=30, t=70 if show_caption else 10, b=60 if show_caption else 40),
        width=750, height=550 if show_caption else 480,
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# 3.  Four-Bar Linkage Types
# ══════════════════════════════════════════════════════════════════

def _parallelogram_positions(k, l, theta):
    """Positions for a standard parallelogram four-bar at angle theta."""
    A = np.array([-k / 2, 0])
    B = np.array([+k / 2, 0])
    disp = l * np.array([np.sin(theta), -np.cos(theta)])
    D = A + disp
    C = B + disp
    return A, B, C, D


def _antiparallelogram_positions(k, l, theta_0, delta):
    """Positions for an antiparallelogram at rod deviation delta."""
    theta = theta_0 + delta
    A = np.array([-k / 2, 0])
    B = np.array([+k / 2, 0])
    D = A + l * np.array([np.sin(theta), -np.cos(theta)])
    phi = theta + 2 * np.arctan2(-k * np.cos(theta), l - k * np.sin(theta))
    C = B + l * np.array([np.sin(phi), -np.cos(phi)])
    return A, B, C, D


def _icr_position(k, l, theta_0, delta):
    """ICR of the antiparallelogram coupler relative to frame."""
    theta = theta_0 + delta
    A = np.array([-k / 2, 0])
    B = np.array([+k / 2, 0])
    D = A + l * np.array([np.sin(theta), -np.cos(theta)])
    phi = theta + 2 * np.arctan2(-k * np.cos(theta), l - k * np.sin(theta))
    C = B + l * np.array([np.sin(phi), -np.cos(phi)])

    # ICR is the intersection of lines AD and BC
    def _line_intersect(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-12:
            return None
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
        return p1 + t * d1

    return _line_intersect(A, D, B, C)


def four_bar_comparison_figure(
    *,
    k: float = 60.0,
    l: float = 150.0,
    show_caption: bool = True,
) -> go.Figure:
    """Side-by-side parallelogram vs. antiparallelogram at multiple poses.

    Shows the fundamental topological difference: uncrossed (parallelogram)
    vs. crossed (antiparallelogram) side links, and how the ICR differs.

    Parameters
    ----------
    k : float
        Frame / coupler length [mm].
    l : float
        Side-link length [mm].

    References
    ----------
    - Uicker et al. (2011) — four-bar classification
    - McCarthy & Soh (2010) — closure equation
    - Dijksman (1977) — antiparallelogram centrode ellipses
    """
    fig = go.Figure()
    theta_0 = np.arcsin(k / l)

    # Horizontal offset between the two mechanisms
    sep = 220

    # Multiple poses for ghost overlay
    angles_para = np.radians([-20, 0, 20])        # parallelogram input angles
    deltas_anti = np.radians([-20, 0, 20])         # antiparallelogram deviations
    alphas = [0.3, 1.0, 0.3]

    # ─── Draw Parallelogram (left) ───
    ox_p = -sep
    for ai, (ang, alpha) in enumerate(zip(angles_para, alphas)):
        A, B, C, D = _parallelogram_positions(k, l, theta_0 + ang)
        A, B, C, D = A + [ox_p, 0], B + [ox_p, 0], C + [ox_p, 0], D + [ox_p, 0]

        # Frame AB
        fig.add_trace(go.Scatter(
            x=[A[0], B[0]], y=[A[1], B[1]],
            mode="lines+markers",
            line=dict(color=FAPS_GRAY, width=4),
            marker=dict(size=7, color=FAPS_GRAY),
            showlegend=False, opacity=alpha,
        ))
        # Left link AD (crank)
        fig.add_trace(go.Scatter(
            x=[A[0], D[0]], y=[A[1], D[1]],
            mode="lines",
            line=dict(color=FAPS_BLUE, width=3),
            showlegend=False, opacity=alpha,
        ))
        # Right link BC (rocker)
        fig.add_trace(go.Scatter(
            x=[B[0], C[0]], y=[B[1], C[1]],
            mode="lines",
            line=dict(color=FAPS_BLUE, width=3),
            showlegend=False, opacity=alpha,
        ))
        # Coupler DC
        fig.add_trace(go.Scatter(
            x=[D[0], C[0]], y=[D[1], C[1]],
            mode="lines+markers",
            line=dict(color=FAPS_GREEN, width=4),
            marker=dict(size=7, color=FAPS_GREEN),
            showlegend=False, opacity=alpha,
        ))
        # ICR for parallelogram is at infinity (parallel links → pure translation)
        # Mark the midpoint of coupler instead
        M = (D + C) / 2
        if alpha == 1.0:
            fig.add_trace(go.Scatter(
                x=[M[0]], y=[M[1]], mode="markers",
                marker=dict(size=6, color=ICR_COLOR, symbol="x"),
                showlegend=False,
            ))

    fig.add_annotation(
        x=ox_p, y=20, text="<b>Parallelogram</b>",
        showarrow=False, font=dict(size=13, color=FAPS_GRAY),
    )

    # ─── Draw Antiparallelogram (right) ───
    ox_a = sep
    # ICR path for the antiparallelogram
    deltas_fine = np.linspace(-0.5, 0.5, 300)
    icr_pts = []
    for d in deltas_fine:
        pt = _icr_position(k, l, theta_0, d)
        if pt is not None:
            icr_pts.append(pt + [ox_a, 0])
    icr_arr = np.array(icr_pts)

    if len(icr_arr) > 0:
        fig.add_trace(go.Scatter(
            x=icr_arr[:, 0], y=icr_arr[:, 1],
            mode="lines",
            line=dict(color=ICR_COLOR, width=2, dash="dot"),
            showlegend=False, opacity=0.6,
        ))

    for ai, (delta, alpha) in enumerate(zip(deltas_anti, alphas)):
        A, B, C, D = _antiparallelogram_positions(k, l, theta_0, delta)
        A, B, C, D = A + [ox_a, 0], B + [ox_a, 0], C + [ox_a, 0], D + [ox_a, 0]

        # Frame AB
        fig.add_trace(go.Scatter(
            x=[A[0], B[0]], y=[A[1], B[1]],
            mode="lines+markers",
            line=dict(color=FAPS_GRAY, width=4),
            marker=dict(size=7, color=FAPS_GRAY),
            showlegend=False, opacity=alpha,
        ))
        # Left link AD — crosses BC
        fig.add_trace(go.Scatter(
            x=[A[0], D[0]], y=[A[1], D[1]],
            mode="lines",
            line=dict(color=ACTIVE_COL, width=3),
            showlegend=False, opacity=alpha,
        ))
        # Right link BC — crosses AD
        fig.add_trace(go.Scatter(
            x=[B[0], C[0]], y=[B[1], C[1]],
            mode="lines",
            line=dict(color=FAPS_BLUE, width=3),
            showlegend=False, opacity=alpha,
        ))
        # Coupler DC
        fig.add_trace(go.Scatter(
            x=[D[0], C[0]], y=[D[1], C[1]],
            mode="lines+markers",
            line=dict(color=FAPS_GREEN, width=4),
            marker=dict(size=7, color=FAPS_GREEN),
            showlegend=False, opacity=alpha,
        ))
        # ICR marker
        icr = _icr_position(k, l, theta_0, delta)
        if icr is not None and alpha == 1.0:
            icr_w = icr + [ox_a, 0]
            fig.add_trace(go.Scatter(
                x=[icr_w[0]], y=[icr_w[1]], mode="markers",
                marker=dict(size=9, color=ICR_COLOR, symbol="x"),
                showlegend=False,
            ))
            fig.add_annotation(
                x=icr_w[0] + 26, y=icr_w[1],
                text="ICR", showarrow=False,
                xanchor="left",
                font=dict(size=11, color=ICR_COLOR),
            )

    fig.add_annotation(
        x=ox_a, y=20, text="<b>Antiparallelogram</b>",
        showarrow=False, font=dict(size=13, color=FAPS_GRAY),
    )

    # ── Legend ──
    for name, color, dash, width in [
        ("Frame (AB = DC)", FAPS_GRAY, "solid", 4),
        ("Crank / rocker (AD, BC)", FAPS_BLUE, "solid", 3),
        ("Crossed link (AD)", ACTIVE_COL, "solid", 3),
        ("Coupler (DC)", FAPS_GREEN, "solid", 4),
        ("ICR path (centrode)", ICR_COLOR, "dot", 2),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=color, width=width, dash=dash), name=name,
        ))

    _apply_schematic_layout(
        fig,
        title="",
        show_caption=show_caption,
        width=950, height=520,
        x_range=[-sep - 130, sep + 130],
        y_range=[-l - 4, 24],
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# 4.  Tensegrity Structure Classification
# ══════════════════════════════════════════════════════════════════

def tensegrity_classes_figure(*, show_caption: bool = True) -> go.Figure:
    """Class-1 tensegrity prism and class-2 antiparallelogram X-joint.

    Left:  Class-1 — 3-strut triangular prism (no strut contact).
    Right: Class-2 — two struts sharing a joint node (the X-joint
           used in the thesis manipulator's elbow).

    References
    ----------
    - Skelton & de Oliveira (2009) — class-k taxonomy
    - Sultan (2009) — tensegrity in engineering
    - Fasquelle et al. (2020) — antiparallelogram as class-2 tensegrity
    """
    fig = go.Figure()
    sep = 200

    # ──── Class-1: Triangular Prism ────
    # Top triangle (rotated 60° from bottom)
    ox1 = -sep
    R = 55       # circumradius
    h = 130      # prism height

    top_angles = np.radians([90, 210, 330])
    bot_angles = top_angles + np.radians(60)  # bottom rotated 60°

    top = np.array([[ox1 + R * np.cos(a), h + R * np.sin(a)] for a in top_angles])
    bot = np.array([[ox1 + R * np.cos(a), R * np.sin(a)] for a in bot_angles])

    # Struts (compression): connect top[i] to bot[i]
    for i in range(3):
        fig.add_trace(go.Scatter(
            x=[top[i, 0], bot[i, 0]], y=[top[i, 1], bot[i, 1]],
            mode="lines+markers",
            line=dict(color=FAPS_GRAY, width=5),
            marker=dict(size=7, color=FAPS_GRAY),
            showlegend=False,
        ))

    # Top cables (tension): top triangle edges
    for i in range(3):
        j = (i + 1) % 3
        fig.add_trace(go.Scatter(
            x=[top[i, 0], top[j, 0]], y=[top[i, 1], top[j, 1]],
            mode="lines",
            line=dict(color=TENDON_T0_COL, width=2),
            showlegend=False,
        ))

    # Bottom cables: bottom triangle edges
    for i in range(3):
        j = (i + 1) % 3
        fig.add_trace(go.Scatter(
            x=[bot[i, 0], bot[j, 0]], y=[bot[i, 1], bot[j, 1]],
            mode="lines",
            line=dict(color=TENDON_T0_COL, width=2),
            showlegend=False,
        ))

    # Diagonal cables: top[i] to bot[(i+1)%3]
    for i in range(3):
        j = (i + 1) % 3
        fig.add_trace(go.Scatter(
            x=[top[i, 0], bot[j, 0]], y=[top[i, 1], bot[j, 1]],
            mode="lines",
            line=dict(color=TENDON_T0_COL, width=1.5, dash="dash"),
            showlegend=False,
        ))

    # Node markers (on top of everything)
    for pts in [top, bot]:
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers",
            marker=dict(size=8, color=FAPS_GRAY,
                        line=dict(width=1, color="white")),
            showlegend=False,
        ))

    fig.add_annotation(
        x=ox1, y=h + R + 22, text="<b>Class-1 Tensegrity</b>",
        showarrow=False, font=dict(size=13, color=FAPS_GRAY),
    )

    # ──── Class-2: Antiparallelogram X-Joint ────
    ox2 = sep
    k = 55.0    # frame/coupler half-width
    l = 130.0   # link length

    # Frame (top bar)
    A2 = np.array([ox2 - k, h])
    B2 = np.array([ox2 + k, h])

    # Links cross: AD and BC
    theta_eq = np.arcsin(2 * k / l)
    D2 = A2 + l * np.array([np.sin(theta_eq), -np.cos(theta_eq)])
    phi2 = theta_eq + 2 * np.arctan2(
        -2 * k * np.cos(theta_eq), l - 2 * k * np.sin(theta_eq)
    )
    C2 = B2 + l * np.array([np.sin(phi2), -np.cos(phi2)])

    # Frame bar
    fig.add_trace(go.Scatter(
        x=[A2[0], B2[0]], y=[A2[1], B2[1]],
        mode="lines+markers",
        line=dict(color=FAPS_GRAY, width=5),
        marker=dict(size=8, color=FAPS_GRAY,
                    line=dict(width=1, color="white")),
        showlegend=False,
    ))

    # Link AD (compression strut 1)
    fig.add_trace(go.Scatter(
        x=[A2[0], D2[0]], y=[A2[1], D2[1]],
        mode="lines",
        line=dict(color=FAPS_GRAY, width=4),
        showlegend=False,
    ))

    # Link BC (compression strut 2)
    fig.add_trace(go.Scatter(
        x=[B2[0], C2[0]], y=[B2[1], C2[1]],
        mode="lines",
        line=dict(color=FAPS_GRAY, width=4),
        showlegend=False,
    ))

    # Coupler bar DC
    fig.add_trace(go.Scatter(
        x=[D2[0], C2[0]], y=[D2[1], C2[1]],
        mode="lines+markers",
        line=dict(color=FAPS_GREEN, width=4),
        marker=dict(size=8, color=FAPS_GREEN,
                    line=dict(width=1, color="white")),
        showlegend=False,
    ))

    # Tendon cables: T₁ from B to D (crossing), T₂ from A to C (crossing)
    fig.add_trace(go.Scatter(
        x=[B2[0], D2[0]], y=[B2[1], D2[1]],
        mode="lines",
        line=dict(color=TENDON_T0_COL, width=2),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[A2[0], C2[0]], y=[A2[1], C2[1]],
        mode="lines",
        line=dict(color=TENDON_T1_COL, width=2),
        showlegend=False,
    ))

    # Joint nodes at crossing region
    for pt in [A2, B2, C2, D2]:
        fig.add_trace(go.Scatter(
            x=[pt[0]], y=[pt[1]], mode="markers",
            marker=dict(size=8, color=FAPS_GRAY,
                        line=dict(width=1, color="white")),
            showlegend=False,
        ))

    # Mark the four corner nodes as class-2 contact points: at every
    # corner two rigid members (frame/coupler bar + strut) meet, which
    # is the defining property of a class-2 tensegrity. The crossed
    # struts AD and BC do NOT touch at the geometric crossing in the
    # middle — they merely pass over each other in the planar diagram.
    for pt in [A2, B2, C2, D2]:
        fig.add_trace(go.Scatter(
            x=[pt[0]], y=[pt[1]], mode="markers",
            marker=dict(size=11, color=ACTIVE_COL, symbol="circle",
                        line=dict(width=2, color="white")),
            showlegend=False,
        ))

    fig.add_annotation(
        x=ox2, y=h + R + 22, text="<b>Class-2 Tensegrity</b>",
        showarrow=False, font=dict(size=13, color=FAPS_GRAY),
    )

    # ── Legend ──
    for name, color, dash, width in [
        ("Compression element (strut)", FAPS_GRAY, "solid", 4),
        ("Edge cable (tension)", TENDON_T0_COL, "solid", 2),
        ("Cross-face cable (tension)", TENDON_T0_COL, "dash", 2),
        ("Antagonistic cable (T₂)", TENDON_T1_COL, "solid", 2),
        ("Coupler / platform", FAPS_GREEN, "solid", 4),
        ("Class-2 contact node", ACTIVE_COL, "solid", 0),
    ]:
        mode = "lines" if width > 0 else "markers"
        marker = dict(size=10, color=color, symbol="circle") if width == 0 else None
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode=mode,
            line=dict(color=color, width=width, dash=dash) if width > 0 else None,
            marker=marker,
            name=name,
        ))

    y_lo = min(bot[:, 1].min(), min(C2[1], D2[1])) - 4
    y_hi = max(top[:, 1].max(), h) + R + 28

    _apply_schematic_layout(
        fig,
        title="",
        show_caption=show_caption,
        width=950, height=560,
        x_range=[-sep - 120, sep + 120],
        y_range=[y_lo, y_hi],
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# 5.  Cable-Driven Parallel Mechanism (CDPM)
# ══════════════════════════════════════════════════════════════════

def cdpm_concept_figure(*, show_caption: bool = True) -> go.Figure:
    """3D schematic of a cable-driven parallel mechanism (CDPM).

    Shows a movable platform suspended by m = 3 active cables from a
    fixed base frame, with the structure-matrix formalism annotated.
    Using 3D so the 120°-spaced cable arrangement is visible.

    References
    ----------
    - Verhoeven (2004) — structure matrix, n+1 theorem
    - Pott (2018) — CDPR fundamentals
    - Nemoto et al. (2022) — 2-DOF cable-driven wrist
    """
    fig = go.Figure()

    # ── Geometry ──
    R_base = 80.0       # base ring radius [mm]
    r_plat = 25.0       # platform ring radius [mm]
    z_base = 0.0        # base height
    z_plat = -76.0      # platform height (below base)
    tilt_psi = np.radians(10)
    tilt_theta = np.radians(8)

    # Rotation matrix (XY fixed Euler: Ry(θ) · Rx(ψ))
    cx, sx = np.cos(tilt_psi), np.sin(tilt_psi)
    cy, sy = np.cos(tilt_theta), np.sin(tilt_theta)
    Rot = np.array([
        [cy,        0,   sy],
        [sx * sy,   cx, -sx * cy],
        [-cx * sy,  sx,  cx * cy],
    ])

    # ── Base ring ──
    n_ring = 60
    ba = np.linspace(0, 2 * np.pi, n_ring + 1)
    bx = R_base * np.cos(ba)
    by = R_base * np.sin(ba)
    bz = np.full_like(bx, z_base)
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines", line=dict(color=FAPS_GRAY, width=5),
        name="Base frame (fixed)", legendgroup="base",
    ))
    fig.add_trace(go.Mesh3d(
        x=np.append(bx, [0]), y=np.append(by, [0]),
        z=np.append(bz, [z_base]),
        color=FAPS_LIGHT, opacity=0.2, showlegend=False,
    ))

    # ── Platform ring (tilted) ──
    plat_body = np.column_stack([
        r_plat * np.cos(ba), r_plat * np.sin(ba),
        np.full(n_ring + 1, z_plat),
    ])
    plat_world = (Rot @ plat_body.T).T
    fig.add_trace(go.Scatter3d(
        x=plat_world[:, 0], y=plat_world[:, 1], z=plat_world[:, 2],
        mode="lines", line=dict(color=FAPS_GREEN, width=5),
        name="Platform (movable)", legendgroup="platform",
    ))

    # ── Cable anchors ──
    cable_angles = np.radians([60, 180, 300])
    base_pts = np.array([
        [R_base * np.cos(a), R_base * np.sin(a), z_base]
        for a in cable_angles
    ])
    plat_body_pts = np.array([
        [r_plat * np.cos(a), r_plat * np.sin(a), z_plat]
        for a in cable_angles
    ])
    plat_world_pts = (Rot @ plat_body_pts.T).T

    # ── Cables ──
    cable_colors = [TENDON_T0_COL, FAPS_BLUE, TENDON_T1_COL]
    cable_names = ["Cable 1 (T₁)", "Cable 2 (T₂)", "Cable 3 (T₃)"]
    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=[base_pts[i, 0], plat_world_pts[i, 0]],
            y=[base_pts[i, 1], plat_world_pts[i, 1]],
            z=[base_pts[i, 2], plat_world_pts[i, 2]],
            mode="lines+markers",
            line=dict(color=cable_colors[i], width=3),
            marker=dict(size=5, color=cable_colors[i]),
            name=cable_names[i],
        ))

    # ── Unit-vector arrows (u_i) ──
    for i in range(3):
        diff = base_pts[i] - plat_world_pts[i]
        unit = diff / np.linalg.norm(diff)
        start = plat_world_pts[i]
        end = start + 22 * unit
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]], y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines+text",
            line=dict(color=cable_colors[i], width=2, dash="dash"),
            text=["", f"u{i+1}"],
            textposition="top center",
            textfont=dict(size=9, color=cable_colors[i]),
            showlegend=False,
        ))

    # ── Anchor labels ──
    fig.add_trace(go.Scatter3d(
        x=base_pts[:, 0], y=base_pts[:, 1], z=base_pts[:, 2],
        mode="text", text=[f"b{i+1}" for i in range(3)],
        textposition="top center",
        textfont=dict(size=9, color=FAPS_GRAY),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=plat_world_pts[:, 0], y=plat_world_pts[:, 1],
        z=plat_world_pts[:, 2],
        mode="text", text=[f"p{i+1}" for i in range(3)],
        textposition="bottom center",
        textfont=dict(size=9, color=FAPS_GREEN),
        showlegend=False,
    ))

    # ── Wrench arrow ──
    # Platform load (e.g. gravity / payload) acts DOWNWARD on the
    # platform; arrow drawn from a point above the platform centre,
    # pointing down into it.
    plat_center = (Rot @ np.array([0, 0, z_plat])).flatten()
    wrench_start = plat_center + np.array([0, 0, 28])
    fig.add_trace(go.Scatter3d(
        x=[wrench_start[0], plat_center[0]],
        y=[wrench_start[1], plat_center[1]],
        z=[wrench_start[2], plat_center[2]],
        mode="lines+text",
        line=dict(color=ICR_COLOR, width=4),
        text=["w", ""], textposition="top center",
        textfont=dict(size=12, color=ICR_COLOR),
        showlegend=False,
    ))
    # Discrete cone-style arrowhead at the platform end.
    fig.add_trace(go.Cone(
        x=[plat_center[0]], y=[plat_center[1]], z=[plat_center[2]],
        u=[0], v=[0], w=[-1], anchor="tip",
        sizemode="absolute", sizeref=8,
        colorscale=[[0, ICR_COLOR], [1, ICR_COLOR]],
        showscale=False, showlegend=False,
    ))

    apply_3d_layout(
        fig,
        title="",
        show_caption=show_caption,
        height=460,
        x_range=[-110, 110], y_range=[-110, 110], z_range=[-95, 25],
        camera=dict(eye=dict(x=1.55, y=1.55, z=0.45)),
    )
    fig.update_layout(margin=dict(l=0, r=180, t=0, b=0))
    return fig

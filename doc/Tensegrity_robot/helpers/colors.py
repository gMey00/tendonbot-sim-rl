"""FAPS color scheme and shared layout utilities for joint visualization."""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go


# ── FAPS Corporate Palette ──────────────────────────────────────────

FAPS_GREEN = "#97C139"
FAPS_BLUE = "#296193"
FAPS_GRAY = "#5F5F5F"
FAPS_LIGHT = "#E6E6E6"


# ── Derived Element Colors ──────────────────────────────────────────

ACTIVE_COL = "#C0392B"       # warm red — active cables / rod BC
ICR_COLOR = "#8E44AD"        # purple — ICR markers
TENDON_T0_COL = "#E74C3C"   # red — tendon T₀
TENDON_T1_COL = "#E67E22"   # orange — tendon T₁
PLATFORM_COL = FAPS_GREEN    # moving platform / coupler
BASE_COL = FAPS_GRAY         # static frame / base
PASSIVE_COL = FAPS_BLUE      # passive cables / rods


# ── Layout Utilities ────────────────────────────────────────────────

def apply_2d_layout(
    fig: go.Figure,
    *,
    title: str = "",
    show_caption: bool = True,
    width: int = 900,
    height: int = 750,
    x_range: Optional[list[float]] = None,
    y_range: Optional[list[float]] = None,
    x_title: str = "x [mm]",
    y_title: str = "y [mm]",
) -> None:
    """Apply consistent FAPS 2D plot styling."""
    fig.update_layout(
        title=dict(
            text=title if show_caption else "",
            font=dict(size=14),
        ),
        xaxis=dict(
            title=x_title,
            scaleanchor="y",
            scaleratio=1,
            range=x_range,
            gridcolor="#eee",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            range=y_range,
            gridcolor="#eee",
            zeroline=False,
        ),
        plot_bgcolor="white",
        legend=dict(x=1.02, y=0.5, font=dict(size=10)),
        margin=dict(
            l=60,
            r=200,
            t=80 if show_caption else 20,
            b=60,
        ),
        width=width,
        height=height,
    )


def apply_3d_layout(
    fig: go.Figure,
    *,
    title: str = "",
    show_caption: bool = True,
    width: int = 950,
    height: int = 750,
    x_range: Optional[list[float]] = None,
    y_range: Optional[list[float]] = None,
    z_range: Optional[list[float]] = None,
    camera: Optional[dict] = None,
) -> None:
    """Apply consistent FAPS 3D plot styling."""
    if camera is None:
        camera = dict(eye=dict(x=1.6, y=1.2, z=0.8))

    fig.update_layout(
        title=dict(
            text=title if show_caption else "",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis=dict(title="X [mm]", range=x_range),
            yaxis=dict(title="Y [mm]", range=y_range),
            zaxis=dict(title="Z [mm]", range=z_range),
            aspectmode="data",
            camera=camera,
        ),
        legend=dict(x=1.02, y=0.5, font=dict(size=9)),
        margin=dict(l=0, r=180, t=80 if show_caption else 10, b=0),
        width=width,
        height=height,
    )


def base_hatching_shapes(half_width: float, y: float = 0.0) -> list[dict]:
    """Rectangle + line shapes simulating ground hatching at the base."""
    return [
        dict(
            type="rect",
            x0=-half_width - 15,
            x1=half_width + 15,
            y0=y + 2,
            y1=y + 12,
            fillcolor=FAPS_LIGHT,
            line=dict(width=0),
        ),
        dict(
            type="line",
            x0=-half_width - 15,
            x1=half_width + 15,
            y0=y + 2,
            y1=y + 2,
            line=dict(color=FAPS_GRAY, width=2),
        ),
    ]


def animation_buttons(play_duration: int = 40) -> list[dict]:
    """Standard Play/Pause buttons for Plotly animations."""
    return [dict(
        type="buttons",
        x=0.08,
        y=-0.06,
        xanchor="left",
        buttons=[
            dict(
                label="▶ Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=play_duration, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0),
                )],
            ),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                )],
            ),
        ],
    )]


def animation_slider(
    labels: list[str],
    *,
    indices: Optional[list[int]] = None,
    prefix: str = "elbow = ",
    suffix: str = "°",
    max_steps: int = 30,
) -> list[dict]:
    """Standard slider for Plotly animations with uniform step sampling."""
    n = len(labels)
    if indices is None:
        step_interval = max(1, n // max_steps)
        indices = list(range(0, n, step_interval))

    steps = [
        dict(
            args=[[str(idx)], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
            )],
            label=labels[idx],
            method="animate",
        )
        for idx in indices
    ]

    return [dict(
        active=0,
        x=0.08,
        len=0.84,
        xanchor="left",
        y=-0.02,
        currentvalue=dict(prefix=prefix, suffix=suffix, font=dict(size=12)),
        steps=steps,
    )]

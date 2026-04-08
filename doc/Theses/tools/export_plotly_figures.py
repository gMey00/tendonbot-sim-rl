#!/usr/bin/env python3
"""Export Plotly figures from kinematic notebooks as SVG for thesis inclusion.

Usage:
    cd doc/Tensegrity_robot && conda run -n env_isaaclab python ../Theses_typst/tools/export_plotly_figures.py

Outputs:
    shared/figures/antiparallelogram_static.svg
    shared/figures/wrist_static_3d.svg
    shared/figures/elbow_comparison_static.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the notebook helpers to the path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HELPERS_DIR = REPO_ROOT / "doc" / "Tensegrity_robot"
sys.path.insert(0, str(HELPERS_DIR))

OUT_DIR = Path(__file__).resolve().parent.parent / "shared" / "figures"

from helpers.linkage import (
    linkage_positions,
    instantaneous_center,
    elbow_angle,
    elbow_sweep_sequence,
    delta_for_elbow,
)
from helpers.elbow_plots import (
    antiparallelogram_static_figure,
    comparison_static_figure,
)
from helpers.wrist_plots import wrist_static_figure


def export(fig, name: str) -> None:
    """Write a Plotly figure as SVG to the output directory."""
    path = OUT_DIR / f"{name}.svg"
    fig.write_image(str(path), format="svg")
    print(f"  ✓ {path.relative_to(REPO_ROOT)}")


def main() -> None:
    # ── Notebook parameters (matching the notebooks) ──────────
    l_e = 150.0   # side-link length [mm]
    k_e = 60.0    # frame/coupler length [mm]
    elbow_max_deg = 75.0

    # Wrist parameters
    R = 72.5     # base ring radius [mm]
    r = 25.0     # platform ring radius [mm]
    h = 76.0     # cable length [mm]
    h_b = 9.0    # base ring height offset [mm]

    print("Exporting Plotly figures for thesis …")

    # 1. Antiparallelogram static 5-pose overview
    fig1 = antiparallelogram_static_figure(
        l_e=l_e, k_e=k_e, elbow_max_deg=elbow_max_deg,
    )
    export(fig1, "antiparallelogram_static")

    # 2. Wrist static 3D overview
    fig2 = wrist_static_figure(R=R, r=r, h=h, h_b=h_b)
    export(fig2, "wrist_static_3d")

    # 3. Elbow comparison (antiparallelogram vs disc) — trajectory + deviation
    fig3 = comparison_static_figure(
        l_e=l_e, k_e=k_e, elbow_max_deg=elbow_max_deg,
    )
    export(fig3, "elbow_comparison_static")

    print("Done.")


if __name__ == "__main__":
    main()

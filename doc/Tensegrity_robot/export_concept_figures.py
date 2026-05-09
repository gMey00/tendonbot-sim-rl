"""Export background concept figures for thesis integration.

Usage:
    cd doc/Tensegrity_robot
    conda run -n env_isaaclab python export_concept_figures.py
"""
import sys, os

sys.path.insert(0, ".")
from helpers.concept_plots import (
    antagonistic_joint_figure,
    torque_stiffness_figure,
    four_bar_comparison_figure,
    tensegrity_classes_figure,
    cdpm_concept_figure,
)

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "Theses", "shared", "figures"
)
os.makedirs(OUT_DIR, exist_ok=True)

figures = [
    ("concept_antagonistic_joint",  antagonistic_joint_figure(show_caption=False)),
    ("concept_torque_stiffness",    torque_stiffness_figure(show_caption=False)),
    ("concept_four_bar_types",      four_bar_comparison_figure(show_caption=False)),
    ("concept_tensegrity_classes",  tensegrity_classes_figure(show_caption=False)),
    ("concept_cdpm",                cdpm_concept_figure(show_caption=False)),
]

for name, fig in figures:
    for fmt in ("png", "svg"):
        path = os.path.join(OUT_DIR, f"{name}.{fmt}")
        fig.write_image(path, scale=2 if fmt == "png" else 1)
        print(f"  {path}")

print(f"\nExported {len(figures)} figures to {OUT_DIR}")

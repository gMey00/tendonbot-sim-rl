"""Centralised configuration for workspace analysis scripts.

To add a new robot:
    1. Add a ``RobotConfig`` entry to ``ROBOTS``.
    2. In ``workspace_sample.py:main()`` add an ``elif`` branch that
       imports the Isaac Lab ``ArticulationCfg`` for the new robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── Robot configuration ───────────────────────────────────────────────────

@dataclass(frozen=True)
class RobotConfig:
    """All workspace-analysis parameters for a single robot."""

    display_name: str
    ee_body_name: str
    controlled_joints: tuple[str, ...]
    mount_height: float
    output_directory: str
    mount_rotations: dict[str, tuple[float, float, float, float]]
    default_mount_direction: str = "down"


ROBOTS: dict[str, RobotConfig] = {
    "tensegrity": RobotConfig(
        display_name="5-DOF Tensegrity Robot",
        ee_body_name="tool_link_0",
        controlled_joints=(
            "base_y_joint", "base_z_joint",
            "elbow_joint", "wrist_y_joint", "wrist_x_joint",
        ),
        mount_height=2.30,
        output_directory="outputs/workspace_analysis_tensegrity",
        mount_rotations={
            "down": (1.0, 0.0, 0.0, 0.0),
            "up": (0.0, 0.0, 1.0, 0.0),
        },
    ),
    "ur10e": RobotConfig(
        display_name="UR10e (6-DOF) + Robotiq 2F-140",
        ee_body_name="robotiq_base_link",
        controlled_joints=(
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ),
        mount_height=1.40,
        output_directory="outputs/workspace_analysis_ur10e",
        mount_rotations={
            "down": (0.0, 0.0, 1.0, 0.0),
            "up": (1.0, 0.0, 0.0, 0.0),
        },
    ),
    "kinova": RobotConfig(
        display_name="Kinova Gen3 (7-DOF) + Robotiq 2F-140",
        ee_body_name="end_effector_link",
        controlled_joints=(
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ),
        mount_height=1.40,
        output_directory="outputs/workspace_analysis_kinova",
        mount_rotations={
            "down": (0.0, 0.0, 1.0, 0.0),
            "up": (1.0, 0.0, 0.0, 0.0),
        },
    ),
}


# ── Task workspace geometry [Klein 2023] ──────────────────────────────────

DESIRED_WS_MIN = np.array([-0.05, -0.40, 0.80])
DESIRED_WS_MAX = np.array([0.35, 0.95, 1.30])

# Gripper tip offset along the EE body's local Z axis (metres).
# Negative Z: tool_link_0's local +Z points toward the ceiling mount, so the
# Robotiq fingertip (below the arm's TCP) is in the local -Z direction.
GRIPPER_TIP_OFFSET = (0.0, 0.0, -0.225)


# ── Sampling defaults ─────────────────────────────────────────────────────

DEFAULT_NUM_SAMPLES = 200_000
DEFAULT_NUM_ENVS = 4096
DEFAULT_VOXEL_SIZE = 0.02
DEFAULT_SLICE_THICKNESS = 0.05

COLLISION_MIN_DISTANCE = 0.05
COLLISION_ADJACENCY_SKIP = 2


# ── Visualisation metadata ────────────────────────────────────────────────

AXIS_LABELS = ["X (m)", "Y (m)", "Z (m)"]

VIEW_CONFIGS = [
    {"name": "Top (XY)", "a": 0, "b": 1, "slice_axis": 2},
    {"name": "Side (XZ)", "a": 0, "b": 2, "slice_axis": 1},
    {"name": "Front (YZ)", "a": 1, "b": 2, "slice_axis": 0},
]

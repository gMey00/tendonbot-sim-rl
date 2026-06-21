"""Centralised configuration for workspace analysis scripts.

To add a new robot:
    1. Add a ``RobotConfig`` entry to ``ROBOTS``.
    2. In ``workspace_sample.py:main()`` add an ``elif`` branch that
       imports the Isaac Lab ``ArticulationCfg`` for the new robot.

Robots
------
* ``tensegrity``          — 5-DOF (2 prismatic base + 3-DOF arm), elbow_approx USD,
                            ImplicitActuator PD drives.
* ``tensegrity_physical`` — 5-DOF (2 prismatic base + physical 4-bar linkage elbow
                            + 2-DOF wrist), body-force tendon actuated arm.
                            FK sampling teleports joint positions directly, so the
                            workspace geometry is captured correctly without needing
                            the tendon control loop.
Floor-standing comparison arms (assembled USDs in ``res/``), all mounted
upright at env-local ``(0.75, 1.0, 0.75)`` with the normal "up" rotation:

* ``ur10_f140``           — UR10 6-DOF + Robotiq 2F-140.
* ``ur10_frankenstein``   — UR10 + tensegrity 2-DOF wrist + Robotiq 2F-140.
* ``ur5e_f140``           — UR5e 6-DOF + Robotiq 2F-140.
* ``ur5e_frankenstein``   — UR5e + tensegrity 2-DOF wrist + Robotiq 2F-140.
* ``kinova_f140``         — Kinova Gen3 7-DOF + Robotiq 2F-140.
* ``kinova_frankenstein`` — Kinova Gen3 + tensegrity 2-DOF wrist + Robotiq 2F-140.

The "frankenstein" variants add the tensegrity ``wrist_x_joint`` / ``wrist_y_joint``
DOFs to the controlled-joint set so the wrist element is part of the analysis.
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
    # Full mount position (x, y, z) in env-local frame.  When None, falls back
    # to ``(0.15, 0.0, mount_height)`` for backward compatibility (this keeps
    # the original tensegrity placement untouched).  Set this to mount a robot
    # at an arbitrary location — e.g. the floor-standing arms at (0.75, 1.0, 0.75).
    mount_position: tuple[float, float, float] | None = None
    # Per-robot gripper tip offset along the EE body's local axes (metres).
    # None -> module-level ``GRIPPER_TIP_OFFSET``.
    gripper_tip_offset: tuple[float, float, float] | None = None
    # ── Physics-divergence rejection bounds ──────────────────────────────
    # Maximum EE distance from the mount point (m).  Rejects PhysX solver
    # blow-ups (samples that fly off to infinity).  The default 1.95 matches
    # the original tensegrity reach bound.
    max_reach: float = 1.95
    # Reject samples whose EE Z is above ``mount_z - 0.15``.  Valid only for
    # overhead/ceiling mounts where the arm always hangs below the mount
    # (tensegrity).  Set False for floor-standing, upward-mounted arms.
    reject_above_mount: bool = True
    # Reject samples whose |EE Y| (env-local) exceeds this bound.  None
    # disables the check.  The default 1.7 matches the original tensegrity
    # lateral bound; the generic ``max_reach`` sphere covers floor-mounted arms.
    max_abs_y: float | None = 1.7
    # Mapping {target_joint: source_joint} — after random sampling,
    # the target joint is copied from the source to enforce kinematic
    # constraints (e.g. antiparallelogram symmetry).
    linked_joints: dict[str, str] | None = None
    # Body whose rotation (around X) is checked post-physics to enforce
    # an elbow-angle limit.  None = no post-filter.
    elbow_filter_body: str | None = None
    elbow_filter_max_rad: float = 0.0
    # When True, the physical antiparallelogram closure condition is used
    # to compute rod_right and coupler_left from rod_left after sampling.
    antiparallelogram_closure: bool = False
    # Number of physics steps per batch.  Closed-loop kinematic constraints
    # (e.g. the antiparallelogram coupler_right loop closure) benefit from
    # extra solver steps after joint teleportation.
    num_settle_steps: int = 1


ROBOTS: dict[str, RobotConfig] = {
    "tensegrity": RobotConfig(
        display_name="5-DOF Tensegrity Robot",
        ee_body_name="tool_link_0",
        controlled_joints=(
            "base_y_joint", "base_z_joint",
            "elbow_joint", "wrist_y_joint", "wrist_x_joint",
        ),
        mount_height=2.30,
        output_directory="outputs/workspace_analysis/tensegrity",
        mount_rotations={
            "down": (1.0, 0.0, 0.0, 0.0),
            "up": (0.0, 0.0, 1.0, 0.0),
        },
    ),
    "tensegrity_physical": RobotConfig(
        display_name="5-DOF Tensegrity Robot (Physical Tendon)",
        ee_body_name="tool_link_0",
        # The physical model replaces elbow_joint with a 4-bar antiparallelogram
        # linkage with a SINGLE kinematic DOF.  rod_left_joint is the independent
        # sampling variable; rod_right_joint and coupler_left_joint are computed
        # from the closure condition (see antiparallelogram_kinematics.ipynb).
        # coupler_right_joint is excluded from the articulation tree.
        controlled_joints=(
            "base_y_joint", "base_z_joint",
            "rod_left_joint", "rod_right_joint", "coupler_left_joint",
            "wrist_y_joint", "wrist_x_joint",
        ),
        antiparallelogram_closure=True,
        num_settle_steps=4,
        mount_height=2.30,
        output_directory="outputs/workspace_analysis/tensegrity_physical",
        mount_rotations={
            "down": (1.0, 0.0, 0.0, 0.0),
            "up": (0.0, 0.0, 1.0, 0.0),
        },
    ),
    # ── Floor-standing comparison arms (F140 gripper) ─────────────────────
    # All mounted upright at (0.75, 1.0, 0.75), normal "up" rotation.
    # These reference the assembled USDs in ``res/`` (UR10/UR5E/KinovaGen3).
    # The EE reference body is the Robotiq ``robotiq_base_link`` (consistent
    # across every variant since they share the same gripper asset).
    "ur10_f140": RobotConfig(
        display_name="UR10 (6-DOF) + Robotiq 2F-140",
        ee_body_name="robotiq_base_link",
        controlled_joints=(
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ),
        mount_height=0.75,
        mount_position=(0.75, 1.0, 0.75),
        default_mount_direction="up",
        output_directory="outputs/workspace_analysis/ur10_f140",
        mount_rotations={
            "down": (0.0, 0.0, 1.0, 0.0),
            "up": (1.0, 0.0, 0.0, 0.0),
        },
        max_reach=2.4,
        reject_above_mount=False,
        max_abs_y=None,
    ),
    "ur10_frankenstein": RobotConfig(
        display_name="UR10 (6-DOF) + Tensegrity Wrist + Robotiq 2F-140",
        ee_body_name="robotiq_base_link",
        controlled_joints=(
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "wrist_x_joint", "wrist_y_joint",
        ),
        mount_height=0.75,
        mount_position=(0.75, 1.0, 0.75),
        default_mount_direction="up",
        output_directory="outputs/workspace_analysis/ur10_frankenstein",
        mount_rotations={
            "down": (0.0, 0.0, 1.0, 0.0),
            "up": (1.0, 0.0, 0.0, 0.0),
        },
        max_reach=2.6,
        reject_above_mount=False,
        max_abs_y=None,
    ),
    "ur5e_f140": RobotConfig(
        display_name="UR5e (6-DOF) + Robotiq 2F-140",
        ee_body_name="robotiq_base_link",
        controlled_joints=(
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ),
        mount_height=0.75,
        mount_position=(0.75, 1.0, 0.75),
        default_mount_direction="up",
        output_directory="outputs/workspace_analysis/ur5e_f140",
        mount_rotations={
            "down": (0.0, 0.0, 1.0, 0.0),
            "up": (1.0, 0.0, 0.0, 0.0),
        },
        max_reach=2.0,
        reject_above_mount=False,
        max_abs_y=None,
    ),
    "ur5e_frankenstein": RobotConfig(
        display_name="UR5e (6-DOF) + Tensegrity Wrist + Robotiq 2F-140",
        ee_body_name="robotiq_base_link",
        controlled_joints=(
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "wrist_x_joint", "wrist_y_joint",
        ),
        mount_height=0.75,
        mount_position=(0.75, 1.0, 0.75),
        default_mount_direction="up",
        output_directory="outputs/workspace_analysis/ur5e_frankenstein",
        mount_rotations={
            "down": (0.0, 0.0, 1.0, 0.0),
            "up": (1.0, 0.0, 0.0, 0.0),
        },
        max_reach=2.2,
        reject_above_mount=False,
        max_abs_y=None,
    ),
    "kinova_f140": RobotConfig(
        display_name="Kinova Gen3 (7-DOF) + Robotiq 2F-140",
        ee_body_name="robotiq_base_link",
        controlled_joints=(
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
        ),
        mount_height=0.75,
        mount_position=(0.75, 1.0, 0.75),
        default_mount_direction="up",
        output_directory="outputs/workspace_analysis/kinova_f140",
        mount_rotations={
            "down": (0.0, 0.0, 1.0, 0.0),
            "up": (1.0, 0.0, 0.0, 0.0),
        },
        max_reach=2.0,
        reject_above_mount=False,
        max_abs_y=None,
    ),
    "kinova_frankenstein": RobotConfig(
        display_name="Kinova Gen3 (7-DOF) + Tensegrity Wrist + Robotiq 2F-140",
        ee_body_name="robotiq_base_link",
        controlled_joints=(
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7",
            "wrist_x_joint", "wrist_y_joint",
        ),
        mount_height=0.75,
        mount_position=(0.75, 1.0, 0.75),
        default_mount_direction="up",
        output_directory="outputs/workspace_analysis/kinova_frankenstein",
        mount_rotations={
            "down": (0.0, 0.0, 1.0, 0.0),
            "up": (1.0, 0.0, 0.0, 0.0),
        },
        max_reach=2.2,
        reject_above_mount=False,
        max_abs_y=None,
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

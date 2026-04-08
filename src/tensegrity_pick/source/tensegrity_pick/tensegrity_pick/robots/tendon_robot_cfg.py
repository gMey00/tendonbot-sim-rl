"""Tendon-driven tensegrity robot configurations.

These configs mirror :mod:`.tensegrity_robot_cfg` but replace the arm's
implicit PD actuator with an explicit effort-passthrough actuator
(:class:`IdealPDActuatorCfg` with zero stiffness and damping).  Joint
torques are computed externally by :class:`.tendon_actuator.TendonEffortAction`
(elbow_approx model) or :class:`.tendon_actuator.PhysicalTendonEffortAction`
(physical antiparallelogram model).

Elbow-approx configs
--------------------
* ``TENS_3DOF_TENDON_CFG`` — 3-DOF arm only, Jacobian-transpose tendon drive
* ``TENS_5DOF_GRIPPER_TENDON_CFG`` — 2-DOF linear base (PD) + 3-DOF tendon arm + Robotiq gripper

Physical (antiparallelogram) configs
-------------------------------------
* ``TENS_3DOF_PHYSICAL_TENDON_CFG`` — 3-DOF arm with 4-bar linkage, body-force elbow tendons
* ``TENS_3DOF_PHYSICAL_TENDON_LO_CFG`` — low-res visualization variant
* ``TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG`` — full manipulator with physical elbow
* ``TENS_5DOF_GRIPPER_PHYSICAL_TENDON_LO_CFG`` — low-res visualization variant

Usage: import the desired config and place it in your scene, then add a
matching ``ActionCfg`` to the task's ``ActionsCfg``.

References
----------
* Klein (2023), §3.2 — cable-driven manipulator hardware
* Klein (2023), §3.3 — Gazebo simulation model with force vectors at
  cable attachment points instead of built-in tendon primitives
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .tensegrity_robot_cfg import PROJ_ASSETS_PATH, TENS_3DOF_CFG, TENS_5DOF_GRIPPER_CFG
from .robotiq_2f140_gripper_cfg import get_gripper_actuators

TARGET_LINK_NAME_3DOF = "tool_link"
CONTROLLED_JOINT_NAMES_3DOF = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]
TARGET_LINK_NAME_5DOF = "tool_link_0"
CONTROLLED_JOINT_NAMES_5DOF = ["base_y_joint", "base_z_joint", "elbow_joint", "wrist_y_joint", "wrist_x_joint"]


# ── Shared tendon arm actuators ────────────────────────────────────────────
# IdealPDActuator with stiffness=0, damping=0 acts as pure effort passthrough.
# The TendonEffortAction sets the effort target every step.
# Effort and velocity limits match physical robot capacity (Klein 2023).

_TENDON_ELBOW_ACTUATOR = IdealPDActuatorCfg(
    joint_names_expr=["elbow_joint"],
    effort_limit=35.0,      # cable-derived: 160 N motor peak × 3:1 MA × 0.0725 m = 34.8 N·m (Klein 2023 §3.2.6 p.53)
    velocity_limit=2.5,     # Klein (2023) §4.2 p.77: 126°/s ≈ 2.2 rad/s measured
    stiffness=0.0,
    damping=0.0,
)

_TENDON_WRIST_ACTUATOR = IdealPDActuatorCfg(
    joint_names_expr=["wrist_y_joint", "wrist_x_joint"],
    effort_limit=3.5,       # cable-derived: 80 N continuous × 0.020 m max lever × 2 = 3.2 N·m (Klein 2023 §3.2.2 p.42)
    velocity_limit=9.0,     # Klein (2023) §4.2 p.76: 504°/s ≈ 8.8 rad/s measured
    stiffness=0.0,
    damping=0.0,
)


# ── 3-DOF tendon-driven (arm only, no base/gripper) ───────────────────────

TENS_3DOF_TENDON_CFG = TENS_3DOF_CFG.replace(
    actuators={
        **TENS_3DOF_CFG.actuators,
        "elbow": _TENDON_ELBOW_ACTUATOR,
        "wrist": _TENDON_WRIST_ACTUATOR,
    },
)
"""3-DOF tensegrity arm with tendon-driven effort-passthrough actuator."""


# ── 5-DOF + gripper tendon-driven ─────────────────────────────────────────

TENS_5DOF_GRIPPER_TENDON_CFG = TENS_5DOF_GRIPPER_CFG.replace(
    actuators={
        **TENS_5DOF_GRIPPER_CFG.actuators,
        "elbow": _TENDON_ELBOW_ACTUATOR,
        "wrist": _TENDON_WRIST_ACTUATOR,
    },
)
"""5-DOF tensegrity with gripper — arm is tendon-driven, base + gripper are PD."""


# ═══════════════════════════════════════════════════════════════════════════
# Physical (antiparallelogram) tendon-driven configs
# ═══════════════════════════════════════════════════════════════════════════
#
# The physical model replaces the single ``elbow_joint`` with a full 4-bar
# antiparallelogram linkage: rod_left_joint, rod_right_joint,
# coupler_left_joint, coupler_right_joint.  Elbow tendons are applied as
# body forces by ``PhysicalTendonEffortAction``, so the linkage joints only
# need zero-stiffness effort-passthrough actuators.
#
# NOTE: coupler_right_joint has ``physics:excludeFromArticulation = True``
# in the USD (loop-closure joint).  It is NOT part of the articulation
# tree and cannot be addressed via the joint API.  Observations and
# actuators use only the tree-branch joints.

TARGET_LINK_NAME_3DOF_PHYSICAL = "tool_link"
CONTROLLED_JOINT_NAMES_3DOF_PHYSICAL = [
    "rod_left_joint", "rod_right_joint",
    "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
]
"""Observable/controllable joints for the 3-DOF physical model (tree branch only)."""

TARGET_LINK_NAME_5DOF_PHYSICAL = "tool_link_0"
CONTROLLED_JOINT_NAMES_5DOF_PHYSICAL = [
    "base_y_joint", "base_z_joint",
    "rod_left_joint", "rod_right_joint",
    "coupler_left_joint",
    "wrist_y_joint", "wrist_x_joint",
]
"""Observable/controllable joints for the 5-DOF physical model (tree branch only)."""


# ── Shared physical-model arm actuators ────────────────────────────────────
# All linkage and wrist joints use IdealPDActuator with zero stiffness/damping.
# Elbow torques come from body forces; wrist torques from J^T effort targets.

_PHYSICAL_LINKAGE_ACTUATOR = IdealPDActuatorCfg(
    joint_names_expr=["rod_.*_joint", "coupler_left_joint"],
    effort_limit=35.0,      # Klein (2023) §3.2.6: 160 N × 3:1 × 0.0725 m = 34.8 N·m
    velocity_limit=2.5,
    stiffness=0.0,
    damping=0.0,
)

_PHYSICAL_WRIST_ACTUATOR = IdealPDActuatorCfg(
    joint_names_expr=["wrist_y_joint", "wrist_x_joint"],
    effort_limit=3.5,       # Klein (2023) §3.2.2 p.42: 80 N × 0.020 m × 2 = 3.2 N·m, rounded up
    velocity_limit=9.0,
    stiffness=0.0,
    damping=0.0,
)


# ── 3-DOF physical tendon-driven (arm only) ───────────────────────────────

_PHYSICAL_3DOF_USD = f"{PROJ_ASSETS_PATH}/Tensegrity/threedof_arm/tensegrity_threedof_arm_physical.usd"

TENS_3DOF_PHYSICAL_TENDON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_PHYSICAL_3DOF_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "rod_left_joint": 0.0,
            "rod_right_joint": 0.0,
            "coupler_left_joint": 0.0,
            "wrist_y_joint": 0.0,
            "wrist_x_joint": 0.0,
        },
        joint_vel={},
    ),
    actuators={
        "linkage": _PHYSICAL_LINKAGE_ACTUATOR,
        "wrist": _PHYSICAL_WRIST_ACTUATOR,
    },
)
"""3-DOF physical (antiparallelogram) arm with body-force tendon drive."""

TENS_3DOF_PHYSICAL_TENDON_LO_CFG = TENS_3DOF_PHYSICAL_TENDON_CFG.replace(
    spawn=TENS_3DOF_PHYSICAL_TENDON_CFG.spawn.replace(
        usd_path=f"{PROJ_ASSETS_PATH}/Tensegrity/threedof_arm/tensegrity_threedof_arm_physical_lo.usd",
    ),
)
"""3-DOF physical arm — low-resolution visualization."""


# ── 5-DOF + gripper physical tendon-driven ─────────────────────────────────
# NOTE: The 5-DOF physical USD must be assembled in Robot Assembler from the
# 3-DOF physical arm + ceiling-mount base + Robotiq 2F-140.  The expected
# output path is:
#   res/Tensegrity/fivedof_manipulator/fivedof_linear_base_physical_robotiq2f140.usd
# Until that USD exists, this config cannot be instantiated in simulation.

_PHYSICAL_5DOF_USD = (
    f"{PROJ_ASSETS_PATH}/Tensegrity/fivedof_manipulator/"
    "fivedof_linear_base_physical_robotiq2f140.usd"
)

TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_PHYSICAL_5DOF_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base_y_joint": 0.0,
            "base_z_joint": 0.0,
            "rod_left_joint": 0.0,
            "rod_right_joint": 0.0,
            "coupler_left_joint": 0.0,
            "wrist_y_joint": 0.0,
            "wrist_x_joint": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_outer_finger_joint": 0.0,
            "right_outer_finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "left_inner_finger_pad_joint": 0.0,
            "right_inner_finger_pad_joint": 0.0,
        },
        joint_vel={},
    ),
    actuators={
        "base_y": TENS_5DOF_GRIPPER_CFG.actuators["base_y"],
        "base_z": TENS_5DOF_GRIPPER_CFG.actuators["base_z"],
        "linkage": _PHYSICAL_LINKAGE_ACTUATOR,
        "wrist": _PHYSICAL_WRIST_ACTUATOR,
        **get_gripper_actuators("original"),
    },
)
"""5-DOF physical tensegrity with gripper — elbow is body-force tendon, base + gripper PD."""

TENS_5DOF_GRIPPER_PHYSICAL_TENDON_LO_CFG = TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG.replace(
    spawn=TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG.spawn.replace(
        usd_path=(
            f"{PROJ_ASSETS_PATH}/Tensegrity/fivedof_manipulator/"
            "fivedof_linear_base_physical_robotiq2f140_lo.usd"
        ),
    ),
)
"""5-DOF physical tensegrity with gripper — low-resolution visualization."""

"""Tendon-driven tensegrity robot configurations.

These configs mirror :mod:`.tensegrity_robot_cfg` but replace the arm's
implicit PD actuator with an explicit effort-passthrough actuator
(:class:`IdealPDActuatorCfg` with zero stiffness and damping).  Joint
torques are computed externally by :class:`.tendon_actuator.TendonEffortAction`.

Configs
-------
* ``TENS_3DOF_TENDON_CFG`` — 3-DOF arm only, tendon-driven
* ``TENS_5DOF_GRIPPER_TENDON_CFG`` — 2-DOF linear base (PD) + 3-DOF tendon arm + Robotiq gripper

Usage: import the desired config and place it in your scene, then add a
:class:`TendonEffortActionCfg` to the task's ``ActionsCfg``.

References
----------
* Klein (2023), §3.2 — cable-driven manipulator hardware
* Klein (2023), §3.3 — Gazebo simulation model with force vectors at
  cable attachment points instead of built-in tendon primitives
"""

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .tensegrity_robot_cfg import PROJ_ASSETS_PATH, TENS_3DOF_CFG, TENS_5DOF_GRIPPER_CFG


# ── Shared tendon arm actuator ─────────────────────────────────────────────
# IdealPDActuator with stiffness=0, damping=0 acts as pure effort passthrough.
# The TendonEffortAction sets the effort target every step.

_TENDON_ARM_ACTUATOR = IdealPDActuatorCfg(
    joint_names_expr=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
    effort_limit=50.0,
    velocity_limit=2.0,
    stiffness=0.0,
    damping=0.0,
)


# ── 3-DOF tendon-driven (arm only, no base/gripper) ───────────────────────

TENS_3DOF_TENDON_CFG = TENS_3DOF_CFG.replace(
    actuators={
        **TENS_3DOF_CFG.actuators,
        "arm": _TENDON_ARM_ACTUATOR,
    },
)
"""3-DOF tensegrity arm with tendon-driven effort-passthrough actuator."""


# ── 5-DOF + gripper tendon-driven ─────────────────────────────────────────

TENS_5DOF_GRIPPER_TENDON_CFG = TENS_5DOF_GRIPPER_CFG.replace(
    actuators={
        **TENS_5DOF_GRIPPER_CFG.actuators,
        "arm": _TENDON_ARM_ACTUATOR,
    },
)
"""5-DOF tensegrity with gripper — arm is tendon-driven, base + gripper are PD."""

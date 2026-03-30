"""Robotiq 2F-140 gripper actuator configurations.

Provides reusable actuator configurations for the Robotiq 2F-140 gripper
across different robotic arms (Kinova Gen3, UR10e, Tensegrity, etc.).

The gripper uses a three-group actuator model:
  1. gripper_drive: Primary actuator (finger_joint) - controls grip force
  2. gripper_finger: Inner finger compliance (left/right inner finger joints)
  3. gripper_passive: Passive four-bar linkage (outer fingers, pads, knuckles)

Configurations
--------------
* ``ROBOTIQ_2F140_ORIGINAL`` — Standard config (K=11.25, D=0.1, etc.)
  Validated baseline across Isaac Lab reference implementations.

* ``ROBOTIQ_2F140_TUNED`` — Alternative tuned config (if needed)
  Currently identical to ORIGINAL. Can be modified for task-specific tuning.
"""

from isaaclab.actuators import ImplicitActuatorCfg

# ═════════════════════════════════════════════════════════════════════════════
# ORIGINAL CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
# Baseline Robotiq 2F-140 config validated across Isaac Lab documentation
# and community setups (isaaclab_assets/robots/universal_robots.py).
# References:
# - IsaacLab Discussion #4226 (Kinova Gen3 baseline)
# - Tensorboard logs show stable grip performance in object manipulation

ROBOTIQ_2F140_ORIGINAL = {
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        effort_limit_sim=10.0,
        velocity_limit_sim=1.0,
        stiffness=11.25,
        damping=0.1,
        friction=0.0,
        armature=0.0,
    ),
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ],
        effort_limit_sim=10.0,
        velocity_limit_sim=10.0,
        stiffness=10.0,
        damping=0.05,
        friction=0.0,
        armature=0.0,
    ),
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[
            "left_inner_finger_pad_joint",
            "right_inner_finger_pad_joint",
            "left_outer_finger_joint",
            "right_outer_finger_joint",
            "right_outer_knuckle_joint",
        ],
        effort_limit_sim=1.0,
        velocity_limit_sim=1.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        armature=0.0,
    ),
}

# ═════════════════════════════════════════════════════════════════════════════
# TUNED CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
# Task-specific tuned config (e.g., for tensegrity or high-speed grasping).
# This is derived from PD tuning experiments documented in doc/pd_tuning_results.md.
# Can be customized based on:
# - Gripper mounting compliance
# - Target object material/texture
# - Required grip speed vs. settling time tradeoff

ROBOTIQ_2F140_TUNED = {
    "gripper_drive": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        effort_limit_sim=10.0,
        velocity_limit_sim=1.0,
        stiffness=11.25,  # Currently same as ORIGINAL; adjust if needed.
        damping=0.1,      # Currently same as ORIGINAL; adjust if needed.
        friction=0.0,
        armature=0.0,
    ),
    "gripper_finger": ImplicitActuatorCfg(
        joint_names_expr=[
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ],
        effort_limit_sim=10.0,
        velocity_limit_sim=10.0,
        stiffness=10.0,   # Higher than IsaacLab ref (0.2) — needed for ceiling-mount grip
        damping=0.05,     # Higher than IsaacLab ref (0.001) — empirically validated
        friction=0.0,
        armature=0.0,
    ),
    "gripper_passive": ImplicitActuatorCfg(
        joint_names_expr=[
            "left_inner_finger_pad_joint",
            "right_inner_finger_pad_joint",
            "left_outer_finger_joint",
            "right_outer_finger_joint",
            "right_outer_knuckle_joint",
        ],
        effort_limit_sim=1.0,
        velocity_limit_sim=1.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
        armature=0.0,
    ),
}


def get_gripper_actuators(variant: str = "original") -> dict:
    """Get gripper actuator configuration by name.

    Args:
        variant: Configuration variant. One of:
            - "original": Standard baseline (ROBOTIQ_2F140_ORIGINAL)
            - "tuned": Task-specific tuning (ROBOTIQ_2F140_TUNED)

    Returns:
        Dict of actuator configurations keyed by group name (gripper_drive, etc.)

    Example:
        >>> from gripper_cfg import get_gripper_actuators
        >>> gripper_acts = get_gripper_actuators("original")
        >>> articulation_cfg.actuators.update(gripper_acts)
    """
    variants = {
        "original": ROBOTIQ_2F140_ORIGINAL,
        "tuned": ROBOTIQ_2F140_TUNED,
    }
    if variant not in variants:
        raise ValueError(
            f"Unknown gripper variant '{variant}'. "
            f"Choose from: {list(variants.keys())}"
        )
    return variants[variant]

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Operational-Space-Control (OSC) action setup for the F140 reach tasks.

Mirrors Isaac Lab's franka ``osc_env_cfg.py`` sample, adapted to the six
comparison variants.  OSC commands an end-effector pose (and, in ``variable_kp``
mode, task-space stiffness) and solves for joint torques via the operational-space
formulation, with explicit **null-space posture control** toward the joint centres
— directly handling the Kinova's 7-DOF redundancy (and the Frankenstein wrist) at
the controller level instead of forcing the policy to discover a redundancy
resolution.  This is the action space the research flags as best for the
downstream contact-rich / cloth tasks.

Differences from the franka sample, by intent:
* gravity is already disabled for reach, matching the sample's requirement;
* the EE body is the Robotiq ``robotiq_base_link`` (per variant);
* the joint position/velocity observations are **kept** (the sample drops them):
  the (sin/cos-encoded) joint state helps the redundant arm learn a good
  null-space posture, which is exactly what we want to exploit here.
"""

from __future__ import annotations

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg

from tensegrity_pick.tasks.manager_based.reach.config.f140_reach_common import F140_EE_BODY


def apply_osc_action(env, controlled_joints: list[str]) -> None:
    """Replace the arm action with an Operational-Space-Control action.

    Also switches the arm (and tensegrity-wrist) actuators to pure effort control
    (zero implicit PD stiffness/damping) so OSC's computed joint torques are not
    fought by an underlying position controller.  Gripper groups keep their
    reach-stabilised gains.
    """
    # OSC computes joint efforts; the implicit position-PD on the controlled
    # joints must be disabled or it competes with the controller (cf. franka OSC,
    # which zeroes the shoulder/forearm gains).
    for name, actuator in env.scene.robot.actuators.items():
        if name.startswith("gripper"):
            continue
        actuator.stiffness = 0.0
        actuator.damping = 0.0

    # Null-space posture control is only valid for a *redundant* arm (more
    # controlled joints than the 6-DOF task).  The 7-DOF Kinova and every
    # tensegrity-wrist ("Frankenstein", +2 DOF) variant are redundant; the plain
    # 6-DOF UR arms are not — enabling it there raises "Null-space control is only
    # applicable for redundant manipulators".  Auto-detect from the joint count.
    redundant = len(controlled_joints) > 6
    nullspace_control = "position" if redundant else "none"
    # A joint posture target is only accepted when null-space control is active.
    nullspace_target = "default" if redundant else "none"

    env.actions.arm_action = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=list(controlled_joints),
        body_name=F140_EE_BODY,
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="variable_kp",
            # NOTE: the franka sample uses inertial_dynamics_decoupling=True, which
            # forms Λ = (J M⁻¹ Jᵀ)⁻¹.  On our arm+Robotiq-gripper articulations that
            # inverse is ill-conditioned; disabling decoupling falls back to a
            # task-space PD (F = Kp·e − Kd·ẋ, τ = Jᵀ F) which is numerically robust
            # and still gives task-space impedance + (for redundant arms) null-space
            # posture control.
            inertial_dynamics_decoupling=False,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=100.0,
            motion_damping_ratio_task=1.0,
            motion_stiffness_limits_task=(50.0, 200.0),
            nullspace_control=nullspace_control,
        ),
        # franka uses "center" (joint-range midpoint), but that is undefined/NaN for
        # the Kinova's continuous (infinite-limit) joints; "default" targets the
        # finite default config and keeps null-space posture control well-defined
        # for every redundant variant ("none" for the non-redundant UR arms).
        nullspace_joint_pos_target=nullspace_target,
        position_scale=1.0,
        orientation_scale=1.0,
        stiffness_scale=100.0,
    )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Differential-IK action setup for the F140 comparison reach tasks.

The joint-space variants drive the arm with an EMA joint-position-to-limits
action.  For a redundant arm (e.g. Kinova 7-DOF with 4 continuous joints) that
joint-space action is a large, ill-conditioned space that PPO struggles with.

The IK variants instead use Isaac Lab's ``DifferentialInverseKinematicsAction``
(damped-least-squares solver, ``command_type="pose"``): the policy commands an
end-effector pose *delta* (relative mode, scaled) and the controller solves the
joint motion.  The action space is Cartesian and the kinematic redundancy is
handled by the DLS solver, which typically converges much faster.

Mirrors Isaac Lab's franka ``ik_rel_env_cfg.py`` sample, adapted to track the
Robotiq ``robotiq_base_link`` over each variant's controlled joints (arm, plus
the tensegrity wrist for the Frankenstein variants).
"""

from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from tensegrity_pick.tasks.manager_based.reach.config.f140_reach_common import F140_EE_BODY


# Stiff, well-damped PD gains for IK joint-target tracking.  Isaac Lab's franka
# IK sample switches to FRANKA_PANDA_HIGH_PD_CFG (stiffness 400, damping 80)
# "for IK tracking to be better"; the robots' native gains (e.g. UR10's USD
# stiffness 1.5e5 with damping ~5) are far too underdamped for the controller to
# track cleanly.  Reach runs gravity-off, so these gains are ample.
IK_ARM_STIFFNESS = 800.0
IK_ARM_DAMPING = 160.0


def apply_ik_rel_action(env, controlled_joints: list[str], scale: float = 0.5) -> None:
    """Replace the arm action with a relative Differential-IK pose action.

    Also stiffens the arm (and tensegrity-wrist) PD gains so the implicit-actuator
    controller tracks the IK-solved joint targets — without this the EE lags and
    reach accuracy collapses.  Gripper groups are left at their reach-stabilised
    gains.

    Parameters
    ----------
    env : the reach env cfg (already configured for joint-space by the variant).
    controlled_joints : joints the IK solver may use (arm [+ tensegrity wrist]).
    scale : scaling on the policy's per-step pose delta (Isaac Lab franka uses 0.5).
    """
    env.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(controlled_joints),
        body_name=F140_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=scale,
    )

    _stiffen_arm_for_ik(env)


def apply_ik_abs_action(env, controlled_joints: list[str]) -> None:
    """Replace the arm action with an *absolute*-pose Differential-IK action.

    Unlike the relative variant, the policy commands the full target pose
    (position + quaternion, 7-D) in the robot root frame and the DLS controller
    solves the joint motion to reach it in one shot.  For a fixed-target reach
    this is typically much easier to learn than relative servoing — the policy
    can essentially regress the commanded target pose.  Mirrors Isaac Lab's
    franka ``ik_abs_env_cfg.py``.
    """
    env.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(controlled_joints),
        body_name=F140_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        ),
    )
    _stiffen_arm_for_ik(env)


def _stiffen_arm_for_ik(env) -> None:
    """Stiffen every non-gripper actuator group for crisp IK joint tracking."""
    for name, actuator in env.scene.robot.actuators.items():
        if name.startswith("gripper"):
            continue
        actuator.stiffness = IK_ARM_STIFFNESS
        actuator.damping = IK_ARM_DAMPING

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

import os

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from tensegrity_pick.tasks.manager_based.reach import mdp
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
    # OPT-IN probe knob (task-space hardening, iteration 20): override the
    # per-step delta scale.  Unset -> the caller's value (baseline unchanged).
    _scale_env = os.environ.get("REACH_IK_REL_SCALE", "")
    if _scale_env not in ("", "0"):
        scale = float(_scale_env)
    # OPT-IN (iteration 21): EMA-smooth the raw task-space actions — the joint
    # space is EMA'd (α=0.2) but task-space was not (report audit item #4).
    _ema = _ts_ema_alpha()
    _cfg_cls = mdp.EMADiffIKActionCfg if _ema is not None else DifferentialInverseKinematicsActionCfg
    _extra = {"ema_alpha": _ema} if _ema is not None else {}
    env.actions.arm_action = _cfg_cls(
        asset_name="robot",
        joint_names=list(controlled_joints),
        body_name=F140_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=scale,
        **_extra,
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

    OPT-IN fixes (iteration 21, research-report Rank 1; both off by default):
    * ``REACH_IKABS_ROT6D=1`` — replace the raw-quaternion action (which Isaac
      Lab consumes UNNORMALISED — a documented SO(3) anti-pattern, cf. Zhou et
      al. 2019) with a 9-D action using a 6-D continuous rotation.
    * ``REACH_TS_EMA=<α>`` — EMA-smooth the raw actions like the joint space.
    """
    _ema = _ts_ema_alpha()
    _rot6d = os.environ.get("REACH_IKABS_ROT6D", "") not in ("", "0")
    if _rot6d:
        _cfg_cls = mdp.SixDRotDiffIKActionCfg
        _extra = {"ema_alpha": _ema}  # None -> off; cfg field exists either way
    elif _ema is not None:
        _cfg_cls, _extra = mdp.EMADiffIKActionCfg, {"ema_alpha": _ema}
    else:
        _cfg_cls, _extra = DifferentialInverseKinematicsActionCfg, {}
    env.actions.arm_action = _cfg_cls(
        asset_name="robot",
        joint_names=list(controlled_joints),
        body_name=F140_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        ),
        **_extra,
    )
    _stiffen_arm_for_ik(env)


def _ts_ema_alpha() -> float | None:
    """Parse the REACH_TS_EMA env knob (e.g. '0.2'); unset/'0' -> None (off)."""
    v = os.environ.get("REACH_TS_EMA", "")
    return float(v) if v not in ("", "0") else None


def _stiffen_arm_for_ik(env) -> None:
    """Stiffen every non-gripper actuator group for crisp IK joint tracking."""
    for name, actuator in env.scene.robot.actuators.items():
        if name.startswith("gripper"):
            continue
        actuator.stiffness = IK_ARM_STIFFNESS
        actuator.damping = IK_ARM_DAMPING

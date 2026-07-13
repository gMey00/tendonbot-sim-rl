# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Operational-Space-Control (OSC) action setup for the F140 reach tasks.

Mirrors Isaac Lab's franka ``osc_env_cfg.py`` sample, adapted to the six
comparison variants.  OSC commands an end-effector pose (and, in ``variable_kp``
mode, task-space stiffness) and solves for joint torques via the operational-space
formulation, with explicit **null-space posture control** toward the joint centres
— directly handling the Kinova's 7-DOF redundancy at the controller level instead
of forcing the policy to discover a redundancy resolution.  The compliant
tensegrity wrist of the "Frankenstein" variants is NOT torque-controlled by OSC
(see ``TENSEGRITY_WRIST_JOINTS`` below).  This is the action space the research
flags as best for the downstream contact-rich / cloth tasks.

Differences from the franka sample, by intent:
* gravity is already disabled for reach, matching the sample's requirement;
* the EE body is the Robotiq ``robotiq_base_link`` (per variant);
* the joint position/velocity observations are **kept** (the sample drops them):
  the (sin/cos-encoded) joint state helps the redundant arm learn a good
  null-space posture, which is exactly what we want to exploit here.
"""

from __future__ import annotations

import os

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg

from tensegrity_pick.tasks.manager_based.reach.config.f140_reach_common import F140_EE_BODY

# The compliant tensegrity-wrist joints of the "Frankenstein" variants.  They are
# soft (PD 400/20) and torque-limited to 3.5 N·m with near-zero reflected inertia:
# running them as pure-effort OSC joints (PD zeroed) removes all dissipation and
# the wrist oscillates, feeding EE error back into the task-space loop until the
# whole arm diverges (ur10/ur5e_frankenstein_osc: 41-65 cm, 0 % success;
# kinova_frankenstein_osc: episodes killed by joint_vel_diverged after ~3 steps).
# Fix: keep these joints on their own implicit PD (holding neutral, like the
# gripper) and exclude them from the OSC Jacobian; OSC corrects any wrist
# deflection with the rigid arm joints instead.
TENSEGRITY_WRIST_JOINTS = ("wrist_x_joint", "wrist_y_joint")


def apply_osc_action(env, controlled_joints: list[str]) -> None:
    """Replace the arm action with an Operational-Space-Control action.

    Also switches the arm actuators to pure effort control (zero implicit PD
    stiffness/damping) so OSC's computed joint torques are not fought by an
    underlying position controller.  Gripper groups and the compliant tensegrity
    wrist (see ``TENSEGRITY_WRIST_JOINTS``) keep their PD gains.
    """
    # OSC controls only the rigid arm joints; the compliant tensegrity wrist
    # stays position-held (excluding it from the Jacobian is well-defined — the
    # action term slices the EE-body Jacobian by the controlled joint ids).
    #
    # OPT-IN (iteration-20 experiment, REACH_OSC_INCLUDE_WRIST): include the wrist
    # in the OSC joint set as a *damped-compliant* joint — stiffness zeroed (so
    # OSC's torque isn't fought by PD) but viscous damping KEPT.  Iteration 15's
    # divergence came from zeroing the wrist PD entirely: a near-massless
    # 3.5 N·m joint with no dissipation oscillates and blows up the task-space
    # loop.  Keeping damping is the middle ground that may let OSC actually use
    # the wrist (which the wrist A/B showed matters for pose capability).
    _include_wrist = os.environ.get("REACH_OSC_INCLUDE_WRIST", "") not in ("", "0")
    if _include_wrist:
        osc_joints = list(controlled_joints)
    else:
        osc_joints = [j for j in controlled_joints if j not in TENSEGRITY_WRIST_JOINTS]

    # OSC computes joint efforts; the implicit position-PD on the controlled
    # joints must be disabled or it competes with the controller (cf. franka OSC,
    # which zeroes the shoulder/forearm gains).
    for name, actuator in env.scene.robot.actuators.items():
        if name.startswith("gripper"):
            continue
        if name == "tendon_wrist":
            if _include_wrist:
                actuator.stiffness = 0.0
                actuator.damping = float(os.environ.get("REACH_OSC_WRIST_DAMPING", "20") or "20")
            continue
        actuator.stiffness = 0.0
        actuator.damping = 0.0

    # Null-space posture control is only valid for a *redundant* arm (more
    # controlled joints than the 6-DOF task).  With the tensegrity wrist excluded
    # from OSC, only the 7-DOF Kinova variants are redundant; the 6-DOF UR arms
    # are not — enabling it there raises "Null-space control is only applicable
    # for redundant manipulators".  Auto-detect from the joint count.
    redundant = len(osc_joints) > 6
    nullspace_control = "position" if redundant else "none"
    # A joint posture target is only accepted when null-space control is active.
    nullspace_target = "default" if redundant else "none"

    # ── OPT-IN OSC stabilisation-sweep overrides (iteration-17 step 1b) ──
    # Defaults reproduce the settled config exactly (env vars unset -> no change),
    # so the position-only baseline stays bit-exact. Used to search for a config
    # that makes OSC seed-robust on the non-redundant UR arms (all four UR-OSC
    # variants are bimodal; see tracking log iteration 17).
    _osc_impedance = os.environ.get("REACH_OSC_IMPEDANCE", "variable_kp")
    _osc_kp = float(os.environ.get("REACH_OSC_STIFFNESS", "100") or "100")
    _osc_kp_min = float(os.environ.get("REACH_OSC_STIFFNESS_MIN", "50") or "50")
    _osc_kp_max = float(os.environ.get("REACH_OSC_STIFFNESS_MAX", "200") or "200")
    _osc_zeta = float(os.environ.get("REACH_OSC_DAMPING_RATIO", "1.0") or "1.0")

    # OPT-IN (iteration 21): EMA-smooth the raw 13-D OSC actions (pose +
    # stiffness), mirroring the joint space's α=0.2 (report audit item #4).
    _ts_ema = os.environ.get("REACH_TS_EMA", "")
    _osc_cfg_cls = OperationalSpaceControllerActionCfg
    _osc_extra: dict = {}
    if _ts_ema not in ("", "0"):
        from tensegrity_pick.tasks.manager_based.reach import mdp as _reach_mdp

        _osc_cfg_cls = _reach_mdp.EMAOSCActionCfg
        _osc_extra = {"ema_alpha": float(_ts_ema)}

    env.actions.arm_action = _osc_cfg_cls(
        asset_name="robot",
        joint_names=list(osc_joints),
        body_name=F140_EE_BODY,
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode=_osc_impedance,
            # NOTE: the franka sample uses FULL inertial decoupling, which forms
            # Λ = (J M⁻¹ Jᵀ)⁻¹.  On our arm+Robotiq-gripper articulations that
            # inverse is ill-conditioned.  PARTIAL decoupling (translational /
            # rotational blocks only) is numerically robust here and scales the
            # task-space gains by the arm's inertia, so the same kp/ζ behave
            # consistently across the light UR5e and the heavy UR10 — without it
            # (plain task-space PD, τ = Jᵀ(Kp·e − Kd·ẋ)) the UR5e was badly
            # under-damped: 11.9 cm/67 % → 2.1 cm/93 % (ur5e_f140_osc), and
            # 4.8 cm/85 % → 3.1 cm/91 % (ur10_f140_osc).
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=True,
            gravity_compensation=False,
            motion_stiffness_task=_osc_kp,
            motion_damping_ratio_task=_osc_zeta,
            motion_stiffness_limits_task=(_osc_kp_min, _osc_kp_max),
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
        **_osc_extra,
    )

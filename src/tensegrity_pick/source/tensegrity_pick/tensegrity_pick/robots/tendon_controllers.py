"""Hierarchical (inner-loop) controller action for the physical tendon model.

Implements the intermediate actuation layer recommended by the project-thesis
discussion (§6, "Recommendations"): instead of exposing raw cable tensions to
the RL policy, the policy commands *joint-space set-points* (effective elbow
angle + two wrist angles) and a validated inner PID loop regulates the cable
tensions at the physics rate.

The inner loop is a vectorized (torch) port of the controller heuristic used
for the body-force model validation
(``scripts/model_validation/common.py`` / ``run_step_response_tendon.py``):

1. Discrete PID per joint (derivative-on-measurement, EMA-filtered D,
   clamped integrator, conditional integration for the wrist).
2. Gravity compensation τ_g = m·g·l_c·sin(θ) (Mukherjee et al., thesis [85]).
3. Block-wise torque→tension distribution:
   - elbow: antagonistic pair, algebraic solution with pre-tension,
   - wrist: pseudo-inverse of the 2×3 wrist Jacobian block with a null-space
     bias toward the previous tensions (temporal smoothness).
4. Force application through the *same* physical channel as the direct
   variant: elbow tensions as body forces at the cable attachment points
   (with cable min/max length limits), wrist tensions via J^T efforts.

The PID gains default to the values validated in the step-response study
(``PHYSICAL_ELBOW_SPEC``: kp=75, ki=6, kd=3; wrist: kp=10, ki=1.5, kd=0.6).

References
----------
* Klein (2023) §3.2.3 (PID), §3.2.4 (tension distribution).
* Project thesis §6 — hierarchical control recommendation for the physical
  tendon variant.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .tendon_actuator import (
    NUM_ELBOW_TENDONS,
    NUM_WRIST_TENDONS,
    PhysicalTendonEffortAction,
    PhysicalTendonEffortActionCfg,
    compute_lower_arm_angle_and_rate,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)

NUM_CONTROLLED_ANGLES = 3  # [elbow, wrist_y, wrist_x]


class HierarchicalPhysicalTendonAction(PhysicalTendonEffortAction):
    """Joint-space set-point action with an inner PID→tension loop.

    The RL agent outputs 3 actions in [-1, 1] which are affine-mapped to
    joint-angle set-points: effective elbow angle (measured from the forearm
    body rotation, see :func:`compute_lower_arm_angle_and_rate`) and the two
    wrist joint angles.  The inner loop runs at the physics rate (every
    ``apply_actions`` call) and converts the set-point errors into cable
    tensions, applied through the physical body-force channel.
    """

    cfg: HierarchicalPhysicalTendonActionCfg

    def __init__(self, cfg: HierarchicalPhysicalTendonActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        B = self.num_envs

        # ── Set-point mapping [-1, 1] → [lo, hi] ───────────────────────────
        lo = torch.tensor(
            [cfg.elbow_setpoint_range[0], cfg.wrist_setpoint_range[0], cfg.wrist_setpoint_range[0]],
            dtype=torch.float32, device=self.device,
        )
        hi = torch.tensor(
            [cfg.elbow_setpoint_range[1], cfg.wrist_setpoint_range[1], cfg.wrist_setpoint_range[1]],
            dtype=torch.float32, device=self.device,
        )
        self._sp_lo = lo.unsqueeze(0)   # (1, 3)
        self._sp_hi = hi.unsqueeze(0)   # (1, 3)

        # ── PID gains (1, 3) — order [elbow, wrist_y, wrist_x] ─────────────
        e_kp, e_ki, e_kd = cfg.elbow_pid
        w_kp, w_ki, w_kd = cfg.wrist_pid
        self._kp = torch.tensor([e_kp, w_kp, w_kp], device=self.device).unsqueeze(0)
        self._ki = torch.tensor([e_ki, w_ki, w_ki], device=self.device).unsqueeze(0)
        self._kd = torch.tensor([e_kd, w_kd, w_kd], device=self.device).unsqueeze(0)

        # ── Gravity compensation constants m·g·l_c per joint ───────────────
        self._grav_gain = torch.tensor(
            [
                cfg.elbow_grav_mass * 9.81 * cfg.elbow_grav_arm,
                cfg.wrist_grav_mass * 9.81 * cfg.wrist_grav_arm,
                cfg.wrist_grav_mass * 9.81 * cfg.wrist_grav_arm,
            ],
            device=self.device,
        ).unsqueeze(0)  # (1, 3)

        # ── Wrist pseudo-inverse + null-space projector ────────────────────
        j_w = self._wrist_jac_t.squeeze(0)                    # (2, 3)
        pinv = torch.linalg.pinv(j_w)                         # (3, 2)
        self._wrist_pinv = pinv
        self._wrist_null = torch.eye(NUM_WRIST_TENDONS, device=self.device) - pinv @ j_w  # (3, 3)

        # ── Set-point slew limits + per-joint integral clamps ──────────────
        self._slew_rates = torch.tensor(cfg.setpoint_slew_rates, device=self.device).unsqueeze(0)  # (1, 3)
        self._integral_clamps = torch.tensor(cfg.integral_clamps, device=self.device).unsqueeze(0)  # (1, 3)

        # ── Controller state buffers ───────────────────────────────────────
        # Parent allocated (B, 5) raw actions for the tension interface;
        # replace with the 3-dim set-point interface.
        self._raw_actions = torch.zeros(B, NUM_CONTROLLED_ANGLES, device=self.device)
        self._setpoints = torch.zeros(B, NUM_CONTROLLED_ANGLES, device=self.device)
        # Slew-limited target actually tracked by the PID (a commanded jump is
        # physically a ramp at the drive's velocity limit).
        self._setpoints_tracked = torch.zeros(B, NUM_CONTROLLED_ANGLES, device=self.device)
        self._integral = torch.zeros(B, NUM_CONTROLLED_ANGLES, device=self.device)
        self._prev_deriv = torch.zeros(B, NUM_CONTROLLED_ANGLES, device=self.device)
        self._prev_wrist_tensions = torch.full(
            (B, NUM_WRIST_TENDONS), cfg.min_tension_wrist, device=self.device,
        )

        logger.info(
            "HierarchicalPhysicalTendonAction — inner PID at %.0f Hz, gains elbow=%s wrist=%s",
            1.0 / env.physics_dt, cfg.elbow_pid, cfg.wrist_pid,
        )

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        return NUM_CONTROLLED_ANGLES

    @property
    def processed_actions(self) -> torch.Tensor:
        """Joint-angle set-points [rad], order [elbow, wrist_y, wrist_x]."""
        return self._setpoints

    # ── Operations ─────────────────────────────────────────────────────────

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        unit = 0.5 * (actions.clamp(-1.0, 1.0) + 1.0)
        self._setpoints[:] = self._sp_lo + unit * (self._sp_hi - self._sp_lo)

    def apply_actions(self) -> None:
        cfg = self.cfg
        dt = self._env.physics_dt
        data = self._asset.data

        # ── Measurements ───────────────────────────────────────────────────
        elbow_q, elbow_qd = compute_lower_arm_angle_and_rate(
            self._asset, self._root_body_idx, self._forearm_body_idx,
        )
        wrist_q = data.joint_pos[:, self._wrist_joint_ids]   # (B, 2) [wrist_y, wrist_x]
        wrist_qd = data.joint_vel[:, self._wrist_joint_ids]
        q = torch.cat([elbow_q.unsqueeze(-1), wrist_q], dim=-1)     # (B, 3)
        qd = torch.cat([elbow_qd.unsqueeze(-1), wrist_qd], dim=-1)  # (B, 3)

        # ── Set-point slew limiting (drive velocity limits) ────────────────
        # The PID tracks a ramp toward the commanded set-point instead of the
        # raw command: a large commanded jump (RL exploration or a fresh IK
        # solution) would otherwise produce kp × O(1 rad) torque demands that
        # slam the cables into saturation and strain the loop closure.
        max_step = self._slew_rates * dt
        self._setpoints_tracked += (self._setpoints - self._setpoints_tracked).clamp(-max_step, max_step)

        # ── PID (derivative-on-measurement, EMA-filtered D) ────────────────
        error = self._setpoints_tracked - q
        # Conditional integration (anti-windup): each joint integrates only
        # inside its integral zone — otherwise an unreachable or far target
        # winds the integrator to its clamp and the arm limit-cycles around
        # the target on saturated tensions.
        izone_mask = torch.ones_like(error, dtype=torch.bool)
        izone_mask[:, 0] = error[:, 0].abs() < cfg.elbow_integral_zone
        izone_mask[:, 1:] = error[:, 1:].abs() < cfg.wrist_integral_zone
        self._integral += error * dt * izone_mask
        self._integral = torch.clamp(self._integral, -self._integral_clamps, self._integral_clamps)

        alpha = cfg.deriv_filter_alpha
        deriv = alpha * (-qd) + (1.0 - alpha) * self._prev_deriv
        self._prev_deriv[:] = deriv

        tau = self._kp * error + self._ki * self._integral + self._kd * deriv
        # Gravity compensation: τ_g = m·g·l_c·sin(θ)
        tau = tau + self._grav_gain * torch.sin(q)

        # ── Torque → tension distribution (block-wise) ─────────────────────
        # Elbow: antagonistic pair with lever arm ±elbow_lever.
        diff = tau[:, 0] / cfg.elbow_lever
        sat_elbow = self._max_tension[0, :NUM_ELBOW_TENDONS]   # (2,)
        t0 = (cfg.min_tension_elbow + diff.clamp(min=0.0)).clamp(max=float(sat_elbow[0]))
        t1 = (cfg.min_tension_elbow + (-diff).clamp(min=0.0)).clamp(max=float(sat_elbow[1]))
        elbow_tensions = torch.stack([t0, t1], dim=-1)          # (B, 2)

        # Wrist: min-norm particular solution + null-space bias toward the
        # previous tensions (removes the flip instability of pure min-norm).
        tau_wrist = tau[:, 1:3]                                  # (B, 2)
        t_min_norm = tau_wrist @ self._wrist_pinv.T              # (B, 3)
        t_raw = t_min_norm + self._prev_wrist_tensions @ self._wrist_null.T
        shift = (cfg.min_tension_wrist - t_raw.min(dim=-1, keepdim=True).values).clamp(min=0.0)
        sat_wrist = self._max_tension[:, NUM_ELBOW_TENDONS:]     # (1, 3)
        wrist_tensions = torch.minimum((t_raw + shift).clamp(min=0.0), sat_wrist)
        self._prev_wrist_tensions[:] = wrist_tensions

        # ── Bookkeeping (parent buffers, for logging/inspection) ───────────
        self._elbow_tensions = elbow_tensions
        self._wrist_tensions = wrist_tensions
        self._tensions = torch.cat([elbow_tensions, wrist_tensions], dim=-1)
        wrist_torques = torch.bmm(
            self._wrist_jac_t.expand(self.num_envs, -1, -1),
            wrist_tensions.unsqueeze(-1),
        ).squeeze(-1)
        self._wrist_torques = wrist_torques

        # ── Apply through the shared physical channel ──────────────────────
        self._apply_tendon_forces(elbow_tensions, wrist_torques)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._setpoints[env_ids] = 0.0
        self._setpoints_tracked[env_ids] = 0.0
        self._integral[env_ids] = 0.0
        self._prev_deriv[env_ids] = 0.0
        self._prev_wrist_tensions[env_ids] = self.cfg.min_tension_wrist


@configclass
class HierarchicalPhysicalTendonActionCfg(PhysicalTendonEffortActionCfg):
    """Configuration for the hierarchical set-point controller action.

    Inherits the physical body-force channel (attachment geometry, per-tendon
    saturation, cable-length limits) from :class:`PhysicalTendonEffortActionCfg`
    and adds the inner-loop controller parameters validated in the
    step-response study (``scripts/model_validation/common.py``).

    Example
    -------
    .. code-block:: python

        arm_tendon = HierarchicalPhysicalTendonActionCfg(
            asset_name="robot",
            joint_names=["wrist_y_joint", "wrist_x_joint"],
        )
    """

    class_type: type[ActionTerm] = HierarchicalPhysicalTendonAction

    elbow_setpoint_range: tuple[float, float] = (-1.0472, 1.0472)
    """Elbow set-point range [rad] (±60°).

    Deliberately inside the ±70° geometric workspace: commanding the boundary
    parks the long cable exactly at its max free length, so any overshoot or
    sway hammers the stretch stop against the drive tension (measured
    2026-07-10: elbow excursions to +84° with ~520 N stop tension, followed by
    persistent branch flips).  ±60° keeps the stop as a rarely-touched safety.
    """

    wrist_setpoint_range: tuple[float, float] = (-0.6981, 0.6981)
    """Wrist set-point range [rad] (±40°).

    Inside the ±50° joint range: with the 1 kg gripper the wrist needs up to
    ~2 N·m near ±50° while the 80 N cables deliver ≤ ~1.5 N·m — the wrist is
    under-actuated at its range edges and enters a saturated chatter cycle
    whose cyclic load branch-flips the elbow four-bar at otherwise benign
    tensions (measured 2026-07-10).
    """

    elbow_pid: tuple[float, float, float] = (75.0, 6.0, 3.0)
    """Elbow PID gains (kp, ki, kd) — validated PHYSICAL_ELBOW_SPEC."""

    wrist_pid: tuple[float, float, float] = (10.0, 1.5, 0.6)
    """Wrist PID gains (kp, ki, kd) — validated WRIST_Y/X_SPEC."""

    setpoint_slew_rates: tuple[float, float, float] = (1.2, 5.0, 5.0)
    """Max set-point tracking rate [rad/s] per joint [elbow, wrist_y, wrist_x].

    The PID tracks a ramp toward the commanded set-point at these rates.
    Without this, commanded jumps (RL exploration noise, fresh IK solutions)
    produce kp × O(1 rad) torque demands that slam the cables into saturation,
    strain the loop closure, and cause the oscillation observed in the
    2026-07-10 GUI review.  The rates sit ~50 % below the hardware velocity
    limits (elbow 2.2 rad/s, wrist 8.8 rad/s measured, Klein §4.2) so the
    tracking error — and hence the commanded tension — stays moderate during
    transit: ramping at the arm's own top speed kept the elbow tension above
    400 N on 27 % of steps and persistently branch-flipped the four-bar in
    ~14 % of episodes (measured 2026-07-10).
    """

    integral_clamps: tuple[float, float, float] = (2.0, 0.8, 0.8)
    """Per-joint symmetric clamp on the PID integrator state [rad·s].

    Sized so the maximum integral torque stays moderate: elbow ki·clamp =
    6 × 2.0 = 12 N·m (≈ ⅓ of the 35 N·m effort scale), wrist 1.5 × 0.8 =
    1.2 N·m (≈ the ~1.5 N·m achievable wrist torque).  The validation study's
    clamp of 50 allowed up to 300 N·m of integral demand — harmless for short
    step tests, but a windup limit-cycle generator on sustained targets.
    """

    elbow_integral_zone: float = 0.35
    """Elbow conditional-integration zone [rad] (anti-windup for far or
    unreachable targets)."""

    wrist_integral_zone: float = 0.175
    """Wrist conditional-integration zone [rad] (anti-windup)."""

    deriv_filter_alpha: float = 0.5
    """EMA coefficient for the derivative term (1 = unfiltered)."""

    elbow_grav_mass: float = 2.49
    """Moving mass below the elbow [kg] (disc 0.192 + forearm 1.9 + EE 0.4)."""

    elbow_grav_arm: float = 0.115
    """Combined CoM lever arm below the elbow [m].

    Calibrated against the measured static balance of the assembled 5-DOF USD
    (tension staircase, 2026-07-09): a 30 N elbow tension holds ≈ 43.7°, i.e.
    m·g·l ≈ 2.8–3.2 N·m.  The 3-DOF validation constant (0.26 m → 6.35 N·m)
    over-compensates roughly 2× on this robot and biases the set-point
    tracking upward.
    """

    wrist_grav_mass: float = 0.40
    """End-effector mass moved by the wrist [kg]."""

    wrist_grav_arm: float = 0.068
    """Wrist gravity lever arm [m] (half wrist-to-tool distance)."""

    elbow_lever: float = 0.0725
    """Elbow tension lever arm [m] for the antagonistic-pair distribution."""

    min_tension_elbow: float = 6.0
    """Elbow cable pre-tension [N] (Klein §3.2.4)."""

    min_tension_wrist: float = 5.0
    """Wrist cable pre-tension [N] (Klein §3.2.4)."""

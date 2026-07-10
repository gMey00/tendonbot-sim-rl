"""Tendon-driven actuator for the tensegrity arm.

Maps *tendon tensions* (N-dim) to *joint torques* (M-dim) via a constant
Jacobian-transpose matrix derived from the physical tendon attachment
geometry defined in ``threedof_manipulator.urdf.xacro``.

This module is robot-specific, not task-specific: any IsaacLab task that
uses a tendon-driven tensegrity robot imports :class:`TendonEffortActionCfg`
from here and plugs it into its action configuration.

Tendon arrangement (xacro Force frames)
----------------------------------------
* **Elbow pair** (antagonistic):
  - Tendon 0  (Force1): attachment at x = +0.0725 m on forearm → positive elbow torque
  - Tendon 1  (Force2): attachment at x = -0.0725 m on forearm → negative elbow torque

* **Wrist triplet** (120° spacing on r = 0.020 m circle, Klein §3.2.2 p.42):
  - Tendon 2  (Force3): (x, y) = (+0.010, +0.017321)
  - Tendon 3  (Force4): (x, y) = (-0.020,  0.0     )
  - Tendon 4  (Force5): (x, y) = (+0.010, -0.017321)

Torque mapping (zero-config Jacobian transpose)::

    τ_elbow   =  +0.0725·T₀  − 0.0725·T₁
    τ_wrist_y = −0.017321·T₂              + 0.017321·T₄
    τ_wrist_x =  +0.010·T₂   − 0.020·T₃  + 0.010·T₄

Tensions are non-negative.  The RL agent outputs actions in [-1, 1] which
are affine-mapped to [0, max_tension].

References
----------
* Klein, M. (2023). *Arbeitsraumanalyse, Simulation und Bewegungsplanung
  eines seilgetriebenen robotischen Manipulators*. Master's thesis,
  Friedrich-Alexander-Universität Erlangen-Nürnberg.  §3.2.4 (cable
  tension distribution), §3.3 (Gazebo force-vector simulation).
* Nemoto, T. et al. (referenced in thesis as [84]) — tendon-driven control
  principle using geometric cable-tension distribution.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


def compute_lower_arm_angle_and_rate(
    asset: Articulation, root_body_idx: int, forearm_body_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Effective elbow angle and angular rate of the lower arm (forearm).

    The physical antiparallelogram model has no single elbow joint — the
    forearm rotation is the *sum* of two linkage joint angles (rod + coupler).
    The correct measurement is therefore the forearm body's rotation relative
    to the upper arm (``root_link``), which the 4-bar constrains to a pure
    twist about the root X axis:

        θ_elbow = 2 · atan2(q_rel.x, q_rel.w),  q_rel = q_root⁻¹ ⊗ q_forearm

    This also stays meaningful if the loop-closure constraint drifts, unlike
    reading a single linkage joint angle.

    Returns
    -------
    angle : (num_envs,) elbow angle in radians, wrapped to (−π, π].
    rate : (num_envs,) relative angular velocity projected on the root X axis.
    """
    data = asset.data
    q_root = data.body_quat_w[:, root_body_idx]        # (B, 4) wxyz
    q_forearm = data.body_quat_w[:, forearm_body_idx]  # (B, 4)
    q_rel = quat_mul(quat_conjugate(q_root), q_forearm)
    angle = wrap_to_pi(2.0 * torch.atan2(q_rel[:, 1], q_rel[:, 0]))

    x_axis_w = quat_apply(q_root, torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(q_root.shape[0], 3))
    omega_rel_w = data.body_ang_vel_w[:, forearm_body_idx] - data.body_ang_vel_w[:, root_body_idx]
    rate = (omega_rel_w * x_axis_w).sum(dim=-1)
    return angle, rate

# ── Default Jacobian transpose (3 joints × 5 tendons) ─────────────────────
# Derived from xacro Force1–5 attachment coordinates (see module docstring).
DEFAULT_JACOBIAN_TRANSPOSE: list[list[float]] = [
    [+0.0725, -0.0725,  0.0,       0.0,      0.0],       # elbow
    [ 0.0,     0.0,     -0.017321,  0.0,     +0.017321],  # wrist_y (r=20 mm, Klein §3.2.2 p.42)
    [ 0.0,     0.0,     +0.010,    -0.020,   +0.010],     # wrist_x
]

NUM_TENDONS = 5


class TendonEffortAction(ActionTerm):
    """Action term that converts tendon tensions into joint effort commands.

    The RL agent outputs ``num_tendons`` raw actions in [-1, 1].  These are
    affine-mapped to non-negative tensions in [0, max_tension] and multiplied
    by the Jacobian transpose to obtain joint torques.

    The resulting torques are applied via ``set_joint_effort_target`` which
    requires an explicit actuator (e.g. ``IdealPDActuator`` with zero stiffness
    and damping) on the corresponding joints.
    """

    cfg: TendonEffortActionCfg
    _asset: Articulation

    def __init__(self, cfg: TendonEffortActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True,
        )
        self._num_joints = len(self._joint_ids)
        logger.info(
            "TendonEffortAction — joints: %s [%s], tendons: %d",
            self._joint_names, self._joint_ids, self.cfg.num_tendons,
        )

        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        jacobian_data = self.cfg.jacobian_transpose
        jacobian_tensor = torch.tensor(jacobian_data, dtype=torch.float32, device=self.device)
        expected_shape = (self._num_joints, self.cfg.num_tendons)
        if jacobian_tensor.shape != expected_shape:
            raise ValueError(
                f"jacobian_transpose shape {jacobian_tensor.shape} does not match "
                f"expected ({self._num_joints}, {self.cfg.num_tendons})"
            )
        # (1, num_joints, num_tendons) — broadcast across envs
        self._jacobian_t = jacobian_tensor.unsqueeze(0)

        # (1, num_tendons) — supports per-tendon saturation limits
        self._max_tension = _resolve_max_tension(self.cfg.max_tension, self.cfg.num_tendons, self.device)
        self._tension_scale = self._max_tension / 2.0
        self._tension_offset = self._max_tension / 2.0

        self._raw_actions = torch.zeros(self.num_envs, self.cfg.num_tendons, device=self.device)
        self._tensions = torch.zeros_like(self._raw_actions)
        self._joint_torques = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        return self.cfg.num_tendons

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_torques

    # ── Operations ─────────────────────────────────────────────────────────

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        # [-1, 1] → [0, max_tension] (per tendon)
        self._tensions = self._raw_actions * self._tension_scale + self._tension_offset
        self._tensions = torch.minimum(self._tensions.clamp(min=0.0), self._max_tension)
        # τ = J^T · T  — batched matmul: (B, J, T) @ (B, T, 1) → (B, J, 1)
        self._joint_torques = torch.bmm(
            self._jacobian_t.expand(self.num_envs, -1, -1),
            self._tensions.unsqueeze(-1),
        ).squeeze(-1)

    def apply_actions(self) -> None:
        self._asset.set_joint_effort_target(self._joint_torques, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._tensions[env_ids] = 0.0
        self._joint_torques[env_ids] = 0.0


@configclass
class TendonEffortActionCfg(ActionTermCfg):
    """Configuration for the tendon-driven arm action term.

    Drop this into any task's ``ActionsCfg`` to replace a standard
    joint-position/effort action with tendon-driven control.  The only
    task-side requirement is an explicit effort-passthrough actuator
    (``IdealPDActuator`` with stiffness=0, damping=0) on the target joints.

    Example
    -------
    .. code-block:: python

        from tensegrity_pick.robots import TendonEffortActionCfg, DEFAULT_JACOBIAN_TRANSPOSE

        @configclass
        class ActionsCfg:
            arm_tendon = TendonEffortActionCfg(
                asset_name="robot",
                joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
                jacobian_transpose=DEFAULT_JACOBIAN_TRANSPOSE,
            )
    """

    class_type: type[ActionTerm] = TendonEffortAction

    joint_names: list[str] = MISSING
    """Target joint names (order must match Jacobian rows)."""

    num_tendons: int = NUM_TENDONS
    """Number of tendon inputs (columns of J^T)."""

    max_tension: float | list[float] = 500.0
    """Maximum per-tendon tension in Newtons (scalar or one value per tendon).

    Raw RL actions in [-1, 1] are mapped to [0, max_tension] per tendon.

    .. note:: The hardware-derived per-tendon saturation is
       :data:`HW_TENDON_MAX_TENSIONS` (``[480, 480, 80, 80, 80]`` N — elbow
       after the 3:1 pulley, wrist direct-drive; Klein 2023 §3.2.5/§3.2.6 and
       ``doc/open_questions.md``).  The scalar 500 N default is kept for
       backward compatibility with existing sim-tendon training runs; switch
       to the hardware list when retraining (TODO M2).
    """

    jacobian_transpose: list[list[float]] = MISSING
    """Jacobian-transpose mapping (num_joints × num_tendons).

    Entry [i][j] gives the torque on joint *i* per unit tension in tendon *j*.
    Use :data:`DEFAULT_JACOBIAN_TRANSPOSE` for the standard 3-DOF arm geometry.
    """


# ── Physical-model tendon constants ───────────────────────────────────────
# Attachment positions in body-local frames, derived from the CAD reference
# points RP1 (upper arm) and RP4 (forearm) in build_tensegrity_arm_usd.py.
#
# root_link origin is at the articulation root (world z = 0 in zero-config).
# forearm_link origin is at world z = −0.4975 m (coupler pivot) in zero-config.
#
# Each sub-list is [x, y, z] in the body's local frame.
# Tendon 0 (left, +Y) and Tendon 1 (right, −Y).

ELBOW_TENDON_ROOT_OFFSETS: list[list[float]] = [
    [0.0, +0.0725, -0.34],   # T0 root attachment (RP1 left)
    [0.0, -0.0725, -0.34],   # T1 root attachment (RP1 right)
]

ELBOW_TENDON_FOREARM_OFFSETS: list[list[float]] = [
    [0.0, +0.0725, -0.02],   # T0 forearm attachment (RP4 left)
    [0.0, -0.0725, -0.02],   # T1 forearm attachment (RP4 right)
]

# Same physical attachment points (world z = −0.5175 at rest), expressed in
# the ELBOW-APPROX forearm frame whose origin sits at the elbow pivot
# (z = −0.430) instead of the four-bar coupler line (z = −0.4975).
ELBOW_TENDON_FOREARM_OFFSETS_APPROX: list[list[float]] = [
    [0.0, +0.0725, -0.0875],
    [0.0, -0.0725, -0.0875],
]

# Wrist-only Jacobian transpose (2 wrist joints × 3 wrist tendons).
# Rows: wrist_y, wrist_x.  Columns: T2, T3, T4.
WRIST_JACOBIAN_TRANSPOSE: list[list[float]] = [
    [-0.017321,  0.0,     +0.017321],   # wrist_y (r=20 mm, Klein §3.2.2 p.42)
    [+0.010,    -0.020,   +0.010],      # wrist_x
]

NUM_ELBOW_TENDONS = 2
NUM_WRIST_TENDONS = 3

# Hardware-derived per-tendon tension saturation [N] (Klein 2023 §3.2.5/§3.2.6,
# doc/open_questions.md §1–3).  The elbow tendons act on the linkage *after*
# the 3:1 pulley (Flaschenzug): 160 N motor-side saturation × 3 ≈ 480 N at the
# attachment point.  The wrist cables are direct-drive: 80 N continuous.
# Order matches the action layout: [T0_elbow, T1_elbow, T2, T3, T4].
HW_TENDON_MAX_TENSIONS: list[float] = [480.0, 480.0, 80.0, 80.0, 80.0]

# Elbow cable length range between the attachment points [m] over the ±70°
# elbow workspace.  Computed from the antiparallelogram closure condition and
# the attachment geometry in ``build_tensegrity_arm_usd.py`` (l_e = 0.150 m,
# k_e = 0.060 m, root attach (0, ±0.0725, −0.34), forearm attach local
# (0, ±0.0725, −0.02)): 0.1775 m at 0° elbow; 0.0913 m (short side) and
# 0.2577 m (long side) at ±70°.  A real cable cannot be wound shorter than
# its fully-spooled length nor paid out beyond its free length, so these act
# as hard actuation limits that keep the linkage inside its physical range.
ELBOW_CABLE_LENGTH_RANGE: tuple[float, float] = (0.0913, 0.2577)

# Antiparallelogram closure constants (build_tensegrity_arm_usd.py):
# rod length l_e, frame/coupler pivot spacing k_e, rest rod angle θ₀.
LINKAGE_ROD_LENGTH = 0.150
LINKAGE_PIVOT_SPACING = 0.060
LINKAGE_THETA_0 = math.asin(LINKAGE_PIVOT_SPACING / LINKAGE_ROD_LENGTH)


def antiparallelogram_closure_angle(rod_left_usd: torch.Tensor) -> torch.Tensor:
    """USD angle of ``rod_right``/``coupler_left`` from ``rod_left`` (closure eq).

    θ = rod_left + θ₀;  φ = θ + 2·atan2(−k·cosθ, l − k·sinθ);  result = φ + θ₀.
    Verified against a numeric continuation solve to machine precision
    (2026-07-09); the effective elbow angle is rod_left + result.
    """
    theta = rod_left_usd + LINKAGE_THETA_0
    phi = theta + 2.0 * torch.atan2(
        -LINKAGE_PIVOT_SPACING * torch.cos(theta),
        LINKAGE_ROD_LENGTH - LINKAGE_PIVOT_SPACING * torch.sin(theta),
    )
    return phi + LINKAGE_THETA_0


def _resolve_max_tension(max_tension: float | list[float], num_tendons: int, device: str) -> torch.Tensor:
    """Return per-tendon max tension as a (1, num_tendons) tensor."""
    if isinstance(max_tension, (int, float)):
        values = [float(max_tension)] * num_tendons
    else:
        values = [float(v) for v in max_tension]
        if len(values) != num_tendons:
            raise ValueError(
                f"max_tension list has {len(values)} entries, expected {num_tendons}"
            )
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


class PhysicalTendonEffortAction(ActionTerm):
    """Action term for the physical antiparallelogram model with body-force elbow tendons.

    Elbow tendons (T0, T1) are applied as external body forces at the actual
    cable attachment points on ``root_link`` and ``forearm_link``.  This captures
    the configuration-dependent force directions of the real 4-bar linkage,
    unlike the constant Jacobian-transpose approximation.

    Wrist tendons (T2–T4) use a constant Jacobian-transpose mapping to joint
    effort targets (identical to :class:`TendonEffortAction`).

    The RL agent outputs 5 actions in [-1, 1] which are affine-mapped to
    non-negative tensions in [0, max_tension] (per tendon).

    Cable-length limits (physical securing)
    ---------------------------------------
    A real elbow cable is inextensible and has a finite spooled length, so the
    distance between its attachment points is physically bounded
    (:attr:`PhysicalTendonEffortActionCfg.elbow_cable_length_range`).  Two
    mechanisms enforce this in simulation:

    * **Wind-up stop (min length):** the commanded tension fades to zero over
      ``cable_slack_ramp`` metres above the minimum length — the spool cannot
      wind the cable any shorter, so no active pull below the limit.
    * **Stretch stop (max length):** when a cable exceeds its maximum free
      length, a stiff passive spring–damper tension is added along the cable.
      This models cable inextensibility and prevents the policy from driving
      the linkage past its geometric workspace (the "linkage breaking"
      exploit observed in RL training).
    """

    cfg: PhysicalTendonEffortActionCfg
    _asset: Articulation

    def __init__(self, cfg: PhysicalTendonEffortActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # ── Resolve wrist joints (for effort targets) ──────────────────────
        self._wrist_joint_ids, self._wrist_joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True,
        )
        self._num_wrist_joints = len(self._wrist_joint_ids)
        if self._num_wrist_joints == self._asset.num_joints:
            self._wrist_joint_ids = slice(None)

        # ── Resolve body indices ───────────────────────────────────────────
        self._root_body_idx = self._asset.find_bodies(self.cfg.root_body_name)[0][0]
        self._forearm_body_idx = self._asset.find_bodies(self.cfg.forearm_body_name)[0][0]
        self._elbow_body_ids = [self._root_body_idx, self._forearm_body_idx]

        logger.info(
            "PhysicalTendonEffortAction — wrist joints: %s [%s], "
            "root body: %s [%d], forearm body: %s [%d]",
            self._wrist_joint_names, self._wrist_joint_ids,
            self.cfg.root_body_name, self._root_body_idx,
            self.cfg.forearm_body_name, self._forearm_body_idx,
        )

        # ── Attachment offsets (constant in body-local frames) ─────────────
        self._root_attach_local = torch.tensor(
            self.cfg.elbow_tendon_root_offsets, dtype=torch.float32, device=self.device,
        )  # (2, 3)
        self._forearm_attach_local = torch.tensor(
            self.cfg.elbow_tendon_forearm_offsets, dtype=torch.float32, device=self.device,
        )  # (2, 3)

        # ── Wrist Jacobian transpose ───────────────────────────────────────
        wrist_jac = torch.tensor(
            self.cfg.wrist_jacobian_transpose, dtype=torch.float32, device=self.device,
        )
        expected_shape = (self._num_wrist_joints, NUM_WRIST_TENDONS)
        if wrist_jac.shape != expected_shape:
            raise ValueError(
                f"wrist_jacobian_transpose shape {wrist_jac.shape} != expected {expected_shape}"
            )
        self._wrist_jac_t = wrist_jac.unsqueeze(0)  # (1, num_wrist_joints, 3)

        # ── Tension mapping (per tendon) ───────────────────────────────────
        self._max_tension = _resolve_max_tension(self.cfg.max_tension, self.cfg.num_tendons, self.device)
        self._tension_scale = self._max_tension / 2.0
        self._tension_offset = self._max_tension / 2.0

        # ── Cable length limits ────────────────────────────────────────────
        self._cable_len_min, self._cable_len_max = self.cfg.elbow_cable_length_range
        if self._cable_len_min >= self._cable_len_max:
            raise ValueError(
                f"elbow_cable_length_range min {self._cable_len_min} must be < max {self._cable_len_max}"
            )

        # ── Kinematic closure mimic (reduced-coordinate four-bar) ──────────
        if self.cfg.mimic_closure:
            self._rod_left_id = self._asset.find_joints(["rod_left_joint"], preserve_order=True)[0]
            self._mimic_ids = self._asset.find_joints(
                ["rod_right_joint", "coupler_left_joint"], preserve_order=True,
            )[0]
        else:
            self._rod_left_id = None
            self._mimic_ids = None

        # ── Pre-allocated buffers ──────────────────────────────────────────
        B = self.num_envs
        self._raw_actions = torch.zeros(B, self.cfg.num_tendons, device=self.device)
        self._tensions = torch.zeros_like(self._raw_actions)
        self._elbow_tensions = torch.zeros(B, NUM_ELBOW_TENDONS, device=self.device)
        self._wrist_tensions = torch.zeros(B, NUM_WRIST_TENDONS, device=self.device)
        self._wrist_torques = torch.zeros(B, self._num_wrist_joints, device=self.device)
        # Body force buffers: (B, 2_bodies, 3)
        self._body_forces = torch.zeros(B, 2, 3, device=self.device)
        self._body_torques = torch.zeros(B, 2, 3, device=self.device)
        # Cable state (updated every sim step in apply_actions)
        self._cable_lengths = torch.full(
            (B, NUM_ELBOW_TENDONS), 0.5 * (self._cable_len_min + self._cable_len_max), device=self.device,
        )
        self._cable_length_rates = torch.zeros(B, NUM_ELBOW_TENDONS, device=self.device)
        self._applied_elbow_tensions = torch.zeros(B, NUM_ELBOW_TENDONS, device=self.device)
        # First-order motor/spool lag state (see cfg.tension_time_constant)
        self._elbow_tensions_filt = torch.zeros(B, NUM_ELBOW_TENDONS, device=self.device)
        self._wrist_torques_filt = torch.zeros(B, self._num_wrist_joints, device=self.device)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        return self.cfg.num_tendons

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._tensions

    @property
    def cable_lengths(self) -> torch.Tensor:
        """Current elbow cable lengths between attachment points.  Shape (B, 2)."""
        return self._cable_lengths

    @property
    def cable_length_rates(self) -> torch.Tensor:
        """Current elbow cable length rates (positive = extending).  Shape (B, 2)."""
        return self._cable_length_rates

    @property
    def applied_elbow_tensions(self) -> torch.Tensor:
        """Actually applied elbow tensions (commanded × wind-stop + stretch stop).  Shape (B, 2)."""
        return self._applied_elbow_tensions

    # ── Operations ─────────────────────────────────────────────────────────

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        # [-1, 1] → [0, max_tension] (per tendon)
        self._tensions = self._raw_actions * self._tension_scale + self._tension_offset
        self._tensions = torch.minimum(self._tensions.clamp(min=0.0), self._max_tension)
        # Split into elbow (T0, T1) and wrist (T2, T3, T4) tensions
        self._elbow_tensions = self._tensions[:, :NUM_ELBOW_TENDONS]
        self._wrist_tensions = self._tensions[:, NUM_ELBOW_TENDONS:]
        # Wrist: τ = J^T · T  — batched matmul: (B, J, 3) @ (B, 3, 1) → (B, J, 1)
        self._wrist_torques = torch.bmm(
            self._wrist_jac_t.expand(self.num_envs, -1, -1),
            self._wrist_tensions.unsqueeze(-1),
        ).squeeze(-1)

    def apply_actions(self) -> None:
        self._apply_tendon_forces(self._elbow_tensions, self._wrist_torques)

    # ── Shared force-application machinery ─────────────────────────────────

    def _compute_cable_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute world-frame cable geometry and rates from current body states.

        Returns
        -------
        root_off_w, forearm_off_w : (B, 2, 3)
            World-frame attachment offsets from the respective link origins.
        cable_dir_w : (B, 2, 3)
            Unit vector from forearm attachment to root attachment.
        cable_len : (B, 2, 1)
            Cable lengths between attachment points.
        cable_len_rate : (B, 2, 1)
            Length rates (positive = cable extending).
        """
        B = self.num_envs
        data = self._asset.data

        root_pos_w = data.body_pos_w[:, self._root_body_idx]          # (B, 3)
        root_quat_w = data.body_quat_w[:, self._root_body_idx]        # (B, 4) wxyz
        forearm_pos_w = data.body_pos_w[:, self._forearm_body_idx]    # (B, 3)
        forearm_quat_w = data.body_quat_w[:, self._forearm_body_idx]  # (B, 4)

        # Rotate local offsets to world frame — (B, 2_tendons, 3)
        root_off_local = self._root_attach_local.unsqueeze(0).expand(B, -1, -1)
        forearm_off_local = self._forearm_attach_local.unsqueeze(0).expand(B, -1, -1)
        root_q_exp = root_quat_w.unsqueeze(1).expand(-1, NUM_ELBOW_TENDONS, -1)
        forearm_q_exp = forearm_quat_w.unsqueeze(1).expand(-1, NUM_ELBOW_TENDONS, -1)

        root_off_w = quat_apply(root_q_exp, root_off_local)            # (B, 2, 3)
        forearm_off_w = quat_apply(forearm_q_exp, forearm_off_local)   # (B, 2, 3)

        root_attach_w = root_pos_w.unsqueeze(1) + root_off_w           # (B, 2, 3)
        forearm_attach_w = forearm_pos_w.unsqueeze(1) + forearm_off_w  # (B, 2, 3)

        cable_vec_w = root_attach_w - forearm_attach_w                 # (B, 2, 3)
        cable_len = cable_vec_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cable_dir_w = cable_vec_w / cable_len                          # (B, 2, 3)

        # Attachment-point velocities: v = v_link + ω × r_offset
        root_lin_w = data.body_lin_vel_w[:, self._root_body_idx].unsqueeze(1)        # (B, 1, 3)
        root_ang_w = data.body_ang_vel_w[:, self._root_body_idx].unsqueeze(1)        # (B, 1, 3)
        forearm_lin_w = data.body_lin_vel_w[:, self._forearm_body_idx].unsqueeze(1)
        forearm_ang_w = data.body_ang_vel_w[:, self._forearm_body_idx].unsqueeze(1)

        v_root_att = root_lin_w + torch.cross(root_ang_w.expand(-1, NUM_ELBOW_TENDONS, -1), root_off_w, dim=-1)
        v_forearm_att = forearm_lin_w + torch.cross(
            forearm_ang_w.expand(-1, NUM_ELBOW_TENDONS, -1), forearm_off_w, dim=-1,
        )
        # d|cable_vec|/dt = dir · d(cable_vec)/dt
        cable_len_rate = ((v_root_att - v_forearm_att) * cable_dir_w).sum(dim=-1, keepdim=True)

        return root_off_w, forearm_off_w, cable_dir_w, cable_len, cable_len_rate

    def _apply_tendon_forces(self, elbow_tensions: torch.Tensor, wrist_torques: torch.Tensor) -> None:
        """Apply elbow tensions as body forces (with cable-length limits) and wrist efforts.

        Parameters
        ----------
        elbow_tensions : (B, 2) commanded elbow tendon tensions [N], ≥ 0.
        wrist_torques : (B, num_wrist_joints) wrist joint effort targets [N·m].
        """
        # ── Anti-stiction solver dither ────────────────────────────────────
        # The GPU tensor-API external-force path exhibits a stiction-like
        # breakaway (~10–20 N·m equivalent) on settled articulations: dynamic
        # actuation is exact, but quasi-static tensions creep instead of
        # rotating the linkage (measured 2026-07-10; not sleep — thresholds
        # zeroed at articulation and body level — and not authored friction).
        # A zero-mean joint-velocity dither of 1e-3 rad/s (≈ 0.06 °/s, far
        # below any signal) measurably restores the quasi-static response;
        # real cable drives use dither against stiction for the same reason.
        if self.cfg.solver_dither > 0.0:
            jv = self._asset.data.joint_vel
            self._asset.write_joint_velocity_to_sim(
                jv + self.cfg.solver_dither * (2.0 * torch.rand_like(jv) - 1.0)
            )

        # ── Motor/spool dynamics: first-order tension lag ──────────────────
        # A real motor cannot step the cable tension instantaneously.  Without
        # this lag, a 0→480 N command in a single physics step produces
        # constraint-impulse spikes through the linkage (and the light wrist
        # dummy link) that destabilise the solver.
        tau = self.cfg.tension_time_constant
        if tau > 0.0:
            alpha = min(self._env.physics_dt / tau, 1.0)
            self._elbow_tensions_filt += alpha * (elbow_tensions - self._elbow_tensions_filt)
            self._wrist_torques_filt += alpha * (wrist_torques - self._wrist_torques_filt)
            elbow_tensions = self._elbow_tensions_filt
            wrist_torques = self._wrist_torques_filt

        root_off_w, forearm_off_w, cable_dir_w, cable_len, cable_len_rate = self._compute_cable_state()

        # ── Cable-length security ──────────────────────────────────────────
        # Wind-up stop: commanded tension fades to zero as the cable
        # approaches the fully-spooled minimum length.
        ramp = max(self.cfg.cable_slack_ramp, 1e-6)
        wind_scale = ((cable_len - self._cable_len_min) / ramp).clamp(0.0, 1.0)   # (B, 2, 1)
        T_active = elbow_tensions.unsqueeze(-1) * wind_scale                       # (B, 2, 1)

        # Stretch stop: passive spring–damper tension when the cable exceeds
        # its maximum free length (cable inextensibility).
        stretch = (cable_len - self._cable_len_max).clamp(min=0.0)                 # (B, 2, 1)
        engaged = stretch > 0.0
        T_stop = (
            self.cfg.cable_stop_stiffness * stretch
            + self.cfg.cable_stop_damping * cable_len_rate * engaged
        ).clamp(min=0.0, max=self.cfg.cable_stop_max_tension) * engaged

        T = T_active + T_stop                                                      # (B, 2, 1)

        # Record cable state for observations / diagnostics
        self._cable_lengths[:] = cable_len.squeeze(-1)
        self._cable_length_rates[:] = cable_len_rate.squeeze(-1)
        self._applied_elbow_tensions[:] = T.squeeze(-1)

        # ── Per-tendon forces in world frame ───────────────────────────────
        F_forearm_w = T * cable_dir_w                               # (B, 2, 3)
        F_root_w = -F_forearm_w                                     # Newton's 3rd law

        # Per-tendon torques about the link origins (τ = r × F).  PhysX applies
        # the wrench at the link frame origin when no position is given, so the
        # force + this torque are equivalent to the force acting at the
        # attachment point.
        tau_forearm_w = torch.cross(forearm_off_w, F_forearm_w, dim=-1)  # (B, 2, 3)
        tau_root_w = torch.cross(root_off_w, F_root_w, dim=-1)           # (B, 2, 3)

        # ── Sum over tendons → net per body, apply via wrench composer ────
        self._body_forces[:, 0] = F_root_w.sum(dim=1)
        self._body_forces[:, 1] = F_forearm_w.sum(dim=1)
        self._body_torques[:, 0] = tau_root_w.sum(dim=1)
        self._body_torques[:, 1] = tau_forearm_w.sum(dim=1)
        # Invalidate the wrench composer's cached link poses so the
        # global→local frame conversion uses current body orientations.
        self._asset._permanent_wrench_composer._link_poses_updated = False
        self._asset.set_external_force_and_torque(
            forces=self._body_forces,
            torques=self._body_torques,
            body_ids=self._elbow_body_ids,
            is_global=True,
        )

        # ── Wrist joint efforts ────────────────────────────────────────────
        self._asset.set_joint_effort_target(
            wrist_torques, joint_ids=self._wrist_joint_ids,
        )

        # ── Kinematic closure: slave rod_right / coupler_left to rod_left ──
        if self._mimic_ids is not None:
            rod_left_q = self._asset.data.joint_pos[:, self._rod_left_id]      # (B, 1)
            target = antiparallelogram_closure_angle(rod_left_q)               # (B, 1)
            self._asset.set_joint_position_target(
                target.expand(-1, 2), joint_ids=self._mimic_ids,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._tensions[env_ids] = 0.0
        self._elbow_tensions[env_ids] = 0.0
        self._wrist_tensions[env_ids] = 0.0
        self._wrist_torques[env_ids] = 0.0
        self._body_forces[env_ids] = 0.0
        self._body_torques[env_ids] = 0.0
        self._cable_lengths[env_ids] = 0.5 * (self._cable_len_min + self._cable_len_max)
        self._cable_length_rates[env_ids] = 0.0
        self._applied_elbow_tensions[env_ids] = 0.0
        self._elbow_tensions_filt[env_ids] = 0.0
        self._wrist_torques_filt[env_ids] = 0.0


@configclass
class PhysicalTendonEffortActionCfg(ActionTermCfg):
    """Configuration for the physical-model tendon action term.

    Elbow tendons apply body forces at attachment points on ``root_link`` and
    ``forearm_link``.  Wrist tendons use a Jacobian-transpose mapping to joint
    effort targets.

    Example
    -------
    .. code-block:: python

        from tensegrity_pick.robots import (
            PhysicalTendonEffortActionCfg,
            ELBOW_TENDON_ROOT_OFFSETS,
            ELBOW_TENDON_FOREARM_OFFSETS,
            WRIST_JACOBIAN_TRANSPOSE,
        )

        @configclass
        class ActionsCfg:
            arm_tendon = PhysicalTendonEffortActionCfg(
                asset_name="robot",
                joint_names=["wrist_y_joint", "wrist_x_joint"],
                wrist_jacobian_transpose=WRIST_JACOBIAN_TRANSPOSE,
            )
    """

    class_type: type[ActionTerm] = PhysicalTendonEffortAction

    joint_names: list[str] = MISSING
    """Wrist joint names for effort targets (order must match wrist Jacobian rows)."""

    num_tendons: int = NUM_TENDONS
    """Total number of tendon inputs (2 elbow + 3 wrist = 5)."""

    max_tension: float | list[float] = HW_TENDON_MAX_TENSIONS
    """Maximum per-tendon tension in Newtons (scalar or one value per tendon).

    Default is the hardware-derived saturation :data:`HW_TENDON_MAX_TENSIONS`
    (``[480, 480, 80, 80, 80]`` N): elbow attachment-point force after the 3:1
    pulley (160 N motor-side × 3), wrist direct-drive continuous limit (80 N).
    Klein (2023) §3.2.5/§3.2.6; see ``doc/open_questions.md`` §1–3.
    """

    elbow_cable_length_range: tuple[float, float] = ELBOW_CABLE_LENGTH_RANGE
    """(min, max) elbow cable length between attachment points in metres.

    Defaults to the geometric range over the ±70° elbow workspace
    (:data:`ELBOW_CABLE_LENGTH_RANGE`).  Commanded tension fades to zero at the
    minimum (spool wind-up stop); a passive spring–damper stop engages beyond
    the maximum (cable inextensibility).
    """

    cable_slack_ramp: float = 0.01
    """Length band [m] above the minimum cable length over which the commanded
    tension linearly fades to zero (wind-up stop smoothing)."""

    cable_stop_stiffness: float = 20000.0
    """Stretch-stop spring stiffness [N/m] applied when a cable exceeds its
    maximum free length (100 N per 5 mm of overstretch)."""

    cable_stop_damping: float = 200.0
    """Stretch-stop damping [N·s/m] on the cable length rate while overstretched."""

    cable_stop_max_tension: float = 500.0
    """Upper bound [N] on the passive stretch-stop tension.

    Bounded near the hardware drive maximum (480 N): an unbounded stop turns
    boundary overshoot into kN-scale impulses through the closed chain, which
    is precisely the load case that tears/branch-flips the loop closure.
    """

    tension_time_constant: float = 0.05
    """First-order lag time constant [s] modelling motor/spool dynamics.

    Applied to the commanded elbow tensions and wrist efforts at the physics
    rate.  A real Maxon EC60 + spool cannot step the cable force
    instantaneously; without the lag, single-step 0→480 N commands produce
    constraint-impulse spikes through the closed-chain linkage.  Set to 0 to
    disable.
    """

    solver_dither: float = 0.0
    """EXPERIMENTAL: zero-mean joint-velocity dither [rad/s] per physics step.

    Anti-stiction knob for the GPU tensor-API external-force path, which
    develops a ~10–20 N·m breakaway on settled articulations (dynamic
    actuation exact, quasi-static tensions creep; measured 2026-07-10 —
    neither sleep thresholds nor authored friction).  An externally injected
    ±1e-3 rad/s jiggle measurably unlocked quasi-static motion in probes, but
    the same write through this action path did not reproduce the effect
    reliably — default OFF pending a deeper PhysX investigation.
    """

    mimic_closure: bool = False
    """Enforce the four-bar closure kinematically in reduced coordinates.

    Every physics step, ``rod_right_joint`` and ``coupler_left_joint`` receive
    position targets from the analytic closure equation of ``rod_left_joint``
    (requires the stiff implicit ``linkage_mimic`` actuator group and the
    ``rigidify_linkage_closure`` prestartup event that disables the PhysX
    loop-closure joint).  The maximal-coordinate closure joint measurably
    absorbs cable torque as elastic drift, admits absorbing parallelogram
    branch-flips, and degrades force response of quiescent articulations
    (2026-07-10); the kinematic mimic keeps the four-bar rigid with
    ``rod_left`` as the free cable-driven DOF.
    """

    root_body_name: str = "root_link"
    """Body name for the upper arm (tendon origin)."""

    forearm_body_name: str = "forearm_link"
    """Body name for the forearm (tendon insertion)."""

    elbow_tendon_root_offsets: list[list[float]] = ELBOW_TENDON_ROOT_OFFSETS
    """Per-tendon attachment offsets on root_link in body-local frame.  Shape (2, 3)."""

    elbow_tendon_forearm_offsets: list[list[float]] = ELBOW_TENDON_FOREARM_OFFSETS
    """Per-tendon attachment offsets on forearm_link in body-local frame.  Shape (2, 3)."""

    wrist_jacobian_transpose: list[list[float]] = WRIST_JACOBIAN_TRANSPOSE
    """Wrist Jacobian transpose (num_wrist_joints × 3).  Default covers wrist_y, wrist_x."""

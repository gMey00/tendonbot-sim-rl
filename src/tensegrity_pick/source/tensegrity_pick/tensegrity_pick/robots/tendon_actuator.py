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
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)

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

        self._max_tension = self.cfg.max_tension
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
        # [-1, 1] → [0, max_tension]
        self._tensions = self._raw_actions * self._tension_scale + self._tension_offset
        self._tensions.clamp_(min=0.0, max=self._max_tension)
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

    max_tension: float = 500.0
    """Maximum per-tendon tension in Newtons.

    Raw RL actions in [-1, 1] are mapped to [0, max_tension].

    .. note:: TODO(motor-params) — This value is derived from Maxon EC60
       motor torque (0.401 Nm continuous) × cable-wrapping gearing ÷ spool
       radius.  The elbow cables wrap 3× around rollers (3:1 mechanical
       advantage).  Confirm spool radius from Klein (2023) hardware section.
       See ``doc/open_questions.md`` for details.
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

# Wrist-only Jacobian transpose (2 wrist joints × 3 wrist tendons).
# Rows: wrist_y, wrist_x.  Columns: T2, T3, T4.
WRIST_JACOBIAN_TRANSPOSE: list[list[float]] = [
    [-0.017321,  0.0,     +0.017321],   # wrist_y (r=20 mm, Klein §3.2.2 p.42)
    [+0.010,    -0.020,   +0.010],      # wrist_x
]

NUM_ELBOW_TENDONS = 2
NUM_WRIST_TENDONS = 3


class PhysicalTendonEffortAction(ActionTerm):
    """Action term for the physical antiparallelogram model with body-force elbow tendons.

    Elbow tendons (T0, T1) are applied as external body forces at the actual
    cable attachment points on ``root_link`` and ``forearm_link``.  This captures
    the configuration-dependent force directions of the real 4-bar linkage,
    unlike the constant Jacobian-transpose approximation.

    Wrist tendons (T2–T4) use a constant Jacobian-transpose mapping to joint
    effort targets (identical to :class:`TendonEffortAction`).

    The RL agent outputs 5 actions in [-1, 1] which are affine-mapped to
    non-negative tensions in [0, max_tension].
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

        # ── Tension mapping ────────────────────────────────────────────────
        self._max_tension = self.cfg.max_tension
        self._tension_scale = self._max_tension / 2.0
        self._tension_offset = self._max_tension / 2.0

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

    # ── Operations ─────────────────────────────────────────────────────────

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        # [-1, 1] → [0, max_tension]
        self._tensions = self._raw_actions * self._tension_scale + self._tension_offset
        self._tensions.clamp_(min=0.0, max=self._max_tension)
        # Split into elbow (T0, T1) and wrist (T2, T3, T4) tensions
        self._elbow_tensions = self._tensions[:, :NUM_ELBOW_TENDONS]
        self._wrist_tensions = self._tensions[:, NUM_ELBOW_TENDONS:]
        # Wrist: τ = J^T · T  — batched matmul: (B, J, 3) @ (B, 3, 1) → (B, J, 1)
        self._wrist_torques = torch.bmm(
            self._wrist_jac_t.expand(self.num_envs, -1, -1),
            self._wrist_tensions.unsqueeze(-1),
        ).squeeze(-1)

    def apply_actions(self) -> None:
        B = self.num_envs

        # ── 1. Get body transforms (link frame) ───────────────────────────
        root_pos_w = self._asset.data.body_pos_w[:, self._root_body_idx]        # (B, 3)
        root_quat_w = self._asset.data.body_quat_w[:, self._root_body_idx]      # (B, 4) wxyz
        forearm_pos_w = self._asset.data.body_pos_w[:, self._forearm_body_idx]   # (B, 3)
        forearm_quat_w = self._asset.data.body_quat_w[:, self._forearm_body_idx] # (B, 4)

        # ── 2. Rotate local offsets to world frame ─────────────────────────
        # Expand: (B, 2_tendons, 3) for batched quat_apply
        root_off_local = self._root_attach_local.unsqueeze(0).expand(B, -1, -1)
        forearm_off_local = self._forearm_attach_local.unsqueeze(0).expand(B, -1, -1)
        root_q_exp = root_quat_w.unsqueeze(1).expand(-1, NUM_ELBOW_TENDONS, -1)
        forearm_q_exp = forearm_quat_w.unsqueeze(1).expand(-1, NUM_ELBOW_TENDONS, -1)

        root_off_w = quat_apply(root_q_exp, root_off_local)        # (B, 2, 3)
        forearm_off_w = quat_apply(forearm_q_exp, forearm_off_local) # (B, 2, 3)

        # World-frame attachment positions
        root_attach_w = root_pos_w.unsqueeze(1) + root_off_w       # (B, 2, 3)
        forearm_attach_w = forearm_pos_w.unsqueeze(1) + forearm_off_w  # (B, 2, 3)

        # ── 3. Cable directions (world frame) ─────────────────────────────
        cable_vec_w = root_attach_w - forearm_attach_w              # (B, 2, 3)
        cable_len = cable_vec_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cable_dir_w = cable_vec_w / cable_len                       # (B, 2, 3)

        # ── 4. Per-tendon forces in world frame ───────────────────────────
        T = self._elbow_tensions.unsqueeze(-1)                      # (B, 2, 1)
        F_forearm_w = T * cable_dir_w                               # (B, 2, 3)
        F_root_w = -F_forearm_w                                     # Newton's 3rd law

        # ── 5. Per-tendon torques in world frame (τ = r × F) ──────────────
        tau_forearm_w = torch.cross(forearm_off_w, F_forearm_w, dim=-1)  # (B, 2, 3)
        tau_root_w = torch.cross(root_off_w, F_root_w, dim=-1)          # (B, 2, 3)

        # ── 6. Sum over tendons → net per body ────────────────────────────
        net_F_root = F_root_w.sum(dim=1)           # (B, 3)
        net_F_forearm = F_forearm_w.sum(dim=1)      # (B, 3)
        net_tau_root = tau_root_w.sum(dim=1)        # (B, 3)
        net_tau_forearm = tau_forearm_w.sum(dim=1)   # (B, 3)

        # ── 7. Apply body forces via wrench composer ──────────────────────
        self._body_forces[:, 0] = net_F_root
        self._body_forces[:, 1] = net_F_forearm
        self._body_torques[:, 0] = net_tau_root
        self._body_torques[:, 1] = net_tau_forearm
        # Invalidate the wrench composer's cached link poses so the
        # global→local frame conversion uses current body orientations.
        self._asset._permanent_wrench_composer._link_poses_updated = False
        self._asset.set_external_force_and_torque(
            forces=self._body_forces,
            torques=self._body_torques,
            body_ids=self._elbow_body_ids,
            is_global=True,
        )

        # ── 8. Wrist joint efforts ────────────────────────────────────────
        self._asset.set_joint_effort_target(
            self._wrist_torques, joint_ids=self._wrist_joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._tensions[env_ids] = 0.0
        self._elbow_tensions[env_ids] = 0.0
        self._wrist_tensions[env_ids] = 0.0
        self._wrist_torques[env_ids] = 0.0
        self._body_forces[env_ids] = 0.0
        self._body_torques[env_ids] = 0.0


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

    max_tension: float = 500.0
    """Maximum per-tendon tension in Newtons.

    .. note:: TODO(motor-params) — See :class:`TendonEffortActionCfg` and
       ``doc/open_questions.md`` for the motor-gearing derivation.
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

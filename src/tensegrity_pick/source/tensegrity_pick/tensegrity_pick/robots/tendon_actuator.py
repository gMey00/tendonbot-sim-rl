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

* **Wrist triplet** (120° spacing on r ≈ 0.016 m circle):
  - Tendon 2  (Force3): (x, y) = (+0.008, +0.013856)
  - Tendon 3  (Force4): (x, y) = (-0.016,  0.0     )
  - Tendon 4  (Force5): (x, y) = (+0.008, -0.013856)

Torque mapping (zero-config Jacobian transpose)::

    τ_elbow   =  +0.0725·T₀  − 0.0725·T₁
    τ_wrist_y = −0.013856·T₂              + 0.013856·T₄
    τ_wrist_x =  +0.008·T₂   − 0.016·T₃  + 0.008·T₄

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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)

# ── Default Jacobian transpose (3 joints × 5 tendons) ─────────────────────
# Derived from xacro Force1–5 attachment coordinates (see module docstring).
DEFAULT_JACOBIAN_TRANSPOSE: list[list[float]] = [
    [+0.0725, -0.0725,  0.0,       0.0,      0.0],       # elbow
    [ 0.0,     0.0,     -0.013856,  0.0,     +0.013856],  # wrist_y
    [ 0.0,     0.0,     +0.008,    -0.016,   +0.008],     # wrist_x
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
    """

    jacobian_transpose: list[list[float]] = MISSING
    """Jacobian-transpose mapping (num_joints × num_tendons).

    Entry [i][j] gives the torque on joint *i* per unit tension in tendon *j*.
    Use :data:`DEFAULT_JACOBIAN_TRANSPOSE` for the standard 3-DOF arm geometry.
    """

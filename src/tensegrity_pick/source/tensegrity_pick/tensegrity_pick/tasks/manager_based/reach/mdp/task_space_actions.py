"""Hardened task-space action terms (iteration 21).

Implements the research report's Rank-1 fixes for the task-space controllers'
pose errors (`doc/reports/RESEARCH_REPORT_taskspace_pose_error.md`):

1. **6-D continuous rotation action for absolute differential IK** —
   Isaac Lab's ``DifferentialInverseKinematicsAction`` in absolute mode consumes
   the policy's raw Gaussian 4-vector as a quaternion *without normalization*
   (source-confirmed).  Quaternions double-cover SO(3) and are discontinuous as
   a ≤4-D chart, a documented anti-pattern for learned rotations (Zhou et al.,
   CVPR 2019; Schuck et al., ICLR 2026).  ``SixDRotDiffIKAction`` instead takes
   a 9-D action (3-D position + 6-D rotation), maps the 6-D part to a rotation
   matrix via Gram–Schmidt and hands the controller a *unit* quaternion.

2. **EMA smoothing for task-space actions** — the joint action space is EMA
   smoothed (α=0.2) but the task-space spaces were not; smoothness is a primary
   driver of action-space quality (Aljalbout et al., RA-L 2024).  All classes
   here accept ``ema_alpha`` (None → off) applied to the *raw* policy actions
   with per-env reset handling.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING  # noqa: F401

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
    OperationalSpaceControllerAction,
)
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


def _rotmat_from_6d(r6: torch.Tensor) -> torch.Tensor:
    """Map a 6-D rotation representation to a rotation matrix (Gram–Schmidt).

    ``r6[:, :3]`` / ``r6[:, 3:]`` are the (unnormalised) first two columns of
    the target rotation matrix (Zhou et al. 2019).  A tiny constant bias keeps
    the construction non-degenerate when the policy outputs near-zero vectors
    (e.g. the very first exploration steps).
    """
    a1 = r6[:, :3] + torch.tensor([1.0e-3, 0.0, 0.0], device=r6.device)
    a2 = r6[:, 3:] + torch.tensor([0.0, 1.0e-3, 0.0], device=r6.device)
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1.0e-6)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1.0e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)  # columns


class _RawActionEMAMixin:
    """Per-env EMA on the raw policy actions (``a_t = α·a + (1−α)·a_{t−1}``).

    ``ema_alpha`` comes from the cfg; None disables (pass-through).  On env
    reset the state is cleared so the first post-reset action passes unfiltered.
    """

    def _ema_setup(self):
        self._ema_alpha = getattr(self.cfg, "ema_alpha", None)
        if self._ema_alpha is not None:
            self._ema_prev = torch.zeros(self.num_envs, self.action_dim, device=self.device)
            self._ema_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _maybe_ema(self, actions: torch.Tensor) -> torch.Tensor:
        if self._ema_alpha is None:
            return actions
        fresh = ~self._ema_started
        if fresh.any():
            self._ema_prev[fresh] = actions[fresh]
            self._ema_started |= True
        self._ema_prev = self._ema_alpha * actions + (1.0 - self._ema_alpha) * self._ema_prev
        return self._ema_prev

    def _ema_reset(self, env_ids: Sequence[int] | None):
        if self._ema_alpha is not None:
            self._ema_started[env_ids] = False


class SixDRotDiffIKAction(_RawActionEMAMixin, DifferentialInverseKinematicsAction):
    """Absolute differential-IK action with a 6-D continuous rotation input.

    Action layout (9-D): ``[x, y, z, r1(3), r2(3)]`` — position in the base
    frame plus the first two rotation-matrix columns.  Converted to the parent
    controller's 7-D ``[pos, unit-quat]`` command.  Only meaningful with
    ``use_relative_mode=False`` and ``command_type="pose"``.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.cfg.controller.use_relative_mode or self.cfg.controller.command_type != "pose":
            raise ValueError("SixDRotDiffIKAction requires an absolute pose IK controller.")
        self._ema_setup()

    @property
    def action_dim(self) -> int:
        return 9  # 3 position + 6-D rotation

    def process_actions(self, actions: torch.Tensor):
        actions = self._maybe_ema(actions)
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale
        pos = self._processed_actions[:, :3]
        quat = math_utils.quat_from_matrix(_rotmat_from_6d(self._processed_actions[:, 3:9]))
        command = torch.cat((pos, quat), dim=-1)
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._ema_reset(env_ids)


class EMADiffIKAction(_RawActionEMAMixin, DifferentialInverseKinematicsAction):
    """Stock differential-IK action (rel or abs) with EMA on the raw actions."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ema_setup()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(self._maybe_ema(actions))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._ema_reset(env_ids)


class EMAOperationalSpaceControllerAction(_RawActionEMAMixin, OperationalSpaceControllerAction):
    """Stock OSC action with EMA on the raw actions (pose + stiffness dims)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ema_setup()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(self._maybe_ema(actions))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._ema_reset(env_ids)


@configclass
class SixDRotDiffIKActionCfg(actions_cfg.DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = SixDRotDiffIKAction
    ema_alpha: float | None = None
    """EMA factor on the raw actions (None -> no smoothing)."""


@configclass
class EMADiffIKActionCfg(actions_cfg.DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = EMADiffIKAction
    ema_alpha: float | None = None
    """EMA factor on the raw actions (None -> no smoothing)."""


@configclass
class EMAOSCActionCfg(actions_cfg.OperationalSpaceControllerActionCfg):
    class_type: type[ActionTerm] = EMAOperationalSpaceControllerAction
    ema_alpha: float | None = None
    """EMA factor on the raw actions (None -> no smoothing)."""

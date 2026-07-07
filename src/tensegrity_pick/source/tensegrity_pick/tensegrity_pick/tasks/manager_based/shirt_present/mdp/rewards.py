"""Reward / observation functions for the shirt present task (pipeline task 2).

Hem-to-hem presentation (study-validated geometry — see shirt_present_env.py):
the arm grasps the OPPOSITE hem corner and pulls it HORIZONTALLY to the
holder's height.  Sequential reward structure (shirt_pick pattern):

  1. Reach:    tanh proximity of the finger tip to the targeted HEM CORNER,
               paid ONLY while the gripper is still OPEN (pre-grasp) and held
               at 1.0 once grasped — so closing the gripper early earns nothing
               (fixes the premature-close reward hack: the deterministic
               proximity-attach latches within 10 cm on closing, so an early
               close + drift-in used to farm reach+grasp without a real
               reach-then-grasp)
  2. Grasp:    per-step bonus while the hand attachment (slot 0) holds
  3. Pull:     tanh proximity of the hand to the HORIZONTAL-PULL target
               (holder height, offset along the camera-plane x) — DIRECTS the
               stretch into the study's horizontal chord (gated on both grasps)
  4. Stretch:  clamped at-grasp-normalised tautness toward taut (both grasps)
  5. Coverage: projected-silhouette coverage in the inspection-camera plane
  6. Present:  per-step success bonus while the full predicate holds
  7. Overstretch: per-step penalty above the validated tautness band
  8. Drop:     one-shot penalty when an established hand grasp is lost
  9. EarlyClose: per-step penalty for commanding the gripper CLOSED while far
               from the target and not yet grasped (fixes the hack directly)
 10. Occlusion: mild per-step penalty for the arm sitting between the −Y camera
               and the cloth (cosmetic — the coverage metric has no camera
               sensor, so it cannot see the arm; best-effort visual term)

dt-scaling (shirt_place lesson): Isaac Lab multiplies rewards by dt (1/60 s),
so per-step weights earn ~ weight x episode-seconds and one-shots earn
weight/60 — the drop penalty is sized ~60x the per-step terms.

All functions guard against the manager term probe during ``load_managers()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Gripper-command threshold above which the deterministic attach is armed
# (matches FINGER_CMD_CLOSE in shared/cloth_sorting_env.py).
_FINGER_CMD_CLOSE = 0.40


def _ready(env: "ManagerBasedRLEnv") -> bool:
    """Env fully constructed (cloth + task-state buffers exist)."""
    return getattr(env, "_stretch_buf", None) is not None


def _zeros(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def _both_attached(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Hand grasp (slot 0) AND holder anchor (slot 1) both live (N,) bool."""
    return env.grasp_active & env.holder_attached


def _gripper_closing(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env bool: the gripper is COMMANDED closed (attach armed)."""
    robot = env.scene["robot"]
    if hasattr(robot.data, "joint_pos_target"):
        cmd = robot.data.joint_pos_target[:, env._finger_joint_idx]
    else:
        cmd = robot.data.joint_pos[:, env._finger_joint_idx]
    return cmd > _FINGER_CMD_CLOSE


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def hand_target_rel_tip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Targeted hem corner relative to the DYNAMIC finger tip (N, 3).

    The exact geometry the deterministic attach trigger uses (give the policy
    the trigger's own error signal) — camera-trivial from depth.
    """
    if not _ready(env):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return env.shirt_grasp_point_w - env._finger_tip_pos()


def pull_target_rel_tip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Horizontal-pull goal relative to the finger tip (N, 3).

    Where the grasped hem corner must be brought: holder height, offset along
    the camera-plane x.  Zero-magnitude before the grasp sets the direction.
    """
    if not _ready(env):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return env.present_pull_target_w - env._finger_tip_pos()


def holder_attached_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1.0 while the holder anchor (slot 1) still pins the cloth (N, 1)."""
    if not _ready(env):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.holder_attached.float().unsqueeze(-1)


def stretch_ratio_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """At-grasp-normalised tautness, 0 until both attachments hold (N, 1)."""
    if not _ready(env):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.stretch_ratio_norm.unsqueeze(-1)


def coverage_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Projected-silhouette coverage of the flat one-sided area (N, 1)."""
    if not _ready(env):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.coverage.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Task rewards
# ---------------------------------------------------------------------------

def reaching_target(env: "ManagerBasedRLEnv", std: float = 0.25) -> torch.Tensor:
    """1 - tanh(||tip - hem target|| / std), paid only with an OPEN gripper.

    Saturates to 1.0 while the hand grasp holds (the hem corner moves once
    lifted; chasing it post-grasp would fight the pull).  Before the grasp the
    term is ZEROED whenever the gripper is commanded closed — removing the
    incentive to close early and drift into the proximity-attach.  The policy
    must therefore approach OPEN and close at the target.
    """
    if not _ready(env):
        return _zeros(env)
    d = torch.norm(env._finger_tip_pos() - env.shirt_grasp_point_w, dim=-1)
    r = 1.0 - torch.tanh(d / std)
    # open-gripper gate pre-grasp; full credit once grasped
    open_pre = (~_gripper_closing(env)) & (~env.grasp_active)
    r = torch.where(env.grasp_active, torch.ones_like(r),
                    torch.where(open_pre, r, torch.zeros_like(r)))
    return r


def grasp_hold(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step bonus while the hand attachment grasp (slot 0) holds."""
    if not _ready(env):
        return _zeros(env)
    return env.grasp_active.float()


def pulling_horizontal(env: "ManagerBasedRLEnv", std: float = 0.20) -> torch.Tensor:
    """1 - tanh(||hand - horizontal-pull target|| / std), gated on both grasps.

    Directs the second grasp to the holder's HEIGHT offset horizontally along
    the camera-plane x — the study's taut horizontal chord that gravity drapes
    below.  Replaces the old undirected tautness-only shaping (which let the
    arm pull in any direction).
    """
    if not _ready(env):
        return _zeros(env)
    d = torch.norm(env._finger_tip_pos() - env.present_pull_target_w, dim=-1)
    return (1.0 - torch.tanh(d / std)) * _both_attached(env).float()


def stretch_progress(
    env: "ManagerBasedRLEnv", lo: float = 0.80, hi: float = 1.02,
) -> torch.Tensor:
    """Clamped MAINTAIN-TAUTNESS term in [0, 1], gated on both attachments."""
    if not _ready(env):
        return _zeros(env)
    prog = torch.clamp((env.stretch_ratio_norm - lo) / (hi - lo), 0.0, 1.0)
    return prog * _both_attached(env).float()


def overstretch_penalty(env: "ManagerBasedRLEnv", limit: float = 1.10) -> torch.Tensor:
    """Tautness excess above ``limit`` (positive; use a negative weight)."""
    if not _ready(env):
        return _zeros(env)
    excess = torch.clamp(env.stretch_ratio_norm - limit, min=0.0)
    return excess * _both_attached(env).float()


def coverage_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Projected-silhouette coverage, gated on both attachments."""
    if not _ready(env):
        return _zeros(env)
    return env.coverage * _both_attached(env).float()


def presented(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step success bonus: the full presentation predicate holds NOW."""
    if not _ready(env):
        return _zeros(env)
    return env.presented_now.float()


def drop_event(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot: 1.0 the step an established hand grasp is lost (weight ~60x)."""
    if not _ready(env):
        return _zeros(env)
    return env.drop_event


def early_close_penalty(
    env: "ManagerBasedRLEnv", clear_dist: float = 0.12,
) -> torch.Tensor:
    """Per-step penalty (positive; negative weight) for closing the gripper early.

    1.0 while the gripper is COMMANDED closed AND no grasp is held AND the tip
    is farther than ``clear_dist`` from the hem target — i.e. exactly the
    premature-close-then-drift-in behaviour that farms the proximity-attach.
    Scales up with distance so a far-away close is punished more.
    """
    if not _ready(env):
        return _zeros(env)
    d = torch.norm(env._finger_tip_pos() - env.shirt_grasp_point_w, dim=-1)
    bad = _gripper_closing(env) & (~env.grasp_active) & (d > clear_dist)
    return bad.float() * torch.clamp(d - clear_dist, min=0.0)


def occlusion_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Mild per-step penalty (positive; negative weight): arm in the camera line.

    The −Y inspection camera (y=2.0) sees the cloth (centroid y≈0.85) occluded
    when the end-effector sits on the camera side of it (ee_y > cloth_y).  The
    coverage metric has NO camera sensor, so it cannot penalise this — a purely
    visual/best-effort term to keep the arm reaching PAST the garment rather
    than in front of it.  Small weight (it fights the fixed base geometry).
    """
    if not _ready(env):
        return _zeros(env)
    ee_y = env._finger_tip_pos()[:, 1]
    cloth_y = env._cloth.centroid_pos_w[:, 1]
    return torch.clamp(ee_y - cloth_y, min=0.0) * _both_attached(env).float()

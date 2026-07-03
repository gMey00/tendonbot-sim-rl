"""Reward / observation functions for the shirt present task (pipeline task 2).

Sequential structure (shirt_pick pattern, adapted from pick-and-hold to
regrasp-and-stretch):

  1. Reach:    tanh proximity of the dynamic finger tip to the LOWEST hanging
               point (the literature-standard second-grasp target and the
               deterministic attach target; saturates to 1 once grasped so the
               term never pulls the arm back after the regrasp)
  2. Grasp:    per-step bonus while the hand attachment (slot 0) holds
  3. Stretch:  clamped progress of the inter-grasp tautness ratio toward taut
               (gated on BOTH attachments — an unattached fling earns nothing)
  4. Coverage: projected-silhouette coverage in the inspection-camera plane
               (gated on both attachments; the honest "inspectable" score —
               folds/bunching count once, hiding the shirt earns nothing)
  5. Present:  per-step bonus while the full success predicate holds — this is
               the success reward; holding the stretched presentation IS the
               task, so no anti-hover fade is needed
  6. Overstretch: per-step penalty above the validated tautness band (the
               two-attachment stretch is only validated stable through 1.15)
  7. Drop:     one-shot penalty when an established hand grasp is lost

dt-scaling (shirt_place lesson): Isaac Lab multiplies rewards by dt (1/60 s),
so per-step weights earn ~ weight x episode-seconds and one-shots earn
weight/60 — the drop penalty is sized ~60x the per-step terms.

All functions guard against the ObservationManager/RewardManager term probe
during ``load_managers()`` (env attributes like ``_cloth`` / the task-state
buffers do not exist yet).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ready(env: "ManagerBasedRLEnv") -> bool:
    """Env fully constructed (cloth + task-state buffers exist)."""
    return getattr(env, "_stretch_buf", None) is not None


def _zeros(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def _both_attached(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Hand grasp (slot 0) AND holder anchor (slot 1) both live (N,) bool."""
    return env.grasp_active & env.holder_attached


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def lowest_point_rel_tip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Lowest cloth particle relative to the DYNAMIC finger tip (N, 3).

    The exact geometry the deterministic attach trigger uses (shirt_pick
    lesson: give the policy the trigger's own error signal, not a nearby
    frame's) — camera-trivial from depth.
    """
    if not _ready(env):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return env.shirt_lowest_point_w - env._finger_tip_pos()


def holder_attached_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1.0 while the holder anchor (slot 1) still pins the cloth (N, 1)."""
    if not _ready(env):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.holder_attached.float().unsqueeze(-1)


def stretch_ratio_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Inter-grasp tautness ratio, 0 until both attachments hold (N, 1).

    Camera-derivable in principle: both grasp points are visible to the
    inspection camera and the garment's flat geometry is known a priori.
    """
    if not _ready(env):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.stretch_ratio.unsqueeze(-1)


def coverage_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Projected-silhouette coverage of the flat one-sided area (N, 1).

    Directly computable from the inspection camera's segmentation mask.
    """
    if not _ready(env):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.coverage.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Task rewards
# ---------------------------------------------------------------------------

def reaching_lowest_point(env: "ManagerBasedRLEnv", std: float = 0.25) -> torch.Tensor:
    """1 - tanh(||tip - lowest point|| / std); held at 1.0 while grasped.

    Unlike shirt_pick's highest point (which becomes the grasped patch and
    tracks the tip), the LOWEST point moves elsewhere on the garment once the
    bottom is lifted — chasing it post-grasp would fight the stretch.  So the
    term saturates to its maximum while the hand grasp holds.
    """
    if not _ready(env):
        return _zeros(env)
    d = torch.norm(env._finger_tip_pos() - env.shirt_lowest_point_w, dim=-1)
    r = 1.0 - torch.tanh(d / std)
    return torch.where(env.grasp_active, torch.ones_like(r), r)


def grasp_hold(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step bonus while the hand attachment grasp (slot 0) holds."""
    if not _ready(env):
        return _zeros(env)
    return env.grasp_active.float()


def stretch_progress(
    env: "ManagerBasedRLEnv", lo: float = 0.50, hi: float = 0.98,
) -> torch.Tensor:
    """Clamped tautness progress in [0, 1], gated on both attachments.

    Linear ramp from ``lo`` (typical slack span right after the second grasp)
    to ``hi`` (taut); saturates there so pulling past taut earns nothing extra
    (the overstretch penalty takes over above the validated band).
    """
    if not _ready(env):
        return _zeros(env)
    prog = torch.clamp((env.stretch_ratio - lo) / (hi - lo), 0.0, 1.0)
    return prog * _both_attached(env).float()


def overstretch_penalty(env: "ManagerBasedRLEnv", limit: float = 1.10) -> torch.Tensor:
    """Tautness excess above ``limit`` (positive; use a negative weight).

    The two-attachment stretch is validated stable only through ratio 1.15
    (Stage-0 de-risk) — penalise before entering untested territory.
    """
    if not _ready(env):
        return _zeros(env)
    excess = torch.clamp(env.stretch_ratio - limit, min=0.0)
    return excess * _both_attached(env).float()


def coverage_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Projected-silhouette coverage, gated on both attachments.

    Rasterized silhouette in the camera (XZ) plane — folds and double layers
    count once, so bunched/hidden cloth cannot farm this.  Gating on the
    bimanual hold keeps the raw-hang coverage (~0.3-0.5) from being farmed
    without the second grasp.
    """
    if not _ready(env):
        return _zeros(env)
    return env.coverage * _both_attached(env).float()


def presented(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step success bonus: the full presentation predicate holds NOW.

    both grasps AND stretch ratio in the taut band AND coverage above the
    threshold AND cloth-centroid speed below the gate (the camera inspects
    the cloth, not the EE — shirt_pick Phase-3 lesson).  Accruing per step
    makes an EARLY, STABLE stretched presentation the optimal policy.
    """
    if not _ready(env):
        return _zeros(env)
    return env.presented_now.float()


def drop_event(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot: 1.0 the step an established hand grasp is lost.

    Weight this ~60x the per-step terms (dt-scaling)."""
    if not _ready(env):
        return _zeros(env)
    return env.drop_event

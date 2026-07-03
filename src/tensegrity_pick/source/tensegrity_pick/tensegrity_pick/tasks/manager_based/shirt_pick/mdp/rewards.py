"""Reward / observation functions for the shirt pick task (pipeline task 1).

Sequential structure (ported from the validated shirt_place design, adapted
from release-into-drum to present-and-HOLD):

  1. Reach:    tanh proximity of the dynamic finger tip to the grasp target
               (the cloth's highest point — the deterministic attach target)
  2. Grasp:    per-step bonus while the attachment grasp holds
  3. Lift:     clamped grasp-point height progress above the belt (gated)
  4. Transport: tanh distance of the grasp point to the presentation pose
               (gated on the grasp — an unattached fling earns nothing)
  5. Present:  per-step bonus while held AT the pose, slow — this is the
               success reward; because holding at the goal IS the task there
               is no anti-hover fade needed (unlike shirt_place, where
               hovering over the drum farmed shaping without releasing)
  6. Drop:     one-shot penalty when the grasp opens before episode end
               (the task never releases)

dt-scaling (shirt_place lesson): Isaac Lab multiplies rewards by dt (1/60 s),
so per-step weights earn ≈ weight x episode-seconds and one-shots earn
weight/60 — the drop penalty is sized ~60x the per-step terms.

All functions guard against the ObservationManager/RewardManager term probe
during ``load_managers()`` (env attributes like ``_cloth`` do not exist yet).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ...shared.cloth_sorting_scene_cfg import (
    CONVEYOR_SURFACE_HEIGHT_M,
    PRESENTATION_POS,
)

BELT_HEIGHT = CONVEYOR_SURFACE_HEIGHT_M


def _ready(env: "ManagerBasedRLEnv") -> bool:
    return getattr(env, "_cloth", None) is not None


def _zeros(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def _presentation_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.scene.env_origins + torch.tensor(
        PRESENTATION_POS, device=env.device, dtype=torch.float32
    ).unsqueeze(0)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def grasp_target_rel_tip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cloth highest point (the deterministic grasp target) relative to the
    dynamic finger tip (N, 3) — the actual quantity the attach trigger uses."""
    if not _ready(env):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return env.shirt_grasp_point_w - env._finger_tip_pos()


# ---------------------------------------------------------------------------
# Task rewards
# ---------------------------------------------------------------------------

def reaching_grasp_target(env: "ManagerBasedRLEnv", std: float = 0.25) -> torch.Tensor:
    """1 − tanh(‖tip − highest point‖ / std) — pre-grasp approach shaping.

    Uses the dynamic finger tip and the highest point (not the centroid): the
    tip must come DOWN onto the cloth for the deterministic attach to fire.
    """
    if not _ready(env):
        return _zeros(env)
    d = torch.norm(env._finger_tip_pos() - env.shirt_grasp_point_w, dim=-1)
    return 1.0 - torch.tanh(d / std)


def grasp_hold(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step bonus while the attachment grasp holds the cloth."""
    if not _ready(env):
        return _zeros(env)
    return env.grasp_active.float()


def lift_progress(env: "ManagerBasedRLEnv", full_height: float = 0.30) -> torch.Tensor:
    """Clamped grasp-point height above the belt, ∈ [0, 1] (grasp-gated).

    Saturates at ``full_height`` so over-lifting earns nothing extra — the
    transport term takes over from there.
    """
    if not _ready(env):
        return _zeros(env)
    z = env.shirt_grasp_point_w[:, 2] - env.scene.env_origins[:, 2]
    prog = torch.clamp((z - BELT_HEIGHT) / full_height, 0.0, 1.0)
    return prog * env.grasp_active.float()


def to_presentation(env: "ManagerBasedRLEnv", std: float = 0.35) -> torch.Tensor:
    """1 − tanh(‖grasp point − presentation pose‖ / std), grasp-gated.

    Gating on the live grasp closes the fling hack (throwing the cloth
    through the pose without holding it earns nothing).
    """
    if not _ready(env):
        return _zeros(env)
    d = torch.norm(env.shirt_grasp_point_w - _presentation_target(env), dim=-1)
    return (1.0 - torch.tanh(d / std)) * env.grasp_active.float()


def presented(
    env: "ManagerBasedRLEnv",
    dist_threshold: float = 0.15,
    vel_threshold: float = 0.20,
) -> torch.Tensor:
    """Per-step success bonus: held AT the presentation pose, cloth still.

    grasp_active AND grasp point within ``dist_threshold`` of the pose AND
    **cloth centroid speed** below ``vel_threshold`` — accruing per step
    makes an EARLY, STABLE hold the optimal policy.

    The speed gate is on the CLOTH, not the EE (changed 2026-07-03, Phase 3):
    at the raised presentation posture the PD arm's residual EE sway is
    0.24–0.27 m/s — permanently above any sane EE gate — while the hanging
    garment low-pass filters it to 0.06–0.20 m/s.  The camera also inspects
    the cloth, not the gripper, so the cloth's stillness is the honest
    "inspectable" criterion (measured: diag_shirt_pick_policy.py).
    """
    if not _ready(env):
        return _zeros(env)
    d = torch.norm(env.shirt_grasp_point_w - _presentation_target(env), dim=-1)
    cloth_speed = env.cloth.centroid_vel_w.norm(dim=-1)
    ok = env.grasp_active & (d < dist_threshold) & (cloth_speed < vel_threshold)
    return ok.float()


def drop_event(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot: 1.0 the step an established grasp is lost (task never
    releases).  Weight this ~60× the per-step terms (dt-scaling)."""
    if not _ready(env):
        return _zeros(env)
    return env.drop_event


# ---------------------------------------------------------------------------
# Metrics (tiny weights — TensorBoard logging via the reward pipeline)
# ---------------------------------------------------------------------------

def metric_presented_now(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Instantaneous presented predicate (for reward-curve inspection)."""
    return presented(env)

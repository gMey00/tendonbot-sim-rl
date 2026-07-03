"""Reward / observation / termination functions for the shirt_distribute task.

Ports the release-into-drum design validated in shirt_place
(``shirt_place/mdp/rewards.py`` + ``doc/reports/shirt_place_optimization_tracking.md``),
retargeted to the goal-conditioned COMMANDED bin (``env.target_bin_pos_w``):

  1. Carry:    tanh XY proximity of the grasp point to the commanded bin,
               gated on holding, faded by ``clearance_fraction`` (anti-hover)
  2. Clear:    whole shirt (lowest particle) above the commanded drum's rim
  3. Release:  small per-step openness hint over the bin + the GRADED one-shot
               ``release_event`` (centering × whole-shirt-lift, ~60× per-step
               weights — dt-scaling lesson)
  4. Success:  per-step fraction of RELEASED cloth inside the commanded drum
               (release-required — closes the "dip the held shirt in" hack)
  5. After:    return_to_neutral gated on ``was_distributed``
  6. Anti:     bad-release one-shot (opened NOT over the goal — includes
               instant drops at episode start), wrong-bin landed fraction,
               dropped-on-floor, carry-time bleed, belt contact

All shirt state reads ``env.shirt_grasp_point_w`` / the cloth tensors via env
properties, matching the shirt_place pattern.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import DRUM_NAMES
from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import (
    FINGER_JOINT_CLOSE_POS,
    get_world_pos as _get_world_pos,
    grasp_center_w as _grasp_center_w,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Observations (task-specific)
# ---------------------------------------------------------------------------

def target_bin_rel_ee(env: "ManagerBasedRLEnv", ee_cfg: SceneEntityCfg) -> torch.Tensor:
    """Commanded bin position relative to the grasp centre (N, 3).

    The carry is EE-centric (the shirt hangs from the gripper), so the goal
    relative to the EE is the direct control-relevant error signal; the
    shared ``target_bin_rel_shirt`` (bin − shirt centroid) stays in the obs
    for the post-release phase where the cloth moves independently.
    """
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    if getattr(env, "_target_bin", None) is None:
        return torch.zeros_like(ee)
    return env.target_bin_pos_w - ee


def was_distributed_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Binary (N, 1): 1.0 once the commanded-bin placement latched."""
    if getattr(env, "_was_distributed", None) is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.was_distributed.float().unsqueeze(-1)


# ---------------------------------------------------------------------------
# Phase 1–2: goal-conditioned carry + clearance
# ---------------------------------------------------------------------------

def approach_target_bin(
    env: "ManagerBasedRLEnv", std: float = 0.4,
) -> torch.Tensor:
    """Tanh XY proximity of the grasp point to the commanded bin (N,).

    Tracks the GRASP POINT (the gripper-held part — the hanging shirt is a
    cone below it; shirt_place lesson: the trailing centroid makes the
    gripper stop at the near rim).  Gated on an active grasp, faded by
    ``clearance_fraction`` so hovering cleared over the right bin is a reward
    desert (anti-hover) — only releasing pays.
    """
    gp = env.shirt_grasp_point_w
    target = env.target_bin_pos_w
    d_xy = torch.norm(gp[:, :2] - target[:, :2], dim=-1)
    proximity = 1.0 - torch.tanh(d_xy / std)
    gate = env.grasp_active & env.was_grasped
    fade = 1.0 - env.clearance_fraction
    return torch.where(gate, proximity, torch.zeros_like(proximity)) * fade


def clearance_over_target_bin(
    env: "ManagerBasedRLEnv",
    drum_top: float,
    rim_clearance: float = 0.04,
    clear_margin: float = 0.05,
    drum_radius: float = 0.32,
) -> torch.Tensor:
    """Dense reward for suspending the WHOLE shirt above the commanded drum.

    Grasp point over the bin footprint AND the lowest cloth particle above
    the rim — the ready-to-drop pose.  Gated on an active grasp.
    """
    target = env.target_bin_pos_w
    gp = env.shirt_grasp_point_w
    d_xy = torch.norm(gp[:, :2] - target[:, :2], dim=-1)
    in_xy = d_xy < drum_radius
    rim = drum_top + rim_clearance
    bottom_z = env.shirt_min_z_w - env.scene.env_origins[:, 2]
    clearance = torch.clamp((bottom_z - rim) / clear_margin, min=0.0, max=1.0)
    gate = in_xy & env.grasp_active & env.was_grasped
    return torch.where(gate, clearance, torch.zeros_like(clearance))


# ---------------------------------------------------------------------------
# Phase 3: release
# ---------------------------------------------------------------------------

def release_openness_over_target(
    env: "ManagerBasedRLEnv",
    finger_cfg: SceneEntityCfg,
    drum_top: float,
    rim_clearance: float = 0.02,
    drum_radius: float = 0.2735,
) -> torch.Tensor:
    """Small per-step openness hint while the carried shirt is over the goal."""
    target = env.target_bin_pos_w
    gp = env.shirt_grasp_point_w
    d_xy = torch.norm(gp[:, :2] - target[:, :2], dim=-1)
    in_xy = d_xy < drum_radius
    local_z = gp[:, 2] - env.scene.env_origins[:, 2]
    above = local_z > (drum_top + rim_clearance)

    robot: Articulation = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
    openness = 1.0 - closure

    gate = in_xy & above & env.was_grasped
    return torch.where(gate, openness, torch.zeros_like(openness))


def release_event_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot graded bonus: gripper opened over the commanded bin (N,).

    Reads the env's ``release_event`` (0.25–1.0 for the single step a good
    release fired, graded by centering × whole-shirt-lift) — a discrete
    committed-drop event, not a per-step state the policy could farm.
    """
    return env.release_event


def bad_release_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot 1.0 the step the grasp opened NOT over the commanded bin.

    Covers the instant-drop failure at episode start (any gripper action
    ≥ 0 on the binary term opens): the episode starts holding, so opening
    anywhere except over the goal is a committed mistake.
    """
    return env.bad_release_event


# ---------------------------------------------------------------------------
# Phase 4: landed cloth
# ---------------------------------------------------------------------------

def fraction_in_target_bin(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step success reward: released-cloth fraction in the commanded drum.

    Gated on ``was_grasped`` AND released — a still-gripped shirt dipped into
    the drum earns nothing (shirt_place hack), and the fraction scores a
    partly-draped shirt honestly.
    """
    frac = env._target_fraction_in_bin()
    gate = env.was_grasped.float() * (~env.grasp_active).float()
    return frac * gate


def fraction_in_wrong_bin(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Released-cloth fraction inside any NON-commanded drum (N,) — penalty.

    A wrong-bin landing must be clearly worse than landing nowhere: the
    physical sorting error the pipeline exists to avoid.
    """
    frac = env._wrong_fraction_in_bin()
    gate = env.was_grasped.float() * (~env.grasp_active).float()
    return frac * gate


def shirt_dropped_on_floor(
    env: "ManagerBasedRLEnv",
    z_threshold: float = 0.70,
    drum_radius: float = 0.40,
) -> torch.Tensor:
    """1.0 while the released shirt centroid lies below belt level OUTSIDE
    every drum footprint — a genuinely dropped shirt (floor / belt gap)."""
    centroid = env.cloth.centroid_pos_w
    local_z = centroid[:, 2] - env.scene.env_origins[:, 2]
    fallen = (local_z < z_threshold) & (~env.grasp_active)
    outside_drums = torch.ones_like(fallen)
    for name in DRUM_NAMES:
        pos_d = _get_world_pos(env.scene[name], env=env)
        d_xy = torch.norm(centroid[:, :2] - pos_d[:, :2], dim=-1)
        outside_drums &= d_xy > drum_radius
    return (fallen & outside_drums).float()


def carry_time_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step time cost while still holding an undistributed shirt.

    The episode starts lifted, so this is a plain hold-time bleed: sustained
    holding must be worse than committing to the drop (anti-hover pressure,
    shirt_place design).
    """
    active = env.grasp_active & (~env.was_distributed)
    return active.float()


# ---------------------------------------------------------------------------
# Phase 5: after the drop
# ---------------------------------------------------------------------------

def return_to_neutral(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    std: float = 0.40,
) -> torch.Tensor:
    """Reward for returning joints to defaults once the shirt is distributed.

    Per-joint normalized deviation (soft-range scaled) → 1 − tanh(rms/std);
    gated on ``was_distributed`` (port of shirt_place's post-place objective —
    prevents lingering over the rim).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    current = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default = robot.data.default_joint_pos[:, asset_cfg.joint_ids]

    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    scale = limits[..., 1] - limits[..., 0]
    valid = scale.isfinite() & (scale > 1e-6)
    scale = torch.where(valid, scale, torch.ones_like(scale))

    dev = (current - default) / scale
    distance = torch.norm(dev, dim=-1) / math.sqrt(dev.shape[-1])
    result = 1.0 - torch.tanh(distance / std)
    return result * env.was_distributed.float()

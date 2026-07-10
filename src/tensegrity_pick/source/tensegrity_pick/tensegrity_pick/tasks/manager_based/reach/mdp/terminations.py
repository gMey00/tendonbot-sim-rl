# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms specific to the physical (antiparallelogram) tensegrity model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Anchor offsets of the loop-closure joint (coupler_right_joint) in body-local
# frames, from the linkage constants in tools/build_tensegrity_arm_usd.py:
#   rod_right_link origin = frame pivot B (0, +0.03, −0.36) in root frame;
#   closure pivot C at rest = (0, −0.03, −0.4975)  →  local (0, −0.06, −0.1375)
#   forearm_link origin = (0, 0, −0.4975)          →  C local (0, −0.03, 0)
_ROD_RIGHT_CLOSURE_ANCHOR_LOCAL = (0.0, -0.06, -0.1375)
_FOREARM_CLOSURE_ANCHOR_LOCAL = (0.0, -0.03, 0.0)


def linkage_closure_broken(
    env: ManagerBasedRLEnv,
    gap_threshold: float = 0.03,
    branch_angle_tolerance: float = 0.3,
    rod_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["rod_right_link"]),
    forearm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["forearm_link"]),
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["root_link"]),
    rod_joints_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["rod_left_joint", "rod_right_joint"]
    ),
    rod_anchor_local: tuple[float, float, float] = _ROD_RIGHT_CLOSURE_ANCHOR_LOCAL,
    forearm_anchor_local: tuple[float, float, float] = _FOREARM_CLOSURE_ANCHOR_LOCAL,
) -> torch.Tensor:
    """Terminate when the four-bar linkage is torn open or branch-flipped.

    Two failure modes of the closed chain are detected:

    1. **Closure gap** — the closure joint (``coupler_right_joint``) is a
       PhysX maximal-coordinate constraint (``excludeFromArticulation``) that
       drifts apart under large body forces.  RL policies exploited this by
       breaking the linkage and steering the dangling forearm with the linear
       base.  The world-space gap between the two closure anchors must stay
       below *gap_threshold*.  Calibration (measured 2026-07-09, 120 Hz): 0 at
       rest, ≲1 mm settled under a sustained 480 N elbow tension, ~1–3 cm
       during violent transients; the exploit opens it much further.
    2. **Branch flip** — during violent transients the chain can re-close in
       the *parallelogram* branch, which satisfies the closure constraint but
       decouples the forearm rotation from the rods.  In the correct
       antiparallelogram branch the effective elbow angle equals
       ``rod_left + rod_right`` (exactly, from the closure condition); in a
       flipped branch they diverge by tens of degrees.  The mismatch must stay
       below *branch_angle_tolerance* (rad; normal elastic drift stays below
       ~0.15 rad even under full tension).
    """
    from tensegrity_pick.robots import compute_lower_arm_angle_and_rate

    asset = env.scene[rod_cfg.name]
    data = asset.data
    rod_idx = rod_cfg.body_ids[0]
    forearm_idx = forearm_cfg.body_ids[0]

    rod_off = torch.tensor(rod_anchor_local, dtype=torch.float32, device=asset.device)
    forearm_off = torch.tensor(forearm_anchor_local, dtype=torch.float32, device=asset.device)

    rod_anchor_w = data.body_pos_w[:, rod_idx] + quat_apply(
        data.body_quat_w[:, rod_idx], rod_off.expand(env.num_envs, 3)
    )
    forearm_anchor_w = data.body_pos_w[:, forearm_idx] + quat_apply(
        data.body_quat_w[:, forearm_idx], forearm_off.expand(env.num_envs, 3)
    )
    gap = (rod_anchor_w - forearm_anchor_w).norm(dim=-1)

    elbow, _ = compute_lower_arm_angle_and_rate(asset, root_cfg.body_ids[0], forearm_idx)
    rod_sum = data.joint_pos[:, rod_joints_cfg.joint_ids].sum(dim=-1)
    branch_mismatch = (elbow - rod_sum).abs()

    return (gap > gap_threshold) | (branch_mismatch > branch_angle_tolerance)

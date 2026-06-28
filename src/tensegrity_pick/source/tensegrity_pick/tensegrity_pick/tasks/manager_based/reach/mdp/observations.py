# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms specific to the reach environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_pos_sin_cos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint positions encoded as ``[sin(q), cos(q)]`` for the selected joints.

    Raw joint angles are discontinuous at the ``±π`` wrap, so two near-identical
    configurations of a continuous (infinite-rotation) joint — e.g. the Kinova
    Gen3's J1/J3/J5/J7 — can appear maximally distant to the policy.  Encoding
    each angle as its sine and cosine removes that artificial discontinuity and
    gives a smooth, unique representation on the circle (standard practice for
    wrap-around joints).  Harmless for bounded joints, so it is applied uniformly.

    Returns a tensor of shape ``(num_envs, 2 * num_selected_joints)`` ordered as
    ``[sin(q_0..q_n), cos(q_0..q_n)]``.
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat((torch.sin(q), torch.cos(q)), dim=-1)

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


def lower_arm_angle(
    env: ManagerBasedEnv,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["root_link"]),
    forearm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["forearm_link"]),
) -> torch.Tensor:
    """Effective elbow angle of the physical antiparallelogram model.  Shape (N, 1).

    Measured as the forearm body's twist about the upper-arm X axis — the
    physically meaningful lower-arm rotation.  This is NOT any single linkage
    joint angle: the forearm rotation is the sum of the rod and coupler joint
    angles, and reading a single joint under-reports it (and becomes
    meaningless if the loop-closure constraint drifts).
    """
    from tensegrity_pick.robots import compute_lower_arm_angle_and_rate

    asset = env.scene[root_cfg.name]
    angle, _ = compute_lower_arm_angle_and_rate(asset, root_cfg.body_ids[0], forearm_cfg.body_ids[0])
    return angle.unsqueeze(-1)


def lower_arm_ang_vel(
    env: ManagerBasedEnv,
    root_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["root_link"]),
    forearm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["forearm_link"]),
) -> torch.Tensor:
    """Effective elbow angular velocity (forearm relative to upper arm, about X).  Shape (N, 1)."""
    from tensegrity_pick.robots import compute_lower_arm_angle_and_rate

    asset = env.scene[root_cfg.name]
    _, rate = compute_lower_arm_angle_and_rate(asset, root_cfg.body_ids[0], forearm_cfg.body_ids[0])
    return rate.unsqueeze(-1)


def tendon_cable_lengths(env: ManagerBasedEnv, action_name: str = "arm_tendon") -> torch.Tensor:
    """Elbow cable lengths between the attachment points [m].  Shape (N, 2).

    Read from the physical tendon action term — exposes the actuator state to
    the policy (evaluation report §9, open issue #1: per-cable length/velocity
    in the observation).
    """
    return env.action_manager.get_term(action_name).cable_lengths


def tendon_cable_length_rates(env: ManagerBasedEnv, action_name: str = "arm_tendon") -> torch.Tensor:
    """Elbow cable length rates [m/s] (positive = extending).  Shape (N, 2)."""
    return env.action_manager.get_term(action_name).cable_length_rates


def tendon_applied_tensions(env: ManagerBasedEnv, action_name: str = "arm_tendon") -> torch.Tensor:
    """Actually applied elbow tensions [N].  Shape (N, 2).

    Differs from the commanded tensions (last_action) through the 50 ms
    motor/spool lag and the cable wind-up/stretch stops — actuator state the
    policy cannot otherwise observe.  Scale with ~1/480 in the ObsTerm.
    """
    return env.action_manager.get_term(action_name).applied_elbow_tensions


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

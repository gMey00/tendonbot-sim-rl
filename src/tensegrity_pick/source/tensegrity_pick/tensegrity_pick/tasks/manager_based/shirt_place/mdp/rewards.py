"""Reward, observation, termination and reset functions for the shirt place task.

Follows the same 6-phase sequential reward structure as cube_place:
  1. Reach:      tanh proximity (dynamic fingertip → shirt proxy)
  2. Grasp:      closure × proximity (fingertip-based, at belt level)
  3. Lift:       binary + height bonus (velocity-gated, closure-gated)
  4. Transport:  tanh XY distance to drum (gated: was_grasped + z > belt)
  5. Release:    gripper openness when shirt above drum
  6. Success:    per-step bonus while shirt rests in drum

Key differences from cube_place:
  - Single object (shirt_proxy) instead of green+red cubes
  - No red cube curriculum or clearance rewards
  - Cloth-specific observations (keypoints, spread) to be added once
    ClothObject tensor API is wired up
  - Higher friction parameters for adhesion-based cloth grasping

The ``shirt_proxy`` is a temporary rigid-body stand-in for the cloth
centroid.  Once ClothObject.update() is implemented, observations and
rewards will read from the actual cloth state tensors instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Shared gripper geometry + observation/reward utilities
from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import (
    FINGER_JOINT_CLOSE_POS,
    FINGER_TIP_CLOSED_Z,
    FINGER_TIP_LOCAL_Z,
    FINGER_TIP_OPEN_Z,
    GRASP_CENTER_LOCAL_Z,
    BinCylinder,
    ConveyorBounds,
    SpawnBox,
    action_rate_l2,
    arm_velocity_bonus,
    base_velocity_l2,
    belt_collision_termination,
    belt_contact_penalty,
    drum_rel_pos,
    dynamic_finger_tip_w as _dynamic_finger_tip_w,
    ee_lin_vel_w,
    ee_pos_w,
    get_world_pos as _get_world_pos,
    grasp_center_w as _grasp_center_w,
    gripper_closure,
    gripper_torque_residual,
    in_upright_cylinder as _in_upright_cylinder,
    joint_torque_penalty,
    joint_vel_l2_controlled,
    joint_vel_out_of_limit,
)


# ---------------------------------------------------------------------------
# Cloth-state accessors
# ---------------------------------------------------------------------------

def _shirt_grasp_point(env: ManagerBasedRLEnv, shirt_name: str) -> torch.Tensor:
    """Deterministic grasp target — the cloth's highest region (N, 3).

    Reads ``env.shirt_grasp_point_w`` (the cloth's highest-point tensor) when the
    cloth env exposes it; falls back to the centroid proxy otherwise (e.g. the
    rigid-cube task, so those rewards stay valid).
    """
    if hasattr(env, "shirt_grasp_point_w"):
        return env.shirt_grasp_point_w
    return env.scene[shirt_name].data.root_pos_w


def _hold_fade(env: ManagerBasedRLEnv) -> torch.Tensor | float:
    """Multiplier that fades per-step positioning shaping out as the shirt clears.

    Returns ``1 - clearance_fraction`` (→ 0 once the shirt is fully cleared over the
    drum), so "positioned over the drum" rewards can't be farmed by hovering — the
    only way to keep gaining reward from that state is to release and drop.
    """
    if hasattr(env, "clearance_fraction"):
        return 1.0 - env.clearance_fraction
    return 1.0


def _shirt_fraction_in_drum(
    env: ManagerBasedRLEnv, shirt_name: str, drum_name: str, bin_geom: "BinCylinder",
) -> torch.Tensor:
    """Fraction of cloth particles inside the drum (N,).

    Uses the env's cloth-aware metric when available; otherwise falls back to a
    binary centroid-in-cylinder test.
    """
    if hasattr(env, "_shirt_particle_fraction_in_drum"):
        return env._shirt_particle_fraction_in_drum()
    return shirt_in_drum(env, shirt_name, drum_name, bin_geom).float()


# ---------------------------------------------------------------------------
# Observations (shirt-specific)
# ---------------------------------------------------------------------------

def shirt_rel_pos(
    env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, shirt_name: str
) -> torch.Tensor:
    """Relative position of the shirt grasp point w.r.t. the grasp centre (N, 3)."""
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    return _shirt_grasp_point(env, shirt_name) - ee


def fingertip_rel_shirt(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    shirt_name: str,
) -> torch.Tensor:
    """Relative position of the shirt grasp point w.r.t. the dynamic fingertip (N, 3)."""
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    return _shirt_grasp_point(env, shirt_name) - tip


def shirt_velocity(env: ManagerBasedRLEnv, shirt_name: str) -> torch.Tensor:
    """Shirt proxy linear velocity (N, 3)."""
    shirt: RigidObject = env.scene[shirt_name]
    return shirt.data.root_lin_vel_w


# ---------------------------------------------------------------------------
# Reset helpers
# ---------------------------------------------------------------------------

def reset_shirt(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    shirt_name: str,
    spawn_box: SpawnBox,
) -> None:
    """Reset the shirt proxy to a random position within the spawn box."""
    device = env.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    x = torch.empty((n,), device=device).uniform_(*spawn_box.x_range)
    y = torch.empty((n,), device=device).uniform_(*spawn_box.y_range)
    z = torch.empty((n,), device=device).uniform_(*spawn_box.z_range)
    yaw = torch.empty((n,), device=device).uniform_(*spawn_box.yaw_range)
    qw = torch.cos(0.5 * yaw)
    qz = torch.sin(0.5 * yaw)
    quat = torch.stack(
        [qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1
    )
    world_pos = torch.stack([x, y, z], dim=-1) + env_origins

    shirt: RigidObject = env.scene[shirt_name]
    shirt.write_root_pose_to_sim(
        torch.cat([world_pos, quat], dim=-1), env_ids=env_ids
    )
    shirt.write_root_velocity_to_sim(
        torch.zeros((n, 6), device=device), env_ids=env_ids
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def shirt_ee_distance(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    shirt_name: str,
    std: float = 0.1,
) -> torch.Tensor:
    """Tanh reward for dynamic-fingertip proximity to the shirt proxy.

    Gated: turns off once the shirt has been grasped (grasp_active=False).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    target = _shirt_grasp_point(env, shirt_name)
    distance = torch.norm(target - tip, dim=-1)
    result = 1.0 - torch.tanh(distance / std)

    if hasattr(env, "grasp_active"):
        result = result * (~env.grasp_active).float()
    return result


def shirt_grasp_reward(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    shirt_name: str,
    std: float = 0.08,
) -> torch.Tensor:
    """Reward for closing the gripper at the shirt's highest point.

    ``closure * (1 - tanh(dist / std))`` for shaping, plus a unit bonus once the
    deterministic grasp has actually latched (``grasp_active``).
    """
    robot: Articulation = env.scene[ee_cfg.name]
    tip = _dynamic_finger_tip_w(robot, ee_cfg, finger_cfg)
    target = _shirt_grasp_point(env, shirt_name)
    distance = torch.norm(target - tip, dim=-1)
    proximity = 1.0 - torch.tanh(distance / std)

    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)

    reward = closure * proximity
    if hasattr(env, "grasp_active"):
        reward = reward + env.grasp_active.float()
    return reward


def shirt_is_lifted(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    shirt_name: str,
    belt_height: float,
    minimal_height: float = 0.06,
    max_distance: float = 0.15,
    finger_cfg: SceneEntityCfg | None = None,
    min_closure: float = 0.20,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Binary reward: 1.0 when the grasped shirt is lifted above the belt.

    Keyed to the grasp point (the welded patch, which rises with the gripper);
    the centroid barely moves for a central pinch on a wide flat shirt.  Gated on
    an active deterministic grasp.
    """
    target = _shirt_grasp_point(env, shirt_name)
    local_z = target[:, 2] - env.scene.env_origins[:, 2]
    is_above = local_z > belt_height + minimal_height

    gate = is_above
    if hasattr(env, "grasp_active"):
        gate = gate & env.grasp_active
    # Fade out once the shirt is cleared over the drum (anti-hover): holding a
    # lifted, cleared shirt earns nothing — only releasing does.
    return torch.where(gate, 1.0, 0.0) * _hold_fade(env)


def shirt_height_bonus(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    shirt_name: str,
    belt_height: float,
    max_height: float = 0.30,
    max_distance: float = 0.15,
    finger_cfg: SceneEntityCfg | None = None,
    min_closure: float = 0.20,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Continuous reward proportional to grasp-point height above the belt.

    Keyed to the grasp point and gated on an active deterministic grasp (the
    centroid barely rises for a central pinch on a wide flat shirt).
    """
    target = _shirt_grasp_point(env, shirt_name)
    local_z = target[:, 2] - env.scene.env_origins[:, 2]
    height_above = torch.clamp(local_z - belt_height, min=0.0, max=max_height)
    normalized = height_above / max_height

    if hasattr(env, "grasp_active"):
        gate = env.grasp_active
        return torch.where(gate, normalized, torch.zeros_like(normalized)) * _hold_fade(env)
    return normalized * _hold_fade(env)


def shirt_approach_target(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    belt_height: float,
    std: float = 0.20,
    lift_threshold: float = 0.04,
) -> torch.Tensor:
    """Bounded tanh reward for XY proximity of a lifted shirt to the drum.

    Gated on ``was_grasped`` to prevent reward hacking.
    """
    pos_d = _get_world_pos(env.scene[drum_name], env=env)

    # Track the *grasp point* (the gripper-held part) to the drum, not the
    # centroid: the grasped shirt hangs from the gripper as a cone, so centring
    # the gripper over the drum opening is what drops the cone in.  The trailing
    # centroid lags behind, which previously made the gripper stop at the near
    # rim and drape the shirt over the edge.
    gp = _shirt_grasp_point(env, shirt_name)
    local_z = gp[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    d_xy = torch.sqrt(
        (gp[:, 0] - pos_d[:, 0]) ** 2 + (gp[:, 1] - pos_d[:, 1]) ** 2
    )
    proximity = 1.0 - torch.tanh(d_xy / std)

    gate = is_lifted
    if hasattr(env, "was_grasped"):
        gate = gate & env.was_grasped

    # Fade out as the shirt clears over the drum (anti-hover): the XY-tracking
    # reward guides the shirt *to* the drum but stops paying once it is there, so
    # the policy can't hover centred over the opening — it must drop.
    return torch.where(gate, proximity, torch.zeros_like(proximity)) * _hold_fade(env)


def shirt_clearance_over_drum(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    belt_height: float,
    rim_clearance: float = 0.08,
    clear_margin: float = 0.08,
    drum_radius: float = 0.32,
) -> torch.Tensor:
    """Dense reward for suspending the *whole* shirt above the drum opening.

    The arm should not reach into the drum; it should lift the shirt high enough
    that the entire hanging garment clears the rim, centre it over the opening,
    and then drop it straight in (regardless of pick orientation).  Reward the
    grasp point being over the drum footprint AND the *lowest* cloth particle
    being above the rim — i.e. the shirt fully hangs above the drum, ready to
    fall in.  Gated on ``was_grasped``.
    """
    pos_d = _get_world_pos(env.scene[drum_name], env=env)
    gp = _shirt_grasp_point(env, shirt_name)
    d_xy = torch.sqrt((gp[:, 0] - pos_d[:, 0]) ** 2 + (gp[:, 1] - pos_d[:, 1]) ** 2)
    in_xy = d_xy < drum_radius

    rim = belt_height + rim_clearance                  # ≈ 0.88 m
    if hasattr(env, "shirt_min_z_w"):
        bottom_z = env.shirt_min_z_w - env.scene.env_origins[:, 2]
    else:  # fallback: assume the shirt hangs ~0.4 m below the grasp point
        bottom_z = (gp[:, 2] - env.scene.env_origins[:, 2]) - 0.40
    # 0 when the shirt's lowest point is at/below the rim, →1 once the whole shirt
    # has been lifted ``clear_margin`` above the rim.
    clearance = torch.clamp((bottom_z - rim) / clear_margin, min=0.0, max=1.0)

    gate = in_xy
    if hasattr(env, "was_grasped"):
        gate = gate & env.was_grasped
    return torch.where(gate, clearance, torch.zeros_like(clearance))


def release_event_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-time bonus (N,) for opening the gripper over the drum at clearance.

    Reads the env's one-shot ``release_event`` flag (1.0 for the single step a
    good release fires — grasp opened while centred over + above the drum rim),
    so the reward is a discrete drop event rather than a per-step openness term
    the policy could farm or avoid.  Anti-hover: it pays only for *committing* to
    the drop.
    """
    if hasattr(env, "release_event"):
        return env.release_event
    return torch.zeros(env.num_envs, device=env.device)


def carry_time_penalty(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    belt_height: float,
    lift_threshold: float = 0.20,
) -> torch.Tensor:
    """Per-step time cost (N,) while holding a *lifted* shirt that isn't placed.

    Anti-hover pressure targeted at the exact failure mode: a grasped shirt held
    up (grasp point well above the belt) without committing to the drop.  Gated on
    ``lift_threshold`` so it never touches the reach/grasp phase — only sustained
    hovering bleeds reward, pushing the policy to finish the placement.
    """
    if not (hasattr(env, "was_grasped") and hasattr(env, "was_placed")):
        return torch.zeros(env.num_envs, device=env.device)
    gp = _shirt_grasp_point(env, shirt_name)
    local_z = gp[:, 2] - env.scene.env_origins[:, 2]
    lifted = local_z > (belt_height + lift_threshold)
    active = env.was_grasped & (~env.was_placed) & lifted
    return active.to(torch.float32)


def placed_and_settled(env: ManagerBasedRLEnv, settle_steps: int = 15) -> torch.Tensor:
    """Terminate ``settle_steps`` control steps after a placement latched.

    Ends the episode shortly after a successful drop so the per-step shaping can
    no longer be farmed by holding — the terminal success reward dominates.
    """
    if hasattr(env, "steps_since_place"):
        return env.steps_since_place >= settle_steps
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def shirt_in_drum(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """True for envs where the shirt proxy is inside the target drum."""
    shirt: RigidObject = env.scene[shirt_name]
    pos_d = _get_world_pos(env.scene[drum_name], env=env)
    return _in_upright_cylinder(shirt.data.root_pos_w, pos_d, bin_geom)


def shirt_in_target(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    """Per-step success reward = fraction of cloth particles in the drum.

    Uses the cloth-aware particle fraction rather than centroid-in-cylinder, so a
    partly-draped shirt is scored honestly and getting *more* cloth in is
    rewarded.  Gated on ``was_grasped`` to prevent reward hacking.
    """
    frac = _shirt_fraction_in_drum(env, shirt_name, drum_name, bin_geom)
    if hasattr(env, "was_grasped"):
        gate = env.was_grasped.to(torch.float32)
        # Only reward *released* cloth in the drum — a still-gripped shirt dipped
        # into the drum earns nothing, so the policy must actually drop it.
        if hasattr(env, "grasp_active"):
            gate = gate * (~env.grasp_active).to(torch.float32)
        return frac * gate
    return frac


def shirt_release_above_target(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    finger_cfg: SceneEntityCfg,
    belt_height: float,
    rim_clearance: float = 0.10,
    drum_radius: float = 0.2735,
) -> torch.Tensor:
    """Reward for opening the gripper when the carried shirt is above the drum."""
    pos_d = _get_world_pos(env.scene[drum_name], env=env)
    # Use the grasp point (the gripper-held part of the shirt) for "above drum".
    pos_s = _shirt_grasp_point(env, shirt_name)

    d_xy = torch.sqrt(
        (pos_s[:, 0] - pos_d[:, 0]) ** 2 + (pos_s[:, 1] - pos_d[:, 1]) ** 2
    )
    in_xy = d_xy < drum_radius

    local_z = pos_s[:, 2] - env.scene.env_origins[:, 2]
    above_rim = local_z > (belt_height + rim_clearance)

    robot: Articulation = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids[0]]
    closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
    openness = 1.0 - closure

    gate = in_xy & above_rim
    if hasattr(env, "was_grasped"):
        gate = gate & env.was_grasped

    return torch.where(gate, openness, torch.zeros_like(openness))


# ---------------------------------------------------------------------------
# Return-to-neutral after placement (mirrors cube_place)
# ---------------------------------------------------------------------------

def return_to_neutral(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 0.25,
) -> torch.Tensor:
    """Reward for returning joints to their defaults once the shirt is placed.

    Uses per-joint normalized deviation (scaled by each joint's soft position
    range) so prismatic (metres) and revolute (radians) joints are
    commensurate, then ``1 - tanh(rms_dev / std)``.  Gated on ``was_placed`` so
    it only activates after a successful placement.  Identical in spirit to the
    cube_place reward of the same name.
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
    if hasattr(env, "was_placed"):
        result = result * env.was_placed.float()
    return result


def was_placed_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Binary observation (N, 1): 1.0 once the shirt has been placed in the drum."""
    if hasattr(env, "was_placed"):
        return env.was_placed.float().unsqueeze(-1)
    return torch.zeros(env.num_envs, 1, device=env.device)


# ---------------------------------------------------------------------------
# Metric-only rewards
# ---------------------------------------------------------------------------

def shirt_place_success_bonus(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    drum_name: str,
    bin_geom: BinCylinder,
) -> torch.Tensor:
    return shirt_in_target(env, shirt_name, drum_name, bin_geom)


def shirt_grasp_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    shirt_name: str,
    belt_height: float,
    lift_threshold: float = 0.06,
    proximity_threshold: float = 0.10,
) -> torch.Tensor:
    """Binary 1.0 when the shirt is grasped and lifted off the belt."""
    target = _shirt_grasp_point(env, shirt_name)
    local_z = target[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = local_z > (belt_height + lift_threshold)

    if hasattr(env, "grasp_active"):
        return (is_lifted & env.grasp_active).to(torch.float32)

    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    is_close = torch.norm(target - ee, dim=-1) < proximity_threshold
    return (is_lifted & is_close).to(torch.float32)


def ee_to_shirt_distance_metric(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    shirt_name: str,
) -> torch.Tensor:
    """Raw distance from grasp centre to the shirt's grasp point (TensorBoard)."""
    ee = _grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    return torch.norm(_shirt_grasp_point(env, shirt_name) - ee, dim=-1)


def shirt_off_conveyor_penalty(
    env: ManagerBasedRLEnv,
    shirt_name: str,
    bounds: ConveyorBounds,
) -> torch.Tensor:
    """Penalty (1.0) when shirt proxy is outside conveyor region."""
    shirt: RigidObject = env.scene[shirt_name]
    local = shirt.data.root_pos_w - env.scene.env_origins
    out_y = (local[:, 1] < bounds.y_min) | (local[:, 1] > bounds.y_max)
    out_z = local[:, 2] < bounds.z_min
    return (out_y | out_z).to(torch.float32)

# rewards_cube_sorting_updated.py
#
# Reward terms for cube sorting task.
# Uses EE cfg (tool_link) directly (no need for body id at config time).

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class BinCylinder:
    radius: float
    height: float


def _asset(env, name: str):
    return env.scene[name]


def _root_pos(asset) -> torch.Tensor:
    # For RigidObjectCollectionData, use object_pos_w
    if hasattr(asset, 'data') and hasattr(asset.data, 'object_pos_w'):
        return asset.data.object_pos_w
    # For RigidObject, use root_pos_w
    elif hasattr(asset, 'data') and hasattr(asset.data, 'root_pos_w'):
        return asset.data.root_pos_w
    # For XformPrimView, use get_world_poses()
    elif hasattr(asset, 'get_world_poses'):
        positions, _ = asset.get_world_poses()
        return positions
    else:
        raise AttributeError(f"Cannot get position from asset of type {type(asset)}")


def _active_mask(env, pos_w: torch.Tensor, parking_x_threshold: float) -> torch.Tensor:
    """Check if objects are active (not parked) by comparing local x position."""
    env_origins = env.scene.env_origins
    if pos_w.ndim == 2:
        local_pos = pos_w - env_origins
        return local_pos[:, 0] < parking_x_threshold
    local_pos = pos_w - env_origins[:, None, :]
    return local_pos[..., 0] < parking_x_threshold


def _passed_threshold(env, pos_w: torch.Tensor, x_threshold: float) -> torch.Tensor:
    """Check if objects have passed the x threshold in local coordinates."""
    env_origins = env.scene.env_origins
    if pos_w.ndim == 2:
        local_pos = pos_w - env_origins
        return local_pos[:, 0] > x_threshold
    local_pos = pos_w - env_origins[:, None, :]
    return local_pos[..., 0] > x_threshold


def _in_upright_cylinder(points_w: torch.Tensor, center_w: torch.Tensor, cyl: BinCylinder) -> torch.Tensor:
    if points_w.ndim == 2:
        rel = points_w - center_w
        r2 = rel[:, 0] ** 2 + rel[:, 1] ** 2
        inside_xy = r2 <= cyl.radius ** 2
        inside_z = (rel[:, 2] >= 0.0) & (rel[:, 2] <= cyl.height)
        return inside_xy & inside_z
    rel = points_w - center_w[:, None, :]
    r2 = rel[..., 0] ** 2 + rel[..., 1] ** 2
    inside_xy = r2 <= cyl.radius ** 2
    inside_z = (rel[..., 2] >= 0.0) & (rel[..., 2] <= cyl.height)
    return inside_xy & inside_z


def green_in_target(env: "ManagerBasedRLEnv", green_name: str, target_bin_name: str, bin_geom: BinCylinder,
                    parking_x_threshold: float = 50.0) -> torch.Tensor:
    green = _asset(env, green_name)
    target = _asset(env, target_bin_name)
    pos_g = _root_pos(green)
    pos_t = _root_pos(target)
    active = _active_mask(env, pos_g, parking_x_threshold)
    inside = _in_upright_cylinder(pos_g, pos_t, bin_geom)
    return torch.sum((inside & active).to(torch.float32), dim=1)


def red_in_target(env: "ManagerBasedRLEnv", red_name: str, target_bin_name: str, bin_geom: BinCylinder,
                  parking_x_threshold: float = 50.0) -> torch.Tensor:
    red = _asset(env, red_name)
    target = _asset(env, target_bin_name)
    pos_r = _root_pos(red)
    pos_t = _root_pos(target)
    active = _active_mask(env, pos_r, parking_x_threshold)
    inside = _in_upright_cylinder(pos_r, pos_t, bin_geom)
    return torch.sum((inside & active).to(torch.float32), dim=1)


def green_missed(env: "ManagerBasedRLEnv", green_name: str, x_passed_threshold: float,
                 target_bin_name: str, bin_geom: BinCylinder,
                 parking_x_threshold: float = 50.0) -> torch.Tensor:
    green = _asset(env, green_name)
    target = _asset(env, target_bin_name)
    pos_g = _root_pos(green)
    pos_t = _root_pos(target)
    active = _active_mask(env, pos_g, parking_x_threshold)
    passed = _passed_threshold(env, pos_g, x_passed_threshold)
    in_target = _in_upright_cylinder(pos_g, pos_t, bin_geom)
    missed = active & passed & (~in_target)
    return torch.sum(missed.to(torch.float32), dim=1)


def ee_to_nearest_green_l2(env: "ManagerBasedRLEnv", ee_cfg: SceneEntityCfg, green_name: str,
                           parking_x_threshold: float = 50.0) -> torch.Tensor:
    robot = _asset(env, ee_cfg.name)
    ee_pos = robot.data.body_pos_w[:, ee_cfg.body_ids[0], :]
    green = _asset(env, green_name)
    pos_g = _root_pos(green)
    active = _active_mask(env, pos_g, parking_x_threshold)

    d2 = torch.sum((pos_g - ee_pos[:, None, :]) ** 2, dim=-1)
    d2 = torch.where(active, d2, torch.full_like(d2, float("inf")))
    d = torch.sqrt(torch.min(d2, dim=1).values.clamp_min(1e-12))
    return torch.where(torch.isfinite(d), d, torch.zeros_like(d))


def cube_lifted_bonus(env: "ManagerBasedRLEnv", ee_cfg: SceneEntityCfg, green_name: str,
                      belt_height: float = 0.8, lift_threshold: float = 0.1,
                      proximity_threshold: float = 0.15,
                      parking_x_threshold: float = 50.0) -> torch.Tensor:
    """Reward for having a cube lifted above the belt when gripper is nearby.
    
    Returns the number of cubes that are:
    1. Within proximity_threshold of the EE
    2. Above belt_height + lift_threshold
    """
    robot = _asset(env, ee_cfg.name)
    ee_pos = robot.data.body_pos_w[:, ee_cfg.body_ids[0], :]  # (N, 3)
    green = _asset(env, green_name)
    pos_g = _root_pos(green)  # (N, M, 3)
    active = _active_mask(env, pos_g, parking_x_threshold)  # (N, M)
    
    # Distance from EE to each cube
    d = torch.sqrt(torch.sum((pos_g - ee_pos[:, None, :]) ** 2, dim=-1))  # (N, M)
    
    # Check if cube is close to EE and lifted
    env_origins = env.scene.env_origins
    local_pos = pos_g - env_origins[:, None, :]  # (N, M, 3) in local coords
    is_lifted = local_pos[..., 2] > (belt_height + lift_threshold)
    is_close = d < proximity_threshold
    
    bonus_mask = active & is_lifted & is_close
    return torch.sum(bonus_mask.to(torch.float32), dim=1)


def cube_approaching_target(env: "ManagerBasedRLEnv", green_name: str, target_bin_name: str,
                            belt_height: float = 0.8, lift_threshold: float = 0.05,
                            parking_x_threshold: float = 50.0) -> torch.Tensor:
    """Dense shaping reward: inversely proportional to distance from lifted cube to target.
    
    Returns negative distance to encourage approaching the target with a lifted cube.
    Only counts cubes that are above belt_height + lift_threshold.
    """
    green = _asset(env, green_name)
    target = _asset(env, target_bin_name)
    pos_g = _root_pos(green)  # (N, M, 3)
    pos_t = _root_pos(target)  # (N, 3)
    active = _active_mask(env, pos_g, parking_x_threshold)  # (N, M)
    
    # Check if cubes are lifted
    env_origins = env.scene.env_origins
    local_pos = pos_g - env_origins[:, None, :]
    is_lifted = local_pos[..., 2] > (belt_height + lift_threshold)
    
    # Distance from each cube to target (XY plane only for horizontal approach)
    d_xy = torch.sqrt((pos_g[..., 0] - pos_t[:, None, 0]) ** 2 + 
                      (pos_g[..., 1] - pos_t[:, None, 1]) ** 2)  # (N, M)
    
    # Only consider active, lifted cubes
    valid = active & is_lifted
    d_xy_valid = torch.where(valid, d_xy, torch.full_like(d_xy, float("inf")))
    
    # Return minimum distance to target (0 if no lifted cubes)
    min_d = torch.min(d_xy_valid, dim=1).values
    return torch.where(torch.isfinite(min_d), min_d, torch.zeros_like(min_d))


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = env.action_manager.action
    ap = env.action_manager.prev_action
    return torch.sum((a - ap) ** 2, dim=1)
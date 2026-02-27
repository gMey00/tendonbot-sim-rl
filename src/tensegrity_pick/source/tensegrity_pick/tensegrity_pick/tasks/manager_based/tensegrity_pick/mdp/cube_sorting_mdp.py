# cube_sorting_mdp_updated.py
#
# MDP helpers for cube-sorting task (conveyor along +X).
# Includes:
#   - EE observations (tool_link)
#   - Nearest-object observations
#   - Pattern-1 cube reset (activate k cubes, park the rest)
#   - IsaacLab-optimized conveyor effect (vectorized velocity injection)
#   - Belt speed randomization stored per-env (tensor [num_envs])
#   - Termination helper: all active cubes passed +X threshold

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def _asset(env, name: str):
    return env.scene[name]


# ---------------------------
# EE observations
# ---------------------------

def ee_pos_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = _asset(env, asset_cfg.name)
    return robot.data.body_pos_w[:, asset_cfg.body_ids[0], :]


def ee_lin_vel_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = _asset(env, asset_cfg.name)
    return robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]


def nearest_obj_rel_pos(env, ee_cfg: SceneEntityCfg, obj_collection_name: str, parking_x_threshold: float = 50.0) -> torch.Tensor:
    ee = ee_pos_w(env, ee_cfg)  # [N,3]
    coll = _asset(env, obj_collection_name)
    pos = coll.data.object_pos_w  # [N,M,3] for collections
    if pos.ndim == 2:
        return pos - ee
    
    # Convert world positions to local to check parking threshold
    env_origins = env.scene.env_origins  # (N, 3)
    local_pos = pos - env_origins[:, None, :]  # (N, M, 3) in local coords
    
    active = local_pos[..., 0] < parking_x_threshold
    d2 = torch.sum((pos - ee[:, None, :]) ** 2, dim=-1)
    d2 = torch.where(active, d2, torch.full_like(d2, float("inf")))
    idx = torch.argmin(d2, dim=1)
    has_any = torch.isfinite(torch.min(d2, dim=1).values)
    rel = pos[torch.arange(pos.shape[0], device=pos.device), idx, :] - ee
    return torch.where(has_any[:, None], rel, torch.zeros_like(rel))


# ---------------------------
# Termination: all active passed along +X
# ---------------------------

def all_active_passed_x(env, obj_collection_names: Tuple[str, ...], x_threshold: float, parking_x_threshold: float = 50.0) -> torch.Tensor:
    done = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    env_origins = env.scene.env_origins  # (N, 3)
    
    for name in obj_collection_names:
        coll = _asset(env, name)
        pos = coll.data.object_pos_w
        
        if pos.ndim == 2:
            # Convert to local coords for threshold checks
            local_pos = pos - env_origins
            active = local_pos[:, 0] < parking_x_threshold
            passed = local_pos[:, 0] > x_threshold
            ok = torch.where(active, passed, torch.ones_like(passed, dtype=torch.bool))
            done &= ok
        else:
            # Convert to local coords for threshold checks
            local_pos = pos - env_origins[:, None, :]  # (N, M, 3)
            active = local_pos[..., 0] < parking_x_threshold
            passed = local_pos[..., 0] > x_threshold
            ok = torch.where(active, passed, torch.ones_like(passed, dtype=torch.bool))
            done &= torch.all(ok, dim=1)
    return done


# ---------------------------
# Pattern-1 cube reset
# ---------------------------

@dataclass
class SpawnBox:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    yaw_range: Tuple[float, float] = (-3.14159, 3.14159)


def reset_cube_pool_pattern1(
    env,
    env_ids: torch.Tensor,
    red_name: str,
    spawn_box: SpawnBox,
    num_green_active: int,
    num_red_active: int,
    green_name: str = "green_cubes",
    parking_pose: Tuple[float, float, float] = (100.0, 100.0, 1.0),
):
    device = env.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    N = env_ids.numel()

    # Get environment origins for the specified env_ids
    # env.scene.env_origins is shape (num_envs, 3)
    env_origins = env.scene.env_origins[env_ids]  # (N, 3)

    def sample_pose(n_envs: int, n_objs: int):
        # Sample local positions
        x = torch.empty((n_envs, n_objs), device=device).uniform_(*spawn_box.x_range)
        y = torch.empty((n_envs, n_objs), device=device).uniform_(*spawn_box.y_range)
        z = torch.empty((n_envs, n_objs), device=device).uniform_(*spawn_box.z_range)
        yaw = torch.empty((n_envs, n_objs), device=device).uniform_(*spawn_box.yaw_range)

        qw = torch.cos(0.5 * yaw)
        qz = torch.sin(0.5 * yaw)
        quat = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)  # (N,k,4) wxyz
        local_pos = torch.stack([x, y, z], dim=-1)  # (N, k, 3)
        
        # Add environment origins to convert to world coordinates
        # env_origins is (N, 3), needs to be (N, 1, 3) for broadcasting
        world_pos = local_pos + env_origins[:, None, :]
        return world_pos, quat

    def apply_pool(name: str, k_active: int):
        pool = _asset(env, name)

        # M = number of objects in the collection
        # pool.data.object_pos_w is typically (num_envs, M, 3)
        M = pool.data.object_pos_w.shape[1]
        k = min(int(k_active), int(M))

        pos_a, quat_a = sample_pose(N, k)

        if M > k:
            # Parking pose in local coordinates, then add env_origins for world coords
            local_parking = torch.tensor(parking_pose, device=device).view(1, 1, 3).repeat(N, M - k, 1)
            pos_p = local_parking + env_origins[:, None, :]  # Add env_origins for per-env parking
            quat_p = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 1, 4).repeat(N, M - k, 1)
            pos = torch.cat([pos_a, pos_p], dim=1)
            quat = torch.cat([quat_a, quat_p], dim=1)
        else:
            pos, quat = pos_a, quat_a

        # API expects object_pose with shape (N, M, 7) - pos and quat concatenated
        object_pose = torch.cat([pos, quat], dim=-1)
        # API expects object_velocity with shape (N, M, 6) - lin_vel and ang_vel concatenated
        object_velocity = torch.zeros((N, M, 6), device=device)

        pool.write_object_pose_to_sim(object_pose, env_ids=env_ids)
        pool.write_object_velocity_to_sim(object_velocity, env_ids=env_ids)

    apply_pool(green_name, num_green_active)
    apply_pool(red_name, num_red_active)


# ---------------------------
# Belt speed storage / randomization (per-env tensor)
# ---------------------------

def sample_and_store_belt_speed(env, env_ids: torch.Tensor, high: float, low: float = 0.2, key: str = "belt_speed"):
    """Sample a belt speed per sub-environment and store it in env.extras[key] as a tensor [num_envs].
    
    Only updates the speeds for the specified env_ids, preserving speeds for other environments.
    """
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    
    # Initialize the full tensor if it doesn't exist
    if key not in env.extras:
        env.extras[key] = torch.empty((env.num_envs,), device=env.device).uniform_(low, high)
    
    # Only update speeds for the specified environment indices
    env_ids_device = env_ids.to(device=env.device, dtype=torch.long)
    new_speeds = torch.empty((len(env_ids_device),), device=env.device).uniform_(low, high)
    env.extras[key][env_ids_device] = new_speeds


# ---------------------------
# IsaacLab-optimized conveyor effect (vectorized)
# ---------------------------

BeltSpeed = Union[float, torch.Tensor, None]


def apply_conveyor_velocity_to_cubes(
    env,
    env_ids: torch.Tensor,
    cube_collection_names: Sequence[str],
    belt_speed: BeltSpeed = None,
    belt_speed_key: str = "belt_speed",
    belt_axis: str = "x",
    belt_height: float = 0.793,
    on_belt_height_tol: float = 0.03,
    lift_disable_height: float = 0.10,
    belt_center_y: float = 0.0,
    belt_half_width: float = 0.40,
    parking_x_threshold: float = 50.0,
):
    """Inject belt tangential velocity to cubes that are on the belt.

    - If belt_speed is None: reads env.extras[belt_speed_key] (tensor [N]) or defaults to 0.5.
    - If belt_speed is float: same speed for all envs.
    - If belt_speed is tensor: expects shape [N].
    """
    if belt_speed is None:
        extras = getattr(env, "extras", None)
        if extras is not None and isinstance(extras, dict) and belt_speed_key in extras:
            belt_speed = extras[belt_speed_key]
        else:
            belt_speed = 0.5

    if isinstance(belt_speed, float) or isinstance(belt_speed, int):
        speeds = torch.full((env.num_envs,), float(belt_speed), device=env.device)
    else:
        speeds = belt_speed  # tensor [N]

    axis_idx = 0 if belt_axis.lower() == "x" else 1
    speeds_bc = speeds[:, None]  # [N,1] broadcast to [N,M]

    # Ensure env_ids is properly shaped - for interval events it may include all envs
    env_ids_device = env_ids.to(device=env.device, dtype=torch.long)

    # Get environment origins for coordinate conversion
    env_origins = env.scene.env_origins[env_ids_device]  # (K, 3)

    for name in cube_collection_names:
        cubes = _asset(env, name)
        pos = cubes.data.object_pos_w  # world positions
        vel = cubes.data.object_lin_vel_w

        if pos.ndim == 2:
            pos = pos[:, None, :]
            vel = vel[:, None, :]

        # Select only the environments being updated
        pos_sel = pos[env_ids_device]  # (K, M, 3) in world coords
        vel_sel = vel[env_ids_device]
        speeds_sel = speeds_bc[env_ids_device]

        # Convert world positions to local positions by subtracting env_origins
        local_pos = pos_sel - env_origins[:, None, :]  # (K, M, 3) in local coords

        # Check bounds in local coordinates
        active = local_pos[..., 0] < parking_x_threshold
        within_width = torch.abs(local_pos[..., 1] - belt_center_y) <= belt_half_width
        on_height = torch.abs(local_pos[..., 2] - belt_height) <= on_belt_height_tol
        not_lifted = local_pos[..., 2] <= (belt_height + lift_disable_height)

        mask = active & within_width & on_height & not_lifted  # [K,M] where K = len(env_ids)

        # --- linear velocity: set belt-axis component for cubes on the belt ---
        vel_new = vel_sel.clone()
        vel_new[..., axis_idx] = torch.where(mask, speeds_sel.expand_as(vel_new[..., axis_idx]), vel_new[..., axis_idx])

        # --- angular velocity: zero out for cubes on the belt ---
        # A real conveyor carries objects without inducing rotation.  If we only
        # inject linear velocity the physics solver sees a sliding contact and
        # generates unphysical forward spin.  Zeroing angular velocity for
        # on-belt cubes prevents this artifact.
        ang_vel_full = cubes.data.object_ang_vel_w
        if ang_vel_full.ndim == 2:
            ang_vel_full = ang_vel_full[:, None, :]
        ang_vel = ang_vel_full[env_ids_device].clone()
        mask_3 = mask.unsqueeze(-1).expand_as(ang_vel)
        ang_vel = torch.where(mask_3, torch.zeros_like(ang_vel), ang_vel)

        # API expects object_velocity with shape (K, M, 6) - lin_vel and ang_vel concatenated
        object_velocity = torch.cat([vel_new, ang_vel], dim=-1)
        cubes.write_object_velocity_to_sim(object_velocity, env_ids=env_ids_device)
# cube_sorting_mdp.py
#
# MDP helpers for cube-sorting task (conveyor along +X).
# Works with a single unified RigidObjectCollection and integer labels.

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


def nearest_obj_rel_pos(
    env,
    ee_cfg: SceneEntityCfg,
    collection_name: str,
    label: int | None = None,
    parking_x_threshold: float = 50.0,
) -> torch.Tensor:
    """Relative position of the nearest (optionally label-filtered) active cube."""
    ee = ee_pos_w(env, ee_cfg)
    coll = _asset(env, collection_name)
    pos = coll.data.object_pos_w
    if pos.ndim == 2:
        return pos - ee

    env_origins = env.scene.env_origins
    local_pos = pos - env_origins[:, None, :]
    active = local_pos[..., 0] < parking_x_threshold

    if label is not None:
        active = active & (env.cube_labels[None, :] == label)

    d2 = torch.sum((pos - ee[:, None, :]) ** 2, dim=-1)
    d2 = torch.where(active, d2, torch.full_like(d2, float("inf")))
    idx = torch.argmin(d2, dim=1)
    has_any = torch.isfinite(torch.min(d2, dim=1).values)
    rel = pos[torch.arange(pos.shape[0], device=pos.device), idx, :] - ee
    return torch.where(has_any[:, None], rel, torch.zeros_like(rel))


# ---------------------------
# Termination: all active cubes passed along +X
# ---------------------------

def all_active_passed_x(
    env,
    collection_name: str,
    x_threshold: float,
    parking_x_threshold: float = 50.0,
) -> torch.Tensor:
    """True when every active cube has passed *x_threshold* in local coords."""
    coll = _asset(env, collection_name)
    pos = coll.data.object_pos_w
    env_origins = env.scene.env_origins

    if pos.ndim == 2:
        local_pos = pos - env_origins
        active = local_pos[:, 0] < parking_x_threshold
        passed = local_pos[:, 0] > x_threshold
        ok = torch.where(active, passed, torch.ones_like(passed, dtype=torch.bool))
        return ok
    else:
        local_pos = pos - env_origins[:, None, :]
        active = local_pos[..., 0] < parking_x_threshold
        passed = local_pos[..., 0] > x_threshold
        ok = torch.where(active, passed, torch.ones_like(passed, dtype=torch.bool))
        return torch.all(ok, dim=1)


def all_targets_placed(env) -> torch.Tensor:
    """True when all active target cubes have been placed in the drum.

    Relies on ``_targets_placed`` and ``_num_active_green`` tracked by
    :class:`TensegrityCubeSortEnv`.
    """
    return env._targets_placed >= max(env._num_active_green, 1)


# ---------------------------
# Cube reset
# ---------------------------

@dataclass
class SpawnBox:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    yaw_range: Tuple[float, float] = (-3.14159, 3.14159)
    min_x_spacing: float = 0.0


def _sample_spaced_x(
    num_envs: int,
    num_cubes: int,
    x_range: Tuple[float, float],
    min_spacing: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample x positions with guaranteed minimum spacing between cubes.

    Uses sorted uniform sampling with forced gaps.  Falls back to uniform
    sampling when spacing constraints cannot be satisfied.
    """
    x_lo, x_hi = x_range
    needed = (num_cubes - 1) * min_spacing
    available = x_hi - x_lo - needed

    if available <= 0 or num_cubes <= 1 or min_spacing <= 0:
        return torch.empty((num_envs, num_cubes), device=device).uniform_(x_lo, x_hi)

    raw = torch.empty((num_envs, num_cubes), device=device).uniform_(0.0, available)
    sorted_raw, _ = raw.sort(dim=1)
    offsets = torch.arange(num_cubes, device=device).float() * min_spacing
    return sorted_raw + offsets + x_lo


def reset_cubes(
    env,
    env_ids: torch.Tensor,
    collection_name: str,
    spawn_box: SpawnBox,
    active_per_label: dict[int | str, int],
    parking_pose: Tuple[float, float, float] = (100.0, 100.0, 1.0),
):
    """Reset the unified cube collection, activating *active_per_label[label]* cubes per label."""
    device = env.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    N = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    pool = _asset(env, collection_name)
    M = pool.data.object_pos_w.shape[1]
    labels = env.cube_labels  # (M,)

    # Build per-cube activation mask (M,) — first `count` cubes of each label
    cube_active = torch.zeros(M, dtype=torch.bool, device=device)
    for lbl, count in active_per_label.items():
        lbl = int(lbl)
        label_indices = (labels == lbl).nonzero(as_tuple=True)[0]
        cube_active[label_indices[:count]] = True

    num_active = int(cube_active.sum().item())
    active_indices = cube_active.nonzero(as_tuple=True)[0]
    num_parked = M - num_active

    # Sample poses for active cubes (with minimum x-spacing to prevent clustering)
    x = _sample_spaced_x(N, num_active, spawn_box.x_range, spawn_box.min_x_spacing, device)

    # Shuffle position assignment across active cubes so no label
    # is systematically assigned the nearest/farthest spawn slot.
    perm = torch.argsort(torch.rand(N, num_active, device=device), dim=1)
    x = x.gather(1, perm)

    y = torch.empty((N, num_active), device=device).uniform_(*spawn_box.y_range)
    z = torch.empty((N, num_active), device=device).uniform_(*spawn_box.z_range)
    yaw = torch.empty((N, num_active), device=device).uniform_(*spawn_box.yaw_range)

    qw = torch.cos(0.5 * yaw)
    qz = torch.sin(0.5 * yaw)
    zeros = torch.zeros_like(qw)
    quat_active = torch.stack([qw, zeros, zeros, qz], dim=-1)
    local_pos_active = torch.stack([x, y, z], dim=-1)
    world_pos_active = local_pos_active + env_origins[:, None, :]

    # Build full (N, M, 7) pose tensor
    parking_world = (
        torch.tensor(parking_pose, device=device).unsqueeze(0).unsqueeze(0)
        + env_origins[:, None, :]
    )
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    pos_all = parking_world.expand(N, M, 3).clone()
    quat_all = identity_quat.unsqueeze(0).unsqueeze(0).expand(N, M, 4).clone()

    pos_all[:, active_indices, :] = world_pos_active
    quat_all[:, active_indices, :] = quat_active

    object_pose = torch.cat([pos_all, quat_all], dim=-1)
    object_velocity = torch.zeros((N, M, 6), device=device)

    pool.write_object_pose_to_sim(object_pose, env_ids=env_ids)
    pool.write_object_velocity_to_sim(object_velocity, env_ids=env_ids)


# ---------------------------
# Belt speed storage / randomization (per-env tensor)
# ---------------------------

def sample_and_store_belt_speed(
    env,
    env_ids: torch.Tensor,
    high: float,
    low: float = 0.2,
    key: str = "belt_speed",
):
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}

    if key not in env.extras:
        env.extras[key] = torch.empty((env.num_envs,), device=env.device).uniform_(low, high)

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
    collection_name: str,
    belt_speed: BeltSpeed = None,
    belt_speed_key: str = "belt_speed",
    belt_axis: str = "x",
    belt_height: float = 0.793,
    on_belt_height_tol: float = 0.03,
    lift_disable_height: float = 0.10,
    belt_center_y: float = 0.0,
    belt_half_width: float = 0.40,
    parking_x_threshold: float = 50.0,
    stop_when_target_in_reach: bool = False,
    target_label: int = 0,
    workspace_x_min: float = -0.05,
    workspace_x_max: float = 0.35,
):
    """Inject belt velocity to on-belt cubes in a single collection.

    When *stop_when_target_in_reach* is True the belt pauses for any
    environment where the most-upstream active target cube (label ==
    *target_label*) has entered the workspace (local x > *workspace_x_min*).
    This gives the agent time to handle each cube before it passes.
    """
    if belt_speed is None:
        extras = getattr(env, "extras", None)
        if extras is not None and isinstance(extras, dict) and belt_speed_key in extras:
            belt_speed = extras[belt_speed_key]
        else:
            belt_speed = 0.5

    if isinstance(belt_speed, (float, int)):
        speeds = torch.full((env.num_envs,), float(belt_speed), device=env.device)
    else:
        speeds = belt_speed

    # ── Conveyor-stop logic ───────────────────────────────────────────
    if stop_when_target_in_reach:
        cubes_all = _asset(env, collection_name)
        pos_all = cubes_all.data.object_pos_w
        if pos_all.ndim == 2:
            pos_all = pos_all[:, None, :]
        local_all = pos_all - env.scene.env_origins[:, None, :]
        labels = env.cube_labels  # (M,)

        is_target = labels == target_label  # (M,)
        is_active = local_all[..., 0] < parking_x_threshold  # (N, M)
        # Most-upstream = smallest local x among active targets
        target_active = is_active & is_target[None, :]  # (N, M)
        local_x = local_all[..., 0]  # (N, M)
        local_x_masked = torch.where(
            target_active, local_x, torch.full_like(local_x, float("inf")),
        )
        most_upstream_x = local_x_masked.min(dim=1).values  # (N,)
        # Stop when most-upstream target has entered workspace
        should_stop = most_upstream_x > workspace_x_min  # (N,)
        speeds = torch.where(should_stop, torch.zeros_like(speeds), speeds)

    axis_idx = 0 if belt_axis.lower() == "x" else 1
    speeds_bc = speeds[:, None]

    env_ids_device = env_ids.to(device=env.device, dtype=torch.long)
    env_origins = env.scene.env_origins[env_ids_device]

    cubes = _asset(env, collection_name)
    pos = cubes.data.object_pos_w
    vel = cubes.data.object_lin_vel_w

    if pos.ndim == 2:
        pos = pos[:, None, :]
        vel = vel[:, None, :]

    pos_sel = pos[env_ids_device]
    vel_sel = vel[env_ids_device]
    speeds_sel = speeds_bc[env_ids_device]

    local_pos = pos_sel - env_origins[:, None, :]

    active = local_pos[..., 0] < parking_x_threshold
    within_width = torch.abs(local_pos[..., 1] - belt_center_y) <= belt_half_width
    on_height = torch.abs(local_pos[..., 2] - belt_height) <= on_belt_height_tol
    not_lifted = local_pos[..., 2] <= (belt_height + lift_disable_height)

    mask = active & within_width & on_height & not_lifted

    vel_new = vel_sel.clone()
    vel_new[..., axis_idx] = torch.where(
        mask, speeds_sel.expand_as(vel_new[..., axis_idx]), vel_new[..., axis_idx],
    )

    ang_vel_full = cubes.data.object_ang_vel_w
    if ang_vel_full.ndim == 2:
        ang_vel_full = ang_vel_full[:, None, :]
    ang_vel = ang_vel_full[env_ids_device].clone()
    mask_3 = mask.unsqueeze(-1).expand_as(ang_vel)
    ang_vel = torch.where(mask_3, torch.zeros_like(ang_vel), ang_vel)

    object_velocity = torch.cat([vel_new, ang_vel], dim=-1)
    cubes.write_object_velocity_to_sim(object_velocity, env_ids=env_ids_device)

# cloth_sorting_mdp.py
#
# Generic observation / reward functions shared by the cloth-sorting pipeline
# tasks (shirt_pick, shirt_present, shirt_distribute).  All shirt state is
# read from the kinematic centroid proxy (``shirt_proxy``) or the env's cloth
# properties, matching the pattern validated in shirt_place.
#
# Camera-realistic observation terms (down-sampled surface points, keypoints
# with visibility flags, projected coverage) are pipeline TODOs — see
# doc/TODO.md and doc/reports/RESEARCH_cloth_sorting_pipeline.md §6.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .gripper_cfg import grasp_center_w, get_world_pos

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def shirt_rel_pos(
    env: "ManagerBasedRLEnv", ee_cfg: SceneEntityCfg, shirt_name: str = "shirt_proxy",
) -> torch.Tensor:
    """Shirt centroid relative to the grasp centre (N, 3)."""
    ee = grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    shirt: RigidObject = env.scene[shirt_name]
    return shirt.data.root_pos_w - ee


def shirt_velocity(env: "ManagerBasedRLEnv", shirt_name: str = "shirt_proxy") -> torch.Tensor:
    """Shirt centroid linear velocity (N, 3)."""
    shirt: RigidObject = env.scene[shirt_name]
    return shirt.data.root_lin_vel_w


def grasp_active_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1.0 while the deterministic attachment grasp holds the cloth (N, 1).

    Falls back to zeros while the ObservationManager probes term shapes during
    ``load_managers()`` — the env's ``ClothObject`` is created only after
    ``super().__init__()`` (same guard pattern as shirt_place).
    """
    if getattr(env, "_cloth", None) is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.grasp_active.float().unsqueeze(-1)


def point_rel_shirt(
    env: "ManagerBasedRLEnv", point: tuple[float, float, float],
    shirt_name: str = "shirt_proxy",
) -> torch.Tensor:
    """Fixed env-local point (e.g. the presentation pose) relative to the shirt (N, 3)."""
    shirt: RigidObject = env.scene[shirt_name]
    target = env.scene.env_origins + torch.tensor(
        point, device=env.device, dtype=torch.float32
    ).unsqueeze(0)
    return target - shirt.data.root_pos_w


def shirt_lowest_point_rel_ee(
    env: "ManagerBasedRLEnv", ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Lowest cloth particle relative to the grasp centre (N, 3).

    The literature-standard second-grasp target on a hanging garment
    (Maitin-Shepard 2010; Doumanoglou 2014) — camera-trivial from depth.
    Requires a ``ClothSortingEnvBase`` env (``shirt_lowest_point_w``).
    """
    ee = grasp_center_w(env.scene[ee_cfg.name], ee_cfg)
    if getattr(env, "_cloth", None) is None:
        # Shape probe during load_managers() — cloth view not yet created.
        return torch.zeros_like(ee)
    return env.shirt_lowest_point_w - ee


def target_bin_rel_shirt(
    env: "ManagerBasedRLEnv", shirt_name: str = "shirt_proxy",
) -> torch.Tensor:
    """Commanded target-bin position relative to the shirt (N, 3).

    Goal-conditioned formulation (research report §4): the condition label
    only selects WHICH bin position is observed — requires the env to expose
    ``target_bin_pos_w`` (see ``ShirtDistributeEnv``).
    """
    shirt: RigidObject = env.scene[shirt_name]
    if getattr(env, "_target_bin", None) is None:
        # Shape probe during load_managers() — goal buffer not yet created.
        return torch.zeros_like(shirt.data.root_pos_w)
    return env.target_bin_pos_w - shirt.data.root_pos_w


# ---------------------------------------------------------------------------
# Rewards (stub-level shaping; task-specific structures are pipeline TODOs)
# ---------------------------------------------------------------------------

def ee_to_shirt_tanh(
    env: "ManagerBasedRLEnv", ee_cfg: SceneEntityCfg, std: float = 0.3,
    shirt_name: str = "shirt_proxy",
) -> torch.Tensor:
    """Smooth approach shaping: 1 − tanh(‖EE − shirt‖ / std) (N,)."""
    d = torch.norm(shirt_rel_pos(env, ee_cfg, shirt_name), dim=-1)
    return 1.0 - torch.tanh(d / std)


def shirt_to_point_tanh(
    env: "ManagerBasedRLEnv", point: tuple[float, float, float], std: float = 0.3,
    shirt_name: str = "shirt_proxy",
) -> torch.Tensor:
    """Shaping toward a fixed env-local point (e.g. the presentation pose) (N,)."""
    d = torch.norm(point_rel_shirt(env, point, shirt_name), dim=-1)
    return 1.0 - torch.tanh(d / std)


def shirt_to_target_bin_tanh(
    env: "ManagerBasedRLEnv", std: float = 0.4, shirt_name: str = "shirt_proxy",
) -> torch.Tensor:
    """Shaping toward the commanded target bin (goal-conditioned) (N,)."""
    d = torch.norm(target_bin_rel_shirt(env, shirt_name), dim=-1)
    return 1.0 - torch.tanh(d / std)

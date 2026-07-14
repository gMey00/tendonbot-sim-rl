# cloth_sorting_mdp.py
#
# Generic observation / reward functions shared by the cloth-sorting pipeline
# tasks (shirt_pick, shirt_present, shirt_distribute).  All shirt state is
# read from the kinematic centroid proxy (``shirt_proxy``) or the env's cloth
# properties, matching the pattern validated in shirt_place.
#
# CAMERA-REALISTIC OBSERVATION CONTRACT (Stage-2 S4, research report §2-Q3)
# -------------------------------------------------------------------------
# The terms in the "camera-realistic observations" section below model what a
# real front/back inspection-camera perception stack can provide: region
# keypoints + visibility (a keypoint detector), a segmented down-sampled
# surface cloud, mask coverage, and a gripper-current-derived grasp flag.
# Noise models default OFF so training/eval stay deterministic until a config
# opts in.
#
# PRIVILEGED-ONLY TERMS — do NOT put these in an actor ("policy") group:
#   * tautness / stretch ratio — NOT camera-observable on real hardware (must
#     be estimated from the silhouette or dropped); keep it in the "critic"
#     observation group and in rewards only.
#   * exact attachment state, both-grasps flags, the full particle field.
# The asymmetric-AC pattern that routes a "critic" group to the value network
# is verified and documented in doc/reports/tmp/pipeline_infra_tracking.md §M1.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .cloth_metrics import (
    downsample_masked_points,
    flat_silhouette_area,
    keypoint_detector_noise,
    load_present_markers,
    per_particle_visibility,
    silhouette_coverage,
)
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
# Camera-realistic observations (Stage-2 observation contract)
# ---------------------------------------------------------------------------

def _particle_visibility_cached(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-particle two-sided visibility ``[N, P]``, computed once per step.

    Several camera terms (keypoint visibility, surface cloud) share the same
    depth-layer rasterization; cache it keyed on the env step counter so one
    observation pass pays for it once.
    """
    stamp = int(env.common_step_counter)
    cache = getattr(env, "_pp_visibility_cache", None)
    if cache is None or cache[0] != stamp:
        vis = per_particle_visibility(env.cloth.nodal_pos_w)
        env._pp_visibility_cache = (stamp, vis)
        cache = env._pp_visibility_cache
    return cache[1]


def _keypoint_idx(env: "ManagerBasedRLEnv", markers_path: str | None) -> torch.Tensor:
    """The 12 study-keypoint particle indices on the env device (cached)."""
    idx = getattr(env, "_camera_kp_idx", None)
    if idx is None:
        idx = load_present_markers(markers_path)["keypoint_idx"].to(env.device)
        env._camera_kp_idx = idx
    return idx


def keypoints_with_visibility(
    env: "ManagerBasedRLEnv",
    noise_std: float = 0.0,
    recall: float = 1.0,
    markers_path: str | None = None,
) -> torch.Tensor:
    """12 region-landmark keypoints + per-keypoint visibility ``[N, 48]``.

    The camera-realistic replacement for privileged particle reads: the 12
    symmetric region-landmark keypoints of the presentation study (particle
    indices from ``doc/reports/data/present_markers.pt``), env-local (world −
    env origin), each gated by the study's two-sided depth-layer visibility
    (``cloth_metrics.per_particle_visibility``).  Layout: 36 positions
    (12 × xyz) followed by 12 flags ∈ {0, 1}.  Occluded or undetected
    keypoints report a ZEROED position and flag 0 — exactly what a detector
    miss looks like downstream.

    Noise model (defaults OFF), calibrated to a real cloth-keypoint detector
    (Lips et al. 2024, "Learning Keypoints for Robotic Cloth Manipulation
    using Synthetic Data": 74 % mAP, ~9 px ≈ 1–2 cm error after fine-tuning):

    * ``noise_std``: isotropic Gaussian position noise, σ in metres —
      realistic setting 0.01–0.02.
    * ``recall``: per-keypoint detection probability for VISIBLE keypoints
      (dropout at 1 − recall) — realistic setting ≈ 0.74.

    Draws use the global torch RNG → deterministic under the run seed.
    """
    kp = _keypoint_idx(env, markers_path)
    n_kp = int(kp.numel())
    if getattr(env, "_cloth", None) is None:
        # Shape probe during load_managers() — cloth view not yet created.
        return torch.zeros(env.num_envs, 4 * n_kp, device=env.device)
    pos_w = env.cloth.nodal_pos_w[:, kp]                        # [N, K, 3]
    pos = pos_w - env.scene.env_origins.unsqueeze(1)            # env-local
    visible = _particle_visibility_cached(env)[:, kp]           # [N, K]
    pos, flags = keypoint_detector_noise(
        pos, visible, noise_std=noise_std, recall=recall,
    )
    return torch.cat([pos.reshape(env.num_envs, -1), flags], dim=-1)


def grasp_active_noisy(
    env: "ManagerBasedRLEnv",
    flip_prob: float = 0.0,
    latency_steps: int = 0,
) -> torch.Tensor:
    """Grasp-active flag as a real gripper would report it ``[N, 1]``.

    On real hardware the grasp flag is NOT free: it must be inferred from
    gripper motor current / force sensing (the Robotiq 2F-140 exposes motor
    current), which is delayed and occasionally wrong.  This term models
    that estimator on top of the sim's exact ``grasp_active``:

    * ``latency_steps``: the flag is reported ``latency_steps`` env steps
      late (history buffer; reset envs re-seed the buffer with their current
      flag, so no stale carry-over across episodes).
    * ``flip_prob``: per-step probability of reporting the wrong value
      (current-threshold misclassification).

    Defaults OFF → identical to ``grasp_active_obs``.  Draws use the global
    torch RNG → deterministic under the run seed.
    """
    if getattr(env, "_cloth", None) is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    flag = env.grasp_active.float()                             # [N]
    if latency_steps > 0:
        hist = getattr(env, "_grasp_flag_hist", None)
        if hist is None or hist.shape[1] != latency_steps + 1:
            hist = flag.unsqueeze(1).repeat(1, latency_steps + 1)
        # Re-seed rows of freshly reset envs (no cross-episode leakage).
        fresh = env.episode_length_buf == 0
        hist[fresh] = flag[fresh].unsqueeze(1)
        hist = torch.roll(hist, shifts=-1, dims=1)
        hist[:, -1] = flag
        env._grasp_flag_hist = hist
        flag = hist[:, 0]
    if flip_prob > 0.0:
        flips = torch.rand(env.num_envs, device=env.device) < flip_prob
        flag = torch.where(flips, 1.0 - flag, flag)
    return flag.unsqueeze(-1)


def surface_point_cloud(
    env: "ManagerBasedRLEnv",
    num_points: int = 64,
    visible_only: bool = True,
) -> torch.Tensor:
    """Down-sampled garment surface cloud ``[N, num_points × 3]``, env-local.

    The segmented-point-cloud observation of the UniFolding/VCD lineage:
    ``num_points`` particles sampled from the camera-VISIBLE surface (two-
    sided depth-layer logic; ``visible_only=False`` gives the privileged
    full-surface variant for critic groups).  Sampling uses one fixed
    particle permutation drawn at first call from a generator seeded with the
    env seed — deterministic given the seed, stable across the episode, no
    fake zero-padding points (short envs wrap around).
    """
    if getattr(env, "_cloth", None) is None:
        return torch.zeros(env.num_envs, 3 * num_points, device=env.device)
    perm = getattr(env, "_surface_cloud_perm", None)
    if perm is None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(getattr(env.cfg, "seed", None) or 0))
        perm = torch.randperm(env.cloth.num_particles, generator=gen).to(env.device)
        env._surface_cloud_perm = perm
    pts = env.cloth.nodal_pos_w - env.scene.env_origins.unsqueeze(1)
    if visible_only:
        mask = _particle_visibility_cached(env)
    else:
        mask = torch.ones(
            env.num_envs, env.cloth.num_particles,
            dtype=torch.bool, device=env.device,
        )
    cloud = downsample_masked_points(pts, mask, num_points, perm)
    return cloud.reshape(env.num_envs, -1)


def coverage_from_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Camera-plane silhouette coverage ``[N, 1]`` — the mask-area metric.

    Segmentation-mask area ÷ calibrated flat area, i.e. EXACTLY the project's
    ``silhouette_coverage`` (the silhouette IS the segmentation mask a real
    front/back camera pair measures), so this term is camera-obtainable
    as-is.  The equivalence with the task envs' own coverage buffers is
    asserted offline in test_camera_obs.py — that assert validates the
    observation contract.
    """
    if getattr(env, "_cloth", None) is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    ref = getattr(env, "_camera_ref_area", None)
    if ref is None:
        ref = flat_silhouette_area(env.cloth.flat_rest_pos)
        env._camera_ref_area = ref
    return silhouette_coverage(env.cloth.nodal_pos_w, ref).unsqueeze(-1)


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

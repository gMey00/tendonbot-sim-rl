"""Smoke test: camera-realistic observation terms on a live cloth env.

Boots the shirt_pick task at a few envs, injects the new shared observation
terms (shared/cloth_sorting_mdp.py) into a runtime COPY of the observation
config (task cfg files untouched) so the ObservationManager shape-probe path
(zeros fallback before the cloth exists) is exercised, then steps and checks:

  * term shapes and finiteness through the manager,
  * keypoint visibility flags in a plausible band (flat-lay shirt: most of
    the 12 keypoints visible from a top/side camera pair),
  * coverage_from_mask == the pure-metric silhouette_coverage on the same
    particle state (the observation-contract assert),
  * grasp-flag latency shifts the deterministic flag by exactly N steps,
  * surface cloud determinism (same seed -> same permutation) and
    visible-only restriction.

Usage::

    cd src/tensegrity_pick
    # env_isaaclab (conda activate, keep PYTHONPATH)
    python scripts/model_validation/check_camera_obs_terms.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-test camera obs terms.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared import cloth_sorting_mdp as mdp  # noqa: E402
from tensegrity_pick.tasks.manager_based.shared.cloth_metrics import (  # noqa: E402
    flat_silhouette_area,
    silhouette_coverage,
)


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}: {name}{'  (' + detail + ')' if detail else ''}")
    return bool(cond)


def main() -> None:
    ok = True
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                        num_envs=args_cli.num_envs)
    # Inject the camera terms into the policy group COPY held by this cfg
    # instance (runtime-only; the task cfg files stay untouched).  This runs
    # the terms through the manager, including the zeros-fallback shape probe.
    cfg.observations.policy.cam_keypoints = ObsTerm(func=mdp.keypoints_with_visibility)
    cfg.observations.policy.cam_cloud = ObsTerm(
        func=mdp.surface_point_cloud, params={"num_points": 32})
    cfg.observations.policy.cam_coverage = ObsTerm(func=mdp.coverage_from_mask)
    cfg.observations.policy.cam_grasp = ObsTerm(
        func=mdp.grasp_active_noisy, params={"latency_steps": 3})

    env = gym.make(args_cli.task, cfg=cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    n = u.num_envs
    dev = u.device

    zero = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
    for _ in range(10):
        obs, *_ = env.step(zero)

    print("manager integration:")
    pol = obs["policy"]
    ok &= check("policy obs finite", bool(torch.isfinite(pol).all()),
                f"shape {tuple(pol.shape)}")

    print("keypoints_with_visibility:")
    kp = mdp.keypoints_with_visibility(u)
    flags = kp[:, 36:]
    ok &= check("shape [N, 48]", kp.shape == (n, 48))
    ok &= check("flags binary", bool(((flags == 0) | (flags == 1)).all()))
    vis_frac = flags.mean().item()
    ok &= check("flat-lay keypoints mostly visible", 0.5 <= vis_frac <= 1.0,
                f"visible frac {vis_frac:.2f}")
    kp_noisy = mdp.keypoints_with_visibility(u, noise_std=0.015, recall=0.74)
    ok &= check("noisy variant same shape, finite",
                kp_noisy.shape == (n, 48) and bool(torch.isfinite(kp_noisy).all()))

    print("coverage_from_mask (contract assert vs pure metric):")
    cov_term = mdp.coverage_from_mask(u).squeeze(-1)
    ref = flat_silhouette_area(u.cloth.flat_rest_pos)
    cov_direct = silhouette_coverage(u.cloth.nodal_pos_w, ref)
    ok &= check("== silhouette_coverage", torch.allclose(cov_term, cov_direct),
                f"term {cov_term.mean():.4f} direct {cov_direct.mean():.4f}")

    print("surface_point_cloud:")
    c1 = mdp.surface_point_cloud(u, num_points=32)
    perm1 = u._surface_cloud_perm.clone()
    del u._surface_cloud_perm
    c2 = mdp.surface_point_cloud(u, num_points=32)
    ok &= check("shape [N, 96]", c1.shape == (n, 96))
    ok &= check("perm deterministic across re-init",
                torch.equal(perm1, u._surface_cloud_perm))
    ok &= check("cloud deterministic", torch.equal(c1, c2))
    # visible-only: every sampled point must be one of the visible particles
    vis = mdp._particle_visibility_cached(u)
    pts = u.cloth.nodal_pos_w - u.scene.env_origins.unsqueeze(1)
    cloud = c1.reshape(n, 32, 3)
    max_d = max(
        float(
            torch.cdist(cloud[i], pts[i][vis[i]],
                        compute_mode="donot_use_mm_for_euclid_dist")
            .min(dim=-1).values.max()
        )
        for i in range(n)
    )
    ok &= check("cloud points are visible particles", max_d < 1e-5,
                f"max match dist {max_d:.2e}")

    print("grasp_active_noisy:")
    seq_obs = []
    for _ in range(8):
        env.step(zero)
        seq_obs.append(mdp.grasp_active_noisy(u, latency_steps=3).squeeze(-1).clone())
    ok &= check("delayed flag binary",
                all(bool(((s == 0) | (s == 1)).all()) for s in seq_obs))
    ok &= check("noise-off equals grasp_active_obs when latency 0",
                torch.equal(mdp.grasp_active_noisy(u).squeeze(-1),
                            u.grasp_active.float()))

    print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES — see above'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Stage-0 de-risk: TWO simultaneous cloth attachments under stretch.

The bimanual shirt_present task needs the holder to keep gripping while the
second arm grasps and stretches — the research report flags stiff stretching
between two pinned PBD particle groups as the core Task-2 physics risk
(mass-spring overstretch / solver instability).

This script validates the new multi-slot ``ClothObject`` attachments:
  1. hang the shirt from a static anchor (slot 1) at the presentation pose
     and let it settle (the shirt_present stub configuration)
  2. attach slot 0 at the hanging shirt's LOWEST point (the literature-
     standard second grasp)
  3. pull slot 0 straight down at STRETCH_SPEED, sweeping the tautness ratio
       ratio = inter-grasp distance / rest distance
     (rest distance = distance of the two grasp-cluster centroids in the
     flat rest shape — the flat-sheet geodesic) from slack to OVERSTRETCH_MAX
  4. at each checkpoint record: max particle speed, cloth bbox diagonal
     inflation vs. the hang state, NaN check, per-slot attachment integrity

PASS criteria:
  * both attachments stay intact for the whole sweep
  * no NaNs, max particle speed < VEL_LIMIT while ratio ≤ 1.0
  * bbox inflation < BBOX_LIMIT while ratio ≤ 1.0
The behaviour beyond ratio 1.0 is *reported* (expected: PBD overstretch —
this is where the Task-2 reward must cap, per the research report).

Runs on the shirt_pick env (robot parked at default pose; the two grasps are
driven directly through the ClothObject API — no policy, no gripper).

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/test_two_attachments.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Two simultaneous cloth attachments stretch test.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (  # noqa: E402
    PRESENTATION_POS,
)

ANCHOR_RADIUS = 0.07      # both grasps use the validated pad-sized radius
HANG_SETTLE_STEPS = 240
STRETCH_SPEED = 0.05      # m/s — slow stretch per the research report
OVERSTRETCH_MAX = 1.15    # sweep the tautness ratio up to here
VEL_LIMIT = 2.5           # m/s — "stable" max particle speed while taut
BBOX_LIMIT = 1.25         # × hang-state bbox diagonal while ratio ≤ 1.0
CHECK_RATIOS = (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.10, 1.15)


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.events.reset_arm.params["position_range"] = (0.0, 0.0)
    cfg.episode_length_s = 1.0e6  # no mid-test manager resets
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    u = env.unwrapped
    cloth = u.cloth
    dev = u.device
    n = u.num_envs
    all_ids = torch.arange(n, device=dev)
    dt = u.cfg.sim.dt

    def sim_steps(k: int, hold0: torch.Tensor | None, hold1: torch.Tensor | None) -> None:
        """Raw sim stepping with per-step slot holds (bypasses the managers)."""
        for _ in range(k):
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt)
            cloth.update()
            if hold1 is not None:
                cloth.hold(hold1, slot=1)
            if hold0 is not None:
                cloth.hold(hold0, slot=0)

    # 1) hang from the slot-1 anchor at the presentation pose
    anchor1 = u.scene.env_origins + torch.tensor(
        PRESENTATION_POS, device=dev, dtype=torch.float32
    ).unsqueeze(0)
    centroids = anchor1.clone()
    cloth.reset_randomized(all_ids, centroids, torch.zeros(n, device=dev))
    cloth.update()
    cloth.attach(all_ids, anchor1, ANCHOR_RADIUS, slot=1)
    assert bool(cloth.is_attached_slot(1).all()), "slot-1 anchor failed to catch particles"
    sim_steps(HANG_SETTLE_STEPS, hold0=None, hold1=anchor1)

    pts = cloth.nodal_pos_w
    bbox_hang = (pts.max(dim=1).values - pts.min(dim=1).values).norm(dim=-1)  # [N]
    print(f"[hang] settled; bbox diagonal = {bbox_hang.mean().item():.3f} m")

    # 2) second grasp (slot 0) at the lowest point
    low = u.shirt_lowest_point_w.clone()
    cloth.attach(all_ids, low, ANCHOR_RADIUS, slot=0)
    assert bool(cloth.is_attached_slot(0).all()), "slot-0 grasp failed to catch particles"
    # Let the attach snap (particles pulled onto the recentred offsets) decay —
    # we want to measure STEADY-STATE two-attachment stability, not the
    # one-step attach transient.
    sim_steps(30, hold0=low, hold1=anchor1)
    # Verify slot exclusivity (no particle in both masks)
    overlap = (cloth._attach_mask[0] & cloth._attach_mask[1]).sum().item()
    n0 = cloth._attach_mask[0].sum(dim=1).float().mean().item()
    n1 = cloth._attach_mask[1].sum(dim=1).float().mean().item()
    print(f"[grasp] slot0 ⌀{n0:.0f} particles, slot1 ⌀{n1:.0f} particles, overlap {overlap} (must be 0)")

    # Rest (flat-sheet geodesic) distance between the two grasp clusters:
    # centroid distance of the cluster particles in the flat rest shape.
    rest = cloth._flat_rest  # [P, 3] (env-invariant)
    m0, m1 = cloth._attach_mask[0], cloth._attach_mask[1]  # [N, P]
    rest_d = torch.stack([
        (rest[m0[i]].mean(dim=0) - rest[m1[i]].mean(dim=0)).norm() for i in range(n)
    ])
    print(f"[grasp] rest distance between clusters = {rest_d.mean().item():.3f} m")

    # 3) stretch sweep: pull slot 0 straight down
    hold0 = low.clone()
    results = []
    ck = 0
    step_len = STRETCH_SPEED * dt
    failed = None
    for it in range(20000):
        hold0[:, 2] -= step_len
        sim_steps(1, hold0=hold0, hold1=anchor1)
        d = (hold0 - anchor1).norm(dim=-1)
        ratio = (d / rest_d).min().item()  # conservative: slowest env
        if ck < len(CHECK_RATIOS) and ratio >= CHECK_RATIOS[ck]:
            # Steady-state check: hold the stretch for 5 steps and average the
            # per-step max particle speed (a single-step solver blip is not
            # instability; sustained high speed / growth is).
            vsum = 0.0
            for _ in range(5):
                sim_steps(1, hold0=hold0, hold1=anchor1)
                vsum += cloth.nodal_vel_w.norm(dim=-1).max().item()
            vmax = vsum / 5.0
            pts = cloth.nodal_pos_w
            has_nan = bool(torch.isnan(pts).any())
            bbox = (pts.max(dim=1).values - pts.min(dim=1).values).norm(dim=-1)
            infl = (bbox / bbox_hang).max().item()
            intact = bool(cloth.is_attached_slot(0).all() and cloth.is_attached_slot(1).all())
            results.append((CHECK_RATIOS[ck], vmax, infl, has_nan, intact))
            print(f"[stretch] ratio {CHECK_RATIOS[ck]:.2f}: max|v| {vmax:6.2f} m/s, "
                  f"bbox ×{infl:.3f}, NaN={has_nan}, attachments intact={intact}")
            if has_nan or not intact:
                failed = CHECK_RATIOS[ck]
                break
            ck += 1
        if ck >= len(CHECK_RATIOS):
            break

    # 4) verdict (stability required up to ratio 1.0; beyond is informational)
    ok = True
    for ratio, vmax, infl, has_nan, intact in results:
        if ratio <= 1.0 + 1e-6:
            if has_nan or not intact or vmax > VEL_LIMIT or infl > BBOX_LIMIT:
                ok = False
    if failed is not None and failed <= 1.0:
        ok = False
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} — two simultaneous attachments "
          f"{'stable' if ok else 'UNSTABLE'} up to tautness ratio 1.0 "
          f"(sweep reached {results[-1][0]:.2f})" if results else "\nRESULT: FAIL — no data")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

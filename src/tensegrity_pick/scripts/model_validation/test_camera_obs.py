"""Offline unit tests for the camera-realistic observation helpers (NO Isaac Sim).

Sibling of test_cloth_metrics.py, covering the Stage-2 S4 additions to
shared/cloth_metrics.py (pure torch):

  * per_particle_visibility — consistency with two_sided_visible_fraction,
    3-layer occlusion ground truth
  * depth_layer_count_in_ball — layer counting at a grasp point
  * keypoint_detector_noise — identity when off, dropout rate ≈ 1 − recall,
    noise magnitude ≈ σ, occluded-keypoint zeroing, seed determinism
  * downsample_masked_points — mask restriction, wrap-around padding,
    determinism given the permutation
  * load_present_markers — study marker format contract
  * coverage-from-mask contract — silhouette_coverage equals an INDEPENDENT
    segmentation-mask-area reimplementation (validates that the observation
    is exactly the mask area a real front/back camera measures)

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    python scripts/model_validation/test_camera_obs.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

_MOD = (
    pathlib.Path(__file__).resolve().parents[2]
    / "source" / "tensegrity_pick" / "tensegrity_pick" / "tasks"
    / "manager_based" / "shared" / "cloth_metrics.py"
)
spec = importlib.util.spec_from_file_location("cloth_metrics", _MOD)
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)


def make_square(n_side: int = 101, size: float = 1.0) -> torch.Tensor:
    """Flat square grid in the XZ plane (camera-facing), y = 0: [P, 3]."""
    lin = torch.linspace(0.0, size, n_side)
    x, z = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([x.flatten(), torch.zeros(n_side**2), z.flatten()], dim=-1)


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}: {name}{'  (' + detail + ')' if detail else ''}")
    return bool(cond)


def main() -> int:
    ok = True
    sq = make_square()
    p = sq.shape[0]

    print("per_particle_visibility:")
    three = torch.cat([sq, sq + torch.tensor([0.0, 0.03, 0.0]),
                       sq + torch.tensor([0.0, 0.06, 0.0])])
    vis = cm.per_particle_visibility(three)                     # [1, 3P]
    frac = cm.two_sided_visible_fraction(three)
    ok &= check("mean equals two_sided_visible_fraction",
                torch.allclose(vis.float().mean(dim=1), frac),
                f"{vis.float().mean().item():.4f} vs {frac.item():.4f}")
    front, mid, back = vis[0, :p], vis[0, p:2 * p], vis[0, 2 * p:]
    ok &= check("front layer visible", bool(front.all()))
    ok &= check("back layer visible", bool(back.all()))
    ok &= check("middle layer hidden", float(mid.float().mean()) < 0.05,
                f"visible frac {mid.float().mean():.3f}")
    ok &= check("single sheet all visible",
                bool(cm.per_particle_visibility(sq).all()))

    print("depth_layer_count_in_ball:")
    center = torch.tensor([[0.5, 0.03, 0.5]])                   # mid of stack
    axis = torch.tensor([0.0, 1.0, 0.0])                        # pinch along Y
    n3 = cm.depth_layer_count_in_ball(three, center, radius=0.07, axis=axis)
    n1 = cm.depth_layer_count_in_ball(sq, torch.tensor([[0.5, 0.0, 0.5]]),
                                      radius=0.07, axis=axis)
    n0 = cm.depth_layer_count_in_ball(sq, torch.tensor([[5.0, 0.0, 5.0]]),
                                      radius=0.07, axis=axis)
    ok &= check("3 stacked layers -> 3", int(n3) == 3, f"got {int(n3)}")
    ok &= check("flat sheet -> 1", int(n1) == 1, f"got {int(n1)}")
    ok &= check("empty ball -> 0", int(n0) == 0, f"got {int(n0)}")
    close = torch.cat([sq, sq + torch.tensor([0.0, 0.005, 0.0])])  # < layer_gap
    nc = cm.depth_layer_count_in_ball(close, torch.tensor([[0.5, 0.0, 0.5]]),
                                      radius=0.07, axis=axis)
    ok &= check("contact layers merge -> 1 (lower bound)", int(nc) == 1,
                f"got {int(nc)}")
    # Batched: two envs, different centers.
    nb = cm.depth_layer_count_in_ball(
        torch.stack([three, three]),
        torch.tensor([[0.5, 0.03, 0.5], [5.0, 0.0, 5.0]]),
        radius=0.07, axis=axis)
    ok &= check("batched [3, 0]", nb.tolist() == [3, 0], f"got {nb.tolist()}")

    print("keypoint_detector_noise:")
    kp = torch.randn(64, 12, 3)
    vis_kp = torch.rand(64, 12) < 0.7
    clean_pos, clean_flag = cm.keypoint_detector_noise(kp, vis_kp,
                                                       noise_std=0.0, recall=1.0)
    ok &= check("noise off: visible positions exact",
                torch.equal(clean_pos[vis_kp], kp[vis_kp]))
    ok &= check("noise off: flags == visibility",
                torch.equal(clean_flag.bool(), vis_kp))
    ok &= check("occluded positions zeroed",
                bool((clean_pos[~vis_kp] == 0).all()))
    torch.manual_seed(7)
    n_pos1, n_flag1 = cm.keypoint_detector_noise(kp, vis_kp, 0.015, 0.74)
    torch.manual_seed(7)
    n_pos2, n_flag2 = cm.keypoint_detector_noise(kp, vis_kp, 0.015, 0.74)
    ok &= check("seed-deterministic", torch.equal(n_pos1, n_pos2)
                and torch.equal(n_flag1, n_flag2))
    big_vis = torch.ones(2000, 12, dtype=torch.bool)
    _, big_flag = cm.keypoint_detector_noise(torch.randn(2000, 12, 3), big_vis,
                                             0.0, recall=0.74)
    rate = big_flag.mean().item()
    ok &= check("dropout rate ~= recall 0.74", 0.71 < rate < 0.77, f"{rate:.3f}")
    det = n_flag1.bool()
    err = (n_pos1[det] - kp[det]).norm(dim=-1)
    ok &= check("noise magnitude ~ sigma*sqrt(3)", 0.015 < err.mean() < 0.04,
                f"mean {err.mean():.4f} m")

    print("downsample_masked_points:")
    pts = torch.randn(3, 200, 3)
    mask = torch.zeros(3, 200, dtype=torch.bool)
    mask[0] = True                       # plenty
    mask[1, :5] = True                   # fewer than num_points -> wrap
    mask[2, 42] = True                   # single point -> repeated
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(200, generator=gen)
    cloud = cm.downsample_masked_points(pts, mask, 16, perm)
    ok &= check("shape [N, 16, 3]", cloud.shape == (3, 16, 3))
    allowed1 = pts[1, mask[1]]
    d1 = torch.cdist(cloud[1], allowed1).min(dim=-1).values
    ok &= check("wrap env: only masked points", float(d1.max()) < 1e-6)
    ok &= check("single-point env: repeated",
                bool((cloud[2] == pts[2, 42]).all()))
    cloud2 = cm.downsample_masked_points(pts, mask, 16, perm)
    ok &= check("deterministic given perm", torch.equal(cloud, cloud2))

    print("load_present_markers:")
    m = cm.load_present_markers()
    ok &= check("12 keypoints", m["keypoint_idx"].shape == (12,)
                and len(m["keypoint_names"]) == 12)
    P = m["border_dist"].shape[0]
    ok &= check("per-particle fields consistent",
                m["particle_region"].shape == (P,)
                and int(m["keypoint_idx"].max()) < P)
    ok &= check("cached singleton", cm.load_present_markers() is m)

    print("coverage-from-mask contract (independent mask-area reimplementation):")
    folded = sq.clone()
    right = folded[:, 0] > 0.5
    folded[right, 0] = 1.0 - folded[right, 0]
    batch = torch.stack([sq, folded])
    flat = sq[:, [0, 2, 1]].clone()
    ref = cm.flat_silhouette_area(flat)
    cov = cm.silhouette_coverage(batch, ref_area=ref, view_axis=1)
    # Independent mask: rasterize the XZ projection into a boolean occupancy
    # grid ("segmentation mask") and take occupied area / reference area.
    cell = cm.DEFAULT_CELL_SIZE
    for i, pts_i in enumerate(batch):
        uv = pts_i[:, [0, 2]]
        ij = ((uv - uv.min(dim=0).values) / cell).long()
        mask_img = torch.zeros(int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1,
                               dtype=torch.bool)
        mask_img[ij[:, 0], ij[:, 1]] = True
        mask_cov = mask_img.sum().item() * cell * cell / ref
        ok &= check(f"batch[{i}] coverage == mask area / ref",
                    abs(mask_cov - cov[i].item()) < 1e-6,
                    f"{mask_cov:.4f} vs {cov[i].item():.4f}")

    print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES — see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

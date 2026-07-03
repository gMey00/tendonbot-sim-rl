"""Offline unit test for shared/cloth_metrics.py (NO Isaac Sim needed).

Validates the silhouette rasterization and stretch helpers against synthetic
particle clouds with known ground truth:

  * flat unit square viewed face-on  → area ≈ 1 m², coverage ≈ 1
  * the same square folded in half   → area ≈ 0.5 m²
  * the same square viewed edge-on   → area ≈ 0 (thin strip)
  * batched input, stretch/rest-distance round trips

The module is pure torch (kept free of isaaclab imports precisely so this
test can run without booting the simulator).

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    python scripts/model_validation/test_cloth_metrics.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

# Load the module by file path — importing the tensegrity_pick package would
# pull in isaaclab (and thus require the simulator app).
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


def check(name: str, value: float, lo: float, hi: float) -> bool:
    ok = lo <= value <= hi
    print(f"  {'PASS' if ok else 'FAIL'}: {name} = {value:.4f} (expected [{lo}, {hi}])")
    return ok


def main() -> int:
    ok = True
    sq = make_square()

    print("silhouette_area:")
    a_face = cm.silhouette_area(sq, view_axis=1)  # camera looks along −Y
    ok &= check("unit square face-on", a_face.item(), 0.95, 1.15)

    folded = sq.clone()
    right = folded[:, 0] > 0.5
    folded[right, 0] = 1.0 - folded[right, 0]     # fold in half (double layer)
    a_fold = cm.silhouette_area(folded, view_axis=1)
    ok &= check("folded square (half area, double layer counts once)",
                a_fold.item(), 0.45, 0.60)

    a_edge = cm.silhouette_area(sq, view_axis=0)  # edge-on: thin strip
    ok &= check("unit square edge-on", a_edge.item(), 0.0, 0.05)

    print("batched input:")
    batch = torch.stack([sq, folded])             # [2, P, 3]
    a_b = cm.silhouette_area(batch, view_axis=1)
    ok &= check("batch[0] == face-on", (a_b[0] - a_face).abs().item(), 0.0, 1e-6)
    ok &= check("batch[1] == folded", (a_b[1] - a_fold).abs().item(), 0.0, 1e-6)

    print("silhouette_coverage:")
    # Flat reference of the square itself, computed like a flat lay (XY plane).
    flat = sq[:, [0, 2, 1]].clone()               # rotate into the XY plane
    ref = cm.flat_silhouette_area(flat)
    cov = cm.silhouette_coverage(batch, ref_area=ref, view_axis=1)
    ok &= check("face-on coverage", cov[0].item(), 0.90, 1.10)
    ok &= check("folded coverage", cov[1].item(), 0.40, 0.60)

    print("plane_extent:")
    ext = cm.plane_extent(batch, view_axis=1)     # [2, 2] (x, z extents)
    ok &= check("face-on width", ext[0, 0].item(), 0.99, 1.01)
    ok &= check("folded width", ext[1, 0].item(), 0.49, 0.51)

    print("stretch_ratio / rest_distance:")
    rd = cm.rest_distance(flat, torch.tensor([0]), torch.tensor([100]))  # corner→corner along one edge
    a = torch.zeros(4, 3)
    b = torch.zeros(4, 3)
    b[:, 0] = torch.tensor([0.5, 1.0, 1.15, 2.0]) * rd
    sr = cm.stretch_ratio(a, b, rd)
    ok &= check("slack 0.5", sr[0].item(), 0.499, 0.501)
    ok &= check("taut 1.0", sr[1].item(), 0.999, 1.001)
    ok &= check("stretched 1.15", sr[2].item(), 1.149, 1.151)

    print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES — see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

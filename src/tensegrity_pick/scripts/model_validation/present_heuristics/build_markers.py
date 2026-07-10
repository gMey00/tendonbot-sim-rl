"""Build the ClothesNet marker asset for the presentation study (offline, once).

The ClothesNet garment ships two extra marker sets alongside the mesh:

  * ``border.obj``            — the garment's boundary loop (silhouette / open
                                edges: neck, cuffs, hem), the "borderpoints";
  * ``kp_<name>.pcd``         — 10 semantic keypoints (collar / shoulders /
                                sleeve ends / armpits / hem corners / centre).

Neither lives in the sim's flat-rest frame, and the sim mesh
(``tshirt_clothesnet.usd``, 11 048 particles) was RE-TESSELLATED from the raw
ClothesNet OBJ, so there is no vertex correspondence.  This script therefore

  1. reads the sim mesh TOPOLOGY straight from the USD and extracts the
     boundary particles (edges used by a single face) → the borderpoints are
     recovered INTRINSICALLY on the sim mesh, no registration needed;
  2. registers each candidate ClothesNet shirt's outline to the flat-rest
     shape with an ICP-refined 2D similarity (rotation + reflection + scale),
     picks the best match (lowest Chamfer — the actual source garment), maps
     its 10 keypoints into the flat-rest frame and snaps each to the nearest
     sim particle;
  3. precomputes, for every particle, its nearest-boundary distance (the
     "border nearness" used by the analysis) and its folded garment region;
  4. records, for each keypoint, the hanging-bank state whose pinned anchor is
     nearest it (so a keypoint can serve as a bank-restored FIRST grasp).

Output: ``doc/reports/data/present_markers.pt`` — consumed by ``markers.py``,
``run_pair_map.py`` (stratified), ``run_keypoint_pick.py`` and ``plot_study.py``
(none of which need pxr).  Validated by ``plot_study.py``'s marker-map figure,
which overlays the mapped ClothesNet border.obj on the recovered mesh boundary.

Run once (needs pxr — the USD crate reader; NOT the full sim app)::

    cd src/tensegrity_pick
    PXR=/home/robot/Isaac/IsaacSim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
    PYTHONPATH="$PXR:$PYTHONPATH" LD_LIBRARY_PATH="$PXR/bin:$CONDA_PREFIX/lib" \
        "$CONDA_PREFIX/bin/python" \
        scripts/model_validation/present_heuristics/build_markers.py
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter

import numpy as np
import torch

REPO = "/home/robot/studentische-arbeiten"
DATA = os.path.join(REPO, "doc", "reports", "data")
CLOTH = os.path.join(REPO, "res", "Props", "Cloth")
USD = os.path.join(CLOTH, "tshirt_clothesnet.usd")

# Force a specific source garment (set from --source in __main__); None = auto-pick.
SOURCE_OVERRIDE: str | None = None

# The 8 mirror-folded region classes (must match plot_study.SYM_CLASSES).
SYM_CLASSES = ["collar", "shoulder", "sleeve", "chest",
               "side", "belly", "hem_corner", "hem_c"]
_SYM_OF = {"collar": "collar", "shoulder_l": "shoulder", "shoulder_r": "shoulder",
           "sleeve_l": "sleeve", "sleeve_r": "sleeve", "chest": "chest",
           "side_l": "side", "side_r": "side", "belly": "belly",
           "hem_l": "hem_corner", "hem_r": "hem_corner", "hem_c": "hem_c"}


def load_obj_v(p: str) -> np.ndarray:
    return np.array([[float(x) for x in ln.split()[1:4]]
                     for ln in open(p) if ln.startswith("v ")])


def load_pcd(p: str) -> np.ndarray:
    pts = []
    for ln in open(p):
        s = ln.split()
        if len(s) == 3:
            try:
                pts.append([float(x) for x in s])
            except ValueError:
                pass
    return np.array(pts)


def usd_boundary_indices(n_particles: int, min_loop: int = 15) -> np.ndarray:
    """Boundary particle ids of the sim garment, read from the USD topology.

    An edge on the garment's open boundary is used by exactly one face.  The
    boundary splits into loops (connected components of the boundary graph):
    the garment's real openings — outer silhouette, neck hole, the two sleeve
    cuffs — are large loops, while the re-tessellation left a handful of tiny
    spurious holes (mesh cracks, ~45 vertices in ≤9-vertex loops) that would
    plant false "border" points in the interior.  We keep only loops with
    >= ``min_loop`` vertices.  The ClothPrim drops the single unreferenced
    trailing vertex (USD has ``n_particles + 1`` points); ids >= ``n_particles``
    are excluded.
    """
    from collections import defaultdict

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(USD)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/mesh"))
    fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    fvc = np.array(mesh.GetFaceVertexCountsAttr().Get())
    edges: Counter = Counter()
    i = 0
    for c in fvc:
        f = fvi[i:i + c]
        i += int(c)
        for k in range(c):
            a, b = int(f[k]), int(f[(k + 1) % c])
            edges[(min(a, b), max(a, b))] += 1
    bnd = [e for e, ct in edges.items() if ct == 1]
    adj: dict = defaultdict(list)
    for a, b in bnd:
        adj[a].append(b)
        adj[b].append(a)
    seen: set = set()
    keep: list = []
    n_small = 0
    for v in list(adj):
        if v in seen:
            continue
        stack, comp = [v], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u])
        if len(comp) >= min_loop:
            keep.extend(comp)
        else:
            n_small += len(comp)
    print(f"[markers] boundary loops kept (>= {min_loop} verts): "
          f"{len(keep)} verts; dropped {n_small} spurious-crack verts")
    bverts = sorted(v for v in keep if v < n_particles)
    return np.array(bverts, dtype=np.int64)


def _norm(P: np.ndarray):
    c = P.mean(0)
    Q = P - c
    s = np.sqrt((Q ** 2).sum(1).mean())
    return c, s, (Q / s).astype(np.float32)


def _umeyama(src: torch.Tensor, dst: torch.Tensor):
    """2D similarity src->dst (rotation, uniform scale, translation; reflection
    permitted — the flat garment has a free front/back flip)."""
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    cov = (D.T @ S) / len(src)
    U, d, Vt = torch.linalg.svd(cov)
    R = U @ Vt
    s = d.sum() / ((S ** 2).sum() / len(src))
    t = md - s * (R @ ms)
    return s, R, t


def register_shirt(cn2d: np.ndarray, sim2d: np.ndarray, dev: str):
    """ICP-refined 2D similarity from ClothesNet plane coords to flat-rest xy.

    Returns (chamfer, scale, R[2,2], t[2]) for ``x_sim = s (x_cn @ R.T) + t``.
    Coarse init over 36 rotations x reflection, each refined by 15 ICP steps.
    """
    rng = np.random.default_rng(0)

    def sub(P, n):
        return P[rng.choice(len(P), min(n, len(P)), replace=False)].astype(np.float32)

    sim = torch.tensor(sub(sim2d, 2500), device=dev)
    cn = torch.tensor(sub(cn2d, 2500), device=dev)
    cnc = cn - cn.mean(0)
    cnc = cnc / cnc.norm(dim=1).mean()
    ssc = (sim - sim.mean(0)).norm(dim=1).mean()
    best = None
    for refl in (1.0, -1.0):
        Rr = torch.tensor([[refl, 0.0], [0.0, 1.0]], device=dev)
        for deg in np.arange(0, 360, 10.0):
            th = np.deg2rad(deg)
            Rk = torch.tensor([[np.cos(th), -np.sin(th)],
                               [np.sin(th), np.cos(th)]],
                              dtype=torch.float32, device=dev)
            cur = (cnc @ (Rk @ Rr).T) * ssc + sim.mean(0)
            for _ in range(15):
                idx = torch.cdist(cur, sim).argmin(1)
                s, R, t = _umeyama(cur, sim[idx])
                cur = s * (cur @ R.T) + t
            d = torch.cdist(cur, sim)
            ch = (d.min(1).values.mean() + d.min(0).values.mean()).item()
            if best is None or ch < best[0]:
                s2, R2, t2 = _umeyama(cn.float(), cur)
                best = (ch, s2.item(), R2.cpu().numpy(), t2.cpu().numpy())
    return best


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fr3 = torch.load(os.path.join(DATA, "present_heuristics_flat_rest.pt"),
                     weights_only=False)
    fr = fr3["flat_rest_pos"].numpy()
    bank_anchor_idx = fr3["bank_anchor_idx"].numpy()
    P = len(fr)
    sim2d = fr[:, :2]

    # ── Regions (for keypoints + the symmetry axis X_CENTER) ───────────
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from regions import (
        REGION_LANDMARKS, REGION_NAMES, X_CENTER, assign_regions,
    )
    lut = np.array([SYM_CLASSES.index(_SYM_OF[n]) for n in REGION_NAMES])
    particle_region = lut[assign_regions(fr[:, 0], fr[:, 1])].astype(np.int64)
    allp = torch.tensor(fr[:, :2], device=dev)

    # 1. BORDERPOINTS — definition A: the garment's OPEN mesh edges (neck, the
    #    two sleeve cuffs, hem; the body side/shoulder seams are folds, not
    #    open edges — same notion ClothesNet's border uses).  The
    #    re-tessellation left the RIGHT sleeve cuff broken (scattered cracks)
    #    while the LEFT half is clean, so we reconstruct a COMPLETE, SYMMETRIC
    #    border from the clean left half mirrored about the symmetry axis
    #    X_CENTER (the outline is symmetric to ~3 mm).  ``border_dist`` is each
    #    particle's distance to this border; ``boundary_idx`` are the particles
    #    that lie ON it.
    raw_bidx = usd_boundary_indices(P)                     # clean-loop open edges
    raw_xy = fr[raw_bidx, :2]
    left_xy = raw_xy[raw_xy[:, 0] <= X_CENTER + 0.006]     # clean left half
    mir_xy = left_xy.copy()
    mir_xy[:, 0] = 2 * X_CENTER - mir_xy[:, 0]
    border_xy = np.vstack([left_xy, mir_xy]).astype(np.float32)
    bxy_t = torch.tensor(border_xy, device=dev)
    border_dist = torch.cdist(allp, bxy_t).min(1).values.cpu().numpy()
    boundary_idx = np.nonzero(border_dist < 0.006)[0].astype(np.int64)
    print(f"[markers] symmetric border: {len(border_xy)} points "
          f"(from {len(left_xy)} clean-left mirrored), "
          f"{len(boundary_idx)} on-border particles")

    # 2. KEYPOINTS — symmetric landmarks that COINCIDE with the Voronoi cells:
    #    one perception-friendly point per garment region, at the region's
    #    landmark snapped to the nearest particle.  Symmetric by construction
    #    (the landmarks are mirror-placed about X_CENTER).  Chosen over
    #    ClothesNet's raw keypoints, which are asymmetric and do not align to
    #    the regions (kept below only as a reference-comparison overlay).
    kp_names = list(REGION_NAMES)                           # 12, one per region
    lm_xy = np.array([REGION_LANDMARKS[n] for n in kp_names], dtype=np.float32)
    d_lm = ((lm_xy[:, None, :] - fr[None, :, :2]) ** 2).sum(-1)
    keypoint_idx = d_lm.argmin(1).astype(np.int64)
    kp_snap_xy = fr[keypoint_idx, :2]
    keypoint_region = particle_region[keypoint_idx]
    # Mirror partner (reflect about X_CENTER, snap): l<->r; central regions map
    # to themselves.  Lets the keypoint chooser reach both sides symmetrically.
    kp_mir_xy = kp_snap_xy.copy()
    kp_mir_xy[:, 0] = 2 * X_CENTER - kp_mir_xy[:, 0]
    d_km = ((kp_mir_xy[:, None, :] - fr[None, :, :2]) ** 2).sum(-1)
    keypoint_mirror_idx = d_km.argmin(1).astype(np.int64)

    # 3. Register the source garment (Ts1_0, forced) — ONLY to map its shipped
    #    keypoints + border.obj into the flat-rest frame for the comparison /
    #    validation figures.
    results = []
    for d in sorted(glob.glob(os.path.join(CLOTH, "TNSC_Tshirt_*"))):
        objs = [o for o in glob.glob(d + "/*.obj") if "border" not in o]
        kpf = glob.glob(d + "/kp_*.pcd")
        if not objs or not kpf:
            continue
        v = load_obj_v(objs[0])
        thin = int((v.max(0) - v.min(0)).argmin())        # garment normal axis
        plane = [i for i in range(3) if i != thin]
        ch, s, R, t = register_shirt(v[:, plane], sim2d, dev)
        results.append((ch, os.path.basename(d), plane, s, R, t, kpf[0]))
    results.sort(key=lambda r: r[0])
    if SOURCE_OVERRIDE is not None:
        picked = [r for r in results if r[1] == SOURCE_OVERRIDE]
        if not picked:
            raise SystemExit(f"--source {SOURCE_OVERRIDE} not among candidates")
        ch, name, plane, s, R, t, kpf = picked[0]
        print(f"[markers] source garment = {name} (chamfer {ch:.4f}) "
              f"[FORCED via --source; auto-pick was {results[0][1]}]")
    else:
        ch, name, plane, s, R, t, kpf = results[0]
        print(f"[markers] source garment = {name} (chamfer {ch:.4f})")
    cn_kp = load_pcd(kpf)[:, plane].astype(np.float32)
    clothesnet_kp_xy = (s * (cn_kp @ R.T) + t).astype(np.float32)
    clothesnet_kp_region = particle_region[
        ((clothesnet_kp_xy[:, None, :] - fr[None, :, :2]) ** 2).sum(-1).argmin(1)]
    border_cn = load_obj_v(os.path.join(CLOTH, name, "border.obj"))[:, plane]
    clothesnet_border_xy = (s * (border_cn.astype(np.float32) @ R.T) + t)

    # 4. Bank state whose pinned anchor is nearest each keypoint (a keypoint as
    #    a bank-restored FIRST grasp).
    anc_xy = fr[bank_anchor_idx, :2]                        # [B, 2]
    d_ab = ((kp_snap_xy[:, None, :] - anc_xy[None, :, :]) ** 2).sum(-1)
    keypoint_nearest_bank = d_ab.argmin(1).astype(np.int64)

    out = {
        "source_shirt": name,
        "register_chamfer": float(ch),
        "boundary_idx": torch.tensor(boundary_idx),          # on-border particles
        "border_xy": torch.tensor(border_xy),                # [Nb, 2] symmetric
        "border_dist": torch.tensor(border_dist),            # [P] metres
        "particle_region": torch.tensor(particle_region),    # [P] 0..7
        "sym_classes": SYM_CLASSES,
        "keypoint_idx": torch.tensor(keypoint_idx),          # [12] region landmarks
        "keypoint_names": kp_names,                           # [12] region names
        "keypoint_xy": torch.tensor(kp_snap_xy),             # [12, 2]
        "keypoint_region": torch.tensor(keypoint_region),    # [12] 0..7
        "keypoint_border_dist": torch.tensor(border_dist[keypoint_idx]),
        "keypoint_nearest_bank": torch.tensor(keypoint_nearest_bank),
        "keypoint_mirror_idx": torch.tensor(keypoint_mirror_idx),  # [12]
        # ClothesNet reference (comparison / validation overlays only):
        "clothesnet_kp_xy": torch.tensor(clothesnet_kp_xy),          # [10, 2]
        "clothesnet_kp_region": torch.tensor(clothesnet_kp_region),  # [10]
        "clothesnet_border_xy": torch.tensor(clothesnet_border_xy),  # [Nb, 2]
    }
    path = os.path.join(DATA, "present_markers.pt")
    torch.save(out, path)
    print(f"[markers] saved {path}")
    print("[markers] keypoints (region landmark, idx, border_dist m):")
    for i in range(len(keypoint_idx)):
        print(f"    {kp_names[i]:11s} idx={keypoint_idx[i]:5d}  "
              f"{SYM_CLASSES[keypoint_region[i]]:11s}  "
              f"border={border_dist[keypoint_idx[i]]:.3f}  "
              f"mirror_idx={keypoint_mirror_idx[i]}")


if __name__ == "__main__":
    _p = argparse.ArgumentParser(description="Build the ClothesNet marker asset.")
    _p.add_argument("--source", type=str, default=None,
                    help="force the source garment (e.g. TNSC_Tshirt_Ts1_0) instead "
                         "of auto-picking the lowest outline-match Chamfer")
    _args = _p.parse_args()
    SOURCE_OVERRIDE = _args.source
    main()

"""All figures + summary stats for the presentation-heuristics study (offline).

Pure-offline analysis: reads the trial CSVs from ``doc/reports/data/``, the
flat-rest dump and the hanging bank, and writes every figure of the findings
report to ``doc/reports/figures/present_heuristics/`` + prints the summary
tables (markdown) to stdout.  No Isaac Sim required — run it in the conda env:

    cd src/tensegrity_pick
    python scripts/model_validation/present_heuristics/plot_study.py

Skips figures whose input CSV does not exist yet, so it can be run
incrementally while the study progresses.

Primary metric everywhere: ``cov_plane`` — the raw camera-plane coverage, the
number the RL task sees.  The horizontal presentation stretch pins the taut
chord along the camera's x axis, so the garment is self-aligned to the image
plane (measured ``cov_yaw_max − cov_plane`` ≈ 0.003); ``cov_yaw_max`` is kept
as the alignment-check column in the summary table.  The free-hang reference
has no alignment, so its best-yaw value is used.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import (  # noqa: E402
    REGION_LANDMARKS, REGION_NAMES, assign_regions, load_metrics, mirror_region,
)

try:
    import markers as mk  # noqa: E402  (built by build_markers.py)
except (FileNotFoundError, ImportError):
    mk = None

REPO = "/home/robot/studentische-arbeiten"
DATA = os.path.join(REPO, "doc", "reports", "data")
FIGS = os.path.join(REPO, "doc", "reports", "figures", "present_heuristics")
BANK = os.path.join(REPO, "res", "Props", "Cloth", "banks",
                    "tshirt_hanging_bank.pt")

# Reference palette (dataviz skill): categorical fixed order + one-hue ramp.
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
       "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 170, "savefig.bbox": "tight",
})

# The 8 mirror-folded region classes (left/right physically equivalent under
# the bank's mirror augmentation) in fixed display order.
SYM_CLASSES = ["collar", "shoulder", "sleeve", "chest",
               "side", "belly", "hem_corner", "hem_c"]
_SYM_OF = {"collar": "collar", "shoulder_l": "shoulder", "shoulder_r": "shoulder",
           "sleeve_l": "sleeve", "sleeve_r": "sleeve", "chest": "chest",
           "side_l": "side", "side_r": "side", "belly": "belly",
           "hem_l": "hem_corner", "hem_r": "hem_corner", "hem_c": "hem_c"}


def sym_class(x, y) -> np.ndarray:
    reg = assign_regions(np.asarray(x), np.asarray(y))
    lut = np.array([SYM_CLASSES.index(_SYM_OF[n]) for n in REGION_NAMES])
    return lut[reg]


def load_csv(name: str) -> pd.DataFrame | None:
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        print(f"[skip] {name} not found")
        return None
    df = pd.read_csv(p)
    df = df[df.valid == 1].copy()
    return df


def free_hang_reference() -> tuple[np.ndarray, np.ndarray]:
    """Sanity anchor: coverage of the unstretched bank hangs (yaw-max, yaw-mean)."""
    cm = load_metrics()
    fr = torch.load(os.path.join(DATA, "present_heuristics_flat_rest.pt"),
                    weights_only=False)
    ref = fr["ref_area"]
    pos = torch.load(BANK, weights_only=False)["pos"]
    covs = []
    for k in range(12):
        a = torch.tensor(np.pi * k / 12)
        q = pos.clone()
        q[..., 0] = torch.cos(a) * pos[..., 0] - torch.sin(a) * pos[..., 1]
        q[..., 1] = torch.sin(a) * pos[..., 0] + torch.cos(a) * pos[..., 1]
        covs.append(cm.silhouette_coverage(q, ref))
    covs = torch.stack(covs)
    return covs.max(0).values.numpy(), covs.mean(0).numpy()


def styled_box(ax, data, positions, color):
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color=INK, lw=1.4),
                    whiskerprops=dict(color=INK2, lw=1.0),
                    capprops=dict(color=INK2, lw=1.0),
                    boxprops=dict(facecolor=color, edgecolor=INK2, lw=0.8,
                                  alpha=0.85))
    return bp


def shirt_outline(ax, fr):
    ax.scatter(fr[:, 0], fr[:, 1], s=0.4, color=GRID, zorder=0)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ── Figure 0: region map ───────────────────────────────────────────────────

def fig_region_map(fr):
    reg = assign_regions(fr[:, 0], fr[:, 1])
    sym = np.array([SYM_CLASSES.index(_SYM_OF[n]) for n in REGION_NAMES])[reg]
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    for i, name in enumerate(SYM_CLASSES):
        m = sym == i
        ax.scatter(fr[m, 0], fr[m, 1], s=1.6, color=CAT[i], rasterized=True)
    for name, (x, y) in REGION_LANDMARKS.items():
        ax.annotate(name, (x, y), ha="center", va="center", fontsize=7.5,
                    color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=GRID, alpha=0.85))
    ax.set_aspect("equal")
    ax.set_xlabel("flat-rest x (m)"), ax.set_ylabel("flat-rest y (m)")
    ax.grid(False)
    ax.set_title("Garment regions (Voronoi cells of hand-placed landmarks;\n"
                 "colors = the 8 left/right-mirrored classes)", fontsize=9)
    fig.savefig(os.path.join(FIGS, "region_map.png"))
    plt.close(fig)


# ── Figure 1: H1 ratio sweep ───────────────────────────────────────────────

def fig_ratio_sweep(h1, hang_max):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ratios = sorted(h1.target_ratio.unique())
    data = [h1[h1.target_ratio == r].cov_plane.values for r in ratios]
    styled_box(ax, data, np.arange(len(ratios)), CAT[0])
    ax.axhline(np.median(hang_max), color=MUTED, ls="--", lw=1.0)
    ax.text(len(ratios) - 0.45, np.median(hang_max) + 0.008,
            "free one-point hang (median)", color=INK2, fontsize=7.5,
            ha="right")
    for i, d in enumerate(data):
        ax.text(i, np.quantile(d, 0.75) + 0.012, f"n={len(d)}",
                ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(np.arange(len(ratios)),
                  [f"{r:.2f}" for r in ratios])
    ax.set_xlabel("target stretch ratio (hold-centre tautness)")
    ax.set_ylabel("silhouette coverage (camera plane)")
    ax.set_title("H1 naive 2-grasp (horizontal stretch): coverage vs tautness",
                 fontsize=9.5)
    fig.savefig(os.path.join(FIGS, "h1_ratio_sweep.png"))
    plt.close(fig)


# ── Figure 2: method comparison ────────────────────────────────────────────

def fig_method_comparison(entries, oracle_ref=None):
    """entries: list of (label, samples) bottom-up; oracle_ref: scalar line."""
    fig, ax = plt.subplots(figsize=(6.0, 0.62 * len(entries) + 1.4))
    ax.grid(axis="y", visible=False)
    for i, (label, d) in enumerate(entries):
        bp = ax.boxplot([d], positions=[i], widths=0.55, vert=False,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.4),
                        whiskerprops=dict(color=INK2, lw=1.0),
                        capprops=dict(color=INK2, lw=1.0),
                        boxprops=dict(facecolor=CAT[0], edgecolor=INK2,
                                      lw=0.8, alpha=0.85))
        ax.text(np.median(d), i + 0.36, f"{np.median(d):.2f}  (n={len(d)})",
                ha="center", fontsize=7, color=INK2)
    if oracle_ref is not None:
        ax.axvline(oracle_ref, color=CAT[5], ls="--", lw=1.2)
        ax.text(oracle_ref + 0.006, -0.42, "oracle top-decile\nmedian",
                color=CAT[5], fontsize=7, va="bottom")
    ax.axvline(1.0, color=MUTED, ls=":", lw=1.0)
    ax.text(1.0, -0.42, "flat face-on = 1.0", color=MUTED, fontsize=7,
            ha="center")
    ax.set_yticks(np.arange(len(entries)), [e[0] for e in entries])
    ax.set_xlabel("silhouette coverage (camera plane)")
    ax.set_title("Presentation quality by method", fontsize=9.5)
    fig.savefig(os.path.join(FIGS, "method_comparison.png"))
    plt.close(fig)


# ── Figures 3+4: pair map ──────────────────────────────────────────────────

def fig_pair_heatmap(h3, min_n=15, out_name="pair_heatmap.png", title=None):
    k = len(SYM_CLASSES)
    top = sym_class(h3.rest_x_top, h3.rest_y_top)
    mov = sym_class(h3.rest_x_move, h3.rest_y_move)
    med = np.full((k, k), np.nan)
    cnt = np.zeros((k, k), dtype=int)
    cov = h3.cov_plane.values
    for i in range(k):
        for j in range(k):
            m = (top == i) & (mov == j)
            cnt[i, j] = m.sum()
            if cnt[i, j] >= min_n:
                med[i, j] = np.median(cov[m])
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    vmin, vmax = np.nanmin(med), np.nanmax(med)
    im = ax.imshow(med, cmap=BLUES, vmin=vmin, vmax=vmax)
    for i in range(k):
        for j in range(k):
            if np.isnan(med[i, j]):
                ax.text(j, i, f"·\nn={cnt[i, j]}", ha="center", va="center",
                        fontsize=6, color=MUTED)
            else:
                frac = (med[i, j] - vmin) / max(vmax - vmin, 1e-9)
                ax.text(j, i, f"{med[i, j]:.2f}\nn={cnt[i, j]}", ha="center",
                        va="center", fontsize=6.5,
                        color="white" if frac > 0.62 else INK)
    ax.set_xticks(range(k), SYM_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(k), SYM_CLASSES)
    ax.set_xlabel("second grasp (stretched)"), ax.set_ylabel("first grasp (hang anchor)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="median coverage (camera plane)")
    ax.set_title(title or ("Grasp-pair quality map (mirror-folded regions;\n"
                           f"cells with n<{min_n} masked)"), fontsize=9.5)
    fig.savefig(os.path.join(FIGS, out_name))
    plt.close(fig)
    return med, cnt


# ── ClothesNet markers: border + keypoints ─────────────────────────────────

def _region_backdrop(ax, fr):
    """Faint Voronoi region colouring as a backdrop for keypoint panels."""
    reg = assign_regions(fr[:, 0], fr[:, 1])
    sym = np.array([SYM_CLASSES.index(_SYM_OF[n]) for n in REGION_NAMES])[reg]
    for i in range(len(SYM_CLASSES)):
        m = sym == i
        ax.scatter(fr[m, 0], fr[m, 1], s=2, color=CAT[i], alpha=0.28,
                   rasterized=True)
    ax.set_aspect("equal"), ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([])


def fig_marker_map(fr):
    """The garment markers on the flat-rest shape, three panels:
    (1) the complete, symmetric borderpoints (definition A = open edges: neck,
    cuffs, hem) + the per-particle border-nearness field; (2) the study's
    symmetric keypoints (one per Voronoi region) they coincide with; (3) the
    ClothesNet shirt's raw keypoints on the same cells — the reference the
    symmetric set replaces (asymmetric, off-cell)."""
    if mk is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.4))
    # (1) borderpoints + field.
    ax = axes[0]
    sc = ax.scatter(fr[:, 0], fr[:, 1], s=2.4, c=mk.BORDER_DIST, cmap=BLUES,
                    rasterized=True)
    ax.scatter(mk.BORDER_XY[:, 0], mk.BORDER_XY[:, 1], s=5, color=CAT[5],
               label=f"borderpoints (n={len(mk.BORDER_XY)})")
    ax.set_aspect("equal"), ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([])
    ax.legend(fontsize=7.5, frameon=False, loc="lower center")
    fig.colorbar(sc, ax=ax, shrink=0.72, label="distance to nearest border (m)")
    ax.set_title("Borderpoints (symmetric open edges)\n+ border-nearness field",
                 fontsize=9.5)
    # (2) the study's symmetric region keypoints.
    ax = axes[1]
    _region_backdrop(ax, fr)
    for k in range(mk.N_KEYPOINTS):
        i = mk.KEYPOINT_IDX[k]
        ax.scatter(fr[i, 0], fr[i, 1], s=150, marker="*", color=CAT[5],
                   zorder=5, edgecolors=INK, linewidths=0.5)
        ax.annotate(mk.KEYPOINT_NAMES[k], (fr[i, 0], fr[i, 1]), fontsize=6.6,
                    weight="bold", xytext=(3, 3), textcoords="offset points")
    ax.set_title("Study keypoints: symmetric, one per region\n"
                 "(coincide with the Voronoi cells)", fontsize=9.5)
    # (3) ClothesNet raw keypoints for comparison.
    ax = axes[2]
    _region_backdrop(ax, fr)
    ck = mk.CLOTHESNET_KP_XY
    for k in range(len(ck)):
        ax.scatter(ck[k, 0], ck[k, 1], s=150, marker="*", color=CAT[3],
                   zorder=5, edgecolors=INK, linewidths=0.5)
        ax.annotate(str(k), (ck[k, 0], ck[k, 1]), fontsize=6.6, weight="bold",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"ClothesNet raw keypoints ({mk.SOURCE_SHIRT})\n"
                 "— asymmetric, off-cell (replaced)", fontsize=9.5)
    fig.suptitle("Garment markers: borderpoints + region-aligned keypoints",
                 fontsize=10.5)
    fig.savefig(os.path.join(FIGS, "marker_map.png"))
    plt.close(fig)


def fig_border_nearness(df, label="pooled random + stratified pairs"):
    """Does grasping nearer the garment border improve presentation quality?
    Coverage vs the SECOND grasp's border distance (the moved/chooseable grasp),
    with the first-grasp trend for contrast."""
    if mk is None or "idx_move" not in df.columns:
        return None
    d = df.dropna(subset=["idx_move", "idx_top"]).copy()
    bd_move = mk.border_dist_of(d.idx_move.astype(int).values)
    bd_top = mk.border_dist_of(d.idx_top.astype(int).values)
    cov = d.cov_plane.values
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for ax, bd, name, col in ((axes[0], bd_move, "second grasp (stretched)", CAT[0]),
                              (axes[1], bd_top, "first grasp (hang anchor)", CAT[3])):
        ax.scatter(bd, cov, s=3, color=col, alpha=0.15, edgecolors="none",
                   rasterized=True)
        bins = np.linspace(0, max(0.12, np.quantile(bd, 0.98)), 9)
        midv, medv = [], []
        for i in range(len(bins) - 1):
            m = (bd >= bins[i]) & (bd < bins[i + 1])
            if m.sum() > 25:
                midv.append(0.5 * (bins[i] + bins[i + 1]))
                medv.append(np.median(cov[m]))
        ax.plot(midv, medv, color=INK, lw=1.8, marker="o", ms=4,
                label="binned median")
        r = np.corrcoef(bd, cov)[0, 1]
        ax.set_xlabel(f"{name}: distance to border (m)")
        ax.set_ylabel("coverage (camera plane)")
        ax.set_title(f"r = {r:.2f}", fontsize=9)
        ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"Border nearness vs presentation quality ({label}, "
                 f"n={len(d)})", fontsize=10)
    fig.savefig(os.path.join(FIGS, "border_nearness.png"))
    plt.close(fig)
    return (np.corrcoef(bd_move, cov)[0, 1], np.corrcoef(bd_top, cov)[0, 1])


def fig_keypoint_analysis(hks, hkp, fr):
    """Are the region-landmark keypoints good grasp targets?
    (a) per-region-keypoint coverage as the SECOND grasp (vs a random bank
    first grasp); (b) the keypoint-pair map (first ≈ keypoint A, second =
    keypoint B), both grouped by the keypoint's garment region (the keypoints
    are the region landmarks, so a keypoint IS identified by its region; the
    mirror partner of a left keypoint is the right keypoint of the same
    region)."""
    if mk is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    K = len(SYM_CLASSES)
    # mean border distance of the landmark(s) in each region.
    reg_bd = np.array([np.mean(mk.KEYPOINT_BORDER_DIST[mk.KEYPOINT_REGION == r])
                       if (mk.KEYPOINT_REGION == r).any() else np.nan
                       for r in range(K)])
    # (a) per-region-keypoint as second grasp.
    ax = axes[0]
    if hks is not None:
        rmove = mk.PARTICLE_REGION[hks.idx_move.astype(int).values]
        cov = hks.cov_plane.values
        stats = sorted((np.median(cov[rmove == r]), r, cov[rmove == r])
                       for r in range(K) if (rmove == r).sum() >= 8)
        for i, (med, r, c) in enumerate(stats):
            col = CAT[6] if SYM_CLASSES[r].startswith("hem") else CAT[0]
            ax.boxplot([c], positions=[i], widths=0.6, vert=False,
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.3),
                       whiskerprops=dict(color=INK2, lw=0.9),
                       capprops=dict(color=INK2, lw=0.9),
                       boxprops=dict(facecolor=col, edgecolor=INK2, lw=0.7,
                                     alpha=0.85))
            ax.text(np.median(c), i + 0.34, f"{np.median(c):.2f}", ha="center",
                    fontsize=6.5, color=INK2)
        ax.set_yticks(range(len(stats)),
                      [f"{SYM_CLASSES[r]}\n(border {reg_bd[r]:.02f}m)"
                       for _, r, _ in stats], fontsize=6.5)
        ax.axvline(0.679, color=MUTED, ls="--", lw=1.0)
        ax.text(0.679, len(stats) - 0.3, "H1 lowest-point\nmedian",
                fontsize=6.5, color=INK2, ha="center", va="top")
        ax.set_xlabel("coverage (camera plane)")
        ax.grid(axis="y", visible=False)
        ax.set_title("Region keypoint as 2nd grasp (random bank 1st grasp)",
                     fontsize=9)
    # (b) keypoint-pair map by region.
    ax = axes[1]
    if hkp is not None:
        anc_xy = fr[hkp.idx_top.astype(int).values][:, :2]
        kA = ((anc_xy[:, None, :] - mk.KEYPOINT_XY[None, :, :]) ** 2).sum(-1).argmin(1)
        rA = mk.KEYPOINT_REGION[kA]
        rB = mk.PARTICLE_REGION[hkp.idx_move.astype(int).values]
        cov = hkp.cov_plane.values
        med = np.full((K, K), np.nan)
        for a in range(K):
            for b in range(K):
                m = (rA == a) & (rB == b)
                if m.sum() >= 5:
                    med[a, b] = np.median(cov[m])
        im = ax.imshow(med, cmap=BLUES)
        for a in range(K):
            for b in range(K):
                if not np.isnan(med[a, b]):
                    fr_ = (med[a, b] - np.nanmin(med)) / max(
                        np.nanmax(med) - np.nanmin(med), 1e-9)
                    ax.text(b, a, f"{med[a, b]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if fr_ > 0.6 else INK)
        ax.set_xticks(range(K), SYM_CLASSES, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(K), SYM_CLASSES, fontsize=6.5)
        ax.set_xlabel("2nd grasp keypoint region")
        ax.set_ylabel("1st grasp keypoint region")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="median coverage")
        ax.set_title("Keypoint↔keypoint pair map", fontsize=9)
    fig.suptitle("Keypoint-guided presentation grasps (region landmarks)",
                 fontsize=10)
    fig.savefig(os.path.join(FIGS, "keypoint_analysis.png"))
    plt.close(fig)


def fig_best_partner(h3, fr, min_n=15):
    """Small multiples: for each first-grasp class, the shirt colored by the
    median coverage achieved with the second grasp in each region."""
    k = len(SYM_CLASSES)
    top = sym_class(h3.rest_x_top, h3.rest_y_top)
    mov = sym_class(h3.rest_x_move, h3.rest_y_move)
    cov = h3.cov_plane.values
    reg_pts = assign_regions(fr[:, 0], fr[:, 1])
    sym_pts = np.array([SYM_CLASSES.index(_SYM_OF[n]) for n in REGION_NAMES])[reg_pts]
    med_all = [[np.median(cov[(top == i) & (mov == j)])
                if ((top == i) & (mov == j)).sum() >= min_n else np.nan
                for j in range(k)] for i in range(k)]
    vmin = np.nanmin(med_all)
    vmax = np.nanmax(med_all)
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 6.4))
    for i, ax in enumerate(axes.flat):
        shirt_outline(ax, fr)
        for j in range(k):
            v = med_all[i][j]
            m = sym_pts == j
            if np.isnan(v):
                ax.scatter(fr[m, 0], fr[m, 1], s=0.6, color=GRID)
            else:
                ax.scatter(fr[m, 0], fr[m, 1], s=0.8,
                           color=BLUES((v - vmin) / max(vmax - vmin, 1e-9)),
                           rasterized=True)
        if np.all(np.isnan(med_all[i])):
            ax.set_title(f"1st: {SYM_CLASSES[i]}\n(no cell ≥ n)", fontsize=8)
            continue
        best = int(np.nanargmax(med_all[i]))
        ax.set_title(f"1st: {SYM_CLASSES[i]}\nbest 2nd: {SYM_CLASSES[best]} "
                     f"({med_all[i][best]:.2f})", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=BLUES,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, shrink=0.7,
                 label="median coverage (camera plane) with 2nd grasp there")
    fig.suptitle("Best second-grasp region for each first grasp (pair-map oracle)",
                 fontsize=10)
    fig.savefig(os.path.join(FIGS, "best_partner_map.png"))
    plt.close(fig)


def fig_pair_drivers(h3):
    """What drives pair quality: rest distance + width-vs-length orientation."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    cov = h3.cov_plane.values
    rd = h3.rest_dist.values
    axes[0].scatter(rd, cov, s=4, color=CAT[0], alpha=0.25, edgecolors="none",
                    rasterized=True)
    axes[0].set_xlabel("pair rest distance (m)")
    axes[0].set_ylabel("coverage (camera plane)")
    r = np.corrcoef(rd, cov)[0, 1]
    axes[0].set_title(f"coverage vs pair separation (r = {r:.2f})", fontsize=9)
    dx = np.abs(h3.rest_x_top - h3.rest_x_move)
    dy = np.abs(h3.rest_y_top - h3.rest_y_move)
    ang = np.degrees(np.arctan2(dy, dx))  # 0° = width-wise, 90° = length-wise
    axes[1].scatter(ang, cov, s=4, color=CAT[0], alpha=0.25, edgecolors="none",
                    rasterized=True)
    bins = np.linspace(0, 90, 10)
    mid = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(cov[(ang >= bins[i]) & (ang < bins[i + 1])])
           if ((ang >= bins[i]) & (ang < bins[i + 1])).sum() > 10 else np.nan
           for i in range(len(mid))]
    axes[1].plot(mid, med, color=INK, lw=1.6, marker="o", ms=3.5,
                 label="binned median")
    axes[1].set_xlabel("pair axis on the flat shirt (° from width-wise)")
    axes[1].legend(fontsize=7.5, frameon=False)
    r2 = np.corrcoef(ang, cov)[0, 1]
    axes[1].set_title(f"coverage vs pair orientation (r = {r2:.2f})", fontsize=9)
    fig.savefig(os.path.join(FIGS, "pair_drivers.png"))
    plt.close(fig)


# ── Summary table ──────────────────────────────────────────────────────────

def summarize(label: str, d: np.ndarray, dmean: np.ndarray | None = None) -> str:
    q = lambda a, p: np.quantile(a, p)  # noqa: E731
    s = (f"| {label} | {len(d)} | {np.median(d):.3f} "
         f"| {q(d, 0.25):.3f}–{q(d, 0.75):.3f} | {q(d, 0.9):.3f} |")
    if dmean is not None:
        s += f" {np.median(dmean):.3f} |"
    return s


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    fr = torch.load(os.path.join(DATA, "present_heuristics_flat_rest.pt"),
                    weights_only=False)["flat_rest_pos"].numpy()
    hang_max, hang_mean = free_hang_reference()
    fig_region_map(fr)

    rows = ["| method | n | median | IQR | p90 | best-yaw med |",
            "|---|---|---|---|---|---|",
            summarize("free 1-point hang (bank, best yaw)", hang_max, hang_max)]
    entries = [("free 1-point hang", hang_max)]

    h1 = load_csv("present_h1_two_grasp.csv")
    if h1 is not None:
        fig_ratio_sweep(h1, hang_max)
        best = h1[h1.target_ratio == 1.05]
        rows.append(summarize("H1 naive 2-grasp @1.05", best.cov_plane.values,
                              best.cov_yaw_max.values))
        entries.append(("H1 naive 2-grasp @1.05", best.cov_plane.values))

    h2 = load_csv("present_h2_three_grasp.csv")
    if h2 is not None:
        for stage, label in [(2, "H2 regrasp (3-grasp)"),
                             (3, "iterated regrasp ×2"),
                             (4, "iterated regrasp ×3")]:
            d = h2[h2.stage == stage]
            if len(d):
                rows.append(summarize(label, d.cov_plane.values,
                                      d.cov_yaw_max.values))
                entries.append((label, d.cov_plane.values))

    fig_marker_map(fr)

    oracle_ref = None
    h3 = load_csv("present_h3_pair_map.csv")
    h3s = load_csv("present_h3s_stratified.csv")
    # Balanced stratified map is the headline pair map when available; the
    # random run (area-weighted, more raw samples) backs the driver/border
    # correlations.  Pool both for the correlation analyses.
    pair_frames = [f for f in (h3, h3s) if f is not None]
    pooled = pd.concat(pair_frames, ignore_index=True) if pair_frames else None
    if h3s is not None:
        fig_pair_heatmap(h3s, min_n=25, out_name="pair_heatmap.png",
                         title="Grasp-pair quality map (stratified: ≥100 "
                               "trials\nper region→region cell; mirror-folded)")
        fig_best_partner(h3s, fr, min_n=25)
    elif h3 is not None:
        fig_pair_heatmap(h3)
        fig_best_partner(h3, fr)
    if h3 is not None:
        fig_pair_heatmap(h3, out_name="pair_heatmap_random.png",
                         title="Grasp-pair quality map (random pairs;\n"
                               "area-weighted, cells with n<15 masked)")
    if pooled is not None:
        fig_pair_drivers(pooled)
        cov = pooled.cov_plane.values
        rows.append(summarize("H3 pair (random+stratified) @1.05", cov,
                              pooled.cov_yaw_max.values))
        entries.append(("H3 grasp pair", cov))
        oracle_ref = float(np.median(cov[cov >= np.quantile(cov, 0.9)]))
        rows.append(f"| H3 oracle top decile | {int(0.1 * len(cov))} "
                    f"| {oracle_ref:.3f} | — | — | — |")
        bn = fig_border_nearness(pooled)
        if bn is not None:
            print(f"[border-nearness] r(cov, 2nd-grasp border dist) = {bn[0]:.3f}, "
                  f"r(cov, 1st-grasp border dist) = {bn[1]:.3f}")

    hks = load_csv("present_hk_second.csv")
    hkp = load_csv("present_hk_pairs.csv")
    if hks is not None or hkp is not None:
        fig_keypoint_analysis(hks, hkp, fr)
    if hks is not None:
        rows.append(summarize("keypoint as 2nd grasp (any kp)",
                              hks.cov_plane.values, hks.cov_yaw_max.values))
        entries.append(("keypoint 2nd grasp", hks.cov_plane.values))
    if hkp is not None:
        rows.append(summarize("keypoint↔keypoint pair (any kp pair)",
                              hkp.cov_plane.values, hkp.cov_yaw_max.values))
        entries.append(("keypoint↔keypoint", hkp.cov_plane.values))

    hs = load_csv("present_h6_shake.csv")
    if hs is not None:
        rows.append(summarize("H1 + shake @1.05", hs.cov_plane.values,
                              hs.cov_yaw_max.values))
        entries.append(("H1 + shake", hs.cov_plane.values))

    ho = load_csv("present_h5_oracle_pick.csv")
    if ho is not None:
        rows.append(summarize("oracle-guided 2nd grasp",
                              ho.cov_plane.values, ho.cov_yaw_max.values))
        entries.append(("oracle-guided 2nd grasp", ho.cov_plane.values))

    h7 = load_csv("present_h7_shoulder_pair.csv")
    if h7 is not None:
        rows.append(summarize("shoulder↔shoulder (scripted)",
                              h7.cov_plane.values, h7.cov_yaw_max.values))
        entries.append(("shoulder↔shoulder", h7.cov_plane.values))

    h8 = load_csv("present_h8_hem_pair.csv")
    if h8 is not None:
        rows.append(summarize("hem_corner↔hem_corner (scripted)",
                              h8.cov_plane.values, h8.cov_yaw_max.values))
        entries.append(("hem↔hem", h8.cov_plane.values))

    h8b = load_csv("present_h8_hem_pair_110.csv")
    if h8b is not None:
        rows.append(summarize("hem_corner↔hem_corner @1.10",
                              h8b.cov_plane.values, h8b.cov_yaw_max.values))
        entries.append(("hem↔hem @1.10", h8b.cov_plane.values))

    fig_method_comparison(entries, oracle_ref)
    print("\n".join(rows))
    print(f"\nsanity: flat face-on = 1.000 (computed offline); free-hang "
          f"yaw-mean median {np.median(hang_mean):.3f}")
    print(f"figures → {FIGS}")


if __name__ == "__main__":
    main()

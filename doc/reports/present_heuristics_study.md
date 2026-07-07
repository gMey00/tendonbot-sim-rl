# Shirt-Presentation Grasping Heuristics — Robot-Free Simulation Study

**Scope:** which grasp points / grasp strategies give the best overall visibility of the
shirt in the inspection-camera plane (front AND back view) — the calibration study for the
`shirt_present` RL task (success threshold + best regrasp target), including the
brute-force **grasp-pair oracle** flagged in [doc/TODO.md](../TODO.md) Task 2.
**Environment:** Isaac Sim 5.1.0 + Isaac Lab, PBD particle cloth (`tshirt_clothesnet.usd`,
11 048 particles), no robots — grasps are static/scripted `ClothObject` attachment anchors.
NVIDIA RTX A6000 (48 GB), FAPS server.
**Date:** 2026-07-04.

Every number is reproducible: each script logs one CSV row per trial (bank state index,
augmentation, particle indices, stretch direction, all metrics) under
[doc/reports/data/](data/), and all figures regenerate offline via `plot_study.py`.
Scripts live in
[`scripts/model_validation/present_heuristics/`](../../src/tensegrity_pick/scripts/model_validation/present_heuristics/)
(usage headers in each file; run from `src/tensegrity_pick` in the `env_isaaclab` conda env).

## TL;DR

| Finding | Number |
|---|---|
| The horizontal presentation works: even the naive heuristic beats the free hang | 0.679 vs 0.563 median coverage |
| Tautness has a real, monotone (but modest) effect | 0.657 → 0.690 median for ratio 0.95 → 1.10 |
| The geometry is **self-aligning** — no yaw control needed | `cov_yaw_max − cov_plane` ≈ 0.003 |
| **Regrasping HURTS** (extremity↔extremity chords span wide but fill poorly) | paired Δ −0.031, only 37 % improve |
| Pair quality is driven by chord LENGTH, and **hem-edge pairs dominate** the oracle map | r = +0.33; side→hem_c 0.83, hem↔hem 0.72 |
| **Best scripted method: grasp both hem corners** (garment upside-down) | **0.820** median @1.05, 0.832 @1.10, p90 0.95 |
| The classic human shoulder↔shoulder grip FAILS here | 0.493 median, 57 % of trials < 0.55 |
| Recommended `shirt_present` success threshold (camera-plane coverage) | **0.65** to start, 0.75 stretch goal |

Full method comparison (camera-plane coverage — the number the RL task sees; all
methods at ratio 1.05 unless noted):

| method | n | median | IQR | p90 |
|---|---|---|---|---|
| shoulder↔shoulder (scripted) | 240 | 0.493 | 0.436–0.707 | 0.758 |
| free 1-point hang (bank, best yaw) | 356 | 0.567 | 0.523–0.606 | 0.633 |
| H3 random pair | 2880 | 0.606 | 0.521–0.697 | 0.787 |
| H2 regrasp (3-grasp) | 288 | 0.653 | 0.618–0.686 | 0.710 |
| iterated regrasp ×2 / ×3 | 288 each | 0.656 / 0.656 | — | 0.721 / 0.730 |
| H1 + shake | 240 | 0.671 | 0.603–0.728 | 0.786 |
| **H1 naive 2-grasp (the task heuristic)** | 240 | **0.679** | 0.621–0.722 | 0.778 |
| oracle-guided 2nd grasp | 240 | 0.713 | 0.628–0.756 | 0.811 |
| **hem_corner↔hem_corner (scripted)** | 240 | **0.820** | 0.701–0.896 | 0.948 |
| hem_corner↔hem_corner @1.10 | 240 | 0.832 | 0.680–0.922 | 0.965 |
| H3 oracle top decile (upper bound) | 288 | 0.842 | — | — |

![Method comparison](figures/present_heuristics/method_comparison.png)

---

## 1. Setup & metric

### 1.1 The visibility objective

The inspection camera at `INSPECTION_CAMERA_POS = (0.15, 2.0, 1.25)` looks along −Y; the
image plane is the world XZ plane (`view_axis=1`). Front + back cameras see the SAME
occluding silhouette, so one scalar serves both views: **`silhouette_coverage`**
([shared/cloth_metrics.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_metrics.py)) —
rasterized projected-silhouette area ÷ one-sided flat area (0.294 m² for this garment).
1.0 = fully unfolded and camera-parallel; folds, bunching and out-of-plane rotation all
reduce it.

Every trial logs three orientation variants of the metric: the raw camera-plane number
`cov_plane`, the best over 12 yaw rotations `cov_yaw_max` (period π), and the
yaw-marginalized `cov_yaw_mean`. **With the horizontal presentation stretch the taut
chord lies along the camera's x axis, which pins the garment's plane: measured
`cov_yaw_max − cov_plane` ≈ 0.001** — so the presentation is self-aligned to the camera
and `cov_plane` (the number the RL task sees) is quoted everywhere below. The yaw-swept
variants remain in every CSV as a built-in check of that alignment. (`cov_yaw_mean` is
much lower by construction: it averages over deliberately rotating the presentation out
of the plane it is aligned to, so it is NOT a meaningful score here.)

### 1.2 Sanity anchors (metric calibration)

| Anchor | Coverage |
|---|---|
| Flat shirt, face-on (computed from `flat_rest_pos`) | **1.000** |
| Free one-point hang, unstretched (356 bank states, best yaw) | 0.563 median, IQR 0.520–0.602 |

### 1.3 Protocol (common to all methods)

Implemented once in
[`study_common.py`](../../src/tensegrity_pick/scripts/model_validation/present_heuristics/study_common.py):

* **Study anchor:** the clean upstream position (`UPSTREAM_SETTLE_POS`), belt + 1.30 m —
  the garment drapes at most ~0.95 m (bank max) below the grasp chord, well clear of the
  belt. No belt/drum/robot interference.
* **Initial states:** restored from `tshirt_hanging_bank.pt` (356 anchor-centred settled
  hangs, 348 distinct anchor particles — uniform over the garment by construction) with
  explicit, logged yaw + mirror augmentation; re-attached at slot 0 with the pad radius
  0.07 m; 45 damped steps to absorb the restore.
* **The horizontal presentation stretch** (the canonical procedure, every method): the
  second grasp attaches at its chosen point on the hanging shirt, then moves on a
  STRAIGHT LINE at 0.45 m/s to a target at the SAME HEIGHT as the first grasp, offset
  along world x — the camera-plane horizontal — by `target_ratio × rest_dist` (rest
  distance of the two grasp particles in the flat rest shape). The taut chord ends up
  horizontal in the image plane and **gravity provides the vertical extension** of the
  garment below it. Pull direction ±x = the side of the grasped point's current
  horizontal offset (never drags the cloth across itself); the regrasp heuristic
  stretches opposite to the previous stage. Tautness guard ≤ 1.15 — the
  Stage-0-validated two-attachment limit (same hold-centre definition as
  `test_two_attachments.py`). With the horizontal chord the tracked-particle tautness
  matches the hold-centre target within ±0.01 (median 0.953 / 1.003 / 1.059 / 1.097 for
  targets 0.95–1.10): the patch-centroid offset of `attach` and the chord's gravity sag
  cancel almost exactly.
* **Settle:** the bank generator's damped protocol (per-step velocity ×0.97) until the
  robust criteria fire (p95 particle speed < 0.10 m/s AND centroid speed < 0.08 m/s, cap
  420 steps), then 30 free steps before measuring.
* **Batching/seeding:** everything vectorized over 48–64 envs (~400 env·steps/s, the
  Stage-0 knee), explicit `torch.manual_seed`, ~45–80 trials/min including all settles.

Secondary metrics per trial: measured tautness, projected extents, drape below the grasp
chord, settle/stretch step counts, stretch direction.

### 1.4 Garment regions

Grasp locations are keyed by flat-rest coordinates and binned into 12 interpretable
regions = Voronoi cells of hand-placed landmarks (a pure x/y-threshold scheme fails — this
mesh's left sleeve angles inward and overlaps the shoulder in x). Left/right regions are
mirror-folded into **8 classes** for aggregation (the bank's mirror augmentation makes
them physically equivalent): collar, shoulder, sleeve, chest, side, belly, hem_corner,
hem_c. Definitions in
[`regions.py`](../../src/tensegrity_pick/scripts/model_validation/present_heuristics/regions.py);
assignment happens offline from the CSVs, so boundaries can be iterated without re-running
simulation.

![Garment regions](figures/present_heuristics/region_map.png)

Mesh-density caveat: the ClothesNet mesh is denser at seams/hems, so uniform-particle
sampling is only approximately uniform over area. Region aggregates are medians
*conditional on the region*, so density only affects per-cell n (reported in every map),
not the values.

---

## 2. Heuristic 1 — naive 2-grasp (the `shirt_present` heuristic)

**Protocol:** random first grasp (bank restore) → second grasp at the LOWEST hanging
point → horizontal presentation stretch, sweeping target ratio 0.95 / 1.00 / 1.05 /
1.10 → settle → measure. 960 trials (240 per ratio), seed 11, all valid.
Script: `run_two_grasp.py`; data: `present_h1_two_grasp.csv`.

![H1 ratio sweep](figures/present_heuristics/h1_ratio_sweep.png)

| target ratio | median | IQR | p90 | measured tautness | settle steps (med) |
|---|---|---|---|---|---|
| 0.95 | 0.657 | 0.603–0.701 | 0.739 | 0.95 | 133 |
| 1.00 | 0.662 | 0.605–0.699 | 0.761 | 1.00 | 127 |
| 1.05 | 0.679 | 0.621–0.722 | 0.778 | 1.06 | 132 |
| 1.10 | **0.690** | 0.611–0.740 | 0.789 | 1.10 | 340 |

**Findings.**

1. **The horizontal presentation works**: the naive heuristic already reaches 0.66–0.69
   median — decisively ABOVE the free one-point hang (0.563). Gravity extends the
   garment below the taut horizontal chord, spanning the silhouette in both image axes
   (projected width ≈ 0.63 m at 1.10, vs 0.59 m flat width).
2. **Tautness has a real, monotone effect** (+0.033 median, +0.050 at p90, from 0.95 →
   1.10): tension straightens the chord, widens the top edge and pulls folds out of the
   upper half. The gradient is genuine but modest — grasp placement still moves the
   needle more (see §4).
3. Trade-off at 1.10: settle time jumps (median 340 steps vs ~130 below — the stiffer
   chord rings longer before the damped criteria fire). 1.05 is the sweet spot per
   settle-time-adjusted throughput; 1.10 buys +0.011 median if time is free.
4. Where the lowest point lands: sleeve end 76 %, hem corner 24 % — unchanged (this is a
   property of the hang, not the stretch).
5. Self-alignment: `cov_yaw_max − cov_plane` = 0.0035 mean over all 960 trials — the
   horizontal chord pins the presentation into the camera plane; no yaw control needed.

Exemplars (inspection-camera view; 3/4 views in
[figures/present_heuristics/renders/](figures/present_heuristics/renders/)):

| worst (0.41) | median (0.68) | best (0.87) | free hang reference (0.57) |
|---|---|---|---|
| ![](figures/present_heuristics/renders/h1_s1_t00515_cam.png) | ![](figures/present_heuristics/renders/h1_s1_t00510_cam.png) | ![](figures/present_heuristics/renders/h1_s1_t00577_cam.png) | ![](figures/present_heuristics/renders/hang_median_cam.png) |

## 3. Heuristic 2 — 3-grasp with regrasp (+ iterated regrasp)

**Protocol:** stage 1 = exactly H1 at ratio 1.05; then the FIRST grasp releases (the
second keeps holding where the stretch left it, at height) → re-settle → regrasp at the
NEW lowest point → horizontal stretch in the OPPOSITE x direction → measure (stage 2).
`--regrasps 3` continues the cycle, alternating directions (stages 3–4). 288 trials × 4
stages, seed 22, all valid. Stage-1 rows double as a PAIRED H1 sample — they reproduce
the independent H1 run's median across seeds (0.677 vs 0.679).
Script: `run_three_grasp.py`; data: `present_h2_three_grasp.csv`.

| stage | median | paired Δ to previous | trials improved |
|---|---|---|---|
| 1 (naive 2-grasp) | **0.677** | — | — |
| 2 (regrasp) | 0.653 | **−0.031** | 37 % |
| 3 (iterate ×2) | 0.656 | +0.004 | 53 % |
| 4 (iterate ×3) | 0.656 | −0.001 | 48 % |

**Findings.**

1. **The regrasp HURTS in the horizontal-presentation geometry** (paired −0.031 median;
   only 37 % of trials improve), and further iterations plateau at the lower level
   instead of recovering.
2. **Why:** after the regrasp, BOTH grasps sit on garment extremities — the former
   lowest point (sleeve end / hem corner) plus the new lowest point (hem corner 78 % of
   stage-2 regrasps). A horizontal chord between two extremal points spans a WIDER
   bounding box (projected width 0.69 vs 0.61 m, drape 0.65 vs 0.55 m) but fills it
   worse: the body hangs as a diagonal fold between the two corners, opening empty
   regions in the projection. The stage-1 pair — a random (often mid-garment) anchor
   plus the lowest point — centres the drape better.
3. Contrast with intuition: regrasping is the textbook move (Maitin-Shepard) for
   *unfolding into a known pose*; for THIS objective (projected coverage of a
   two-point-held presentation) it trades fill for span at a net loss. The productive
   use of a third grasp is grasp-point CHOICE, not repetition — see §4/§5.

| worst (0.40) | median (0.65) | best (0.78) |
|---|---|---|
| ![](figures/present_heuristics/renders/h2_s2_t00227_cam.png) | ![](figures/present_heuristics/renders/h2_s2_t00215_cam.png) | ![](figures/present_heuristics/renders/h2_s2_t00077_cam.png) |

## 4. Heuristic 3 — the grasp-pair oracle map

**Protocol:** random first grasp (bank restore — the bank's pinned particle is uniform
over the garment by construction) → second grasp at a UNIFORM-RANDOM particle →
horizontal presentation stretch at ratio 1.05 → settle → measure. 2 880 trials, seed 33,
all valid. Sampling: uniform over particles ≈ uniform over area (≈5 mm spacing; seams
are denser — affects only per-cell n, reported in every map).
Script: `run_pair_map.py`; data: `present_h3_pair_map.csv`.

Headline numbers: random pairs median **0.606** (IQR 0.521–0.697) — BELOW the
lowest-point heuristic (0.679): in the horizontal-presentation geometry the lowest-point
rule is a decent chooser, because it reliably picks a LONG chord. But the map's upper
tail is far above both: **top-decile median 0.842**, best sampled pair 1.03 (≈ the flat
face-on reference — slight overstretch + raster edge effects push it past 1.0).

![Pair heatmap](figures/present_heuristics/pair_heatmap.png)

The pair map (first grasp = hang anchor, rows × second grasp = stretched, columns;
mirror-folded regions, cells n<15 masked):

* **Best cells — hem/edge pairs dominate:** side→hem_c **0.83**, side→hem_corner
  **0.78**, hem_c→hem_corner **0.73**, hem_corner→hem_corner **0.72**, sleeve→hem_corner
  **0.70**. Holding the garment by its bottom edge (or side seam + bottom) puts the
  taut chord along a long garment edge, and the whole body + sleeves drape below it as
  one wide curtain.
* **Worst rows:** belly-first (0.61 best cell) and any pair with a mid-panel second
  grasp — a chest/belly grasp gathers fabric mid-panel and folds the garment around it.
* **Oracle upper bound:** top-decile median **0.842**, single best 1.03.

![Best partner per first grasp](figures/present_heuristics/best_partner_map.png)

**What drives pair quality:** in this geometry it is chord LENGTH (r = +0.33 with pair
rest distance), NOT orientation on the garment (r = 0.05). The mechanism is simple: the
horizontal chord is the presentation's top edge — the longer it is, the wider the
curtain gravity hangs below it. (The discarded vertical-stretch variant had exactly the
opposite signature — orientation r = 0.41, distance r = −0.18 — a reminder that pair
"quality" is a property of the presentation procedure, not of the pair alone.)

![Pair drivers](figures/present_heuristics/pair_drivers.png)

Extreme sampled pairs (camera view):

| best sampled pair (1.03) | worst sampled pair (0.29) |
|---|---|
| ![](figures/present_heuristics/renders/h3_best_cam.png) | ![](figures/present_heuristics/renders/h3_worst_cam.png) |

## 5. Further methods

Five follow-ups were tried and measured; two win, three are recorded negatives.

### 5.1 Iterated lowest-point regrasp — no gain (see §3, stages 3–4)

Alternating-direction regrasp cycles plateau at the (lower) single-regrasp level;
coverage never recovers the stage-1 starting point.

### 5.2 Oracle-guided second grasp — +0.034 from arbitrary first grasps

**Protocol:** exactly H1 except the second grasp is chosen by the pair map instead of
the lowest point: classify the first grasp's region → look up the best partner region
(policy table [present_oracle_policy.json](data/present_oracle_policy.json), from §4:
shoulder/sleeve/hem_c → hem_corner, side/chest → hem_c, belly → side, collar → sleeve,
hem_corner → hem_corner) → grasp the particle at that region's landmark (l/r variant
farther from the first grasp). Ground-truth particle lookup stands in for perception →
an upper bound for any real region-targeting chooser. 240 trials, seed 55.
Script: `run_oracle_pick.py`; data: `present_h5_oracle_pick.csv`.

**Result: median 0.713 (IQR 0.628–0.756, p90 0.811)** — +0.034 over the lowest-point
rule. Per-first-region medians: hem_corner 0.78, hem_c 0.74, sleeve 0.73, chest 0.70 —
the policy notably RESCUES mid-panel first grasps (chest-first reaches 0.70 by pairing
with the bottom-centre hem; under the lowest-point rule such trials populate H1's lower
quartile).

| worst (0.43) | median (0.71) | best (0.97) |
|---|---|---|
| ![](figures/present_heuristics/renders/h5_oracle_s1_t00112_cam.png) | ![](figures/present_heuristics/renders/h5_oracle_s1_t00177_cam.png) | ![](figures/present_heuristics/renders/h5_oracle_s1_t00201_cam.png) |

### 5.3 Scripted shoulder↔shoulder — the human-intuition grip FAILS (negative)

Both grasps at the shoulder landmarks (first grasp restricted to shoulder-anchored bank
states, second at the opposite shoulder), 240 trials, seed 77
(`run_oracle_pick.py --fixed_target shoulder --first_regions shoulder`).
**Median 0.493; 57 % of trials below 0.55** — the WORST method tested, despite being how
a person shows a shirt. The distribution is strongly bimodal (p75 0.707, p90 0.758):
when it works it looks like the human presentation, but in over half the trials the
short shoulder chord (≈ 0.35 m rest distance — the shortest of any scripted pair) lets
the body fall into a vertical fold, and the sleeves drape over the front. A person
implicitly fixes this with micro-adjustments; a static two-point hold does not.

| worst (0.36) | median (0.49) |
|---|---|
| ![](figures/present_heuristics/renders/h7_shoulder_s1_t00043_cam.png) | ![](figures/present_heuristics/renders/h7_shoulder_s1_t00016_cam.png) |

### 5.4 Scripted hem_corner↔hem_corner — the WINNER

Same protocol with both grasps at the hem-corner landmarks (garment held upside-down by
its bottom edge), 240 trials each at ratio 1.05 (seed 88) and 1.10 (seed 99).
Data: `present_h8_hem_pair.csv`, `present_h8_hem_pair_110.csv`.

**Median 0.820 @1.05 / 0.832 @1.10, p90 0.95–0.97** — within 0.02 of the ORACLE TOP
DECILE (0.842) using one fixed, perception-friendly rule (hem corners are rigid,
seam-marked, easy-to-detect features). Why it wins: the taut chord IS the garment's
longest continuous fabric edge, so the entire body panel hangs below it as one wide
curtain, with the sleeves falling outside the torso silhouette instead of over it.
Tautness beyond 1.05 adds little (+0.012 median) — the edge is already straight.

| worst (0.50) | median (0.82) | best (1.09) |
|---|---|---|
| ![](figures/present_heuristics/renders/h8_hem_s1_t00197_cam.png) | ![](figures/present_heuristics/renders/h8_hem_s1_t00191_cam.png) | ![](figures/present_heuristics/renders/h8_hem_s1_t00078_cam.png) |

### 5.5 Out-of-plane shake before the final settle — no effect (negative)

H1 + a ±0.04 m sinusoidal Y oscillation of the moving grasp (3 cycles) between stretch
and final settle, 240 trials, seed 66 (`run_two_grasp.py --shake_amp 0.04`):
median 0.671 vs 0.679 without — no improvement (if anything, slight residual sway at
measurement). Kept for the record; the folds that cost coverage in this geometry are
held by gravity + the grasp choice, not by stiction that agitation could release.

## 6. Recommendation for the `shirt_present` RL task

**Success-threshold calibration** (camera-plane `silhouette_coverage`, the number the
task's reward/metric sees — the presentation is self-aligned, §1.1, so no swing
correction is needed):

| percentile | H1 naive scripted (the task's heuristic) | hem↔hem scripted (best) |
|---|---|---|
| p10 | 0.555 | 0.560 |
| p25 | 0.621 | 0.701 |
| median | 0.679 | 0.820 |
| p75 | 0.722 | 0.896 |
| p90 | 0.778 | 0.948 |

1. **Success threshold: coverage ≥ 0.65** (just under the scripted heuristic's median —
   a policy matching the naive heuristic succeeds ~60 % of episodes, so the signal is
   learnable from the start), with **0.75 as the curriculum end stage / report
   headline** (naive p90 — reaching it consistently requires learning better grasp
   placement or tauter, cleaner stretches).
2. **Reward tautness up to ≈ 1.05–1.10, gently.** The gradient is real (+0.033 median
   over the sweep) but modest; keep the ≤ 1.15 guard. Note 1.10 roughly doubles the
   settle time (pendulum ringing) — if the episode has a stability latch, 1.05 is the
   better operating point.
3. **Do NOT add a regrasp phase for visibility** (−0.031 paired, §3) — this also answers
   TODO Task-2 open question #2: robot 1 should KEEP HOLDING through the whole stretch;
   releasing and regrasping lowers coverage.
4. **The big lever is grasp-point choice** (+0.14 median over the naive rule): target
   the HEM CORNERS. If the upstream pick (Task 1 / shirt_pick) can be biased toward hem
   corners — or the presentation policy can select the second grasp by region — the
   scripted hem↔hem presentation already sits within 0.02 of the oracle top decile.
   Hem corners are also the perception-friendliest features on the garment. Second-best
   general rule from an ARBITRARY first grasp: pair-map policy (+0.034, §5.2), i.e.
   second grasp toward hem_corner/hem_c for almost every first-grasp region.
5. **No yaw control needed**: the horizontal chord self-aligns the presentation to the
   camera plane (measured gap 0.003) — spend no reward terms on orientation.

---

## Iteration log

* **2026-07-04 — study setup.** Booted `Template-Shirt-Pick-Tensegrity-v0` with manual
  stepping (bank-generator pattern); anchor at belt + 1.3 m (bank drape max 0.95 m).
  Own copy of the bank restore with explicit yaw/mirror logging so `anchor_idx` stays
  known per trial (the env-base helper randomizes internally). Smoke run (8 envs): all
  trials valid, ~400 env·steps/s end-to-end.
* **2026-07-04 — tautness definition checked** against Stage-0
  (`test_two_attachments.py`): its 1.15 limit is hold-centre-based; the tracked-particle
  ratio reads higher at the same hold-centre target (patch-centroid recentring in
  `attach`). Guard kept on the hold-centre definition; particle ratio logged as
  `meas_ratio`.
* **2026-07-04 — GEOMETRY CORRECTION (full rerun).** The first complete pass implemented
  the stretch **along the pair's current separation direction** (≈ vertically downward,
  since the second grasp hangs below the first) — a misread of the intended procedure.
  Georg's clarification: the second grasp must be brought UP to the first grasp's height
  and stretched HORIZONTALLY along the camera-plane x axis, so gravity supplies the
  vertical extension. All vertical-stretch results were discarded and every experiment
  rerun with the corrected geometry. Two artifacts of the wrong geometry are worth
  remembering: the vertical taut chord left a free yaw DOF (the garment swung
  arbitrarily out of the camera plane) and produced narrow ~0.27 m-wide silhouettes
  scoring BELOW the free hang — both disappear with the horizontal chord
  (`cov_yaw_max − cov_plane` ≈ 0.001 measured, i.e. self-aligned).
* **2026-07-04 — render lesson.** Writing saved particle states into the referenced
  garment mesh's USD points renders blank: with the sim playing, Hydra reads Fabric, so
  USD point writes never reach the camera. Fix: author the garment as real PBD cloth
  (the `render_shirt_check.py` scene) and push states through
  `ClothObject.write_nodal_state_to_sim` every frame (GPU → Fabric → renderer), zero
  velocity to pin the pose. Camera focal 35 mm.
* **2026-07-04 — calibration wobble noted.** Each boot re-adopts the settled rest shape
  (`recompute_flat_rest_from_current`), so the reference flat area varies 0.292–0.294 m²
  across processes → ~±0.7 % cross-run coverage normalization noise. All reported
  effects are ≥ 3 % — unaffected.
* **2026-07-04 — corrected H1 (960).** Tautness now monotone (0.657 → 0.690), naive
  heuristic decisively above the free hang; self-alignment confirmed at scale (yaw gap
  0.0035). The tracked-particle vs hold-centre tautness offset of the vertical variant
  (+0.07) vanished — patch-centroid offset and chord sag cancel horizontally.
* **2026-07-04 — corrected H2 (288 × 4).** REVERSAL: the regrasp that helped the
  vertical geometry (+0.031) HURTS the horizontal one (−0.031, 37 % improve) —
  extremity↔extremity chords span wider but fill worse. Stage-1 medians again reproduce
  H1 across seeds (0.677 / 0.679).
* **2026-07-04 — corrected H3 (2 880).** Second reversal: random pairs (0.606) now sit
  BELOW the lowest-point rule (0.679) — long chords win and the lowest point reliably
  provides one. Driver flipped from orientation (vertical: r = 0.41) to chord length
  (horizontal: r = 0.33). Hem/edge pairs dominate the map (side→hem_c 0.83); policy
  JSON rebuilt.
* **2026-07-04 — optimizations.** Oracle-guided second grasp 0.713 (+0.034, rescues
  mid-panel first grasps); scripted hem↔hem 0.820 @1.05 / 0.832 @1.10 (p90 0.95 —
  within 0.02 of the oracle top decile); scripted shoulder↔shoulder 0.493 (the
  human-intuition grip fails — bimodal, short chord folds the body); shake 0.671 vs
  0.679 (nil). Success threshold recalibrated to 0.65 / 0.75.

# Task: Robot-free study of shirt-presentation grasping heuristics — FAPS-server agent

You are a coding agent with full access to this repo (`/home/robot/studentische-arbeiten`),
running on the FAPS server (single ~48 GB GPU). Your job is a **simulation study, not an RL
task**: using the established PBD cloth pipeline (no robots — grasps are static/scripted solver
anchors), quantify **which grasp points / grasp strategies give the best overall visibility of
the shirt in the inspection-camera plane (front AND back view)**, produce thesis-quality
figures + example renders, and write a findings report. Your results directly inform the
`shirt_present` RL task an agent is developing in parallel on the Alex cluster.

Two sibling agents work in parallel on Alex (`shirt_present`, `shirt_distribute`) —
**coordination rules in §7 are hard requirements.**

---

## 1. Setting & the visibility objective

- Camera geometry: inspection camera at `INSPECTION_CAMERA_POS = (0.15, 2.0, 1.25)` looking
  along −Y; the image plane is the world **XZ plane** (`view_axis=1`). A back camera mirrors it,
  so front+back see the SAME occluding silhouette — the scalar objective is
  **`silhouette_coverage`** from `shared/cloth_metrics.py`: rasterized projected-silhouette area
  ÷ one-sided flat area (`flat_silhouette_area(cloth.flat_rest_pos)`). 1.0 = fully unfolded and
  camera-parallel; folds/bunching/out-of-plane rotation all reduce it. The module is pure torch
  and unit-tested (`scripts/model_validation/test_cloth_metrics.py`).
- Since the study is robot-free, normalize orientation: perform stretches **within the camera
  plane** (or report the coverage of the plane-aligned pose), so heuristics are compared by
  achievable *shape*, not by an arbitrary yaw. Report both the plane-aligned number and the
  yaw-marginalized one if they differ meaningfully.
- Also track secondary metrics per trial: `stretch_ratio` between grasps (tautness — stay ≤ 1.15,
  the validated two-attachment stability limit), `plane_extent`, drape length below the top
  grasp, settle time.

## 2. The three heuristics to investigate (in this order)

1. **Naive 2-grasp:** first grasp at a RANDOM shirt particle → hang/settle → second grasp at the
   **lowest hanging point** → stretch until taut (sweep the target stretch ratio, e.g.
   0.95 / 1.0 / 1.05 / 1.1). This is exactly the `shirt_present` task's heuristic — your measured
   coverage distribution is that task's success-threshold calibration.
2. **3-grasp with regrasp:** as (1), then **drop the first grasp**, let the shirt hang from the
   second → **regrasp at the new lowest point** → stretch again. (Two attachment slots exist —
   a regrasp sequence never needs more than two simultaneous anchors.)
3. **Random-pair map:** grasp two RANDOM particles → stretch to taut → measure. Sample enough
   pairs (batched across envs; think ≥ 2–5 k trials) to build a **grasp-pair quality map**:
   which particle-pair REGIONS give the best presentation. Key the map by the particles'
   rest-shape coordinates (`cloth.flat_rest_pos` — i.e. positions on the flat shirt), and
   aggregate into interpretable regions (collar / shoulders / sleeve ends / hem corners /
   center …). This is the brute-force oracle flagged in `doc/TODO.md` (Task 2, "grasp-pair
   oracle"): it upper-bounds all heuristics and labels good second-grasp regions.

**Afterwards:** propose and try further methods/optimizations that beat the above — e.g.
lowest-point regrasp iterated to convergence (Maitin-Shepard-style), choosing the SECOND grasp
by the oracle map instead of the lowest point, small dynamic shakes before stretching to release
folds, or gravity-assisted unfolding between regrasps. Timebox exploration; measure everything
you try.

## 3. What is already built for you

- **Grasp/anchor machinery** (`shared/cloth_object.py::ClothObject`): two attachment slots,
  `attach(env_ids, centers, radius, slot)` (pad-sized radius 0.07 = "one grasp point"),
  `hold(centers, slot)` per step to move an anchor along a scripted trajectory (that IS the
  stretch), `detach`, `is_attached_slot`. Static anchors need no robot.
- **The generation pattern to copy:** `scripts/asset_generation/generate_hanging_bank.py` — boots
  `Template-Shirt-Pick-Tensegrity-v0`, sets `episode_length_s = 1e6`, steps the sim manually
  (`scene.write_data_to_sim(); sim.step(render=False); scene.update(dt); cloth.update()`),
  hangs the shirt from a random particle at a clean UPSTREAM anchor (away from all scene
  geometry) and settles with a **damped phase** (synthetic air drag, per-step velocity scale
  0.97 — PBD cloth pendulums for tens of seconds otherwise) + free re-equilibration, with
  ROBUST settle criteria (p95 particle speed + centroid speed — the max never converges; a few
  particles jitter at ~1 m/s forever). Reuse this pattern wholesale; do your experiments at the
  same upstream anchor so no belt/drum/robot interferes.
- **Initial states:** the hanging bank `res/Props/Cloth/banks/tshirt_hanging_bank.pt`
  (`pos`/`vel` anchor-centred, `anchor_idx` = the pinned particle, `drape`) — restore via
  `cloth.write_nodal_state_to_sim(pos, vel, env_ids)` + `attach` at the anchor (or through
  `ClothSortingEnvBase._reset_cloth_hanging_from_bank`) instead of re-settling from scratch;
  bit-exact cache-restore is validated (`test_cache_restore_reset.py`). For heuristic 3 you can
  also hang directly from chosen particles (generate fresh, the bank generator shows how).
- **Rendering for the report:** `scripts/model_validation/render_shirt_check.py` is the
  established camera-capture pattern (offscreen render to PNG). Render exemplars from the
  camera's actual viewpoint (−Y at `INSPECTION_CAMERA_POS` geometry, or the equivalent at your
  upstream anchor) AND a 3/4 view so the grasp configuration is visible.
- Efficiency facts: cloth throughput peaks at 32–64 envs (~400 env·steps/s, falls above —
  Stage-0 benchmark); batch trials across envs; boot takes 1–2 min (pre-settle) so do MANY
  trials per process launch; settling a hang takes ~2 s sim with the damped protocol.

## 4. Method requirements

- **Batched + seeded:** everything vectorized over envs, explicit `torch.manual_seed`, trial
  parameters logged so any figure is reproducible from a CSV.
- **Stretch procedure** (define once, use for all methods): move the second anchor away from the
  first along their current separation direction (rotated into the camera plane for the
  plane-aligned variant) at a gentle speed (~0.3–0.6 m/s hold-target steps), stop at the target
  stretch ratio, damped-settle, measure. Guard: tautness ≤ 1.15.
- **Statistics before conclusions:** ≥ ~200 trials per heuristic variant (bank + fresh hangs),
  report distributions (median/IQR), not single runs. The pair map (heuristic 3) needs the
  bigger budget — plan the sampling (uniform pairs vs stratified by garment region) and justify
  it.
- Sanity checks: coverage of (a) the flat shirt face-on ≈ 1.0, (b) an unstretched center hang
  (expect ~0.3–0.5) — report these anchors so the reader can calibrate the scale.

## 5. Deliverables (thesis-grade — this feeds a Master's-thesis chapter)

1. **`doc/reports/present_heuristics_study.md`** — the findings report (repo style: dated
   iteration log at the end, measured numbers, decisions + rationale, losers kept for the
   record). Structure suggestion: setup & metric definition → per-heuristic results →
   pair-map analysis → additional methods → recommendation for the `shirt_present` RL task
   (success threshold + best regrasp target).
2. **Figures** under `doc/reports/figures/present_heuristics/` (+ raw CSVs under
   `doc/reports/data/`). Think about what best serves a thesis; strong candidates:
   - coverage distributions per heuristic (box/violin, common axis, n stated);
   - coverage vs stretch-ratio sweep curves;
   - the **grasp-pair heatmap** on the flat-shirt outline (first-grasp region × second-grasp
     region → median coverage; plus a "best partner for each first grasp" map) — this is the
     centerpiece, iterate on its readability;
   - a method-comparison summary bar with the oracle upper bound as a reference line.
3. **Example renders** (PNG, camera view + 3/4 view) for each heuristic: typical AND best/worst
   cases, annotated with their coverage value — the method and result must be visible at a
   glance. Embed them in the report.
4. **Reusable scripts** in `scripts/model_validation/present_heuristics/` (new dir), each with a
   usage-comment header, so every figure/number is re-runnable: e.g. `run_two_grasp.py`,
   `run_three_grasp.py`, `run_pair_map.py`, `render_exemplars.py`, `plot_study.py`
   (matplotlib via `_isaac_sim/python.sh` or the conda env; tensorboard is NOT installed).
5. Tick/annotate the "grasp-pair oracle" and related items in `doc/TODO.md` (Task 2 section).

## 6. How to run (FAPS server)

```bash
cd /home/robot/studentische-arbeiten/src/tensegrity_pick
source /home/robot/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && unset VIRTUAL_ENV
env PYTHONUNBUFFERED=1 "$CONDA_PREFIX/bin/python" -u <script> --headless <args>
```
- **ONE Isaac Sim process at a time** — check `nvidia-smi` first; Georg's own GUI session (an
  `isaacsim.exp.full.kit` process) may hold the GPU — never kill processes you did not start.
- carb traps SIGTERM: stop your own runs with `kill -9`. Expect the harmless recurring USD
  `Sublayer ... has cycles` warning — grep it out.

## 7. Coordination & constraints (hard rules)

- You share the working tree with Georg on `project/tendonbot-sim-rl`. **No git commits unless
  Georg asks.**
- **Do NOT edit** `tasks/manager_based/shared/**` or any task package (`shirt_present/**`,
  `shirt_distribute/**`, `shirt_pick/**`, …) — parallel agents own those. Your writable areas:
  `scripts/model_validation/present_heuristics/**`, `doc/reports/present_heuristics_study.md`,
  `doc/reports/figures/present_heuristics/**`, `doc/reports/data/**`, your TODO items.
  If a shared-code change seems unavoidable, note it in the report and work around it locally
  (copy the function into your script dir).
- GPU-time discipline: batch trials; don't leave idle sim processes running.

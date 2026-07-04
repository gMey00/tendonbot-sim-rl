# Shirt Present Task — Optimization Tracking

**Task:** `Template-Shirt-Present-Kinova-F140-v0` / `Template-Shirt-Present-UR5e-F140-v0` (pipeline task 2)
**Robot:** Kinova Gen3 / UR5e + Robotiq 2F-140 (second robot, pedestal mount `(0.75, 1.0, 0.75)`); passive tensegrity holder
**Framework:** Isaac Lab + SKRL (PPO), Isaac Sim 5.1.0
**Hardware:** NVIDIA RTX A6000 (48 GB)

**Current state (2026-07-03):** runnable stub — the shirt hangs from a static solver
anchor at the presentation pose (idealized task-1 terminal state); zero/random agents pass
for both robot variants.  The learning arm cannot yet grasp (single attachment slot).
No training runs yet.  Pipeline context:
[RESEARCH_cloth_sorting_pipeline.md](RESEARCH_cloth_sorting_pipeline.md) ·
staged plan in [doc/TODO.md](../TODO.md) · task docs in the
[shirt_present README](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_present/README.md).

---

## Phase 0: Task Stub with Hanging-Anchor Cloth

**Date:** 2026-07-03

### Summary

Created the shirt_present task (bimanual inspection stretch) as a robot-agnostic stub on the
shared cloth-sorting infrastructure (central scene + `ClothSortingEnvBase`, see the
[shirt_pick tracking Phase 0](shirt_pick_optimization_tracking.md) for that foundation).
The task-2-specific piece is the **hanging-anchor reset**: instead of lying on the belt, the
shirt is teleported to the presentation pose and its centre cluster (0.07 m radius) is
pinned as PBD solver anchors — the same `ClothObject.attach`/`hold` mechanism the validated
shirt_place grasp uses — so it drapes under gravity into a centre-hung garment.  This is an
idealized stand-in for "the retriever holds the shirt at its former highest point".

### Changes

#### Task files
- **`shirt_present/shirt_present_env.py`** — `ShirtPresentEnv(ClothSortingEnvBase)`:
  `enable_hand_grasp = False` (the single ClothObject attachment slot is taken by the holder
  anchor), `_reset_cloth` override (teleport to `PRESENTATION_POS (0.15, 0.50, 1.10)` +
  anchor pin), `_update_grasp` override (gripper drive + static `hold`).
- **`shirt_present/shirt_present_env_cfg.py`** — robot-agnostic cfg; scene adds the passive
  `holder_robot` (tensegrity at its hanging mount, PD-holds its default pose, no actions) as
  visual scenery.  Observations include `lowest_point_rel` — the literature-standard second
  grasp target on a hanging garment (lowest cloth particle, camera-trivial from depth;
  Maitin-Shepard 2010 / Doumanoglou 2014), computed from `ClothSortingEnvBase.shirt_lowest_point_w`.
- **`shirt_present/config/{kinova_f140, ur5e_f140}/`** — robot variants at the showcase
  second-robot mount (UR5e yaw +90°), joint-position actions (scale 0.5, default-offset),
  gym registrations, skrl PPO yaml (shirt_place determinism profile).

### Verification (2026-07-03)

| Check | Result |
|---|---|
| `zero_agent` Kinova, 4 envs headless | **PASS** — obs `(4, 37)`, act `(4, 8)`, ≥45 s stepping, no traceback |
| `random_agent` UR5e, 4 envs headless | **PASS** — obs `(4, 34)`, act `(4, 7)` |

### Stub simplifications (by design, tracked in TODO)
- **Initial state is the idealized centre-hang**, not the shirt_pick terminal-state bank —
  the report flags skill-chaining distribution shift as the pipeline's #1 risk; training
  against the idealized hang only would bake it in.
- **The learning arm cannot capture cloth yet.** `ClothObject` manages one attachment set
  per env (`_attach_mask`/`_attach_offset`); the holder anchor occupies it.  Multi-slot
  attachments + a two-attachment stretch-stability test (PBD overstretch between two pinned
  particle groups) is the core Task-2 physics work.  Fallback per report: handover.
- The passive holder is scenery — its gripper is not at the anchor position.  Posing it from
  the task-1 terminal bank comes with the bank work.
- `Metrics/grasp_rate` currently reflects the holder anchor (≈ 1.0 by construction) —
  placeholder until the second grasp exists.
- No inspectability reward yet (stub: reach shaping + regularisers only).

### Known design constraints for Phase 1+ (from the research report)
- Coverage must be measured from the **actual front/back camera viewpoints**, not top-down —
  top-down coverage is gamed by bunching/hiding the shirt.
- Tautness proxy = inter-grasp Euclidean distance / rest (geodesic) distance between the
  grasped particle groups; cap reward before ratio ≈ 1.0 (PBD overstretch inflates area).
- Reference success band: ICRA-2024 cloth-competition top three reached ~0.55–0.60 average
  coverage (real dual-UR5e; use as rough reference only).

---

## Update (2026-07-03, Stage-0 de-risk): two attachments validated

The core physics risk from Phase 0 is retired: `ClothObject` now supports two
independent attachment slots (slot 0 = hand, slot 1 = holder anchor; this env
now anchors on slot 1), and the two-attachment stretch test is **stable
through tautness ratio 1.15** (steady-state max particle speed ≤ 0.7 m/s,
bbox +5 %, both grasps intact, no NaN) — no handover fallback needed.  Full
results: [cloth_stage0_physics_derisk.md](cloth_stage0_physics_derisk.md) §2;
script: `scripts/model_validation/test_two_attachments.py`.  The shirt_place
regression stayed 5/5 after the refactor.  Note for the future MDP: the
one-step attach transient (~2.8 m/s particle snap) decays within ~5 steps —
ignore the first steps after a grasp in reward/termination predicates.

## Planned Phase 1: Bank Reset + Hand Grasp + Coverage Reward

Per the staged plan ([doc/TODO.md](../TODO.md)):
1. ~~Multi-slot attachments + stretch validation~~ — **DONE** (see update above).
2. Reset from the shirt_pick terminal-state bank
   (`ShirtPickEnv.snapshot_terminal_states` provides the capture side).
3. Enable the slot-0 hand grasp here; lowest-point regrasp **heuristic
   baseline**, then RL.
4. Projected-coverage + tautness reward; brute-force grasp-pair oracle script for the
   upper bound.  Calibrate the tautness cap against measured strain — the
   straight-line rest distance underestimates the fabric path (Stage-0 finding).
5. Snapshot stretched terminal states → shirt_distribute initial-state bank.

---

## Phase 1: Full regrasp-and-stretch MDP (Alex agent)

**Date:** 2026-07-04 · **Branch:** `project/shirt-present` (isolated git worktree, see ops note)
**Hardware:** Alex `rtxpro6k` (1× RTX PRO 6000 per job)

### Goal
Implement the naive two-grasp presentation heuristic as a trainable MDP (§1 of the agent
prompt): deterministic second grasp at the lowest hanging point (slot 0), stretch between the
grasps, honest silhouette-coverage success, windowed latch ≥ 0.9 deterministic.

### Changes (commit `a7e5c97`)
- **`shirt_present_env.py`** — `enable_hand_grasp = True`; `shirt_grasp_point_w` →
  `shirt_lowest_point_w` (the intended base-env hook — `_update_grasp` is NOT copied, the
  override calls `super()._update_grasp()` then re-pins the slot-1 holder anchor).  New
  per-step task-state buffers (recomputed after the physics step, consumed with the usual
  one-step delay):
  - **stretch ratio** = ‖patch-centroid₀ − patch-centroid₁‖ / ‖flat-rest separation‖ — the
    exact definition `test_two_attachments.py` validated stable through **1.15**;
  - **coverage** = `silhouette_coverage(nodal_pos_w, flat_silhouette_area(flat_rest), view_axis=1)`
    (camera XZ plane, folds count once).
  - Success predicate `presented_now`: `grasp_active ∧ holder_attached ∧
    stretch ∈ [0.90, 1.10] ∧ coverage ≥ 0.55 (placeholder, see calibration) ∧
    cloth-centroid speed < 0.20`.  Windowed latch ≥ 80 % of the last 60 steps
    (shirt_pick Phase-3 lesson: consecutive latches are brittle; EE speed gates
    unharvestable at raised postures — 0.24–0.27 m/s residual sway vs 0.06–0.20 on the cloth).
  - `drop_event` one-shot + `was_dropped`; `Metrics/{present_rate, drop_rate,
    final_coverage, final_stretch_ratio}` logged in `_reset_idx`;
    `snapshot_terminal_states` (both grasp states) for the Task-2→3 bank.
- **`mdp/rewards.py`** — sequential rewards (weights × dt = 1/60 s, one-shots ~60× per-step):
  | Term | w | Notes |
  |---|---|---|
  | reaching_lowest_point (tanh, std 0.25, dyn. finger tip) | 2 | saturates to 1 while grasped (lowest point migrates post-grasp — chasing it would fight the stretch) |
  | grasp_hold | 5 | slot-0 attached |
  | stretch_progress (lo 0.50 → hi 0.98, clamped) | 8 | gated on BOTH attachments |
  | coverage_reward | 10 | gated on both attachments (anti-fling/bunch) |
  | presented (full predicate) | 30 | dominant per-step success term |
  | overstretch (ratio − 1.10, proportional) | −40 | safety margin under the validated 1.15 |
  | drop one-shot | −120 | ≈ −2 after dt-scaling |
  | action_rate / joint_vel | −1e-4 → curriculum −3e-3 / −2e-3 @ 2000 steps | shirt_place profile |
- **`mdp/__init__.py`** — `from .rewards import *` (NOT `from . import rewards` — silently
  keeps isaaclab's own `rewards` bound; lesson 5).
- **`shirt_present_env_cfg.py`** — obs add: lowest point rel DYNAMIC finger tip (the attach
  trigger's own geometry), both grasp flags, stretch ratio, coverage (all camera-derivable in
  principle); episode 8.0 s (reach ~1.5 s + grasp + stretch + 1 s latch window + swing slack);
  curriculum added.  Action space: stub's `JointPositionActionCfg` (scale 0.5,
  default-offset) per the prompt's decision — EMA variant is the registered fallback.
- **`scripts/model_validation/baseline_shirt_present.py`** — scripted two-grasp baseline:
  FD **action-space** Jacobian (columns = tip m / action unit, re-estimated every 200
  approach steps — stale-J lesson), P-servo to the lowest point → close → pull along the
  anchor→grasp ray with live tautness-ratio stop feedback → hold.  Measures raw-hang vs
  stretched coverage/tautness distributions (threshold calibration) and per-env reachability.

### Ops note: sibling-agent collision → isolated worktree
The shirt_distribute agent works in the SAME `$HOME/studentische-arbeiten` checkout; its
`git checkout -b project/shirt-distribute` (between this agent's branch creation and first
commit) put commit `a7e5c97` on the wrong branch, and any later branch switch would have
rewritten the other agent's files mid-job.  Fixed: branches repointed
(`shirt-present → a7e5c97`, `shirt-distribute → 14b6278`, sibling's uncommitted files
untouched), and ALL shirt_present work now runs from an isolated **git worktree**
`$HOME/studentische-arbeiten-present` with `PYTHONPATH` pinning imports to the worktree
(the conda env's PEP-660 editable finder is `sys.meta_path.append`ed, so `PathFinder` +
`PYTHONPATH` wins — verified).  ⚠️ Slurm jobs of the two agents are otherwise independent.

### Validation (2026-07-04)

**Cloth env-count re-benchmark on the RTX PRO 6000** (`bench_cloth_env_count.py
--task Template-Shirt-Present-UR5e-F140-v0`, zero actions, job 3809914):

| N envs | construct (s) | steps/s | env·steps/s | driver GPU (GB) | sane |
|---|---|---|---|---|---|
| 32 | 19.8 | 36.3 | **1162** | 4.9 | ✓ |
| 64 | 39.5 | 16.7 | 1066 | 6.0 | ✓ |
| 128 | 96.6 | 6.4 | 819 | 8.4 | ✓ |

Same qualitative shape as the A6000 Stage-0 result (throughput peaks at
N = 32–64 and FALLS above), ~2.8× faster in absolute terms.  **Decision:
train at 64 envs** — 8 % below peak throughput for 2× the PPO batch.

| Check | Result |
|---|---|
| `zero_agent` UR5e, 4 envs headless (job 3809913) | **PASS** — obs `(4, 38)` (+4 task-state dims vs stub), act `(4, 7)`, ~4 min stepping, no traceback |

### Scripted-baseline iterations & the three measured MDP corrections (2026-07-04)

| Run | Controller / change | reached <7 cm | grasped | Key finding |
|---|---|---|---|---|
| v1 (3809926) | FD action-space Jacobian, act clamp ±1 | 0/16 | 0/16 | min≈final dist 0.66 m — **action space saturated**: `JointPositionActionCfg` scale 0.5 gives only ±0.5 rad from the ready pose |
| v2 (3809927) | act clamp ±2.5 (=±1.25 rad) | 5/16 | 4/16 | reaches, but max \|a\| pinned at 2.5; **at-grasp stretch ratio read 1.01–1.58 on slack cloth** → flat-Euclidean rest normalisation over-reads wrap-around grasp pairs |
| v3 (3810090) | + geodesics from render-mesh faces | — | — | **CUDA device-side assert**: the render mesh has more vertices than the welded particle set; OOB index poisoned the context → job hung to timeout (carb traps signals) |
| v3.1 (3810135) | + EMA to-limits actions; springs still wrong path | 3/16 | 3/16 | FD servo saturates at joint limits over the full range; **frozen saturated actions after grasp dragged the cloth** (ratio 2.9, coverage 0.86) — hold must command the current posture |
| v4 | analytic PhysX Jacobian (fresh every step) + posture-hold via inverted to-limits mapping + geodesics from `physxParticle:springIndices/RestLengths` | *pending* | | |

**MDP changes locked in from these measurements** (commits `77b9ae1..`):
1. **Action space → `EMAJointPositionToLimitsActionCfg(α=0.2)`** (both variants) — the
   reach-grid joint-space winner.  Justification: reaching the lowest hanging point needs
   joint targets ±1.25 rad from the ready pose (v2), far outside the stub action's ±0.5 rad
   and outside the ±1 band a Gaussian policy explores well; to-limits puts every reachable
   posture inside [-1, 1] by construction.
2. **Stretch ratio normalised by the GEODESIC rest distance** from the holder patch
   (vectorised multi-source Bellman-Ford over the authored PBD spring graph
   `physxParticle:springIndices`/`springRestLengths`, refreshed per reset on GPU; scipy is
   not installed in the cluster env).  Flat-Euclidean read 1.31–1.58 on slack wrap-around
   pairs (v2) — a taut straight span should read ≤ 1.0.
   ⚠️ Do NOT derive the graph from the render mesh's face-vertex indices (v3 crash).
3. **Coverage threshold 0.55 → 0.50.**  Measured raw-hang coverage over the bank
   (16 hangs): mean 0.441, p50 0.425, p90 0.530; naive ray-pull stretched holds reach
   ≤ 0.512.  0.50 exceeds the raw median while remaining achievable; the ICRA-2024
   band (0.55–0.60) stays the aspirational reference — revisit after training.

### Baseline v4 (job 3810139) + the at-grasp-normalised tautness fix

Analytic-Jacobian servo: **8/16 reached < 7 cm, 11/16 grasped, 11/11 held** through the
2-s measurement hold (cloth speed ≤ 0.053 m/s — the stillness gate is comfortably passable).
The geodesic sanity check then exposed the LAST metric flaw: on the raw hang,
anchor→lowest Euclid/geodesic = 1.11–1.37 (mean 1.29) — the lowest-point span is ALREADY
gravity-taut at grasp (raw at-grasp ratio 1.10–1.43, mean 1.30; contributions: holder-patch
spread ≤ 7 cm from the anchor point + real PBD gravity strain).  An absolute [0.9, 1.1]
band can therefore never pass.  **Fix:** normalise per env by the at-grasp ratio r0
(recorded at the attach rising edge): slack < 0.92·r0, overstretch > 1.10·r0 (Stage-0
margin).  The stretch reward becomes maintain-tautness (1.0 at grasp, decaying when the
span droops); COVERAGE is the term that drives the actual opening/orienting of the garment.
Under the corrected predicate the naive ray-pull baseline passes coverage ≥ 0.50 in
~3/11 held envs — a genuine nonzero scripted baseline for RL to beat with oriented
stretches.

### Baseline v4.1 — MDP VALIDATED (job 3810144, 16 bank hangs)

| Metric | Value |
|---|---|
| reached < 7 cm / grasped / held through 2-s hold | 8/16 / **11/16** / 11/11 |
| at-grasp raw ratio r0 (held) | 1.10–1.43 (mean 1.30) — as analysed |
| held normalised ratio | 0.98–1.03 (pull target 1.02) — normalisation behaves as designed |
| held cloth speed | ≤ 0.052 m/s (gate 0.20 — comfortably passable) |
| held coverage | 0.36–0.59 (mean 0.44); envs ≥ 0.50: presented_frac 1.00 |
| **scripted present rate** | **2/16** (fails split: 5 approach-reach, 9 coverage < 0.50) |

Every gate is reachable, the naive scripted heuristic scores nonzero, and coverage is the
axis RL must improve (oriented stretch vs the baseline's blind ray-pull).  → Training
authorised per the workflow; run 1: `train.py --num_envs 64`, PPO profile from the yaml
(20 k trainer steps), job 3810160.

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

---

## Phase 2: Training

### Run 1 — PPO, 64 envs, 20 k trainer steps (job 3810160, 34 min)

Stochastic rollouts: grasp_rate → 1.00 within 2 k steps; present_rate climbing 0.33 → 0.68
and still rising at the 20 k cap; final_coverage → 0.56; drop_rate 0.05–0.21.
**Deterministic eval** (agent_20000, seed 7, 96 episodes, job 3810211):

| present_rate | grasp_rate | drop_rate | final_coverage | mean\|act\| |
|---|---|---|---|---|
| **0.625** | 0.927 | 0.167 | 0.574 | 0.520 |

vs scripted baseline 0.125.  No determinism gap (deterministic ≈ late stochastic rollouts).
Leaks to 0.9: drops (0.167 — the policy sometimes reopens the gripper) and residual
non-grasps (0.073).  Coverage at 0.574 already exceeds the 0.50 gate and sits in the
ICRA reference band.  → Run 2: same config, 60 k trainer steps (job 3810212) — run 1 was
still improving at cap; revisit the drop penalty only if run 2's drop_rate stays > ~0.1
(diagnose gates first, lesson: don't touch weights blindly).

### Run 2 (60 k, job 3810212) + gate diagnosis (job 3810430)

Deterministic evals (seed 7 × 96 eps): agent_60000 present **0.708** / drop 0.229;
best_agent 0.646 / drop 0.135 — grasp_rate 1.000 both.  Gate fractions on agent_60000
(8 envs × 480 steps, both-grasped-conditioned): stretch-in-band 0.858, **coverage 0.546
(weakest gate — the policy hovers at the 0.50 threshold)**, stillness 0.963; reach superb
(min 0.7–1.8 cm).  Drops (grasp_frac ~0.93 with re-grasp slack episodes) are the other leak
→ **Run 3 = drop one-shot −120 → −240**, otherwise identical (job 3810431).

### Run 3 (drop −240, 60 k, job 3810431)

drop_rate → 0.00 from ~24 k (penalty worked); stochastic present_rate plateaus 0.70–0.86
over 36–50 k, then **collapses after 52 k** (0.25 → 0.04 → 0.00; stretch_norm drifting to
1.20) — late-run policy collapse, textbook case for eval-based checkpoint selection
(shirt_place rule).  Candidates: agent_38000, agent_48000 → deterministic evals jobs
3810659/3810660.

### Run-3 checkpoint evals + failure taxonomy (jobs 3810659/3810660/3810664)

| Checkpoint | present | grasp | drop | final coverage |
|---|---|---|---|---|
| agent_38000 | 0.740 | 1.000 | 0.000 | 0.591 |
| agent_48000 | **0.750** | 0.958 | 0.000 | 0.647 |

Diag (agent_48000, 8 envs): 5/8 latched.  The three failures are three DIFFERENT
marginal-gate modes: (1) reach-miss — tip stalled 0.103 m, attach trigger (0.10) never
fired; (2) overstretch — held at ratio 1.134, just above the 1.10 band edge (the reward
plateau 0.97–1.10 has no gradient); (3) coverage 0.445 < 0.50.  No structural failure —
consistent with "more training + checkpoint selection", so next step is a 2-seed sweep at
the identical config capped at 48 k (jobs 3810665/3810666, seeds 1/2) before any further
weight surgery.  If no checkpoint reaches 0.9: overstretch-limit param 1.10 → 1.05 (creates
a gradient moat under the predicate band edge) + coverage weight bump are the queued levers.

### Run 4 (2-seed sweep, identical cfg, cap 48 k — jobs 3810665/3810666)

Deterministic (seed 7 × 96): s1/agent_28000 0.781 (grasp 0.979), s1/agent_40000 **0.792**
(grasp 1.000, drop 0.010), s2/agent_44000 0.635.  Confirms run-3 level; no seed luck ≥ 0.9.

### Run 5 (coverage 14, overstretch onset 1.05 — jobs 3811009/3811010)

Stochastic: s42 climbs to 0.84 at the 48 k cap and is STILL RISING (drop 0.00 at cap);
s1 peaks 0.81 @ 32 k then declines.  Deterministic (seed 7 × 96):

| Checkpoint | present | grasp | drop | coverage |
|---|---|---|---|---|
| s42/agent_48000 | **0.833** | 0.948 | 0.010 | 0.560 |
| s1/agent_32000 | 0.719 | 0.771 | 0.000 | 0.549 |

Monotone progress across iterations (0.625 → 0.750 → 0.792 → 0.833); the s42 leak is now
~5 % non-grasp episodes + latch margins.  → Run 6: same cfg, 96 k steps, seed 43
(job 3811137) — s42 was still climbing at cap; watching for the run-3-style late collapse
(checkpoint selection handles it either way).

### Run 6 — SUCCESS (96 k steps, seed 43, job 3811137)

Same config as run 5 (drop −240, coverage 14, overstretch onset 1.05).  Stochastic
present_rate holds 0.87–1.00 over the whole back half (48–96 k) with drop_rate 0.00 and
grasp_rate 1.00 — no run-3-style collapse.  **Deterministic checkpoint selection**
(mean actions, 96 episodes each):

| Checkpoint | eval seed | present_rate | grasp | drop | final coverage |
|---|---|---|---|---|---|
| agent_96000 | 7 | **0.927** | 0.990 | 0.000 | 0.682 |
| agent_96000 | 11 | **0.927** | 0.990 | 0.000 | 0.682 |
| agent_90000 | 7 | 0.917 | 0.990 | 0.000 | 0.664 |
| agent_90000 | 11 | 0.927 | 0.990 | 0.000 | 0.669 |

**SELECTED: `2026-07-04_15-21-09_ppo_torch_seed43/checkpoints/agent_96000.pt`** —
windowed present latch ≥ 0.9 across 2 eval seeds × 96 episodes ✓ (§2 success criterion).
Final coverage 0.66–0.68 sits ABOVE the ICRA-2024 reference band (0.55–0.60) and far above
both the raw hang (0.44) and the scripted baseline's holds (≤ 0.51): the policy learned an
oriented stretch the blind ray-pull cannot do.  Mean coverage of presented terminal states:
0.71–0.72 (snapshot job 3811352).

Iteration ladder (deterministic present_rate): 0.625 (run 1, 20 k) → 0.708 (run 2, 60 k)
→ 0.750 (run 3, drop −240) → 0.792 (run 4 sweep) → 0.833 (run 5, coverage 14 +
overstretch moat) → **0.927** (run 6, 96 k).

### Task-2 → Task-3 terminal bank hook — VALIDATED (job 3811352)

`snapshot_terminal_states` field shapes/finiteness/attachment invariants all pass on live
rollouts of the selected checkpoint (2 rounds × 32 envs: presented 30/32, 28/32; holder
anchor never lost).  Sample bank: 58 presented states (pos+vel + BOTH grasp masks +
stretch/coverage metadata) → `scripts/model_validation/snapshot_shirt_present_terminal.py`
regenerates at any size for shirt_distribute.

### Figures

`shirt_present/figures/ur5e_f140/01–06` (run 6, plotted via
`scripts/plotting/plot_shirt_present_training_results.py`).

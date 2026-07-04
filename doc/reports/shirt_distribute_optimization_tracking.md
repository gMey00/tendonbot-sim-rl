# Shirt Distribute Task — Optimization Tracking

**Task:** `Template-Shirt-Distribute-Kinova-F140-v0` / `Template-Shirt-Distribute-UR5e-F140-v0` (pipeline task 3)
**Robot:** Kinova Gen3 / UR5e + Robotiq 2F-140 (second robot, pedestal mount `(0.75, 1.0, 0.75)`)
**Framework:** Isaac Lab + SKRL (PPO), Isaac Sim 5.1.0
**Hardware:** NVIDIA RTX A6000 (48 GB)

**Current state (2026-07-03):** runnable stub with working goal-conditioning plumbing
(per-episode random target bin, `target_bin_rel` observation,
`Metrics/distribute_success_rate`); zero/random agents pass for both robot variants.
Shirt starts on the belt edge (not yet from the task-2 terminal bank); rewards are stub
shaping.  No training runs yet.  Pipeline context:
[RESEARCH_cloth_sorting_pipeline.md](RESEARCH_cloth_sorting_pipeline.md) ·
staged plan in [doc/TODO.md](../TODO.md) · task docs in the
[shirt_distribute README](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_distribute/README.md).

---

## Phase 0: Task Stub with Goal-Conditioned Bin Selection

**Date:** 2026-07-03

### Summary

Created the shirt_distribute task (label-conditioned sorting throw/place) as a
robot-agnostic stub on the shared cloth-sorting infrastructure (central scene +
`ClothSortingEnvBase`, see the
[shirt_pick tracking Phase 0](shirt_pick_optimization_tracking.md)).  The
task-3-specific piece implemented up front is the **goal-conditioned formulation**
recommended by the research report (§4, TossingBot precedent): the condition label never
enters the observation — it only selects *which* of the three drum positions is observed.
The per-episode target bin is resampled uniformly, so "nearest bin" and "correct bin"
diverge and the policy cannot shortcut the goal.  This kills the main Task-3 hack mode
before any reward tuning happens.

### Changes

#### Task files
- **`shirt_distribute/shirt_distribute_env.py`** — `ShirtDistributeEnv(ClothSortingEnvBase)`:
  `_target_bin ∈ {0,1,2}` resampled per episode **before** the parent reset (reset-step
  observations already see the new goal); `target_bin_pos_w` property; success = after a
  release, ≥ 15 % of cloth particles inside the commanded drum's interior cylinder
  (radius 0.2735 m, height 0.85 m — shirt_place geometry: draped cloth rarely reaches the
  drum bottom).  Requiring the release closes the "lower the still-gripped shirt into the
  drum" hack found and fixed in shirt_place.  Shirt rest pose `(0.75, 0.30)` — belt edge
  closest to the second robot.
- **`shirt_distribute/shirt_distribute_env_cfg.py`** — robot-agnostic cfg: 35/38-dim
  observations incl. `target_bin_rel` and `grasp_active`, stub rewards (reach 2.0 +
  goal-conditioned to-target-bin 4.0 + regularisers), timeout 6 s + divergence terminations.
- **`shirt_distribute/config/{kinova_f140, ur5e_f140}/`** — robot variants at the showcase
  second-robot mount (UR5e yaw +90°), joint-position actions (scale 0.5), gym
  registrations, skrl PPO yaml (shirt_place determinism profile).

#### Scene (three bins, from the central scene)
`drum_reusable (0.15, 1.0)` / `drum_recyclable (1.35, 1.0)` / `drum_trash (0.75, 1.6)` —
showcase positions around the second robot (left / right / behind).  Names are semantic for
readability; only positions matter to the policy (label → bin mapping is a deployment-time
wrapper concern).

### Verification (2026-07-03)

| Check | Result |
|---|---|
| `zero_agent` UR5e, 4 envs headless | **PASS** — obs `(4, 35)`, act `(4, 7)`, ≥45 s stepping, no traceback |
| `random_agent` Kinova, 4 envs headless | **PASS** — obs `(4, 38)`, act `(4, 8)` |

### Stub simplifications (by design, tracked in TODO)
- **Initial state is "shirt on the belt edge, ungrasped"** — the real task starts holding
  the stretched shirt (task-2 terminal bank).  The stub therefore contains an implicit
  pick sub-task that will disappear once bank initialization lands.
- Rewards are placeholder shaping; the graded one-shot `release_event`
  (centering × height quality) and the anti-hover fade must be ported from shirt_place,
  with dt-scaled weights (one-shots ~60× per-step shaping).
- The deterministic grasp's finger-tip offsets were calibrated on the tensegrity
  `tool_link_0` frame — UR/Kinova EE frames need calibration before the stub's implicit
  pick works reliably.
- Fixed three-drum layout; bin-layout randomization (so nearest ≠ correct generalizes)
  comes with the real reward work.
- Reach check pending: `drum_recyclable` sits 0.6 m lateral from the robot base — if
  placing proves kinematically tight, switch to a TossingBot-style conditioned throw.

---

## Phase 1: Grasped-Hang Init + shirt_place Reward Port (Alex agent)

**Date:** 2026-07-04 · **Platform:** Alex `rtxpro6k` (RTX PRO 6000, driver 610.43) ·
**Branch:** `project/shirt-distribute`

### Goal

Replace the flat-on-belt stub MDP with the real Task-3 start (shirt already in the
learning robot's own gripper at a plausible end-of-Task-2 pose), port the validated
shirt_place release-into-drum reward design goal-conditioned, and validate the MDP
with a scripted baseline before training.

### Initial-state design (documented per the prompt)

The episode starts with the shirt welded (slot 0) at one random particle patch to the
robot's own CLOSED gripper, arm in a sampled "holding" pose:

- **Holding-pose sweep at env init** (`_build_holding_pose_bank`): guided rejection
  sampling with in-sim FK — write sampled arm joints (+ closed gripper), step 2 sim
  steps, read the calibrated dynamic fingertip, accept if it lands in the env-local
  **tip box `x [0.10, 1.00], y [0.65, 1.30], z [1.20, 1.75]`** (presentation area
  `(0.15, 0.90, 1.60)` ↔ home region) with fingers pointing down
  (EE→tip world-z drop ≥ 0.10 m).  Explore rounds sample ±min(half-range, π) around
  the default pose; after ≥ 8 seeds, exploit rounds perturb accepted poses ±0.30 rad.
  Up to 200 rounds, target 256 poses.  Recorded tips are the true PD equilibria, so
  the hang anchor and the first-step fingertip agree.
- **Pose-dependent drape limit** (`_allowed_drape`): the hang below the tip must clear
  only what is actually below it — drum tops (0.88 m + 0.04) within 0.55 m xy of a
  drum centre, the belt (0.80 m) for tip y < 0.72, otherwise nothing (floor).  Poses
  admitting no bank state are rejected at sweep time.  The hanging bank's drape spans
  0.474–0.946 m (mean 0.736, 356 states) — a relaxed one-point hang of this garment
  is never shorter, so tips directly over a drum need z ≥ ~1.47.
- **Reset** (`_reset_cloth`): sample pose → `write_joint_state_to_sim` + PD targets →
  `_force_gripper_closed` → `_reset_cloth_hanging_from_bank(slot=0)` at the recorded
  fingertip, `max_drape` bucketed to `(0.55, 0.62, 0.72, 0.82, 0.95)` (the helper
  takes one bound per call).
- **PIPELINE SEAM:** replace sweep+restore with "sample (joint_pos, cloth_state) from
  the real Task-2 terminal bank" when it exists — the rest of the MDP is agnostic.

**Failed first attempt (kept for the record):** tip box z ∈ [1.55, 1.80] + a blunt
global "clear the drum tops" bound → the sweep found **0/960 poses**: fingers-down
puts the flange 0.10–0.24 m above the tip, i.e. at 1.65–2.04 m, at/beyond the UR5e
flange ceiling of ≈ 1.83 m (base 0.75 + shoulder 0.16 + 0.425 + 0.392 + wrist ≈ 0.1).
The acceptance volume sat at the workspace boundary.  Lesson: *derive tip boxes from
the arm's envelope, not from the retriever's presentation height* — task 2 hands over
at the SECOND robot's grasp point, which is necessarily lower.

### Bug found by reading, not running: stale-FK reset drag

`robot.data.body_pos_w` is NOT refreshed by `write_joint_state_to_sim` (lazy buffers
refresh on `scene.update` after a sim step), so on the reset step the base
`_update_grasp`'s `hold(tip)` would compute the fingertip from the PRE-reset pose and
drag the freshly restored hang there for one step (velocity spike through the PBD
weld).  Fixed by an `_update_grasp` override substituting the recorded reset anchor
for exactly one step (`_reset_anchor_pending`).

### Action-space change (measured/structural need, per §3 of the task charter)

`JointPositionActionCfg(use_default_offset=True)` commands `default + 0.5·action` —
from a sampled far-from-default holding pose a ZERO action yanks the arm home at full
PD speed on step 1, whipping the held cloth (and making "zero action" maximally
destructive for exploration).  Switched to **`RelativeJointPositionActionCfg`
(target = current + scale·action), scale 0.05** (≈ 3 rad/s cap at 60 Hz): the null
action holds the sampled pose — the correct semantics for a task that starts
mid-pipeline.  Gripper stays the binary term; note its convention: **action ≥ 0 ⇒
OPEN**, so a zero/random agent releases immediately — legitimate (release is the
final act), handled by the `bad_release` one-shot penalty instead of a grace hack.

### Rewards (shirt_place port, goal-conditioned; dt-scaled — one-shots ~60×)

| Term | Weight | Notes |
|---|---|---|
| `approach_bin` / `_fine` | 15 / 6 | grasp-point→commanded-bin XY tanh (std 1.0 / 0.2), grasp-gated, faded by `clearance_fraction` (anti-hover) |
| `clearance_over_bin` | 4 | lowest particle above commanded rim while gp over footprint |
| `release_hint` | 5 | per-step openness over the commanded bin |
| `release_event` | 240 | **one-shot graded** (centering × whole-shirt-lift, 0.25–1.0) ≈ ≤ 4 total |
| `bad_release` | −120 | **one-shot**: opened NOT over the goal (incl. instant t=0 drops) ≈ −2 |
| `in_target_bin` | 120 | per-step released-cloth fraction in the commanded drum (release-required) |
| `in_wrong_bin` | −60 | released fraction in any non-commanded drum (the sorting error) |
| `dropped_on_floor` | −5 | centroid < 0.70 m outside all drum footprints |
| `carry_time` | −1.5 | holding an undistributed shirt bleeds (anti-hover pressure) |
| `return_to_neutral` | 200 | post-success, gated `was_distributed` (shirt_place Bug-A lesson) |
| `action_rate` / `joint_vel` | −3e-4 → curriculum −3e-3 / −2e-3 @ 2000 steps | shirt_place profile |
| `belt_contact` | −10 | fingertip below belt |

**No settled-early termination**: shirt_place measured that terminating after the drop
CUT total return below hover-to-timeout and PPO drifted back to hovering; post-drop
per-step rewards make releasing strictly dominant instead.  Terminations: timeout 8 s,
joint-vel divergence, belt collision (0.12 m).

### Observations (42 dims, UR5e)

Stub set + `target_bin_rel_ee` (the carry is EE-centric), `shirt_lowest_rel`
(the drape length the clearance terms key on, camera-trivial from depth),
`was_distributed` (post-success phase flag for `return_to_neutral`).

### Metrics

`distribute_success_rate` (+ **per-bin** `distribute_success_bin{0,1,2}`),
`release_rate`, `max_target_fraction`, `grasp_rate` (base); exact per-episode
batch stash (`last_finished_batch`) for the eval script's per-bin accounting.

### Verification (job 3809918, RTX PRO 6000, 2026-07-04)

Scripts (usage headers in each): `scripts/model_validation/check_shirt_distribute_env.py`
(pose bank / reset integrity / 120-step hold / release / diversity),
`scripts/model_validation/baseline_shirt_distribute.py` (DLS-IK carry→release per drum
THROUGH the training action path), `scripts/skrl/evaluate_shirt_distribute.py`
(deterministic, per-bin).

**Env check (8 envs):**

| Check | Result |
|---|---|
| pose_bank | **FAIL first run** — 6 poses / 1600 samples (0.4 % blind acceptance); the exploit phase gated on 8 seeds never engaged → fixed: exploit from the FIRST seed, mixed 50/50 explore/exploit rounds (±0.25 rad), 300-round cap; re-check pending (job 3810091) |
| reset_attached | PASS 8/8 |
| reset_at_tip | PASS — grasp-point↔tip max 0.077 m |
| reset_drape_clear | PASS — drape ≤ pose's allowed (max 0.759 m vs allowed min 0.574), bottom min 0.523 m over free space |
| hold_120_steps | PASS — 0/8 drops, tip dist max 0.034 m, p95 particle speed 0.165 m/s |
| release_drops | PASS — 8/8 released and fell |
| reset_diversity | PASS — tip std (0.14, 0.15, 0.12) m, bins [8, 9, 15] |

**Zero/random agents (4 envs, 420 s each):** PASS — obs `(4, 42)`, act `(4, 7)`,
stepped to timeout, no traceback.  (They RELEASE immediately — binary gripper
action ≥ 0 ⇒ open — which is expected, and is why `bad_release` exists.)

**Scripted baseline (8 envs, 6-pose bank):**

| Commanded bin | Success | Fraction (mean/min) | Carry time (mean/max) | Held@arrival |
|---|---|---|---|---|
| 0 `drum_reusable` (0.15, 1.0) | **8/8** | 0.83 / 0.18 | 0.65 s / 0.82 s | 8/8 |
| 1 `drum_recyclable` (1.35, 1.0) | **5/8** | 0.54 / 0.00 | 0.43 s / 0.55 s | 8/8 |
| 2 `drum_trash` (0.75, 1.6) | **8/8** | 0.86 / 0.34 | 0.35 s / 0.45 s | 8/8 |
| **overall** | **21/24 (0.88)** | | | |

All three drums are reachable through the training action path (final xy err
≤ 0.06 m, grasp held everywhere) — **no TossingBot throw needed**.  Carry times
≤ 0.82 s ⇒ the 8 s episode has ample headroom.  The recyclable misses are
release-quality misses (fraction 0.00 — draped outside), not reachability:
exactly what the graded release event should teach the policy to avoid.

### Re-check + RTX PRO 6000 env-count benchmark (job 3810091, 2026-07-04)

**Fixed pose sweep: 7/7 PASS** — 256 poses (target met), tips
x [0.38, 0.98], y [0.65, 1.30], z [1.34, 1.66]; reset integrity, 120-step
hold (0/16 drops, p95 particle speed 0.088 m/s), release, diversity all PASS.
Note the presentation-side corner (x < 0.38) stays unpopulated: it lies within
0.55 m of `drum_reusable`, so it needs tip z ≥ 1.47 with fingers down —
reachable but rare; acceptable width for now (the Task-2 bank will define the
true distribution at the seam).

**Env-count benchmark (`bench_cloth_env_count.py --task Template-Shirt-Distribute-UR5e-F140-v0`),
RTX PRO 6000 (driver 610.43), 0 PhysX overflow warnings:**

| N envs | construct (s) | steps/s | env·steps/s | CUDA reserved (GB) | driver used (GB) |
|---|---|---|---|---|---|
| 32 | 19.8 | 9.79 | 313 | 0.17 | 4.87 |
| **64** | 44.0 | 5.58 | **357** | 0.23 | 6.04 |
| 128 | 101.5 | 1.37 | 175 | 0.33 | 8.31 |

The A6000 lesson holds on the RTX PRO 6000: throughput peaks at N = 64 and
FALLS at 128 (−51 %).  **Train at 64 envs.**

### Training run 1 (`sd_train1`, job 3810092, seed 42, 64 envs, 20 000 timesteps)

Completed in 1 h 07 (93 % GPU util, ≈ 5.1 steps/s — matches the benchmark).
**Metrics bug found:** all `Metrics/distribute_*` writes were wiped — they were
written BEFORE `super()._reset_idx()`, which rebuilds `extras["log"]`
(only the base class's `grasp_rate`, written after its super(), survived).
Fixed; run 1 is otherwise interpretable through the reward components:

| Episode_Reward (mean) | start | 25 % | 50 % | 75 % | end |
|---|---|---|---|---|---|
| `in_target_bin` (w 120) | 0.0 | 35.4 | 15.7 | 39.5 | 17.6 |
| `return_to_neutral` (w 200) | 0.0 | 58.4 | 24.0 | 51.3 | 37.2 |
| `release_event` (w 240, ≤ 4 max) | 0.0 | 0.30 | 0.10 | 0.10 | 0.08 |
| `bad_release` (w −120) | −0.01 | −0.29 | −0.32 | −0.11 | −0.19 |
| `in_wrong_bin` (w −60) | 0.0 | 0.0 | −0.52 | −0.03 | −2.06 |
| `grasp_rate` | 0.05 | 1.0 | 1.0 | 1.0 | 1.0 |
| Policy std | 0.61 | 0.59 | 0.58 | 0.58 | 0.58 |
| mean episode steps (of 480) | 192 | 457 | 322 | 315 | 319 |

Reading: correct-bin drops ARE learned (`in_target_bin` + `return_to_neutral`
positive and large), but release QUALITY is poor (`release_event` ≈ 2 % of
max — releases are rim-grazes, not centred cleared drops), wrong-bin landings
grow late, the policy std barely anneals (0.61 → 0.58 — the shirt_place
determinism watch-item), and mean episode length ~319/480 suggests a
termination firing early (diagnose: joint_vel vs belt_collision — termination
tags did not reach TB either).  Deterministic per-bin eval of run-1
checkpoints running (job 3810140); 2-seed retrain with fixed metrics queued
(`sd_train2`, job 3810141, seeds 1/2).

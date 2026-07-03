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

## Planned Phase 1: Bank Init + Release Reward + First Training

Per the staged plan ([doc/TODO.md](../TODO.md)):
1. Initialize grasped from the shirt_present terminal-state bank.
2. Port shirt_place's release design: graded one-shot bonus, anti-hover fade,
   terminate-shortly-after-drop; sparse landing-in-correct-bin bonus.
3. First PPO runs per robot variant; select checkpoints by **deterministic** eval
   (shirt_place lesson: training curves don't predict deterministic quality).
4. Bin-layout randomization; success target > 85 % correct-bin across labels/layouts.

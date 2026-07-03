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

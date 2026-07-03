# Shirt Pick Task — Optimization Tracking

**Task:** `Template-Shirt-Pick-Tensegrity-v0` (pipeline task 1; further retriever variants planned)
**Robot:** 5-DOF tensegrity manipulator + Robotiq 2F-140 gripper (hanging mount above the belt)
**Framework:** Isaac Lab + SKRL (PPO), Isaac Sim 5.1.0
**Hardware:** NVIDIA RTX A6000 (48 GB)

**Current state (2026-07-03):** runnable stub — env constructs, cloth + deterministic
highest-point grasp inherited fully functional from shirt_place, zero/random agents pass.
No training runs yet.  Pipeline context:
[RESEARCH_cloth_sorting_pipeline.md](RESEARCH_cloth_sorting_pipeline.md) ·
staged plan in [doc/TODO.md](../TODO.md) · task docs in the
[shirt_pick README](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/README.md).

---

## Phase 0: Task Stub, Central Scene & Shared Cloth Machinery

**Date:** 2026-07-03

### Summary

Created the shirt_pick task as the first of the three master-thesis pipeline tasks
(pick → present → distribute), robot-agnostic like the reach task.  The heavy machinery is
*not* new code: the generic cloth handling proven in shirt_place (deterministic
highest-point attachment grasp, kinematic Robotiq four-bar drive, centroid-proxy sync,
one-off pre-settle, flat-lay reset) was extracted into a shared base env that all three
pipeline tasks inherit.  A new central scene — imported by all three tasks so layout changes
propagate everywhere — rebuilds the hand-authored showcase stage
`res/Scenes/cloth_sorting_showcase.usd` (positions extracted from its USD xformOps).

### Changes

#### New shared infrastructure (used by all three pipeline tasks)
- **`shared/cloth_sorting_scene_cfg.py`** — central scene: retriever mount `(0.15, 0, 2.30)`,
  second-robot pedestal mount `(0.75, 1.0, 0.75)` (= the validated F140-reach
  workspace-analysis pose; UR5e yaw +90° per showcase), three sorting drums at
  `(0.15, 1.0)` / `(1.35, 1.0)` / `(0.75, 1.6)`, inspection-camera frame constants
  `(0.15, 2.0, 1.25)` looking −Y (no camera sensor spawned), presentation pose
  `(0.15, 0.50, 1.10)` — empirically reachability-checked, see Phase 1, cloth cfg + belt box collider + centroid proxy (shirt_place pattern),
  and `configure_cloth_sim()` (the validated 60 Hz / 24-PBD-iteration / large-GPU-buffer profile).
  The base scene's single `drum_target` is removed via the `None`-entity mechanism.
- **`shared/cloth_sorting_env.py`** — `ClothSortingEnvBase(ManagerBasedRLEnv)`: ClothObject
  lifecycle, pre-settle, proxy sync, gripper mimic drive, deterministic grasp
  (attach trigger 0.10 m / weld radius 0.07 m, unchanged from shirt_place), `was_grasped`
  latch + `Metrics/grasp_rate`.  Subclass hooks: `shirt_rest_xy`, `enable_hand_grasp`,
  `_reset_cloth`.
- **`shared/cloth_sorting_mdp.py`** — shared obs/reward functions (`shirt_rel_pos`,
  `shirt_velocity`, `grasp_active_obs`, `point_rel_shirt`, `shirt_lowest_point_rel_ee`,
  `target_bin_rel_shirt`, tanh shaping helpers).

#### Task files
- **`shirt_pick/shirt_pick_env_cfg.py`** — robot-agnostic cfg (`_set_robot_params` pattern):
  32-dim observations, stub rewards (reach 2.0 + to-presentation 4.0 + regularisers),
  timeout 6 s + joint-vel-divergence terminations, cloth prestartup events.
- **`shirt_pick/shirt_pick_env.py`** — `ShirtPickEnv(ClothSortingEnvBase)` + `present_rate`
  metric (grasp point within 0.15 m of the presentation pose while held).
- **`shirt_pick/config/tensegrity/`** — robot variant (mount/actions identical to the trained
  shirt_place tensegrity variant), gym registration, skrl PPO yaml (copied from shirt_place
  incl. the determinism profile: `entropy_loss_scale 0.0`, `initial_log_std −0.5`,
  `max_log_std 0.5`).

### Bugs found & fixed during bring-up
- **Observation probe vs. env attributes:** `ObservationManager._prepare_terms` calls every
  obs function during `load_managers()` — *before* the env subclass `__init__` creates
  `_cloth` / `_target_bin`.  Obs functions reading env attributes need a zeros fallback
  (same guard pattern shirt_place used for `was_placed_obs`).

### Verification (2026-07-03)

| Check | Result |
|---|---|
| `zero_agent` 4 envs headless | **PASS** — obs `(4, 32)`, act `(4, 6)`, ≥45 s stepping, no traceback |
| `random_agent` 4 envs headless | **PASS** |
| ClothObject | 4 envs × 11 048 particles acquired; pre-settle OK |

Ops notes from the verification session (recorded for future runs): python stdout is
block-buffered when redirected → healthy runs look stalled; use `PYTHONUNBUFFERED=1`.
Cold-cache cloth-task startup ≈ 10–15 min (warm ≈ 3 min).  Carb traps SIGTERM
(crash-handler prompt) → kill with `-9` / `timeout -s KILL`.

### Stub simplifications (by design, tracked in TODO)
- Shirt starts **flat** at a fixed pose (shirt_place first-iteration setup) — the crumpled
  drop-and-settle state bank is the report's key Task-1 item.
- `present_rate` is a distance latch, not yet a stable-hold criterion.
- Rewards are placeholder shaping; the anti-hover design (fade at goal, dt-scaled one-shots)
  from shirt_place is not yet ported.

---

## Phase 1: Stage-0 De-Risk, Crumpled Bank, Heuristic Baseline & Real MDP

**Date:** 2026-07-03

### Summary

Closed all Stage-0 physics prerequisites (deterministic cache-restore resets,
two-attachment stability, env-count benchmark — full findings in
[cloth_stage0_physics_derisk.md](cloth_stage0_physics_derisk.md)), generated
and wired the **crumpled-state bank**, ran the **scripted heuristic baseline**
(which caught an unreachable presentation pose before any GPU-hours were
spent on it), and replaced the stub MDP with the real sequential
pick-and-present reward structure.  Debug training (16 envs × 960 timesteps)
runs clean with all reward terms live.

### Crumpled-state bank (initial-state distribution)

- **Generator:** [scripts/generate_crumpled_bank.py](../../src/tensegrity_pick/scripts/generate_crumpled_bank.py)
  — SoftGym drop-and-settle (random pick, lift 0.15–0.45 m + up to 0.15 m lateral drag at
  0.5 m/s, release, settle ≤300 steps) in 64 parallel envs on the free upstream belt.
- **Bank:** 256 states → `res/Props/Cloth/banks/tshirt_crumpled_bank.pt` (65 MB, env-local
  xy-centred positions+velocities + protocol metadata incl. git rev). Pile heights ⌀0.07 m
  (max 0.13) vs ~0.02 m flat. Generation: 4.5 min.
- **Restore path:** `ClothSortingEnvBase._reset_cloth_from_bank` — per-reset random state +
  uniform yaw + 50 % x-mirror (reflections preserve PBD constraint distances) + ±3 cm XY
  jitter; writes pos+vel via the Stage-0-validated `write_nodal_state_to_sim` (1.2 ms).
  `bank_fraction` mixes flat lays as a curriculum knob (1.0 for now).
- **Verified:** zero agent in shirt_pick with the bank (repeated bank resets, no errors);
  the settle criterion (<0.01 m/s) never triggers due to persistent single-particle contact
  jitter (Stage-0 §1 finding) — the 300-step cap rules, some states carry ≤2 m/s single-particle
  residual velocity (physically consistent; restored exactly).

### Heuristic baseline & the unreachable-pose catch

[scripts/model_validation/baseline_shirt_pick.py](../../src/tensegrity_pick/scripts/model_validation/baseline_shirt_pick.py)
— P-servo pick (align over highest point → descend → close → lift → servo to pose → hold),
with auto-calibrated elbow→tip-x sign.

- **First run caught a reachability bug** (the shirt_place "gate every predicate on reachable
  geometry" lesson repeating): the original `PRESENTATION_POS (0.15, 0.45, 1.25)` was outside
  the tensegrity carry envelope (max grasp-point z observed ≈ 1.23; shirt_place memo ≈ 1.2).
  → Moved to **(0.15, 0.50, 1.10)** — also keeps the task-2 hanging shirt (grasp − ~0.35 m)
  clear of the belt footprint (collider y ≤ 0.45).
- **Baseline results** (8 envs, crumpled bank): ever-grasped **6/8**, presented **1/8**
  (best env: hold at 0.042 m, stable-hold fraction 1.00) → the pose is REACHABLE and the RL
  bar is 1/8. The 2/8 grasp misses are piles whose highest point the crude servo could not
  chase — exactly the alignment skill RL should learn (it observes `grasp_target_rel`).

### Real MDP (replacing the stub)

- **Rewards** ([shirt_pick/mdp/rewards.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/mdp/rewards.py)):
  reaching (tip→highest, 2.0) → grasp_hold (5.0) → lift (clamped 0–0.30 m, 8.0) →
  to_presentation (grasp-gated tanh, 15.0) → **presented** (held < 0.15 m and < 0.2 m/s,
  30.0/step — the success reward; no anti-hover fade needed because holding at the goal IS
  the task) → **drop** (one-shot −120 ≈ −2 dt-scaled) + belt_contact (−10) + curriculum-ramped
  regularisers (shirt_place profile, @2000 trainer steps).
  Grasp-gating the transport term closes the fling hack.
- **Env bookkeeping** ([shirt_pick_env.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/shirt_pick_env.py)):
  `drop_event` one-shot (consumed next step, like shirt_place's `release_event`),
  **stable-hold present latch** (30 consecutive steps < 0.15 m and < 0.2 m/s), `drop_rate`
  metric, and the `snapshot_terminal_states()` hook for the Task-1→2 bank.
- **Observations:** +`grasp_target_rel` (highest point relative to the dynamic finger tip —
  the exact attach-trigger geometry) → 35 dims.
- **Terminations:** + belt_collision (0.12 m, validated in shirt_place).
  Episode length 6 → **7 s** (serial baseline needs ~7.5 s; RL overlaps phases but needs the
  0.5 s hold on top).

### Debug bugs found & fixed
- `from . import rewards` in the task mdp package silently no-ops when the isaaclab star
  import already bound a `rewards` attribute — must use `from .rewards import *`
  (the shirt_place pattern). Cost one benchmark sweep.
- `env.reset()` / articulation state writes after inference-mode stepping need
  `torch.inference_mode()` wrappers (Isaac buffers become inference tensors).

### Verification
- Debug PPO run (16 envs, 960 timesteps, 61 s): all reward terms live —
  `grasp_rate` already 1.0 with an untrained policy (the deterministic attach fires),
  `drop_rate` 1.0 (expected untrained), `presented` 0, no NaN/crash.

---

## Phase 2: First Training Run — 85 % Deterministic Stable-Present

**Date:** 2026-07-03
**Run:** `logs/skrl/shirt_pick/2026-07-03_02-47-24_ppo_torch_seed1` — 128 envs
(Stage-0 benchmark optimum band), seed 1, 20 000 trainer timesteps (~4.8 h
wall; ~1 s/it with PPO updates — roughly half the zero-action benchmark
rate), skrl PPO with the shirt_place determinism profile (entropy 0,
initial_log_std −0.5, log_std ∈ [−3, 0.5]).  **All episodes start from the
crumpled bank** (bank_fraction 1.0) — no flat-lay curriculum was needed.

### Training curves

Figures: [shirt_pick/figures/tensegrity/](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/figures/tensegrity/)
(generated by [scripts/plot_shirt_pick_training_results.py](../../src/tensegrity_pick/scripts/plot_shirt_pick_training_results.py)).
Clean sequential skill acquisition: reach/grasp/lift/transport saturate by
~7.5 k; `grasp_rate` hits 1.0 by step 800; `drop_rate` collapses 0.94 → ~0.05
by 5 k (the one-shot penalty works); the `presented` success term keeps
climbing **linearly through 20 k** (0.03 → 14.8 episode reward) — the run was
still improving at the cap.

### The presented-metric recalibration (measured, not assumed)

Training-time `present_rate` stayed 0.000 all run.
[diag_shirt_pick_policy.py](../../src/tensegrity_pick/scripts/model_validation/diag_shirt_pick_policy.py)
showed the DETERMINISTIC policy parks the shirt **5–8 cm** from the pose with
the presented predicate true for ~60 % of all steps (min distances 0.4–4.7 cm)
— but the EE sways at ~0.18 m/s with jitter spikes, so the original latch
(30 CONSECUTIVE steps below 0.20 m/s) never fired: longest streaks 9–17.
The metric was miscalibrated for a PD arm's residual sway, not the policy.
→ Latch redefined as **windowed**: ≥ 80 % of the last 60 steps (1 s) with
dist < 0.15 m ∧ EE speed < 0.25 m/s ∧ grasped.  (The per-step `presented`
REWARD gates are unchanged — the policy was trained and paid correctly.)

### Deterministic checkpoint evaluation (mean actions, 96 episodes/eval, crumpled bank)

| Checkpoint | Seed | present_rate | grasp_rate | drop_rate | mean \|act\| |
|---|---|---|---|---|---|
| agent_16000 | 7 | 0.729 | 0.885 | 0.000 | 0.94 |
| **agent_20000** | 7 | **0.844** | 0.896 | 0.021 | 1.20 |
| **agent_20000** | 12 | **0.854** | 0.906 | 0.021 | 1.21 |

Script: [scripts/skrl/evaluate_shirt_pick.py](../../src/tensegrity_pick/scripts/skrl/evaluate_shirt_pick.py)
(episode-weighted metric accumulation; note: `torch.manual_seed` must be set
explicitly — `env_cfg.seed` alone did not differentiate rollouts).

**vs. baseline:** scripted P-servo presented 1/8 (0.125) → RL **0.85** on the
same crumpled distribution.  The residual failures are dominated by grasp
misses on hard piles (grasp_rate ≈ 0.90 caps present_rate); drops are rare
(2 %).  No determinism gap this time — the shirt_place PPO profile prevented
it from the start (stochastic-rollout metrics *under*-reported the policy).

### Fine-tune attempt — REJECTED by deterministic eval

`presented` was still climbing at the 20 k cap → continuation run from
agent_20000 (`2026-07-03_08-11-04_ppo_torch_seed2`, 128 envs, +6 000
timesteps via `--checkpoint … --max_iterations 125`; note `--max_iterations`
is multiplied by the 48-step rollout length — an unconverted value silently
schedules a 48× longer run).

| Checkpoint | present_rate | grasp_rate | drop_rate | present ∕ grasped |
|---|---|---|---|---|
| base agent_20000 | **0.844–0.854** | 0.896–0.906 | 0.021 | ≈ 0.94 |
| fine-tune agent_4000 | 0.792 | 0.833 | 0.021 | ≈ 0.95 |
| fine-tune agent_6000 | 0.802 | 0.812 | **0.000** | ≈ **0.99** |

The fine-tune perfected the hold (zero drops, 99 % of grasped episodes
present) but **lost grasp robustness** (0.90 → 0.81) — the dominant
`presented` gradient polished the post-grasp behaviour at the expense of the
grasp acquisition on hard crumple geometries.  Net present_rate went DOWN, so
per the selection rule the fine-tune is rejected.

**Selected checkpoint:**
`logs/skrl/shirt_pick/2026-07-03_02-47-24_ppo_torch_seed1/checkpoints/agent_20000.pt`
— deterministic present 0.85, grasp 0.90, drop 0.02.

**Path to > 90 %** (next iteration): the binding constraint is the grasp on
hard piles, not the hold.  Candidates: strengthen the pre-grasp alignment
signal (reach shaping weight or a dedicated xy-alignment term on
`grasp_target_rel`), re-grasp behaviour after a miss (the drop penalty
currently discourages retry-friendly exploration), or a mild bank curriculum
(`bank_fraction` < 1 early).

### Task-1 → Task-2 handoff

`ShirtPickEnv.snapshot_terminal_states()` captures per-env particle pos/vel +
slot-0 attach mask + grasp point + joint state + presented flag; generating
the shirt_present initial-state bank from the trained policy's presented
terminal states is the next pipeline step (see doc/TODO.md).

---

## Phase 3: GUI Review Fixes — Real Presentation Pose + Trash-Toss Bank

**Date:** 2026-07-03. GUI review of the Phase-2 policy raised two issues:
(1) the bank shirts were not crumpled enough for a trash conveyor, and
(2) the presentation pose was wrong — the workspace analysis from the Project
Thesis says **(0.15, 0.9, 1.6)** is feasible.

### Presentation pose: probe before believing "unreachable"

New script [probe_present_pose.py](../../src/tensegrity_pick/scripts/model_validation/probe_present_pose.py):
grasps the shirt (flat lay + alignment servo), then servos the grasp point to
a candidate pose with **UNCLIPPED actions** (only physical limits constrain)
using a numeric-Jacobian damped-least-squares controller, and reports
(a) hold distance, (b) joints at physical position limits, (c) peak applied
torque / effort-limit fraction under the shirt's load.

**Result for (0.15, 0.9, 1.6): REACHABLE — 8/8 envs, hold distance 0.000 m.**
Hold configuration: elbow +1.19 (of ±1.5), wrist_x +0.14, wrist_y +0.10,
base_y +0.19 (of ±0.5), base_z −0.04 — **no joint near a limit, well inside
the existing task action clips**.  Peak torque fractions with the shirt held:
wrist_x 0.78, elbow 0.22, base ≤ 0.15 — **no torque saturation**, so the
Klein effort limits were NOT raised and the action clips were NOT widened
(nothing to change beyond the pose constant in
`shared/cloth_sorting_scene_cfg.py`, where the probe findings are quoted).

Two probe lessons (documented for future reach questions):
1. The earlier "carry max ≈ 1.2 m / pose unreachable" conclusions were
   **artifacts of crude scripted probes**: a single hang-pose Jacobian goes
   stale as the arm rises and the servo stalls far from the target — the
   stall reads exactly like an envelope limit.  Probe verdicts need adaptive
   J re-estimation (or the workspace-analysis FK, as the user did).
2. A torque fraction of 1.00 at a joint sitting AT its position limit is the
   PD pressing into the hard stop, not actuator weakness — strength verdicts
   must come from limit-free configurations.

### Crumpled bank v2: trash-toss protocol

The v1 protocol (lift 0.15–0.45 m) only partially lifted the ~0.7 m garment →
near-flat piles (⌀0.07 m) that did not read as "trash tossed onto a belt".
v2 ([generate_crumpled_bank.py](../../src/tensegrity_pick/scripts/generate_crumpled_bank.py)):

- **Full lift-off tosses:** pick height 0.45–0.80 m at 1.2 m/s with lateral
  drag, released at full speed (the dangling cloth keeps momentum and wads on
  landing); 50 % of lays start half-folded (`reset_randomized` fold), 50 % of
  rounds get a second toss.
- **v2.0 mistake (caught by the new stats):** unconstrained 1.2 m/s tosses
  threw most shirts clear off the 0.9 m belt — mean pile TOP ended up below
  belt height (shirts on the floor), and some captures were mid-flight
  (max |v| 4.3 m/s).  → v2.1 biases drag direction toward the belt centre
  (±~60°), clamps release points to the belt interior, and **validates every
  state** (top above belt, nothing >0.10 m below belt level, max particle
  speed < 1 m/s) before capture.
- **Final bank (seed 4, 64 envs × 5 rounds, ~10 min):** **288 valid states**
  (~10 % rejected off-belt/unsettled), pile height **⌀0.10 m / max 0.15 m**
  (v1: 0.07/0.13), xy footprint shrunk to **0.71–0.77× the flat lay** — the
  footprint-vs-flat ratio was added as the honest "crumpledness" metric.

### Phase-3 retraining

From scratch (new initial-state distribution + new goal): 128 envs, seed 1,
20 000 timesteps.  Results appended below.

### Phase-3 run 1 (`2026-07-03_12-05-56_ppo_torch_seed1`, 20 k steps) — diagnosis

Training (with the user's Isaac Sim GUI session sharing the GPU for part of
the run — thermal throttling, ~2× slower iterations): grasp_rate 1.0 by
step 1 k, drop_rate 0.00 at the end (one transient exploration spike to 0.61
around 15 k, self-recovered), `presented` climbing but far from converged at
the cap (3.6 vs Phase-2's 14.8 episode reward).

Deterministic eval: agent_16000 present 0.000 / grasp 0.76; **agent_20000
present 0.37 / grasp 0.87 / drop 0.00** (2 seeds: 0.365/0.375) — steeply
improving at the cap, so the run was extended rather than re-rolled.

**Gate diagnosis** (diag_shirt_pick_policy.py): distances are solved — holds
at 0.06–0.12 m from the pose (dist-gate satisfied 68 % of ALL steps) — but
**EE speed at the raised posture is 0.24–0.27 m/s**, permanently above the
0.20 reward gate (satisfied only 16 % of steps): the elbow-extended lever
sways more than the old low posture (0.17–0.19 m/s).  The success reward was
barely harvestable → the last skill stalled.

**Fix (measured, physically honest):** the presented speed gate (reward AND
metric) now gates on the **cloth centroid speed** instead of the EE — the
hanging garment low-pass filters the arm sway to 0.06–0.20 m/s (satisfied
72 % of steps at gate 0.20), and the camera inspects the cloth, not the
gripper.  Windowed latch unchanged (≥ 80 % of the last 1 s).

### Phase-3 run 2 — continuation with the cloth-speed gate

From run-1 agent_20000: 128 envs, seed 3, +12 000 timesteps
(`--max_iterations 250`).  Results appended below.

### Phase-3 results — 100 % deterministic stable-present at the real pose

Continuation run `2026-07-03_19-33-17_ppo_torch_seed3` (+12 k timesteps from
run-1 agent_20000, cloth-speed gate active).  The unlocked success reward
consolidated the hold immediately.  Deterministic eval (96 episodes/eval,
trash-toss bank v2.1, pose (0.15, 0.9, 1.6)):

| Checkpoint | Seed | present_rate | grasp_rate | drop_rate |
|---|---|---|---|---|
| run-1 agent_20000 (EE gate) | 7/12 | 0.365 / 0.375 | 0.865 / 0.875 | 0.000 |
| continuation agent_8000 | 7/12 | 0.979 / 0.979 | 0.990 / 0.990 | 0.021 / 0.010 |
| **continuation agent_12000 (selected)** | 7/12 | **1.000 / 1.000** | **1.000 / 1.000** | 0.073 / 0.052 |

**Selected checkpoint:**
`logs/skrl/shirt_pick/2026-07-03_19-33-17_ppo_torch_seed3/checkpoints/agent_12000.pt`
— perfect stable-present and grasp on both eval seeds.  Trade-off note:
agent_12000 loses the grasp at some point in 5–7 % of episodes (after the
present latch); **agent_8000** is the low-drop alternative (1–2 % drops,
0.979 present) if the Task-1→2 handoff prefers held-at-end robustness over
the perfect present score — decide when the terminal-state bank is generated.

Figures: `shirt_pick/figures/tensegrity/` (run 1) and
`figures/tensegrity_continuation/` (continuation).

Phase-3 summary vs the user review: both issues closed —
(1) trash-toss bank v2.1 (piles ⌀0.10 m, footprint 0.72× flat, all states
validated on-belt), (2) presentation at the workspace-analysis pose
(0.15, 0.9, 1.6), probe-verified with **no effort-limit or action-clip
changes needed** (joints bind nowhere; peak torque fraction 0.78 wrist_x).

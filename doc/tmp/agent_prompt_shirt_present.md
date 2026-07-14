# Task: Implement, validate, debug and train `shirt_present` (cloth-sorting pipeline, Task 2) — Alex HPC agent

You are a coding agent with full access to this repo, running on the **Alex NHR HPC cluster**
(read `doc/alex_quickstart.md` first — Slurm-only GPU access, `tools/train_alex.sh` wrapper,
Isaac Sim/Lab + conda env `env_isaaclab` installed under `$HPCVAULT` via
`tools/install_IsaacLab_Alex.sh`). Your job is the **implement → validate (scripted baseline) →
train → analyze** loop for the `shirt_present` task until it satisfies §2, then hand back a clear
summary. Two sibling agents work in parallel on `shirt_distribute` (also on Alex) and on a
robot-free presentation-heuristic study (FAPS server) — **coordination rules in §8 are hard
requirements.**

---

## 1. The task

The shirt arrives **hanging at the presentation pose** `PRESENTATION_POS = (0.15, 0.90, 1.60)`,
pinned at **one random particle patch** (attachment **slot 1** — a static solver anchor standing
in for the retriever robot's grip; the passive `holder_robot` in the scene is visual only). The
learning arm must execute the **naive two-grasp presentation heuristic**:

1. **Second grab at the lowest hanging point** (`env.shirt_lowest_point_w` — the
   literature-standard target: Maitin-Shepard 2010, Doumanoglou 2014).
2. **Stretch** the garment between the two grasp points so front/back inspection cameras can
   assess it (camera at `INSPECTION_CAMERA_POS = (0.15, 2.0, 1.25)` looking along −Y; no camera
   sensor is spawned — observations must stay camera-derivable).

Part of your job is to **figure out the right success metrics, observations and cloth metrics**
(starting points in §4) — document your reasoning and the measured alternatives in the tracking
report.

## 2. Success criteria

- A deterministic (mean-action) checkpoint that from the random-particle hang reliably: reaches
  the lowest point → grasps (slot 0) → stretches taut → **holds the stretched, camera-facing,
  still presentation**, with a windowed success latch ≥ **0.9** across seeds/episodes.
- Success must be *honest*: silhouette-coverage-based (bunched/hidden cloth must not score), taut
  but not overstretched (stretch ratio ≈ 0.95–1.1; two-attachment stretch is validated stable
  through **1.15** — beyond that you are in untested territory), cloth still (centroid-speed
  gate, NOT EE speed — see §5).
- Zero/random agents keep working; a scripted baseline (§6.2) validates the MDP **before** any
  training run.
- Everything documented per §7.

## 3. Robot & action space (decided — deviate only with measured justification)

- **Robot: UR5e-F140** (`Template-Shirt-Present-UR5e-F140-v0`). Reasoning: best variant of the
  24-run reach comparison grid (1.8 cm / 96 % — `doc/reports/tracking/reach_optimization_tracking.md`,
  Iterations 14–15), and the showcase scene already mounts the UR5e at the second-robot pedestal
  `(0.75, 1.0, 0.75)` (yaw +90°, facing the belt) — the exact pose the reach grid used.
  Registered fallback if 6-DOF proves limiting around the hanging cloth: `Kinova-F140` (7-DOF
  redundancy; 2.4–3.7 cm / 87–90 % in reach).
- **Action space: joint-space** — the best reach action space (joint 3.1 cm / 91 % vs ik-rel
  3.4 / 88, ikabs 4.0 / 84, osc 3.3 / 90). The stub currently uses `JointPositionActionCfg`
  (delta from default, scale 0.5) — the pattern proven on the cloth tasks; the reach winner was
  `EMAJointPositionToLimitsActionCfg` (α = 0.2, `reach/config/f140_reach_common.py`). Start with
  the stub's action term; if approach precision or jerk is a problem, try the EMA variant.
  Document whichever you settle on and why. Fallback: Differential-IK relative
  (`ik_reach_common.py`) — close second in reach and natural for a precision regrasp.
- Gripper: identical Robotiq 2F-140 on all arms, kinematically driven by the base env; the grasp
  is the **deterministic attachment grasp** (binary gripper action + proximity trigger), not
  contact physics.

## 4. What is already built for you (shared infra — read before writing code)

Task package: `src/tensegrity_pick/source/.../tasks/manager_based/shirt_present/`
Shared: `.../tasks/manager_based/shared/` — **read-only for you** (§8).

- `shared/cloth_sorting_env.py::ClothSortingEnvBase` — cloth machinery proven in
  shirt_place/shirt_pick: PBD ClothObject (11 048 particles), pre-settle, proxy sync, Robotiq
  four-bar drive, deterministic attach/hold/detach in `_update_grasp` (trigger dist 0.10, weld
  radius 0.07, close/open threshold 0.40 on the finger command).
  - The grasp trigger targets **`shirt_grasp_point_w`** (property, default = highest point).
    **`ShirtPresentEnv` must override it to the lowest hanging point** and set
    `enable_hand_grasp = True` — that is the intended hook, do not copy `_update_grasp`.
  - `grasp_active` = slot 0 (the learning arm). The holder anchor is slot 1
    (`cloth.is_attached_slot(1)`); `ShirtPresentEnv._update_grasp` already re-pins it each step.
- **Hanging-state bank** (initial-state distribution, already wired): 356-state bank at
  `res/Props/Cloth/banks/tshirt_hanging_bank.pt` (regenerate on Alex with
  `scripts/asset_generation/generate_hanging_bank.py --headless --num_envs 32 --rounds 12` if the `.pt` is not in
  your checkout), restored per reset by
  `ClothSortingEnvBase._reset_cloth_hanging_from_bank` with yaw+mirror augmentation and a
  `max_drape` filter (`MAX_HANG_DRAPE` in `shirt_present_env.py` keeps long hangs out of the
  reusable drum that sits directly under the presentation pose — do not remove it without
  checking drum clearance).
- **Two attachment slots** (Stage-0 de-risk, `doc/reports/cloth_stage0_physics_derisk.md`):
  particle-exclusive capture, union ANCHOR masses, stretch stable through tautness ratio 1.15.
- **`shared/cloth_metrics.py`** (pure torch, unit-tested by
  `scripts/model_validation/test_cloth_metrics.py`): `silhouette_area` / `silhouette_coverage`
  (rasterized projection onto the camera plane, `view_axis=1`; folds count once — the honest
  "inspectable" score; reference area via `flat_silhouette_area(cloth.flat_rest_pos)`),
  `plane_extent` (smooth cheaper cousin for shaping), `stretch_ratio` / `rest_distance`
  (tautness from `cloth.flat_rest_pos`).
- `shared/cloth_sorting_mdp.py`: `shirt_lowest_point_rel_ee`, `grasp_active_obs`, etc. — note the
  zeros-fallback guard pattern (ObservationManager probes term shapes before the env subclass
  `__init__` finishes; every custom obs/reward reading env attributes needs it).
- Finger-tip frames are **calibrated for the UR/Kinova `robotiq_base_link`**
  (`gripper_cfg.EE_FRAME_TIP_SIGN`, probe: `scripts/model_validation/check_f140_finger_tip.py`)
  — `_finger_tip_pos`, belt penalties etc. are frame-correct as of 2026-07-04.
- Reference MDP to port from: `shirt_pick/` (windowed present latch, drop_event one-shot,
  metrics logging in `_reset_idx`, reward structure + curriculum) — its README and
  `doc/reports/tracking/shirt_pick_optimization_tracking.md` Phase 3 are the style/lesson reference.

## 5. Hard-won lessons (violate these and you will re-discover them expensively)

1. **dt-scaling:** Isaac Lab multiplies rewards by dt (1/60 s). Per-step weights earn
   ≈ weight × episode-seconds; one-shots earn weight/60 → size one-shot weights ~60× per-step.
2. **Speed gates on the CLOTH CENTROID, never the EE** — at raised postures residual PD sway is
   0.24–0.27 m/s (unharvestable, stalls learning at ~0.4 success); the hanging garment low-pass
   filters to 0.06–0.20. Cameras inspect the cloth anyway.
3. **Windowed success latch** (≥ 80 % of the last ~60 steps), never consecutive-step latches.
4. **Validate the MDP with a scripted baseline before training** (task 1's probe caught two
   unreachable "goal poses" that would have silently capped success). But: never trust a scripted
   probe's "unreachable" verdict from a stale-Jacobian servo — re-estimate J every ~200 steps.
5. **`mdp/__init__.py` must import local rewards via `from .rewards import *`** — a plain
   `from . import rewards` silently keeps isaaclab's own `rewards` module bound.
6. **Cloth env count:** throughput peaks at N = 32–64 (~400 env·steps/s on the A6000) and FALLS
   above; never train above 128 without re-benchmarking
   (`scripts/model_validation/bench_cloth_env_count.sh`) — re-run it once on the RTX PRO 6000
   and record the numbers in your tracking report.
7. **Deterministic eval:** mean actions; `torch.manual_seed` explicitly (env seed alone gives
   identical rollouts); episode-weighted metric accumulation — copy
   `scripts/skrl/evaluate_shirt_pick.py`. skrl `train.py --max_iterations` is ×48-rollout units.
8. Ops: `PYTHONUNBUFFERED=1` for redirected logs; carb traps SIGTERM → `kill -9`; wrap state
   writes after inference stepping in `torch.inference_mode()`; boot takes 1–2 min (cloth
   pre-settle); ignore the recurring USD `Sublayer ... has cycles` warning.

## 6. Suggested work plan

1. **MDP design.** Success predicate proposal (measure, then decide):
   `grasp_active(slot 0) ∧ holder still attached ∧ stretch_ratio ∈ [0.9, 1.1] ∧
   silhouette_coverage ≥ threshold ∧ cloth centroid speed < 0.20`, windowed latch.
   Calibrate the coverage threshold empirically: log the coverage distribution of (a) the raw
   hang (expect ~0.3–0.5) and (b) hand-scripted stretches (§6.2) before fixing it.
   Rewards (shirt_pick pattern): reach-lowest-point tanh → grasp_hold → stretch/coverage
   progress (grasp-gated) → presented per-step bonus → drop one-shot penalty (~60×). Consider
   `plane_extent` for smooth shaping where the rasterized coverage is too step-like.
   Observations: `shirt_lowest_point_rel_ee`, grasp states (both slots), stretch ratio, coverage,
   joint pos/vel, previous action. Everything camera-derivable in principle.
2. **Scripted baseline** (`scripts/model_validation/baseline_shirt_present.py`, modeled on
   `baseline_shirt_pick.py`): P-servo to the lowest point, close, pull to a taut pose. Purpose:
   (a) prove the second grasp + stretch is physically achievable from bank states, (b) measure
   achievable coverage/tautness → success thresholds, (c) catch kinematic-reach problems early.
3. **Zero/random agent checks** after every MDP change (PASS = "Gym observation space" printed +
   ≥45 s stepping, `scripts/agents/zero_agent.py --num_envs 4 --headless`; they loop forever — kill).
4. **Train** (`tools/train_alex.sh Template-Shirt-Present-UR5e-F140-v0 --headless --num_envs 64`,
   adjust after your env-count re-benchmark). Diagnose stalls with a gate-fraction diagnostic
   (copy `diag_shirt_pick_policy.py`) BEFORE touching weights blindly.
5. **Deterministic eval** across ≥ 2 seeds × ≥ 48 episodes; select the checkpoint; record the
   `snapshot_terminal_states`-style hook for the Task-2→3 terminal bank (mirror
   `ShirtPickEnv.snapshot_terminal_states`, including both grasp states).

## 7. Documentation & reporting (repo style — mandatory)

- **`doc/reports/tracking/shirt_present_optimization_tracking.md`** (stub exists): dated iteration
  entries — goal, changes, experiments (including the ones that LOST, kept for the record),
  results tables, root-cause analyses. Update as you go, not at the end.
- **`shirt_present/README.md`** (stub exists): task definition, MDP tables (obs/rewards/
  terminations with weights), decisions with one-line rationale, final results + figures
  (plot via a `scripts/plot_shirt_present_training_results.py` port of the shirt_pick plotter;
  figures under `shirt_present/figures/<variant>/`).
- **Revalidation scripts** with usage-comment headers in `scripts/model_validation/` for every
  claim you make (baseline, diagnostics, eval) — the repo standard is that findings are
  re-runnable.
- Tick/annotate your items in `doc/TODO.md` (Task 2 section).
- Cite measured numbers, never adjectives.

## 8. Coordination & constraints (hard rules)

- **Work on your own branch** `project/shirt-present` (from `project/tendonbot-sim-rl`); commit
  in logical increments and push regularly — Georg merges.
- **Do NOT edit `tasks/manager_based/shared/**`, `shirt_pick/**`, `shirt_distribute/**`,
  `reach/**` or other agents' files.** If a shared-code change seems unavoidable, implement the
  minimal override inside `shirt_present/` instead and flag it in your summary + tracking report.
- One Isaac Sim process per allocated GPU; all GPU work through Slurm (`train_alex.sh`, or
  `salloc` + the documented conda activation for interactive debugging).
- Training is authorized. Do not modify robot effort limits or cloth solver parameters without a
  measured justification documented in the config comment (upstream precedent:
  `PRESENTATION_POS` comment in `cloth_sorting_scene_cfg.py`).

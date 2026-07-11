# TODO — Tensegrity Pick & Place Project

Priority markers: 🔴 High · 🟡 Medium · 🟢 Low · ⬛ Blocked (requires Isaac Sim / Lab runtime)

---

## Physical Tendon Variant — Reach Rework (2026-07-09)  🔴

Fixes for the RL failure of the body-force (physical antiparallelogram) variant
(reach eval 2026-05-23: 31.4 % ± 14.4 % success; policies "hacked" the task by
tearing the loop-closure joint open and steering the dangling forearm with the
linear base).  Full change log:
[physical_variant_fix_report.md](reports/physical_variant_fix_report.md).

- ✅ ~~**Per-tendon cable saturation for the physical action**~~\
  **DONE (2026-07-09):** `PhysicalTendonEffortActionCfg.max_tension` defaults to the
  hardware-derived `[480, 480, 80, 80, 80]` N (`HW_TENDON_MAX_TENSIONS`); both tendon
  action terms now accept scalar or per-tendon lists (M2 mechanism).
- ✅ ~~**Cable min/max length between the attachment points**~~\
  **DONE (2026-07-09):** geometric range over the ±70° elbow workspace computed from the
  four-bar closure: **[0.0913, 0.2577] m** (0.1775 m at 0°). Wind-up stop (tension fades
  to zero at min length) + passive spring–damper stretch stop at max length in
  `PhysicalTendonEffortAction._apply_tendon_forces`.
- ✅ ~~**Linkage-breaking exploit**~~\
  **DONE (2026-07-09):** `linkage_closure_broken` termination — closure-anchor gap > 3 cm
  (elastic solver drift under full 480 N stays ≲1 cm at 120 Hz) OR parallelogram
  **branch-flip** detection (|elbow − (rod_L+rod_R)| > 0.3 rad; the flipped branch
  satisfies the closure with gap ≈ 0 but decouples the forearm). Verified: sustained
  480 N now parks the elbow at the ~70° stop with the closure intact (<1 mm).
- ✅ ~~**Sim-fidelity fixes found during verification**~~ **(2026-07-09):**
  physical variants now run **120 Hz physics** (decimation 4; closure drifts cm at 60 Hz),
  `enable_external_forces_every_iteration`, **PhysX sleep disabled** (body-force-driven arm
  freezes when asleep — tensor-API forces don't wake it), **50 ms tension lag** (motor/spool
  dynamics; kills constraint-impulse spikes), `joint_vel_diverged` 100→500 rad/s (1-gram
  wrist dummy link spiked past 100 on reset transients → silent truncation reset-loops),
  hierarchical elbow gravity comp recalibrated to the measured 5-DOF static balance
  (6.35 → ≈2.8 N·m).
- ✅ ~~**Lower-arm rotation measured at the right points**~~\
  **DONE (2026-07-09):** `mdp.lower_arm_angle` / `lower_arm_ang_vel` measure the forearm
  body twist about the upper-arm X axis (`compute_lower_arm_angle_and_rate`) — the elbow
  angle is the *sum* of two linkage joints, not any single joint. Raw linkage joint
  angles removed from the policy observation (base + wrist joints kept).
- ✅ ~~**Per-cable length/velocity in the policy observation**~~ (eval report §9 #1)\
  **DONE (2026-07-09):** `mdp.tendon_cable_lengths` / `tendon_cable_length_rates` read the
  cable state from the action term.
- ✅ ~~**Hierarchical controller variant**~~ (thesis §6 recommendation)\
  **DONE (2026-07-09):** `HierarchicalPhysicalTendonAction` — joint-space set-points
  tracked by the validated inner PID→tension loop (gains from the step-response study:
  elbow 75/6/3, wrist 10/1.5/0.6, gravity comp, block-wise tension distribution with
  null-space smoothing). Registered as `Template-Reach-Tensegrity-Physical-Hierarchical-v0`
  (+ `-Play-v0`).
- 🔴 **Retrain both physical reach variants on the CLUSTER** (5 seeds each, direct +
  hierarchical) + eval matrix incl. the new IK+PID heuristic baseline; compare against the
  31.4 % baseline and the 98 % sim-tendon reference. Everything is prepared:
  [run_physical_reach_pipeline.sh](../src/tensegrity_pick/scripts/training/run_physical_reach_pipeline.sh)
  (resumable train → eval → plots; heuristic included), local seeds discarded after the
  2026-07-10 controller/RL fixes (see tracking iteration 16b). Also fixed on the way:
  `evaluate_reach.py` checkpoint sort (lexicographic → numeric; the 2026-05 study actually
  scored agent_9600/agent_90000) and the eval inference-tensor crash.
- 🔴 **PhysX quasi-static breakaway on the four-bar** (~10–20 N·m; measured 2026-07-10,
  survives sleep-threshold zeroing, zero-g, closure removal — GPU-solver-level). Candidates:
  CPU-pipeline comparison, PhysX version sweep, NVIDIA report. `solver_dither` cfg knob is
  experimental. See the [fix report addendum](reports/physical_variant_fix_report.md).
- 🟡 Re-run the physical step-response validation with the 80 N wrist saturation
  (was 600 N in the validated study) — expect slower wrist rise times.
- 🟡 Sensitivity sweep: cable-stop stiffness/damping (5 kN/m / 100 N·s/m defaults) and
  `cable_slack_ramp` vs. training stability.

---

## Master-Thesis Goal Tasks — Cloth-Sorting Pipeline  🔴

Implementation of the three-task condition-sorting pipeline
(`shirt_pick` → `shirt_present` → `shirt_distribute`) per the research report
[RESEARCH_cloth_sorting_pipeline.md](reports/RESEARCH_cloth_sorting_pipeline.md).
Task stubs, the central scene and all robot-variant registrations were created 2026-07-03.


### Already in place — reuse, don't rebuild

- ✅ **Stage 0 physics de-risk (largely done via shirt_place):** PBD particle-cloth state I/O
  (`ClothPrim` view, `reset_randomized`, `write_nodal_pos_to_sim`), deterministic ANCHOR/WELD
  attachment grasp, and per-env cloth at 512 envs are validated —
  [shared/cloth_object.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_object.py),
  regression suite [test_shirt_fixes.py](../src/tensegrity_pick/scripts/model_validation/test_shirt_fixes.py) (5/5 PASS).
- ✅ **Highest-point pick** (report §2 — the canonical, depth-camera-trivial Task-1 grasp) is
  implemented and trained to 100 % deterministic success in
  [shirt_place](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_place/README.md);
  the generic machinery is now shared via
  [shared/cloth_sorting_env.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_sorting_env.py).
- ✅ **Robot-agnostic task pattern + action-space abstraction** (report "Robot-set considerations"):
  the reach task provides joint-pos / IK-Rel / IK-Abs / OSC variants for all comparison arms
  ([reach/config/](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/config/)),
  and the second-robot mount `(0.75, 1.0, 0.75)` is the validated workspace-analysis pose from
  [f140_reach_common.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/config/f140_reach_common.py).
- ✅ **Cloth sim profile + PPO determinism profile:** 60 Hz/decimation-1, 24 PBD iterations, GPU
  buffer sizing, and the entropy-0/log-std-capped skrl profile (fixes the deterministic-play gap) come
  from shirt_place — see
  [shirt_place_optimization_tracking.md](reports/shirt_place_optimization_tracking.md) and the
  reward **dt-scaling** lesson (one-shot weights need ~60× per-step weights).
- ✅ **Central scene** shared by all three tasks:
  [shared/cloth_sorting_scene_cfg.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_sorting_scene_cfg.py)
  — rebuilds [res/Scenes/cloth_sorting_showcase.usd](../res/Scenes/cloth_sorting_showcase.usd)
  (3 drums, pedestal + second-robot mount, inspection-camera frame constants, cloth + belt collider).
- ✅ **Task stubs (zero/random agents verified):**
  [shirt_pick](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/README.md)
  (`Template-Shirt-Pick-Tensegrity-v0`),
  [shirt_present](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_present/README.md)
  (`…-Present-Kinova-F140-v0`, `…-Present-UR5e-F140-v0`),
  [shirt_distribute](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_distribute/README.md)
  (`…-Distribute-Kinova-F140-v0`, `…-Distribute-UR5e-F140-v0`).

### Stage 0 — remaining physics de-risk  ✅ CLOSED (2026-07-03)

All three items validated — full findings, methodology and reproduction commands in
[cloth_stage0_physics_derisk.md](reports/cloth_stage0_physics_derisk.md).

- ✅ ~~**Benchmark max stable parallel cloth env count**~~\
  **DONE:** 32–256 envs all stable (no NaN, zero PhysX overflow warnings, ≤12 GB GPU).
  Throughput scales SUB-linearly: env·steps/s peaks at N=32–64 (~400–420) and *falls* to 223 at
  N=256 → train at **64–128 envs**, never more on this GPU.
  Script: [bench_cloth_env_count.sh](../src/tensegrity_pick/scripts/model_validation/bench_cloth_env_count.sh);
  data: [cloth_env_benchmark.csv](reports/data/cloth_env_benchmark.csv).
- ✅ ~~**Validate TWO simultaneous attachments**~~\
  **DONE:** `ClothObject` generalised to two attachment slots (slot 0 = hand, slot 1 = holder;
  particle-exclusive, union-composed ANCHOR masses; existing callers unchanged — shirt_place
  regression still 5/5). Stretch test stable through tautness ratio 1.15 (steady-state max
  particle speed ≤0.7 m/s, bbox +5 %, no NaN, both grasps intact) → **no handover fallback needed**.
  Script: [test_two_attachments.py](../src/tensegrity_pick/scripts/model_validation/test_two_attachments.py).
- ✅ ~~**Deterministic cache-restore reset check**~~\
  **DONE:** bit-exact (0.0 m cross-trial deviation over 120 steps), restored settled states stay
  settled (0.7 mm drift), restore costs 1.2 ms (≈4600× cheaper than settling). New
  `ClothObject.write_nodal_state_to_sim` writes pos+vel together (positions-only restores leave
  stale solver velocities).
  Script: [test_cache_restore_reset.py](../src/tensegrity_pick/scripts/model_validation/test_cache_restore_reset.py).

### Crumpled-state bank (Task-1 initial-state distribution)  ✅ CLOSED (2026-07-03)

- ✅ ~~**Drop-and-settle generator**~~\
  **DONE:** [scripts/asset_generation/generate_crumpled_bank.py](../src/tensegrity_pick/scripts/asset_generation/generate_crumpled_bank.py)
  — SoftGym protocol (random pick, lift 0.15–0.45 m + lateral drag, release, settle ≤300 steps)
  in parallel envs on the free upstream belt. Generated bank: **256 states**
  (64 envs × 4 rounds, 4.5 min) at `res/Props/Cloth/banks/tshirt_crumpled_bank.pt` (65 MB,
  env-local xy-centred pos+vel + full protocol metadata). Pile heights ⌀0.07 m (max 0.13) vs
  ~0.02 m flat — properly crumpled. Note: the <0.01 m/s SoftGym settle criterion never triggers
  (persistent single-particle contact jitter, see the Stage-0 report §1) — the 300-step cap rules.
- ✅ ~~**Bank augmentation (yaw + mirror)**~~\
  **DONE:** applied at load time in
  [cloth_sorting_env.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_sorting_env.py)
  `_reset_cloth_from_bank` (uniform yaw, 50 % x-mirror — reflections preserve PBD constraint
  distances — plus ±3 cm XY jitter); restores pos+vel via `write_nodal_state_to_sim`
  (the Stage-0-validated deterministic path). Verified with zero agent in shirt_pick
  (bank loads, repeated bank resets, no errors).
- 🟡 **Curriculum over the bank:** `bank_fraction` mixes flat lays with bank states per reset
  (mechanism implemented, `ShirtPickEnv.bank_fraction = 1.0` trains crumpled-only for now);
  stage by episode return if crumpled-only training stalls.

### Task 1 — shirt_pick (pick from belt → present to camera)

Progress log: [shirt_pick_optimization_tracking.md](reports/shirt_pick_optimization_tracking.md).

- ✅ Stub: robot-agnostic cfg + tensegrity variant
  ([shirt_pick_env.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/shirt_pick_env.py)).
- ✅ ~~**Scripted heuristic baseline before RL**~~\
  **DONE (2026-07-03):** [baseline_shirt_pick.py](../src/tensegrity_pick/scripts/model_validation/baseline_shirt_pick.py)
  — P-servo align→descend→close→lift→carry→hold. On the crumpled bank: ever-grasped 6/8,
  **presented 1/8** (the RL bar). Also served as the reachability gate: caught that
  `PRESENTATION_POS (0.15, 0.45, 1.25)` was outside the carry envelope → moved to
  **(0.15, 0.50, 1.10)** (and keeps the task-2 hang clear of the belt).
- ✅ ~~**Reward structure**~~\
  **DONE (2026-07-03):** sequential reach→grasp→lift→transport→**presented** (per-step success,
  no anti-hover fade needed — holding at the goal IS the task) + one-shot **drop** penalty
  (dt-scaled −120) + belt contact/collision + shirt_place curriculum profile. Transport is
  grasp-gated (closes the fling hack). See
  [shirt_pick/mdp/rewards.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/mdp/rewards.py).
- ✅ ~~**Stable-hold success criterion**~~\
  **DONE:** `present_rate` = grasp point < 0.15 m of the pose AND EE < 0.2 m/s for 30
  consecutive steps (0.5 s); `drop_rate` metric added.
- ✅ ~~**First training run + deterministic checkpoint eval**~~\
  **DONE (2026-07-03):** 128 envs, 20 k timesteps, crumpled-bank-only episodes.
  **Selected: `…02-47-24_ppo_torch_seed1/checkpoints/agent_20000.pt` — deterministic
  present_rate 0.85, grasp 0.90, drop 0.02** (96 eps × 2 seeds; baseline 0.125).
  The training-time present metric needed recalibration (windowed stable-hold latch — a
  consecutive-steps latch was too brittle for PD-arm sway); a +6 k fine-tune was evaluated
  and REJECTED (perfect holds but grasp 0.90→0.81). Full log:
  [shirt_pick tracking Phases 1–2](reports/shirt_pick_optimization_tracking.md).
- ✅ ~~**Push present_rate > 90 %**~~ / ~~GUI-review fixes~~\
  **DONE (2026-07-03, Phase 3):** after the user GUI review, the presentation pose moved to the
  workspace-analysis pose **(0.15, 0.9, 1.6)** (probe-verified reachable+strong enough — NO
  effort/clip bumps needed, see [probe_present_pose.py](../src/tensegrity_pick/scripts/model_validation/probe_present_pose.py))
  and the bank was regenerated with the **trash-toss** protocol (piles ⌀0.10 m, footprint 0.72×
  flat, on-belt-validated states). After re-gating `presented` on the **cloth centroid speed**
  (the raised posture's EE sway 0.24–0.27 m/s made an EE gate unharvestable; the hanging cloth
  filters to 0.06–0.20 m/s), the retrained policy reaches **deterministic present 1.000 / grasp
  1.000** (96 eps × 2 seeds). Selected:
  `logs/skrl/shirt_pick/2026-07-03_19-33-17_ppo_torch_seed3/checkpoints/agent_12000.pt`
  (low-drop alternative agent_8000: present 0.979, drop 0.01–0.02).
- 🟡 **Post-present grasp slips** (5–7 % of episodes lose the grasp after the present latch with
  the selected checkpoint) — decide agent_12000 vs agent_8000 when generating the Task-1→2
  terminal bank (held-at-end matters there), or add a small held-to-end bonus.
- 🟡 **Terminal-state bank generation** for the Task-1→2 handoff — the capture hook
  `ShirtPickEnv.snapshot_terminal_states()` (particle pos/vel + attach mask + joint state +
  presented flag) exists; needs a rollout script that runs the trained policy and saves
  presented-only states.
- 🟡 Camera-realistic observations: `grasp_target_rel` (highest point vs finger tip) is in;
  still open: N down-sampled surface points, cloth bbox. NO semantic keypoints on crumpled
  cloth (report §2: unreliable).
- 🟡 Retriever robot variants: UR5e / UR10 / Kinova (+ Frankenstein tensegrity-wrist) configs —
  copy the [config/tensegrity](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/config/tensegrity/) pattern with the per-robot mounts from
  [proj_base_scene_cfg.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/proj_base_scene_cfg.py);
  calibrate the finger-tip offsets for the UR/Kinova EE frames first.
- 🟡 Decide: conveyor moving or stopped during the pick (open question #7; belt drive is stubbed
  off — conveyor materials are visual stubs, see [project memory](reports/shirt_place_optimization_tracking.md)).

### Task 2 — shirt_present (bimanual stretch for inspection)  → 🤖 Alex agent ([prompt](agent_prompt_shirt_present.md))

- ✅ Stub: hanging-anchor cloth reset at `PRESENTATION_POS`, passive holder robot, lowest-point
  observation ([shirt_present_env.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_present/shirt_present_env.py)).
- ✅ **Random-particle hang init** (2026-07-04): hanging-state bank (356 states,
  [generate_hanging_bank.py](../src/tensegrity_pick/scripts/asset_generation/generate_hanging_bank.py)) restored
  per reset via `_reset_cloth_hanging_from_bank` (slot-1 anchor, yaw+mirror augmentation,
  `max_drape` filter keeps long hangs out of the reusable drum under the pose).
- ✅ **Shared visibility/stretch metrics** (2026-07-04):
  [shared/cloth_metrics.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_metrics.py)
  — rasterized `silhouette_coverage` in the camera plane, `stretch_ratio`/`rest_distance`
  tautness (offline-tested: `scripts/model_validation/test_cloth_metrics.py`).
- ✅ ~~**Naive lowest-point second grab + stretch MDP**~~ **DONE (2026-07-04, Alex agent,
  branch `project/shirt-present`):** slot-0 grasp via `shirt_grasp_point_w` override,
  windowed present latch, at-grasp-normalised geodesic tautness, EMA to-limits actions
  (measured: delta-from-default saturated).  Trained UR5e-F140 to deterministic
  present **0.927** on 2 eval seeds × 96 episodes (`agent_96000`, run 6) vs scripted
  baseline 0.125 — see
  [shirt_present_optimization_tracking.md](reports/shirt_present_optimization_tracking.md).
- ✅ ~~**Reward:** projected coverage from camera viewpoints, tautness bonus, centroid speed
  gate~~ **DONE:** silhouette coverage (camera XZ plane, gated on both grasps), maintain-taut
  band on the GEODESIC at-grasp-normalised ratio (flat-Euclidean over-reads wrap-around pairs
  1.3–1.6 — measured), overstretch moat from 1.05, drop one-shot −240; final coverage 0.68
  (above the ICRA 0.55–0.60 band).  NB: the RL run's coverage gate (0.50) is below the FAPS
  study's recommended 0.65 — a candidate re-threshold now that both are merged (`final_coverage`
  is logged, so re-scoreable).
- 🟡 **Hem-to-hem presentation — MERGED into `project/tendonbot-sim-rl` (2026-07-10) as the new
  hem↔hem baseline for continued RL exploration** (redesign 2026-07-08, Alex agent).  After Georg's
  visual inspection of `agent_96000`, replaced the naive lowest-point grasp with the study's
  **hem↔hem horizontal-pull** geometry + fixed the 6 findings.  The policy learns to grasp the
  accessible hem corner + pull a taut horizontal presentation, reaching **det. present 0.380 @
  coverage 0.637, grasp 0.73** (seed43 resume `agent_104000`, 2 eval seeds×96ep) — a first step,
  BELOW the naive lowest-point's **0.927 @ 0.68** but the chosen direction for the "perfect hem↔hem"
  policy.  The `shirt_present` env is now the hem↔hem geometry; the naive `agent_96000` (0.927) is
  superseded as a policy (its env changed) but its checkpoint is kept in `theses_logs/` for
  reference.  Best hem↔hem checkpoint in `need_visual_verification/` (play with the current env).
  **Next (open — better hem↔hem strategies):** the two limiters to beat are (1) the high 2nd-hem-corner
  grasp is RL-unlearnable (grasp_rate <0.1) — needs a curriculum or a different second-grasp target;
  (2) the random-holder→low-corner coverage (0.64) is below the naive stretch's (0.68) — needs a
  better holder/second-grasp PAIR (the study's side→hem_c / hem↔hem cells score 0.73–0.82).
  Retained lessons:
  tautness must use flat rest distance (not geodesic+r0); silhouette grid must be clamped at 512
  envs (or train at 64); and **never penalise "gripper closed while far" — it teaches the policy to
  never close the gripper (grasp_rate 0); the attach is target-tied so early-close is cosmetic**.
  Full arc + numbers: tracking report Phase 2 section.
- ✅ **Brute-force grasp-pair oracle + 2-grasp/3-grasp heuristic comparison** (2026-07-04,
  FAPS heuristic-study agent): [present_heuristics_study.md](reports/present_heuristics_study.md)
  — horizontal presentation stretch (2nd grasp raised to the holder's height, pulled along the
  camera-plane x axis; self-aligning, yaw gap 0.003). Naive lowest-point heuristic 0.679
  median coverage (beats free hang 0.567); regrasping HURTS (−0.031 paired → answer to open
  question #2: robot 1 keeps holding); 2 880-pair oracle map (top decile 0.842): chord LENGTH
  drives quality, hem-edge pairs win — **scripted hem_corner↔hem_corner reaches 0.820–0.832
  (p90 0.95)**, shoulder↔shoulder fails (0.493); oracle-guided 2nd grasp from arbitrary first
  grasp 0.713 ([policy JSON](reports/data/present_oracle_policy.json)). Recommended success
  threshold **0.65** (0.75 stretch goal); reward tautness gently up to 1.05.
  Scripts: `scripts/model_validation/present_heuristics/`.
- 🟡 Task-2→3 terminal bank: hook + validation script DONE
  (`snapshot_shirt_present_terminal.py`, 58-state presented-filtered sample with both grasp
  masks) — Task-3 agent regenerates at the size it needs from `agent_96000`.
- 🟡 Later: reset from the REAL Task-1 terminal-state bank
  (`ShirtPickEnv.snapshot_terminal_states` hook exists; decide agent_12000 vs agent_8000 —
  post-present grasp slips 5–7 % vs 1–2 %) — skill-chaining distribution shift is the report's
  #1 flagged risk.
- 🟡 Decide open question #2: does robot 1 keep holding during the whole stretch?

### Task 3 — shirt_distribute (label-conditioned bin placement)  → 🤖 Alex agent ([prompt](agent_prompt_shirt_distribute.md))

- ✅ Stub: goal-conditioned formulation — `target_bin_rel` observation, per-episode random bin,
  `distribute_success_rate` metric (release required, mirrors shirt_place)
  ([shirt_distribute_env.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_distribute/shirt_distribute_env.py)).
- ✅ Init infrastructure (2026-07-04): hanging bank restorable at the OWN finger tip (slot 0)
  + `_force_gripper_closed` reset helper — the "shirt already in the gripper" start.
- ✅ **Init grasped at sampled end-of-Task-2 poses** (2026-07-06): in-sim holding-pose sweep →
  cached pose bank (`res/Props/Cloth/banks/ur5e_f140_distribute_pose_bank.pt`) + fingertip
  hang restore (slot 0); `RelativeJointPositionAction` (zero action holds the sampled pose);
  Task-2 terminal-bank seam documented in `shirt_distribute_env.py`.
- ✅ **Reward** (2026-07-06): shirt_place release design retargeted to the commanded bin —
  graded one-shot `release_event`, anti-hover clearance fade, release-required `in_target_bin`,
  `bad_release` (discovery curriculum), wrong-bin penalty, `settle_after_success`, dt-scaled.
- ✅ **Scripted baseline validates the MDP** (2026-07-06): `baseline_shirt_distribute.py`
  places **21/24 = 0.88** across the three bins; all drums reachable through the training action
  path (no throw needed). Env-count benchmark: peak 357 env·steps/s @ 64 envs on RTX PRO 6000.
- ✅ **Goal-conditioned mode collapse SOLVED** (2026-07-10). The §2 blocker — every PPO seed
  hard-zeroing one of {recyclable, trash} (best 0.596 = 0.77/0.42/0.60) — was diagnosed by a
  [literature review](reports/RESEARCH_REPORT_goal_conditioned_mode_collapse) as cross-goal
  critic interference under an aggregate return normalizer, and fixed with the report's stacked
  levers (all in `shirt_distribute/learning/`): **per-goal value/advantage normalization +
  per-goal (multi-head) critic + goal one-hot** (`PerGoalPPO`/`GoalMultiHeadValue`), then **FiLM
  goal-gating + per-goal actor heads** (`FiLMGoalPolicy`, `…-PerGoal-FiLM-v0`). Monotonic
  **0.596 → 0.856 → 0.882**; collapse gone on all seeds.
- 🟡 **§2 met on 2 of 3 bins; bin 2 at a task ceiling.** Best checkpoint `seed1_filmU/agent_92000`
  (1080-ep eval): mean **0.882**, bins **0.910 / 0.899 / 0.832** — bins 0/1 clear ≥ 0.85 robustly,
  all three on eval seed 8. Holdout is **bin 2 (trash, behind the pedestal arm) at 0.83**, a
  single-drum reachability ceiling (persists across FiLM seeds + B5; scripted baseline itself
  0.88). Full record: [tracking Phases 2/2b/2c](reports/shirt_distribute_optimization_tracking.md).
- 🟢 Selected policy staged for playback:
  [logs/skrl/need_visual_verification/shirt_distribute/](../src/tensegrity_pick/logs/skrl/need_visual_verification/shirt_distribute/README.md).
- 🟡 **Lift bin 2 over 0.85 (task-side, not a learning lever):** TossingBot-style release-velocity
  conditioning for the far/behind drum, a bin-2-specific approach/release reward, or wider
  holding-pose coverage for reaching behind the base.
- 🟡 **Bin-layout randomization** so nearest ≠ correct generalizes (target >85 % correct-bin
  across labels and layouts).
- 🟢 Retire/merge the old [shirt_sort](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_sort/shirt_sort_env_cfg.py) template — superseded by shirt_distribute.

### Cross-cutting

- ✅ **Asymmetric actor–critic** (2026-07-12, INF) — skrl 2.1.0 **separates** policy vs critic
  observation groups: runtime probe
  [check_skrl_asymmetric_ac.py](../src/tensegrity_pick/scripts/model_validation/check_skrl_asymmetric_ac.py)
  proved the value net consumes the `"critic"` group (dim + content instrumented); config
  pattern + the `state_preprocessor` remap gotcha in
  [pipeline_infra_tracking.md](reports/pipeline_infra_tracking.md) §M1. No RSL-RL fallback needed.
- 🟡 **Camera-realistic observation library** in
  [shared/cloth_sorting_mdp.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_sorting_mdp.py):
  N-point surface sampling, keypoints + per-keypoint visibility flags, projected coverage.
- 🟡 **Task-space action variants** (IK-Rel/IK-Abs/OSC) for the cloth tasks — reuse
  [ik_reach_common.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/config/ik_reach_common.py) /
  [osc_reach_common.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/config/osc_reach_common.py)
  so one policy family swaps across arms.
- ✅ **Finger-tip offset calibration for UR/Kinova EE frames** (2026-07-04) — measured with
  [check_f140_finger_tip.py](../src/tensegrity_pick/scripts/model_validation/check_f140_finger_tip.py):
  `tool_link_0` and `robotiq_base_link` are co-located with antiparallel Z on every F140 arm →
  frame-aware sign table `gripper_cfg.EE_FRAME_TIP_SIGN`; `_finger_tip_pos`, `grasp_center_w`,
  belt penalty/termination helpers now resolve the frame per EE body (they were projecting
  22.5 cm to the WRONG side for the UR/Kinova arms).
- 🟢 **Stage 4 integration:** chain the three trained policies via the state banks, end-to-end
  metrics, robot-swap matrix; tiled-render cameras only after state-based policies work.
- 🟢 Register the three pipeline tasks + launch shortcuts in
  [src/tens_pick.sh](../src/tens_pick.sh) and the top-level README.

---

## Pipeline Stage 2 — Cooperative, Camera-Realistic Rework  🔴 (research report 2026-07-12)

Roadmap from [RESEARCH_REPORT_shirt_sorting_stage2.md](reports/RESEARCH_REPORT_shirt_sorting_stage2.md)
(design + citations) — execution plan, agent split, GPU schedule and gates in
[pipeline_stage2_execution_plan.md](pipeline_stage2_execution_plan.md).
Four parallel agents: **T1** = [pick grasp head](agent_prompt_pick_grasp_head.md),
**T2** = [present coop](agent_prompt_present_coop.md),
**T3** = [distribute polish](agent_prompt_distribute_polish.md),
**INF** = [pipeline infra](agent_prompt_pipeline_infra.md) (sole owner of `shared/**`).
Headline finding: the presentation ceiling is set by the GRASP PAIR — the first grasp is chosen
in Task 1 and robot 1's holding pose is the binding constraint, so the leverage is Task-1 grasp
choice + a cooperative Task 2, not more robot-2 training.

### S0 — Chaining seams & re-threshold (thesis-critical)  → 🤖 T1 + T2

- 🔴 **Task-1→2 terminal bank** [T1]: rollout script over the trained pick policy
  (`agent_12000`), snapshot at the present latch via the existing
  `ShirtPickEnv.snapshot_terminal_states` hook, **including the 5–7 % post-latch slip
  episodes** (they are the real distribution Task 2 must survive — report §3 Stage 0).
- 🔴 **Task-2 success gate 0.50 → 0.65 + tautness-pulse legitimacy** [T2]: raise the coverage
  gate to the study's threshold; don't penalize overshoot-then-relax trajectories (study §5.6:
  pulse halves settle time at no coverage cost). Retrain folds into S1 (one retrain, not two).
- 🟡 **Seam gate:** Task 2 retrained on the REAL terminal bank vs the synthetic hanging bank —
  fallback if the real bank destabilizes training: keep synthetic + inject the slip distribution.

### S1 — Task-2 hem↔hem curriculum (thesis-critical)  → 🤖 T2

- 🔴 **Hem-biased hanging-bank regeneration**: the current 356-state bank has only **19
  hem-corner-anchored states** (measured) — regenerate with region-biased anchor sampling
  (`generate_hanging_bank.py` + anchor-region option) so the curriculum has data.
- 🔴 **Target-conditioned grasp curriculum**: start with the accessible/low hem corner
  (bank-filtered starts), raise the target corner as grasp rate improves; reach-assist resets.
  *Success:* hem↔hem grasp rate > 0.6 (from < 0.1). *Abort < 0.3:* fall back to
  oracle/accessible-corner target and bank 0.679–0.713.

### S2 — Task-1 learned grasp head (thesis-critical)  → 🤖 T1

- 🔴 **Grasp-choice MDP**: 12 region-landmark keypoint observations (privileged reads of ONLY
  the 12 landmark particles — camera-contract-compatible) + border-distance feature; make the
  pick target policy-selectable (candidate set around the depth-trivial highest point);
  terminal bonus ∝ predicted downstream coverage of the achieved hold region (lookup table
  derived offline from the study's stratified pair map). *Gate:* realized Task-2 coverage from
  Task-1 holds > **0.713** (arbitrary-first-grasp oracle) within ~100 k steps × 3 seeds, else
  revert to highest-point + oracle-target Task 2 (report §2-Q1 decision gate).

### S3 — Cooperative Task 2 (thesis-critical → stretch)  → 🤖 T2, after G1 gate

- 🔴 **Movable scripted holder first**: robot 1 repositions its holding pose (base y/z +
  wrist; NO x translation — hardware constraint) toward study-informed poses; keep holding
  throughout (regrasp hurts with any target: −0.031/−0.051, study §5.7).
- 🔴 **Single joint policy over both arms** (FlingBot/SpeedFolding precedent): concatenated
  action space in the SAME manager-based env — no DirectMARLEnv rewrite. Robot-1 action =
  low-dim holding-pose adjustment. Shared presentation reward + light per-arm shaping;
  tautness capped at the validated 1.15. *Trigger:* grasp rate > 0.6 but coverage ceiling
  < 0.75 with fixed holder. *Success:* coverage ≥ 0.65 → 0.75 @ grasp ≥ 0.9.
  *Abort:* ship the single-agent oracle-target policy.
- 🟢 **MAPPO/IPPO ablation (S6, stretch)**: only on a measured coordination failure of the
  joint policy; known risk: skrl MARL breaks on gymnasium ≥ 1.0.0 (`env.state()`) — pin
  0.29.1; maturity unverified in-repo.

### S4 — Observation contract + asymmetric actor–critic (thesis-critical)  → 🤖 INF

- ✅ **Verify skrl 2.1.0 asymmetric AC** (2026-07-12, INF): **SEPARATED** — instrumented
  runtime probe on the reach task (injected `critic` group, constant-content check inside the
  value net's inputs) passed on every call; evidence + copyable env-cfg/agent-yaml pattern in
  [pipeline_infra_tracking.md](reports/pipeline_infra_tracking.md) §M1. RSL-RL fallback NOT
  needed (rsl_rl is not installed in `env_isaaclab`; revisit only if the pattern breaks).
- 🔴 **Camera-realistic observation library** (`shared/cloth_sorting_mdp.py`): 12 keypoints +
  per-keypoint visibility flags (noise model: ~1–2 cm Gaussian + dropout at 1−recall, Lips
  2024), coverage-from-mask (≡ silhouette coverage), noisy/delayed grasp-active flag
  (gripper-current proxy), N-point down-sampled surface cloud. **Tautness leaves the actor**
  (not directly observable in reality) → privileged critic only.
- 🟡 Per-task integration of the contract (T1/T2 own their cfg edits; INF delivers terms).

### S5 — Grasp-fidelity evaluation gates (thesis-critical, low cost)  → 🤖 INF

- 🔴 **Pre-condition gate** (approach alignment, finger clearance, single- vs multi-layer
  detection at the grasp point) + **stochastic misgrasp/slip gate** calibrated to published
  parallel-jaw cloth stats (DRAPER: 16–22 % misgrasp) — **evaluation-only, opt-in flags,
  default off** (training keeps the deterministic attachment; SoftGym-lineage precedent).
- 🔴 **Ranking-preservation study**: best checkpoints under idealized vs gated grasp; promote
  gates into training ONLY if success drops > 0.15 absolute AND checkpoint ranking reorders.

### Task 3 polish (independent)  → 🤖 T3

- 🔴 Lift **bin 2 ≥ 0.85** (release-velocity conditioning / bin-2 approach shaping / wider
  holding-pose coverage — existing item, now assigned).
- 🔴 **Bin-layout randomization** (nearest ≠ correct; target > 85 % across labels + layouts).
- 🟡 **Hold-presented-until-commanded gate** (preserve the inspected state until the drop
  decision — report §2-Q1).
- 🟡 Consume the REAL Task-2 terminal bank once S1/S3 produce it (seam gate as in S0).

### Stretch (post-gates)

- 🟢 **S7 teacher–student vision distillation** (DAgger point-cloud student; in-loop tiled
  rendering only if distillation loses > 0.1 coverage — throughput knee forbids it otherwise).
- 🟢 **S8 contact-physics pinch validation** (one-off PhysX adhesion+friction study,
  robot-free rig like the heuristics study; NOT a training change).

---

## Cloth Integration  ⬛ Requires Isaac Sim / Lab

These items cannot be done without a running Isaac Sim 5.1 instance.

### USD Asset Preparation

The first-attempt `tshirt_mod` / short-sleeve shells and their `make_pbd_cloth.py`
/ `make_xpbd_cloth.py` / `make_cloth_checked.py` authoring pipeline did not settle
or self-collide correctly and have been **retired** to
[res/Props/Cloth/Deprecated/](../res/Props/Cloth/Deprecated/). The active garment
is a flat-laid, welded **ClothesNet** shirt built from `TNSC_Tshirt_Ts1_0`.

- ✅ ~~**Validate active T-shirt mesh**~~\
  **DONE:** `tshirt_clothesnet.usd` (welded to a manifold mesh, ~11 k verts, mean edge ≈ 0.0098 m) passes readiness/schema checks. See [Cloth README](../res/Props/Cloth/README.md).
- ✅ ~~**Build the PBD particle-cloth asset**~~\
  **DONE:** `tshirt_clothesnet.usd` — referenced as `TSHIRT_USD_PATH` in `shirt_place_scene_cfg.py`. The decisive fix was keying the collision offsets to the mesh particle spacing (`particle_contact_offset = mean edge`, `solid_rest = ½ edge`).
- ✅ ~~**Build the XPBD surface-deformable asset**~~\
  **DONE:** `tshirt_clothesnet_xpbd.usd` (schema + material pre-authored offline, flat `restShapePoints`) — referenced as `TSHIRT_XPBD_USD_PATH`; needs `/physics/enableDeformableBeta` (set automatically by the startup event). Selectable via `SHIRT_CLOTH_BACKEND`.
- ✅ ~~**Verify both backends settle correctly**~~\
  **DONE:** PBD drapes into a recognizable t-shirt (~3–10 cm soft drape, on belt, bounded); XPBD settles to a flat ~1.4 cm lay. Validated standalone (DexGarmentLab-style settle) and in-task; renders in `./renders/`.
- 🟢 **Port the working welded/offset-aware build steps into `res/Props/Cloth/Scripts/`** — the scripts that produced the ClothesNet assets are not yet committed (only the validators remain in `Scripts/`); needed before re-authoring a different shirt from the candidate pool.

### ClothObject implemented ✅

- ✅ ~~**`ClothObjectCfg` + backend-aware `ClothObject`**~~\
  **DONE:** [shared/cloth_object.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_object.py) — `ClothBackend` enum (PBD/XPBD), `PBDClothParams` + `XPBDClothParams`, `ClothObjectCfg`. The runtime methods (`update`, `reset`, `reset_randomized`, `write_nodal_pos_to_sim`) are fully implemented (no more `NotImplementedError`).
- ✅ ~~**Spawn + ingest cloth per env**~~\
  **DONE:** `apply_cloth_startup_event` (prestartup) references the garment per env, bakes transforms into the mesh points, and authors the chosen backend (PBD particle system / XPBD surface deformable). `disable_complex_colliders_event` swaps the conveyor's triangle-mesh colliders for a box.
- ✅ ~~**Acquire backend tensor view + state I/O**~~\
  **DONE:** `__init__` creates a `ParticleClothView` (PBD) or `surface_deformable_body_view` (XPBD); `_get/_set_positions`/`_velocities` dispatch on backend. `nodal_pos_w`, `nodal_vel_w`, `centroid_pos_w` exposed.
- ✅ ~~**Reset / randomized lay-flat / write-back**~~\
  **DONE:** `reset` restores spawned default; `reset_randomized` lays the shirt flat at a random pose (+ optional half-fold); `write_nodal_pos_to_sim` injects arbitrary states (used by the validation scripts and curriculum resets).

### Scene Integration  (`TensegrityShirtPlaceEnv`)  — mostly done

- ✅ ~~**Wire `ClothObject` into the env**~~\
  **DONE:** the env instantiates `ClothObject`, calls `update()`/`reset_randomized()`, and keeps `shirt_proxy` synced to the cloth centroid each step (`_sync_proxy_to_cloth`).
- ✅ ~~**`replicate_physics=False`**~~ **DONE** in `ShirtPlaceSceneCfg` (per-env physics for cloth).
- 🟡 **Retire `shirt_proxy`** — reward/observation functions still read the kinematic proxy. Once cloth-native observations are added, point them at `cloth.centroid_pos_w` and remove the proxy `RigidObject`.

### MDP Rewards / Observations (`shirt_place/mdp/rewards.py`)

- 🔴 **Add cloth-specific observations** once `ClothObject` tensor API is live:
  - `shirt_keypoints(env, cloth)` → `(N, num_keypoints × 3)` keypoint positions relative to EE
  - `shirt_spread(env, cloth)` → scalar: max pairwise keypoint distance (measures how open/folded)
  - `shirt_centroid_vel(env, cloth)` → `(N, 3)` centroid velocity from nodal mean
- 🟡 **Tune cloth grasp reward** — adhesion-based grasping may produce different fingertip-proximity distributions than rigid cubes; re-tune `std` parameter in `shirt_grasp_reward`
- 🟡 **Add cloth deformation penalty** — penalize excessive stretching (`max(stretch_ratio) > threshold`) to prevent gripper from tearing the mesh

### Shirt Sort Task (`shirt_sort/`)

- 🔴 **Full implementation of `shirt_sort_env_cfg.py`** (currently template with empty obs/rewards/terminations — [shirt_sort_env_cfg.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_sort/shirt_sort_env_cfg.py))
- 🔴 **Cloth spawning pool** — `shirt_sort_scene_cfg.py` needs `ClothObjectCfg` entries for clean/dirty shirt variants
- 🔴 **Conveyor integration** — confirm cloth particles do not fall through active conveyor belt surface (conveyor belt is off in `shirt_place` but active in `shirt_sort`)

---

## Hardware Parameters

- ✅ ~~**Confirm spool radius**~~ (see [doc/open_questions.md](open_questions.md) §1)\
  **DONE:** All spools (elbow and wrist) have radius **5 mm**.\
  Derived: τ_nominal / F_max = 0.401 Nm / 80 N = 0.005 m.\
  Cross-references: `tendon_actuator.py:194`, `tendon_robot_cfg.py:52,60`
- ✅ ~~**Confirm wrist cable wrapping ratio**~~ (see [doc/open_questions.md](open_questions.md) §2)\
  **DONE:** Elbow uses 3:1 pulley system (Flaschenzug principle, Klein 2023 §3.2.5 p.46).\
  Wrist confirmed as direct-drive (no pulley, no MA).
- ✅ ~~**Verify continuous vs. peak motor torque**~~ ([doc/open_questions.md](open_questions.md) §3)\
  **DONE:** Continuous torque 0.401 Nm → 80 N per motor at spool. PSU peak 0.79 Nm → 158 N.\
  Klein (2023) §3.2.6 p.53. Elbow saturation 160 N (motor-side, ~2× continuous).\
  Wrist saturation 80 N (continuous limit). See [doc/Tensegrity_robot/hardware_parameters.md](Tensegrity_robot/hardware_parameters.md).
- ✅ ~~**Refine elbow `effort_limit`**~~ in `tendon_robot_cfg.py` — set to **35.0 N·m**\
  Derivation: 160 N (motor peak) × 3 (MA) × 0.0725 m (lever) = 34.8 N·m. Klein (2023) §3.2.6 p.53.
- ✅ ~~**Refine wrist `effort_limit`**~~ — set to **3.5 N·m**\
  Derivation: 80 N (continuous) × 0.020 m (max lever, Klein §3.2.2 p.42) × 2 (cable contributions) = 3.2 N·m,\
  rounded up with PSU headroom margin.
- ✅ ~~**Split `max_tension` per tendon group**~~\
  **DONE (2026-07-09):** `TendonEffortAction` and `PhysicalTendonEffortAction` both accept
  scalar or per-tendon `max_tension`; `HW_TENDON_MAX_TENSIONS = [480, 480, 80, 80, 80]` N
  is the default for the physical action. The sim-tendon task cfgs keep the scalar 500 N
  until their retrain (see Methodology M2 below) so existing checkpoints stay valid.

---

## RL Training

- 🟡 **Cube sort**: agent has not converged on the release phase — reward shaping needs tuning\
  Context: `reaching`/`transport` work but `place_success_rate` near zero ([cube_sort/README.md](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/README.md))
- ✅ ~~**Shirt place proxy validated**~~\
  **DONE (2026-04-04):** Both PD variant (obs=38, act=6) and physical tendon variant (obs=44, act=8) pass random agent verification (50 steps, 4 envs). Training verification with 32 envs/48-step rollout also passes. See [shirt_place tracking](reports/shirt_place_optimization_tracking.md).
- ✅ ~~**Shirt place**: run full baseline training~~\
  **DONE:** trained past the proxy stage to full PBD cloth with 100 % deterministic
  place success — see [shirt_place tracking](reports/shirt_place_optimization_tracking.md).
- ✅ ~~**Physical tendon**: validate sim fidelity (step response) before long runs~~\
  **DONE:** step-response validation ran via
  `scripts/model_validation/run_step_response_tendon.py --variant physical`
  (data: `src/tensegrity_pick/outputs/model_validation/tendon_physical/`, thesis appendix A.3);
  reach training ran 2026-05-22 (5 seeds, 31.4 % success —
  [reach_evaluation_findings.md](reports/reach_evaluation_findings.md)). Follow-ups live in
  the *Physical Tendon Variant — Reach Rework* section at the top.
- 🟢 **Hyperparameter sweep**: rollouts, learning rate, network depth for cloth task (higher observation complexity than cube tasks)
- 🟢 **Domain randomisation**: add cloth parameter randomisation (stiffness ±20%) and mass randomisation once cloth is working

---

## Tooling / Infrastructure

- 🟡 **Fix pre-commit `mypy` exclusion** ([.pre-commit-config.yaml:45](../src/tensegrity_pick/.pre-commit-config.yaml)) — PyTorch `Tensor | dict` type alias triggers false positive; suppress properly rather than excluding the whole rule
- 🟡 **Fix pre-commit VPN stall** ([.pre-commit-config.yaml:54](../src/tensegrity_pick/.pre-commit-config.yaml)) — `pre-commit` network hook intermittently hangs under VPN; investigate mirror or offline mode
- 🟢 **Register `shirt_place` in top-level `README.md`** — still shows "template — not yet implemented"
- 🟢 **Register `shirt_sort` in top-level `README.md`** — same
- ✅ ~~**Add `shirt_place` training launch shortcut** to [src/tens_pick.sh](../src/tens_pick.sh)~~\
  **DONE (2026-04-04):** Added zero/random/train/play commands for both PD and physical tendon shirt_place variants.

---

## Tests

- ✅ ~~**Add test for `ClothObject` stub interface**~~\
  **OBSOLETE:** `ClothObject` is fully implemented (no `NotImplementedError` paths remain);
  the runtime behaviour is covered by the shirt regression suite
  ([test_shirt_fixes.py](../src/tensegrity_pick/scripts/model_validation/test_shirt_fixes.py), 5/5 PASS).
- ✅ ~~**Integration test for `TensegrityShirtPlaceEnv` construction**~~\
  **DONE (2026-04-04):** Manually verified via headless random agent (4 envs, 50 steps) and training rollout verification (32 envs, 48 steps). Both PD and physical tendon variants pass. Obs space shape confirmed (38 PD, 44 tendon), no MISSING fields at init.
- 🟢 **Add test for cloth scripts** (`check_cloth_readiness.py`) — exercise `validate_cloth_mesh()` on a trivial synthetic USD mesh without Isaac Sim

---

## Documentation

- ✅ ~~**Cloth directory README**~~\
  **DONE (2026-04-04):** Created `res/Props/Cloth/README.md` documenting meshes, scripts, workflow (validate → apply → verify), and simulation backend comparison.
- 🟢 **`shirts_sort/README.md`** — still template; fill observations, rewards, and variants tables once task is implemented
- 🟢 **`doc/tendon_simulation.md`** — add section on cloth co-simulation constraints (PBD particle cloth & rigid-body gripper contact model)
- 🟢 **`doc/workflow_guide.md`** — add "Cloth object workflow" section: validate mesh (`check_cloth_readiness.py`) → build welded ClothesNet asset → verify both backends in Sim (`run_shirt_validation.py`) → train with proxy-synced `ClothObject`

---

## Methodology Re-validation  🟡 Project Thesis follow-ups

Action items raised by the systematic methodology review of the Project Thesis (`reports/methodology_review/`). These translate documentation fixes already applied to the chapters into matching code/training changes.

### Per-tendon cable saturation (M2)

- ✅ ~~**Switch tendon actions to per-tendon `max_tension`**~~\
  **DONE (2026-07-09):** `TendonEffortActionCfg` / `PhysicalTendonEffortActionCfg` accept
  `float | list[float]`; hardware list `HW_TENDON_MAX_TENSIONS = [480, 480, 80, 80, 80]` N
  exported from [robots/tendon_actuator.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/tendon_actuator.py)
  and default for the *physical* action.
- 🔴 **Wire `HW_TENDON_MAX_TENSIONS` into the sim-tendon task cfgs** (reach/cube/shirt
  `TendonEffortActionCfg` instances still use the scalar 500 N default) and verify
  Klein-style scaling in the step-response run (wrist saturation column should drop
  from `600 N` to `80 N`). Do this together with the retrain below — it invalidates
  existing sim-tendon checkpoints.
- 🟡 **Retrain tendon variants after the saturation update**\
  - `Template-Tensegrity-Reach-Tendon-v0` (PPO, 2 048 envs, headless)\
  - `Template-Tensegrity-Cube-Place-Tendon-v0` (PPO, 2 048 envs, headless)\
  Compare reward curves against the existing 500 N baselines and update the §5 results tables once converged.
- 🟡 **Re-run step-response validation with the new saturations** ([test/test_step_response.py](../test/test_step_response.py))\
  Regenerate the per-axis CSVs and the aggregate metrics that feed appendix A.7.

### Wrist effort limit (M1)

- ✅ ~~**Code already at 3.5 Nm wrist effort**~~\
  Verified in `tensegrity_robot_cfg.py`; chapter §4.2/§4.4 prose, table, and equations now match (3.5 Nm with $\approx 10\%$ controller headroom).
- 🟢 **Audit §5 results tables** for any remaining `3.0 Nm` references and replace with `3.5 Nm` when the next results revision is written.

### Documentation review report

- 🟡 **Close out remaining Major/Minor issues**\
  See [`reports/methodology_review/reports/00_chapter_scope_structure.md`](../reports/methodology_review/reports/00_chapter_scope_structure.md) — Blocker (B1–B3) and Major (M1–M9) items are now fixed in the chapters; Minor (m1–m11) and Cross-section (X1–X12) items are still open.

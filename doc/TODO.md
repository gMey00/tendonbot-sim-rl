# TODO — Tensegrity Pick & Place Project

Priority markers: 🔴 High · 🟡 Medium · 🟢 Low · ⬛ Blocked (requires Isaac Sim / Lab runtime)

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
- 🔴 **Init grasped at sampled end-of-Task-2 poses** (sampled joint configs + fingertip hang
  restore; later replaced by the real Task-2 terminal bank) — *delegated: Alex agent, UR5e-F140,
  joint actions; recipe in the prompt.*
- 🔴 **Reward:** sparse landing-in-correct-bin + graded release event — port the shirt_place
  `release_event` quality grading and dt-scaling weights directly — *delegated (same agent)*.
- 🟡 **Bin-layout randomization** so nearest ≠ correct generalizes (target >85 % correct-bin
  across labels and layouts).
- 🟡 TossingBot-style throw (release-velocity conditioning) if bins prove outside comfortable reach.
- 🟢 Retire/merge the old [shirt_sort](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_sort/shirt_sort_env_cfg.py) template — superseded by shirt_distribute.

### Cross-cutting

- 🔴 **Asymmetric actor–critic:** verify installed skrl 2.1.0 actually separates policy vs critic
  observation groups (`"critic"` obs group → `num_states`); if aliased, switch to RSL-RL for the
  pipeline tasks (report §6). Privileged critic obs: full particle field, true attachment state.
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
- 🟡 **Split `max_tension` per tendon group** — currently a single 500 N for all 5 tendons.\
  Elbow needs ~480 N (after 3:1 MA), wrist needs ~80 N (no MA). A per-group\
  max_tension would improve wrist action-space resolution. Requires refactoring\
  `TendonEffortAction` and `PhysicalTendonEffortAction`.

---

## RL Training

- 🟡 **Cube sort**: agent has not converged on the release phase — reward shaping needs tuning\
  Context: `reaching`/`transport` work but `place_success_rate` near zero ([cube_sort/README.md](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/README.md))
- ✅ ~~**Shirt place proxy validated**~~\
  **DONE (2026-04-04):** Both PD variant (obs=38, act=6) and physical tendon variant (obs=44, act=8) pass random agent verification (50 steps, 4 envs). Training verification with 32 envs/48-step rollout also passes. See [shirt_place tracking](reports/shirt_place_optimization_tracking.md).
- 🟡 **Shirt place**: run full baseline training with proxy cube to validate reward pipeline produces learning signal
- 🟡 **Physical tendon**: no training runs yet; validate sim fidelity (step response) before long runs
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

- 🟡 **Add test for `ClothObject` stub interface** — verify `NotImplementedError` is raised correctly and `ClothObjectCfg` fields are valid
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

- 🔴 **Switch `TendonActuatorCfg` to per-tendon `max_tension`** ([src/tensegrity_pick/.../tendon_actuator.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/tendon_actuator.py))\
  Replace the scalar `max_tension: float = 500.0` with a per-tendon list (`[480, 480, 80, 80, 80]` N) so the elbow and wrist tendons saturate at their Klein-derived hardware limits (160 N motor-side $\times$ 3:1 elbow pulley, 80 N for the wrist cables).
- 🔴 **Update `tensegrity_robot_cfg.py` actuator definition** ([src/tensegrity_pick/.../tensegrity_robot_cfg.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/tensegrity_robot_cfg.py))\
  Wire the new per-tendon saturation list into the `IdealPDActuator`-based tendon actuator group. Verify Klein-style scaling in `step_response_test.py` (cable saturation column should drop from `600 N` to `80 N` for wrist tests).
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

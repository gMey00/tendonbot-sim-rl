# Shirt Place Task — Optimization Tracking

**Task:** `Template-Tensegrity-Shirt-Place-v0` / `Template-Tensegrity-Shirt-Place-Physical-Tendon-v0`
**Robot:** 5-DOF tensegrity manipulator + Robotiq 2F-140 gripper (ceiling-mounted)
**Framework:** Isaac Lab + SKRL (PPO), Isaac Sim 5.1.0
**Hardware:** NVIDIA RTX A6000 (48 GB)

**Final result (2026-07-02):** deterministic policy places the shirt in
**100 %** of evaluation episodes with **~93 %** of cloth particles inside the
drum and returns the arm toward neutral.
Checkpoint: `logs/skrl/shirt_place/2026-07-02_15-10-58_ppo_torch_seed1/checkpoints/agent_4000.pt`.

---

## Phase 0: Infrastructure & Proxy-Based Environment

**Date:** 2026-04-04

### Summary

Implemented the shirt_place task as a proxy-based environment where a rigid blue cube (5 cm, 0.20 kg) stands in for the cloth T-shirt. This allows full reward pipeline development and training validation before cloth simulation is enabled. Created the `ClothObject` infrastructure (stub implementation) for future cloth integration.

### Changes

#### New files
- **`shared/cloth_object.py`**: `ClothBackend` enum (PBD/XPBD), `PBDClothParams` and `XPBDClothParams` dataclasses with cotton T-shirt defaults, `ClothObjectCfg` for prim/mesh/backend configuration, and `ClothObject` stub class (all runtime methods raise `NotImplementedError`).
- **`res/Props/Cloth/README.md`**: Documentation for cloth meshes, scripts, workflow (validate → apply → verify), and simulation backend comparison.

#### Bug fixes
- **`shirt_place_scene_cfg.py`**: Fixed `TSHIRT_MESH_PRIM` from `/World/tshirt_mod/mesh` to `/root/World/mesh/Mesh` (verified via `check_cloth_readiness.py`). Fixed `shirt_proxy` prim_path from `f"{ENV_NS}ShirtProxy"` to `"{ENV_REGEX_NS}/ShirtProxy"` (Isaac Lab convention). Removed unused `ENV_NS` variable.
- **`shirt_place/config/tensegrity_tendon/joint_pos_env_cfg_physical.py`**: Fixed `PhysicalTendonEffortActionCfg` keyword arguments from `root_offsets`/`forearm_offsets` to `elbow_tendon_root_offsets`/`elbow_tendon_forearm_offsets` (matching actual dataclass fields).
- **`cube_place/config/tensegrity_tendon/joint_pos_env_cfg_physical.py`**: Same `PhysicalTendonEffortActionCfg` keyword fix (pre-existing bug in cube_place too).

#### Updated files
- **`shared/__init__.py`**: Added exports for `ClothBackend`, `ClothObject`, `ClothObjectCfg`, `PBDClothParams`, `XPBDClothParams`.

### Verification

- PD variant random agent: **PASS** (50 steps, 4 envs); training rollout: **PASS**.
- Physical-tendon variant random agent: **PASS**.
- `check_cloth_readiness.py`: both T-shirt meshes **PASS**.

---

## Phase 1: PBD Cloth Integration, Deterministic Grasp & First Trainable Policy

**Date:** ~2026-06 → 2026-07-01 (commits `668ebef`, `4e2525e`)

### Summary

Replaced the rigid proxy with a real PBD particle-cloth T-shirt, solved the
cloth-stability and cloth-grasping problems, redesigned the rewards around the
release requirement, and trained a first working policy (~89 % place success —
*stochastic actions only*, see Phase 2).

### Cloth physics (the hard part)

- **New asset**: flat-laid, welded, manifold ClothesNet shirt
  (`tshirt_clothesnet.usd`, ~11 048 particles, mean edge ≈ 0.0098 m) replacing
  the inflated `tshirt_mod` shells.
- **The decisive stability fix**: collision offsets must track particle
  spacing (`particle_contact_offset = mean edge`, `solid_rest_offset = ½ edge`).
  Larger offsets put spring-connected neighbours inside each other's contact
  radius → energy pumping → divergence / self-collision explosions.
- Aerodynamic drag/lift **zeroed** (cloth floated like foil), adhesion
  **zeroed** (cloth glued to belt and drum rim), mass 0.30 kg, damping 0.5,
  24 solver iterations, self-collision on, CCD off.
- One-off **pre-settle** at env init: the raw mesh is relaxed into a flat
  sheet on the free upstream belt and adopted as the per-reset rest shape.
- The conveyor USD's triangle-mesh/convex colliders are **disabled** (PBD
  cloth tunnels through them) and replaced by an invisible static box with its
  top face at belt height (`conveyor_collider`).
- 60 Hz physics / decimation 1 (halved cloth cost vs 120 Hz/dec-2).

### Deterministic grasp (friction cannot lift PBD cloth — validated)

GarmentLab/DexGarmentLab-style attachment, batched in
`ClothObject.attach/hold/detach`: when the gripper is commanded closed with
the dynamic fingertip within 0.10 m of the cloth's highest region, the
particle cluster within 0.07 m of the tip is grabbed and its offsets recentred
onto the tip; held each step; released on open.  Two switchable modes —
`WELD` (teleport, stretch ratio 1.03) and **`ANCHOR`** (PBD solver anchors via
inverse-mass ≈ 0, stretch ratio 1.00, active).  A literal runtime
`PhysxPhysicsAttachment` is **not viable** (GPU physics baked at
`sim.reset()`; verified empirically).  The Robotiq linkage is driven
kinematically (`_drive_gripper`) because the native PD drives fight the
four-bar linkage.

### Reward design battles (anti-hover)

- Robot **dragged** the shirt to the drum → transport gated on carry height
  (`lift_threshold` 0.02 → 0.25 m).
- Policy **dipped the held shirt** into the drum to farm the fraction metric →
  success (`shirt_in_target`) requires the cloth to be **released**.
- **Hover traps**: (a) terminate-after-place backfired (cut future return
  below hovering-to-timeout) — removed; (b) clearance/goal-tracking became
  hover attractors → positioning shaping faded by `1 − clearance_fraction`,
  one-shot `release_event` bonus added, `carry_time` penalty added.
- Success measured as **particle-fraction-in-drum** (centroid-in-drum lies for
  a draped garment).

### Verification tooling

`scripts/model_validation/test_shirt_fixes.py` — scripted grasp→lift→carry→
release cycle asserting 5 fixes (stretch ratio, released fall, linkage
coherence, tip-based grasp, small weld cluster).

### Result

Run `2026-07-01_20-36-06_ppo_torch_seed1`, `agent_6000`: grasp 1.0, place
~0.89, drum-fraction ~0.5 — **with stochastic actions**.  Post-peak regression
after iter ~6 000.  Two open bugs: arm does not return to start after
placement (Bug A); gripper appears to rest inside the conveyor (Bug B).

---

## Phase 2: Determinism, Bug Fixes & Reward Accounting Rework

**Date:** 2026-07-01 → 2026-07-02

### Problem diagnosis (measured, not assumed)

1. **Determinism gap — the headline defect.**  The deployed policy plays the
   deterministic mean action, but `agent_6000` scored **0 %** deterministically
   (mean parks the tip at the default pose, never descends) vs ~100 %
   stochastically.  Root cause: `entropy_loss_scale 0.005` +
   `max_log_std 2.0` froze the policy std at ~0.96 for the entire run — the
   policy was a *saturated-noise controller* (mean |action| ≈ 0.98, actions
   constantly slamming the clip limits).
2. **Bug B was a misdiagnosis.**  A dedicated probe
   (`scripts/model_validation/diag_belt_collision.py`) showed the conveyor box
   **does** stop the gripper: collision meshes rest at 0.794 m vs belt
   0.800 m while the base_z drive saturates at 200 N and the fingers bend
   back.  The "piercing" perception came from the *virtual* fingertip constant
   reading ~0.12 m below the real meshes in the jammed state (free-air
   closure sweep confirmed the constants themselves are correct), plus the
   post-place dive behaviour.  The belt-collision termination threshold
   (0.20 m) was **kinematically unreachable** (max ~0.125 m) — dead code.
3. **Reward accounting bugs.**  Isaac Lab multiplies rewards by dt (1/60 s):
   a per-step weight `w` ≈ `w×5`/episode, a one-shot `w` ≈ `w/60` — the
   "dominant" `release_event=80` was actually worth **1.3** (vs ~165 for
   `shirt_in_target`).  The smoothness curriculum (`num_steps=150000`) never
   fired in a 20 000-timestep run (`num_steps` counts trainer timesteps).
   The anti-hover clearance fade required a shirt-bottom height of 1.08 m —
   unreachable (max lift ≈ 0.93 m) — so it *never engaged*.
   `shirt_off_conveyor` penalised the legitimate carry (centroid y > 0.4)
   **and** every step the shirt rested inside the drum (z < 0.7) — −5/step
   for succeeding.
4. **Bug A (return-to-neutral)**: reward std 0.25 → flat gradient far from
   neutral; reach/grasp shaping still active after placement kept pulling the
   arm toward the cloth *inside the drum*; weight 100 made "park over the
   drum" acceptable (~50 % of the term's max).

### Changes applied

| Area | Change |
|---|---|
| PPO (`skrl_ppo_cfg.yaml`) | `entropy_loss_scale` 0.005 → **0.0**, `initial_log_std` 0 → **−0.5**, `max_log_std` 2.0 → **0.5**, `min_log_std` −2.5 → **−3.0** |
| `release_event` | weight 80 → **240**, magnitude **graded ×(0.25…1.0)** by centering × whole-shirt-lift quality |
| Clearance fade | `clear_margin` 0.20 → **0.05** (fade completes at the reachable 0.93 m) |
| `shirt_off_conveyor` | penalise only *fell below belt outside the drum footprint* |
| Curriculum | `num_steps` 150 000 → 8 000 (→ 2 000 for fine-tunes), `action_rate` target −2e-3 → −3e-3 |
| `return_to_neutral` | std 0.25 → **0.40**; later weight 100 → **200** (run 4) |
| Post-place gating | `reaching_shirt`, `reaching_shirt_fine`, `grasping`, `arm_utilization` all **off after placement** (run 3) |
| Belt termination | `max_penetration` 0.20 → **0.12** (catches the full-force grind state) |
| Eval tooling | `_eval_diag.py` extended: live drum-fraction, placed count, neutral-pose deviation |

Mechanics regression stayed **5/5 PASS** throughout (no cloth-physics changes
were needed).

### Training runs & deterministic results

All evaluations: deterministic mean actions, 16 envs × 3 episodes, **unseen**
seed 7 (final checkpoint additionally seed 12).

| Run | What | Deterministic outcome |
|---|---|---|
| `2026-07-01_20-36-06` (`agent_6000`) | Phase-1 baseline | grasp 0/8, place **0 %** (stoch: 100 %/frac 0.5) |
| `2026-07-02_00-09-06`, 20 k steps | reworked rewards + PPO fix | **`agent_14000`: place 0.98, frac 0.91 (0.77–1.00), \|act\| 0.37**; post-place drift to dev ≈ 0.40 |
| `2026-07-02_06-38-15`, 20 k steps | + tighter `goal_tracking_fine` (std 0.15, w 8) | **regressed**: frac 0.37–0.49 — release-at-rim attractor despite *identical training curves*; change reverted |
| `2026-07-02_12-40-34`, 8 k fine-tune from `agent_14000` | + post-place gating | **`agent_4000`: place 1.00, frac 0.97 (0.95–1.00)**; return still partial (dev ≈ 0.39); `agent_6000/8000` oscillated (0.70 / 0.85) |
| `2026-07-02_15-10-58`, 4.8 k fine-tune from run-3 `agent_4000` | + `return_to_neutral` ×2 | **`agent_4000` = FINAL: place 1.00 (96/96, seeds 7+12), frac 0.93 (0.89–0.96), \|act\| 0.36, best return-home (dev 0.13 → ≈0.3)** |

Training figures: `tasks/manager_based/shirt_place/figures/tensegrity/`
(main run) and `figures/tensegrity_finetune/` (final fine-tune), generated by
`scripts/plot_shirt_training_results.py` (tensorboard-free).

### Verification of the train/play pipeline (skrl 2.1.0)

Audited for the historical "train and play scale differently" failure class:
checkpoints store the native `observation_preprocessor` key and `agent.load()`
restores it in play (the legacy `state_preprocessor` migration shim in
`play.py`/`_eval_diag.py` is now a harmless no-op); `PPO.act()` calls the
scaler with `train=False` (no stat drift during play); play's
`act(obs, env.state(), timestep=…, timesteps=…)` matches the 2.1.0 signature;
`mean_actions` is present for the Gaussian head → play is genuinely
deterministic; both tasks register the same agent YAML → no config divergence.

### Lessons learned (transferable)

1. **Budget one-shot rewards as `w/60`, per-step as `w×5`** (dt-scaling).
2. **`modify_reward_weight.num_steps` counts trainer timesteps**, not samples.
3. **An entropy bonus can permanently freeze the Gaussian std** → the policy
   exploits its own noise and the mean learns nothing.  For mean-action
   deployment: entropy 0, capped `log_std`, moderate initial std.
4. **Check every gate/threshold against reachable geometry** — two of ours
   (clearance fade, belt termination) were silently dead.
5. **Training curves do not predict deterministic quality.**  Select
   checkpoints only by deterministic eval on unseen seeds; fine-tune
   checkpoints oscillate hard (0.97 → 0.70 → 0.85 within 4 k steps).
6. **Gate pre-goal shaping off after the goal** or it fights the post-goal
   behaviour (reach-pull toward the placed cloth vs return-to-neutral).
7. **Fine-tuning from a competent checkpoint** applies targeted behavioural
   fixes in ~4 k timesteps at low risk — cheaper and safer than re-rolling.

### Remaining gaps / possible future work

- Post-place neutral deviation settles at ~0.3 (arm returns *near* home but
  leans); a longer return-focused fine-tune could tighten it.
- Drum-fraction 0.93 vs 0.97 trade-off against return quality (both
  checkpoints kept; see table above).
- Shirt pose randomisation (XY/yaw, folds) is still disabled — the flat
  fixed-pose simplification from Phase 1.
- Physical-tendon variant untrained on the cloth task.

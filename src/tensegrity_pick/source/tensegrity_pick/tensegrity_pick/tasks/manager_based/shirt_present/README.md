# Shirt Present Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

**Task 2 of the cloth-sorting pipeline.** The second robot grasps a second
holding point on the **hanging** T-shirt and stretches it so front/back
inspection cameras can assess the garment's condition (reusable / recyclable
/ trash).  Classification itself is a black box — this task only has to make
the cloth *inspectable*.

> **Status: HEM-TO-HEM REDESIGN, training in progress (2026-07-07).**  After
> Georg's visual inspection of the first-pass `agent_96000` (naive lowest-point
> grasp, det. present 0.927), the geometry was rebuilt around the FAPS
> heuristics study's winning **hem-corner ↔ hem-corner horizontal pull**
> (scripted coverage 0.820 vs 0.679 for the naive rule) and six visual-
> inspection findings were fixed.  Phase-2 training runs on `project/shirt-present`;
> deterministic checkpoint selection (≥ 0.9 present latch at the 0.65 coverage
> gate) is pending — see the
> [tracking report Phase 2](../../../../../../../../doc/reports/shirt_present_optimization_tracking.md#phase-2-hem-to-hem-presentation-redesign).
>
> The shirt hangs pinned at a HEM point (slot-1 solver anchor, restored from the
> 43 bottom-edge-anchored hanging-bank states) so it hangs upside-down; the
> learning arm's slot-0 grasp targets the **opposite hem corner** (latched at
> reset) and pulls it HORIZONTALLY to the holder's height — gravity drapes the
> body below the taut chord (self-aligning to the camera, study yaw gap 0.003).
> Success = windowed presented latch (both grasps ∧ taut ∧ silhouette coverage
> ≥ 0.65 ∧ cloth still).  Two simultaneous attachments are validated stable
> through tautness ratio 1.15
> ([Stage-0 report](../../../../../../../../doc/reports/cloth_stage0_physics_derisk.md) §2).

## Table of Contents

- [Goal](#goal)
- [Variants](#variants)
- [Scene](#scene)
- [Reset Mechanics: the Hanging Bank](#reset-mechanics-the-hanging-bank)
- [Controlled Joints](#controlled-joints)
- [Actions](#actions)
- [Observations (policy group)](#observations-policy-group)
- [Rewards (stub)](#rewards-stub)
- [Terminations](#terminations)
- [Reset Events](#reset-events)
- [Simulation Parameters](#simulation-parameters)
- [Planned Work (stub → trainable task)](#planned-work-stub--trainable-task)
- [Running](#running)
- [Related](#related)

## Goal

Train an RL agent (the second arm) to maximize garment inspectability:

1. **Reach** the hanging shirt (held by the retriever, gripping a HEM point at
   the presentation pose, so the garment hangs upside-down).
2. **Regrasp** the OPPOSITE **hem corner** — the study's winning grasp pair
   (hem-edge chords span the garment's longest continuous edge; +0.14 median
   coverage over the naive lowest-point rule).  The two hem corners are found
   deterministically from the flat-rest shape (landmark Voronoi scheme,
   `mdp/present_geometry.py`); the accessible one is latched at reset.
3. **Stretch HORIZONTALLY**: pull the grasped corner to the holder's HEIGHT,
   offset along the camera-plane x by ~1.05 × the flat rest span, so the taut
   chord is horizontal and gravity drapes the body below it.

**Success predicate** (`presented_now`, latched over a window — ≥ 80 % of the
last 60 steps): `grasp_active(slot 0) ∧ holder_attached(slot 1) ∧
stretch_ratio ∈ [0.90, 1.15] ∧ silhouette_coverage ≥ 0.65 ∧ cloth-centroid
speed < 0.20 m/s`.  Rationale (measured decisions):

- **Silhouette coverage** = rasterized projection of all particles onto the
  camera (world-XZ) plane / flat one-sided rest area
  (`shared/cloth_metrics.py`) — folds count once, so bunching/hiding cannot
  score.  A front+back inspection sees the SAME occluding silhouette, so one
  area serves both viewpoints.  Reference band: the FAPS heuristics study puts
  scripted hem↔hem median coverage at 0.820 (p25 0.701), so **0.65** is a
  learnable success gate (0.75 curriculum goal); ICRA-2024 top-three ≈ 0.55–0.60.
- **Stretch ratio** = ‖patch-centroid₀ − patch-centroid₁‖ / **FLAT rest
  distance** between the two grasp patches (`cloth_metrics.stretch_ratio` — the
  heuristics study's definition; clean for the hem-corner pair, no wrap-around
  over-reading).  At the horizontal chord it reads ~1.05 (measured baseline:
  1.051); band [0.90, 1.15] (Stage-0 stability margin).  NB: the first pass
  normalised a GEODESIC ratio by the at-grasp r0, which mis-read the hem↔hem
  hang→horizontal configuration change (a correct pull → 0.82 normalised, below
  the taut gate) — hence the switch.
- **Speed gate on the cloth centroid, never the EE** (shirt_pick Phase-3
  lesson: residual PD sway at raised postures is 0.24–0.27 m/s while the
  hanging garment low-pass filters to 0.06–0.20; the camera inspects the
  cloth anyway).

Logged metrics: `Metrics/{present_rate, grasp_rate, drop_rate,
final_coverage, final_stretch_ratio, final_stretch_norm}`.

Terminal states (stretched shirt) form the **initial-state bank of
shirt_distribute**.

## Variants

| Environment ID | Robot | Description |
|---|---|---|
| `Template-Shirt-Present-Kinova-F140-v0` | Kinova Gen3 (7 DOF) + Robotiq 2F-140 | Joint-position control |
| `Template-Shirt-Present-Kinova-F140-Play-v0` | 〃 | 50-env play/eval configuration |
| `Template-Shirt-Present-UR5e-F140-v0` | UR5e (6 DOF) + Robotiq 2F-140 | Joint-position control; mounted yaw +90° (showcase pose) |
| `Template-Shirt-Present-UR5e-F140-Play-v0` | 〃 | 50-env play/eval configuration |

Planned: UR10 variant and "Frankenstein" (+ tensegrity wrist) versions of all
second-robot arms, plus task-space (IK/OSC) action variants reusing the reach
task's common configs.

## Scene

Uses the **central cloth-sorting scene**
[shared/cloth_sorting_scene_cfg.py](../shared/cloth_sorting_scene_cfg.py)
(see the [shirt_pick README](../shirt_pick/README.md#scene) for the full
entity table).  Task-specific additions:

| Entity | Description |
|---|---|
| `robot` | Second robot on the pedestal at `(0.75, 1.0, 0.75)` — the validated F140 reach workspace-analysis pose; UR5e yaw-rotated +90° per the showcase |
| `holder_robot` | Passive 5-DOF tensegrity, now mounted directly above the local anchor pointing straight down so its gripper GRIPS the anchor patch (finding #3 fix; measured rest drop 0.98 m; the actual hold is still the solver anchor) |
| shirt | Hangs from the LOCAL anchor `(0.50, 0.85, 1.20)` — a shirt_present-only override of the shared `PRESENTATION_POS (0.15, 0.90, 1.60)`: x=0.50 halves the cross-body reach (base at 0.75) and clears the drum+pedestal; z=1.20 keeps the horizontal chord in the UR5e envelope.  Coverage is translation-invariant, so this is metric-neutral. |

## Reset Mechanics: the Hanging Bank

At each reset ([shirt_present_env.py](shirt_present_env.py)):

1. A relaxed hang is restored from the **hem-anchored subset** of the bank
   (`present_geometry.holder_region_mask` keeps the 43 of 356 states pinned on
   the garment bottom edge, so the retriever grips a HEM point → upside-down
   hang), yaw+mirror augmented; the pinned patch is re-attached at the local
   anchor on **slot 1** (`ClothObject.attach`, radius 0.07 m).  Set
   `use_hem_holder=False` to fall back to the full random-anchor bank.
2. The **hand target is latched**: the more accessible of the two hem-corner
   particles (farther from the anchor) is chosen once from the settled hang and
   held for the episode (per-step selection flip-flops between the corners).
3. Each control step re-pins the holder patch (`ClothObject.hold`, slot 1)
   while the base env runs the learning arm's deterministic attach/hold/detach
   on **slot 0**, targeting the latched hem corner.  The stretch ratio's rest
   normaliser is the FLAT rest distance between the two grasp patches (no
   geodesic graph — clean for the hem pair).

Two simultaneous attachments are Stage-0 validated (stable through +15 %
tautness, `scripts/model_validation/test_two_attachments.py`).

## Controlled Joints

| Variant | Arm joints | Gripper |
|---|---|---|
| Kinova Gen3 | `joint_1` … `joint_7` | `finger_joint` (+ 7 mimic joints) |
| UR5e | `shoulder_pan/lift`, `elbow`, `wrist_1/2/3` | `finger_joint` (+ 7 mimic joints) |

## Actions

| Term | Dims (Kinova / UR5e) | Type | Scale |
|---|---|---|---|
| `arm_action` | 7 / 6 | **EMAJointPositionToLimits** (α = 0.2 — the reach-grid joint-space winner; the stub's delta-from-default scale 0.5 saturated: reaching the hang needs ±1.25 rad, measured job 3809927) | to-limits |
| `gripper_action` | 1 | BinaryJointPositionAction | open 0.0 / close 0.7854 |

Total: **8 dims (Kinova) / 7 dims (UR5e)**.

## Observations (policy group)

**44 dims (Kinova) / 41 dims (UR5e)**, no noise corruption (the hem-to-hem
redesign added `pull_target_rel`, +3 vs the first pass):

| Term | Dims (Kinova / UR5e) | Description |
|---|---|---|
| `joint_pos_rel` | 8 / 7 | Controlled joint positions relative to defaults (arm + finger) |
| `joint_vel_rel` | 8 / 7 | Controlled joint velocities |
| `ee_pos_w` | 3 | Grasp-centre position (env-local) |
| `shirt_rel` | 3 | Shirt centroid relative to grasp centre |
| `hand_target_rel` | 3 | **Targeted hem corner** relative to the DYNAMIC finger tip — the exact geometry the deterministic attach trigger uses |
| `pull_target_rel` | 3 | **Horizontal-pull goal** (holder height, offset along camera-plane x) relative to the finger tip |
| `shirt_vel` | 3 | Shirt centroid velocity |
| `grasp_active` | 1 | Hand grasp (slot 0) attached |
| `holder_attached` | 1 | Holder anchor (slot 1) still pinned |
| `stretch_ratio` | 1 | At-grasp-normalised tautness (0 until both attached) |
| `coverage` | 1 | Camera-plane silhouette coverage |
| `gripper_closure` | 1 | Normalized closure |
| `actions` | 8 / 7 | Previous action |

All task-state terms are camera-derivable in principle (both grasp points
visible, garment flat geometry known a priori, coverage = segmentation-mask
area ratio).  Still open (report §6): shoulder keypoints with visibility
flags; privileged particle field in a critic-only group.

## Rewards

Sequential shirt_pick pattern.  dt-scaling: per-step weight *w* earns
≈ *w* × episode-seconds; one-shots earn *w*/60 → the drop penalty is sized
~60× the per-step terms.

| Term | Weight | Description |
|---|---|---|
| `reaching` | 2.0 | `1 − tanh(‖tip − hem corner‖ / 0.25)`; paid ONLY with an OPEN gripper pre-grasp (fixes finding #1's premature-close hack), saturates to 1 while grasped |
| `grasp_hold` | 5.0 | Per-step while the slot-0 attachment holds |
| `pull` | 10.0 | `1 − tanh(‖tip − horizontal-pull target‖ / 0.20)`, gated on BOTH grasps — DIRECTS the stretch into the study's horizontal chord (replaces the old undirected tautness-only shaping) |
| `stretch` | 4.0 | Maintain-tautness on the flat-rest ratio (0.80 → 1.02 ramp), gated on both attachments |
| `coverage` | 14.0 | Camera-plane silhouette coverage, gated on both attachments (anti-fling/bunch) |
| `presented` | 30.0 | Full success predicate per step — dominant term; no anti-hover fade (holding IS the task) |
| `overstretch` | −40.0 | Proportional above ratio 1.10 — moat under the 1.15 predicate edge (Stage-0 stability) |
| `drop` | −240.0 | One-shot when an established hand grasp is lost (≈ −4 after dt) |
| `early_close` | −15.0 | Per-step (distance-scaled) for commanding the gripper closed while far + ungrasped (finding #1) |
| `occlusion` | −2.0 | Mild: EE on the camera side of the cloth (finding #5; cosmetic — the coverage metric has no camera sensor) |
| `action_rate` | −1e-4 → −3e-3 | Curriculum ramp at 2000 trainer steps (shirt_place profile) |
| `joint_vel` | −1e-4 → −2e-3 | 〃 |

## Terminations

| Term | Condition |
|---|---|
| `time_out` | Episode length 6.0 s |
| `joint_vel_diverged` | Any controlled joint > 100 rad/s |

Planned: drop (either attachment lost) = failure termination.

## Reset Events

Same event set as shirt_pick (conveyor collider swap, cloth prestartup,
scene/arm/gripper reset); the cloth reset is the env-owned hanging-anchor
procedure described above.

## Simulation Parameters

Identical to shirt_pick — `configure_cloth_sim()` profile (60 Hz, 24 PBD
iterations, self-collision on, enlarged GPU buffers, `replicate_physics=False`).
See the [shirt_pick table](../shirt_pick/README.md#simulation-parameters).

## Planned Work (stub → trainable task)

Full staged plan: [doc/TODO.md — Master-Thesis Goal Tasks](../../../../../../../../doc/TODO.md).
Headlines:

1. **Reset from the shirt_pick terminal-state bank** (skill-chaining
   distribution shift is the pipeline's #1 risk — do not train against the
   idealized centre-hang only).  `ShirtPickEnv.snapshot_terminal_states`
   provides the capture side.
2. ~~Second attachment in `ClothObject` + stretch-stability validation~~ —
   **DONE (Stage 0):** multi-slot attachments implemented and stable through
   tautness ratio 1.15; no handover fallback needed.  Remaining: enable the
   slot-0 hand grasp in this env (attach trigger on the hanging shirt).
3. **Lowest-point regrasp heuristic baseline**, then RL.
4. **Coverage/tautness reward** (front+back viewpoints).
5. **Brute-force grasp-pair oracle** script (spanned area over sampled vertex
   pairs) for an upper bound and second-grasp region labels.
6. Decide: does the retriever keep holding throughout (open question #2)?

## Findings addressed (2026-07 visual inspection)

Six findings from live playback of the first-pass `agent_96000`
(`logs/skrl/need_visual_verification/shirt_present/findings.md`):

1. **Premature gripper-close hack** — `reaching` pays only with an OPEN gripper
   pre-grasp + `early_close_penalty`.
2. **Cloth/robot clipping** — mitigated by the horizontal geometry (arm works
   beside the chord, not through the drape) + the relocated anchor; residual is
   collision-fidelity (UR5e self-collision is off), GUI-verify only.
3. **Holder didn't grip the anchor** — holder mounted above the anchor pointing
   down (measured rest drop 0.98 m → gripper at the anchor, 0.087 m).
4. **"Self-collision, move to x=0.8"** — reconciled/REJECTED: x=0.8 drapes the
   shirt through the robot's own pedestal; anchor moved to x=0.50 (halves the
   cross-body reach), the real self-fold driver.  Self-collision is disabled so
   the fold is cosmetic.
5. **Arm occludes the −Y camera** — mild `occlusion_penalty` (best-effort; the
   coverage metric has no camera sensor).
6. **Cloth too stretchy** — investigated, no change (shared cloth identical to
   the validated shirt_pick).

## Results

**First pass (naive lowest-point, superseded):** det. present **0.927**
(agent_96000, run 6) — kept for the terminal-bank hook and as the ceiling
reference.  **Hem-to-hem redesign (Phase 2):** scripted in-scene baseline
present 0.125 / grasp 0.41 / coverage ceiling 0.82; RL training in progress
(job 3821742).  Full history + deterministic checkpoint selection:
[tracking report Phase 2](../../../../../../../../doc/reports/shirt_present_optimization_tracking.md#phase-2-hem-to-hem-presentation-redesign).

## Running

```bash
# from src/tensegrity_pick, conda env env_isaaclab (GPU via Slurm on Alex)

# Smoke tests
python scripts/agents/zero_agent.py --task=Template-Shirt-Present-UR5e-F140-v0 --num_envs 4 --headless

# Scripted baseline + threshold calibration
python scripts/model_validation/baseline_shirt_present.py --headless --num_envs 16

# Train (Alex/Slurm; 512 envs, 2-seed array — see tools/train_alex.sh)
./tools/train_alex.sh -s "43 44" -t 08:00:00 -i 2500 \
    Template-Shirt-Present-UR5e-F140-v0 --headless --num_envs 512

# Deterministic evaluation / gate diagnosis / terminal-state bank
python scripts/skrl/evaluate_shirt_present.py --headless --num_envs 32 \
    --num_episodes 96 --seed 7 --checkpoint <ckpt.pt>
python scripts/model_validation/diag_shirt_present_policy.py --headless \
    --num_envs 8 --checkpoint <ckpt.pt>
python scripts/model_validation/snapshot_shirt_present_terminal.py --headless \
    --num_envs 32 --rounds 2 --checkpoint <ckpt.pt> --out <bank.pt>
```

## Related

- [Optimization tracking](../../../../../../../../doc/reports/shirt_present_optimization_tracking.md)
- [Research report (pipeline plan)](../../../../../../../../doc/reports/RESEARCH_cloth_sorting_pipeline.md)
- [shirt_pick](../shirt_pick/README.md) — pipeline task 1 (produces this task's initial states)
- [shirt_distribute](../shirt_distribute/README.md) — pipeline task 3 (consumes this task's terminal states)
- [shirt_place](../shirt_place/README.md) — grasp mechanics / cloth model / PPO profile reference
- [Shared scene](../shared/cloth_sorting_scene_cfg.py) · [Shared env base](../shared/cloth_sorting_env.py)

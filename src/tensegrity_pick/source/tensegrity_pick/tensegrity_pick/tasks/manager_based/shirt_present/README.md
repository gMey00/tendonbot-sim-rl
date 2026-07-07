# Shirt Present Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

**Task 2 of the cloth-sorting pipeline.** The second robot grasps a second
holding point on the **hanging** T-shirt and stretches it so front/back
inspection cameras can assess the garment's condition (reusable / recyclable
/ trash).  Classification itself is a black box — this task only has to make
the cloth *inspectable*.

> **Status: TRAINED & VALIDATED (2026-07-04).**  Selected checkpoint
> `logs/skrl/shirt_present/2026-07-04_15-21-09_ppo_torch_seed43/checkpoints/agent_96000.pt`
> (UR5e-F140): deterministic windowed present latch **0.927** on both eval
> seeds (7 & 11, 96 episodes each), grasp 0.990, drop 0.000, final coverage
> 0.68 — above the ICRA-2024 reference band.  Scripted baseline: 0.125.
> Full iteration history in the tracking report.
>
> The shirt hangs pinned at ONE RANDOM particle patch (356-state hanging bank,
> slot-1 solver anchor at the presentation pose); the learning arm's slot-0
> deterministic grasp is ENABLED and targets the **lowest hanging point**.
> Success = windowed presented latch (both grasps ∧ taut ∧ silhouette
> coverage ∧ cloth still).  Two simultaneous attachments are validated stable
> through tautness ratio 1.15
> ([Stage-0 report](../../../../../../../../doc/reports/cloth_stage0_physics_derisk.md) §2).
> Training/eval results: see the
> [tracking report](../../../../../../../../doc/reports/shirt_present_optimization_tracking.md).

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

1. **Reach** the hanging shirt (held by the retriever at the presentation pose).
2. **Regrasp** a second point — literature standard: the hanging shirt's
   **lowest point** (Maitin-Shepard 2010; Doumanoglou 2014; camera-trivial
   from depth). Shoulder-shoulder only if keypoints prove reliable.
3. **Stretch** the cloth taut-but-not-overstretched so front and back cameras
   see a maximal spanned area.

**Success predicate** (`presented_now`, latched over a window — ≥ 80 % of the
last 60 steps): `grasp_active(slot 0) ∧ holder_attached(slot 1) ∧
stretch_ratio/r0 ∈ [0.92, 1.10] ∧ silhouette_coverage ≥ 0.50 ∧ cloth-centroid
speed < 0.20 m/s`.  Rationale (measured decisions):

- **Silhouette coverage** = rasterized projection of all particles onto the
  camera (world-XZ) plane / flat one-sided rest area
  (`shared/cloth_metrics.py`) — folds count once, so bunching/hiding cannot
  score.  A front+back inspection sees the SAME occluding silhouette, so one
  area serves both viewpoints.  Reference band: ICRA-2024 cloth-competition
  top-three coverage ≈ 0.55–0.60.
- **Stretch ratio** = ‖patch-centroid₀ − patch-centroid₁‖ / GEODESIC rest
  distance from the holder patch (PBD spring graph, GPU multi-source
  Bellman-Ford per reset), normalised by the at-grasp value r0 (the
  lowest-point span is already gravity-taut at grasp: raw r0 = 1.10–1.43
  measured).  < 0.92·r0 = slack, > 1.10·r0 = overstretch (Stage-0 margin).
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
| `holder_robot` | Passive 5-DOF tensegrity at its hanging mount `(0.15, 0.0, 2.30)` — visual scenery in the stub (the actual hold is the solver anchor); planned: posed from the shirt_pick terminal bank |
| shirt | Hangs from the anchor at `PRESENTATION_POS = (0.15, 0.50, 1.10)` |

## Reset Mechanics: the Hanging Bank

At each reset ([shirt_present_env.py](shirt_present_env.py)):

1. A relaxed random-particle hang is restored from the 356-state bank
   (`res/Props/Cloth/banks/tshirt_hanging_bank.pt`, yaw+mirror augmented,
   `max_drape` filtered to clear the drum under the pose); the pinned patch
   is re-attached at the anchor on **slot 1** (`ClothObject.attach`, radius
   0.07 m).  Missing bank → idealized centre-hang fallback.
2. Each control step re-pins the holder patch (`ClothObject.hold`, slot 1)
   while the base env runs the learning arm's deterministic attach/hold/
   detach on **slot 0**, retargeted to the lowest hanging point.
3. Per reset, the holder-patch **geodesic distance field** over the PBD
   spring graph is recomputed (GPU multi-source Bellman-Ford) — the rest
   normaliser of the stretch ratio.

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

**41 dims (Kinova) / 38 dims (UR5e)**, no noise corruption:

| Term | Dims (Kinova / UR5e) | Description |
|---|---|---|
| `joint_pos_rel` | 8 / 7 | Controlled joint positions relative to defaults (arm + finger) |
| `joint_vel_rel` | 8 / 7 | Controlled joint velocities |
| `ee_pos_w` | 3 | Grasp-centre position (env-local) |
| `shirt_rel` | 3 | Shirt centroid relative to grasp centre |
| `lowest_point_rel` | 3 | **Lowest cloth particle** relative to the DYNAMIC finger tip — the exact geometry the deterministic attach trigger uses |
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
| `reaching` | 2.0 | `1 − tanh(‖tip − lowest point‖ / 0.25)`; **saturates to 1 while grasped** (the lowest point migrates post-grasp — chasing it would fight the stretch) |
| `grasp_hold` | 5.0 | Per-step while the slot-0 attachment holds |
| `stretch` | 8.0 | Maintain-tautness on the at-grasp-normalised ratio (0.80 → 0.97 ramp), gated on BOTH attachments |
| `coverage` | 14.0 | Camera-plane silhouette coverage, gated on both attachments (anti-fling/bunch); 10→14 after gate diagnosis (weakest gate) |
| `presented` | 30.0 | Full success predicate per step — dominant term; no anti-hover fade needed (holding IS the task) |
| `overstretch` | −40.0 | Proportional above normalised ratio 1.05 — gradient moat under the 1.10 predicate edge (validated stability ends at +15 %) |
| `drop` | −240.0 | One-shot when an established hand grasp is lost (≈ −4 after dt); −120→−240 after run-2 drops capped present at 0.65–0.71 |
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

## Results (UR5e-F140, RTX PRO 6000, 2026-07-04)

| Stage | det. present_rate | Notes |
|---|---|---|
| Scripted baseline (v4.1) | 0.125 | 11/16 grasped; coverage is what the blind ray-pull cannot raise |
| Run 1 (20 k) | 0.625 | still climbing at cap |
| Run 3 (drop −240) | 0.750 | drops 0.23 → 0.00 |
| Run 5 (coverage 14, overstretch moat 1.05) | 0.833 | |
| **Run 6 (96 k, seed 43) — agent_96000** | **0.927 / 0.927** (eval seeds 7 / 11) | grasp 0.990, drop 0.000, coverage 0.682 |

Figures: [figures/ur5e_f140/](figures/ur5e_f140/) (total reward, task metrics,
cloth metrics, reward decomposition, penalties, episode length).
Terminal-state bank hook validated:
`scripts/model_validation/snapshot_shirt_present_terminal.py` (58-state sample,
presented-filtered, both grasp masks included).

## Running

```bash
# from src/tensegrity_pick, conda env env_isaaclab (GPU via Slurm on Alex)

# Smoke tests
python scripts/agents/zero_agent.py --task=Template-Shirt-Present-UR5e-F140-v0 --num_envs 4 --headless

# Scripted baseline + threshold calibration
python scripts/model_validation/baseline_shirt_present.py --headless --num_envs 16

# Train (64 envs = measured RTX PRO 6000 sweet spot, see tracking report)
python scripts/skrl/train.py --task=Template-Shirt-Present-UR5e-F140-v0 \
    --algorithm=PPO --num_envs 64 --headless --max_iterations 2000

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

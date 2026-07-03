# Shirt Present Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

**Task 2 of the cloth-sorting pipeline.** The second robot grasps a second
holding point on the **hanging** T-shirt and stretches it so front/back
inspection cameras can assess the garment's condition (reusable / recyclable
/ trash).  Classification itself is a black box — this task only has to make
the cloth *inspectable*.

> **Status: stub, physics de-risked (2026-07-03).** The environment
> constructs and runs (zero/random agents verified for both robot variants).
> The shirt hangs from a **static slot-1 solver anchor** at the presentation
> pose — an idealized stand-in for the shirt_pick terminal state — and the
> passive retriever (`holder_robot`) is visual scenery.  **Two simultaneous
> attachments are now supported and validated** (multi-slot `ClothObject`,
> stable through tautness ratio 1.15 — see the
> [Stage-0 report](../../../../../../../../doc/reports/cloth_stage0_physics_derisk.md) §2);
> the learning arm's slot-0 grasp stays off only until the Task-2 regrasp MDP
> is implemented — see [Planned Work](#planned-work-stub--trainable-task).

## Table of Contents

- [Goal](#goal)
- [Variants](#variants)
- [Scene](#scene)
- [Stub Mechanics: the Hanging Anchor](#stub-mechanics-the-hanging-anchor)
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

Success metric (planned): **projected coverage** of the cloth onto the
inspection-camera image planes (front `INSPECTION_CAMERA_POS = (0.15, 2.0, 1.25)`
looking −Y, plus a back view), with a tautness bonus and an overstretch
penalty.  Reference band: ICRA-2024 cloth-competition top-three coverage
≈ 0.55–0.60.  Current stub metric: `Metrics/grasp_rate` (reflects the holder
anchor, ≈ 1.0 by construction — placeholder).

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

## Stub Mechanics: the Hanging Anchor

At each reset ([shirt_present_env.py](shirt_present_env.py)):

1. The flat rest-shape shirt is teleported so its centroid sits at
   `PRESENTATION_POS`.
2. The particle cluster within **0.07 m** of the anchor is pinned as PBD
   solver anchors (`ClothObject.attach`, same mechanism as the validated
   shirt_place grasp).
3. Each control step re-pins the cluster at the static anchor
   (`ClothObject.hold`), so the sheet drapes under gravity into a
   **centre-hung garment** — an idealized "held at the former highest point"
   state.

The anchor uses **attachment slot 1**; slot 0 is reserved for the learning
arm's own grasp.  The two-simultaneous-attachments physics risk flagged by
the research report is **retired**: the Stage-0 stretch test holds both
grasps stable through tautness ratio 1.15 with no solver instability
(`scripts/model_validation/test_two_attachments.py`).  `enable_hand_grasp`
stays `False` only until the Task-2 regrasp/stretch MDP (rewards, lowest-point
targeting) is implemented.

## Controlled Joints

| Variant | Arm joints | Gripper |
|---|---|---|
| Kinova Gen3 | `joint_1` … `joint_7` | `finger_joint` (+ 7 mimic joints) |
| UR5e | `shoulder_pan/lift`, `elbow`, `wrist_1/2/3` | `finger_joint` (+ 7 mimic joints) |

## Actions

| Term | Dims (Kinova / UR5e) | Type | Scale |
|---|---|---|---|
| `arm_action` | 7 / 6 | JointPositionAction (delta from default) | 0.5 |
| `gripper_action` | 1 | BinaryJointPositionAction | open 0.0 / close 0.7854 |

Total: **8 dims (Kinova) / 7 dims (UR5e)**.

## Observations (policy group)

**37 dims (Kinova) / 34 dims (UR5e)**, no noise corruption:

| Term | Dims (Kinova / UR5e) | Description |
|---|---|---|
| `joint_pos_rel` | 8 / 7 | Controlled joint positions relative to defaults (arm + finger) |
| `joint_vel_rel` | 8 / 7 | Controlled joint velocities |
| `ee_pos_w` | 3 | Grasp-centre position (env-local) |
| `shirt_rel` | 3 | Shirt centroid relative to grasp centre |
| `lowest_point_rel` | 3 | **Lowest cloth particle** relative to grasp centre — the regrasp target |
| `shirt_vel` | 3 | Shirt centroid velocity |
| `gripper_closure` | 1 | Normalized closure |
| `actions` | 8 / 7 | Previous action |

Planned (camera-realistic, report §6): inter-grasp distance, tautness proxy
(inter-grasp distance / rest geodesic distance), projected coverage estimate,
shoulder keypoints **with visibility flags**; privileged particle field in a
critic-only group.

## Rewards (stub)

| Term | Weight | Description |
|---|---|---|
| `reaching_shirt` | 2.0 | `1 − tanh(‖EE − shirt‖ / 0.3)` — approach the regrasp region |
| `action_rate` | −1e-4 | Action-rate L2 |
| `joint_vel` | −1e-4 | Joint-velocity L2 |

Planned reward structure (with known hack modes to guard against):

- **Projected coverage** from the actual front/back camera viewpoints — NOT
  top-down (top-down coverage is gamed by bunching/hiding the shirt).
- **Taut-but-not-overstretched bonus**: cap reward when the tautness ratio
  exceeds ~1.0 (PBD mass-spring overstretch inflates area artificially).
- **Strain penalty** on particle-spring stretch.
- Both grasps must remain attached for coverage to pay.

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

## Running

```bash
# from src/tensegrity_pick, conda env env_isaaclab

# Smoke tests (verified 2026-07-03)
python scripts/zero_agent.py   --task=Template-Shirt-Present-Kinova-F140-v0 --num_envs 4 --headless
python scripts/random_agent.py --task=Template-Shirt-Present-UR5e-F140-v0   --num_envs 4 --headless

# Train (once the real MDP is implemented)
python scripts/skrl/train.py --task=Template-Shirt-Present-Kinova-F140-v0 --num_envs 128 --headless
```

## Related

- [Optimization tracking](../../../../../../../../doc/reports/shirt_present_optimization_tracking.md)
- [Research report (pipeline plan)](../../../../../../../../doc/reports/RESEARCH_cloth_sorting_pipeline.md)
- [shirt_pick](../shirt_pick/README.md) — pipeline task 1 (produces this task's initial states)
- [shirt_distribute](../shirt_distribute/README.md) — pipeline task 3 (consumes this task's terminal states)
- [shirt_place](../shirt_place/README.md) — grasp mechanics / cloth model / PPO profile reference
- [Shared scene](../shared/cloth_sorting_scene_cfg.py) · [Shared env base](../shared/cloth_sorting_env.py)

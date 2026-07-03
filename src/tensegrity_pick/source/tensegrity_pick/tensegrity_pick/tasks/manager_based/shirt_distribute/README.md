# Shirt Distribute Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

**Task 3 of the cloth-sorting pipeline.** After the (black-box) condition
classification, the second robot throws/places the T-shirt into the drum
matching its label (reusable / recyclable / trash).

**Goal-conditioned formulation** (research report §4, TossingBot precedent):
the condition label never enters the observation directly — it only selects
**which bin's position** is observed (`target_bin_rel`).  The per-episode
target bin is resampled uniformly, so "nearest bin" and "correct bin" diverge
and the policy cannot ignore the goal.

> **Status: stub (2026-07-03).** The environment constructs and runs
> (zero/random agents verified for both robot variants) with working
> goal-conditioning plumbing (random target bin, goal observation,
> `distribute_success_rate` metric).  The shirt currently starts flat on the
> belt edge (not yet grasped from the shirt_present terminal bank), and the
> reward is stub shaping — see [Planned Work](#planned-work-stub--trainable-task).

## Table of Contents

- [Goal](#goal)
- [Variants](#variants)
- [Scene](#scene)
- [Goal Conditioning](#goal-conditioning)
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

Train an RL agent (the second arm) to sort the inspected garment:

1. Start **holding the shirt** (stretched-inspection terminal state; stub:
   pick it up from the belt edge first).
2. **Carry** it toward the commanded bin.
3. **Release** it over the bin opening so the garment falls inside.

Success metric (TensorBoard `Metrics/…`): `distribute_success_rate` — after a
release (`~grasp_active`), ≥ 15 % of cloth particles inside the **commanded**
drum's interior cylinder (radius 0.2735 m, height 0.85 m — the same
interior-depth geometry as shirt_place, since draped cloth rarely reaches the
drum bottom).  Requiring the release closes the "lower the still-gripped
shirt into the drum" reward hack found in shirt_place.  Also logged:
`grasp_rate`.  Target from the staged plan: **> 85 % correct-bin placement**
across all three labels (and randomized layouts, once added).

## Variants

| Environment ID | Robot | Description |
|---|---|---|
| `Template-Shirt-Distribute-Kinova-F140-v0` | Kinova Gen3 (7 DOF) + Robotiq 2F-140 | Joint-position control |
| `Template-Shirt-Distribute-Kinova-F140-Play-v0` | 〃 | 50-env play/eval configuration |
| `Template-Shirt-Distribute-UR5e-F140-v0` | UR5e (6 DOF) + Robotiq 2F-140 | Joint-position control; mounted yaw +90° (showcase pose) |
| `Template-Shirt-Distribute-UR5e-F140-Play-v0` | 〃 | 50-env play/eval configuration |

Planned: UR10 and "Frankenstein" (+ tensegrity wrist) variants; task-space
(IK/OSC) action variants.

## Scene

Uses the **central cloth-sorting scene**
[shared/cloth_sorting_scene_cfg.py](../shared/cloth_sorting_scene_cfg.py)
(full entity table in the [shirt_pick README](../shirt_pick/README.md#scene)).
Task-specific configuration:

| Entity | Description |
|---|---|
| `robot` | Second robot on the pedestal at `(0.75, 1.0, 0.75)`; UR5e yaw +90° |
| `drum_reusable` | `(0.15, 1.0, 0)` — left of the robot (showcase position) |
| `drum_recyclable` | `(1.35, 1.0, 0)` — right of the robot |
| `drum_trash` | `(0.75, 1.6, 0)` — behind the robot |
| `holder_robot` | not spawned (`None`) |
| shirt | Stub: flat on the belt edge at `(0.75, 0.30)` — the closest belt point to the second robot |

## Goal Conditioning

Implemented in [shirt_distribute_env.py](shirt_distribute_env.py):

- `self._target_bin ∈ {0, 1, 2}` is resampled **uniformly per episode**
  (before the parent reset, so reset-step observations already see the new
  goal).
- `target_bin_pos_w` gathers the commanded drum's world position;
  the observation `target_bin_rel` = commanded bin − shirt centroid.
- The semantic label → bin mapping is a wrapper concern at deployment time;
  during training only the bin *position* matters, which is exactly what
  makes the policy transfer to any label assignment or (later) randomized
  bin layout.

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

**38 dims (Kinova) / 35 dims (UR5e)**, no noise corruption:

| Term | Dims (Kinova / UR5e) | Description |
|---|---|---|
| `joint_pos_rel` | 8 / 7 | Controlled joint positions relative to defaults (arm + finger) |
| `joint_vel_rel` | 8 / 7 | Controlled joint velocities |
| `ee_pos_w` | 3 | Grasp-centre position (env-local) |
| `shirt_rel` | 3 | Shirt centroid relative to grasp centre |
| `shirt_vel` | 3 | Shirt centroid velocity |
| `target_bin_rel` | 3 | **Commanded bin relative to the shirt** (the goal) |
| `gripper_closure` | 1 | Normalized closure |
| `grasp_active` | 1 | 1.0 while the attachment grasp holds |
| `actions` | 8 / 7 | Previous action |

## Rewards (stub)

| Term | Weight | Description |
|---|---|---|
| `reaching_shirt` | 2.0 | `1 − tanh(‖EE − shirt‖ / 0.3)` |
| `to_target_bin` | 4.0 | `1 − tanh(‖shirt − commanded bin‖ / 0.4)` — goal-conditioned shaping |
| `action_rate` | −1e-4 | Action-rate L2 |
| `joint_vel` | −1e-4 | Joint-velocity L2 |

Planned structure (port from shirt_place — it solved exactly this
release-into-drum problem):

- **Graded one-shot release event** (centering × height quality, dt-scaled
  weight ~60× the per-step shaping).
- **Anti-hover fade**: positioning shaping fades once the shirt is cleared
  over the bin, so only releasing pays.
- Sparse landing-in-correct-bin bonus; wrong-bin landings earn nothing (the
  goal-conditioned shaping already points at the right bin).

## Terminations

| Term | Condition |
|---|---|
| `time_out` | Episode length 6.0 s |
| `joint_vel_diverged` | Any controlled joint > 100 rad/s |

Planned: terminate N steps after a successful drop (shirt_place
`PLACE_SETTLE_STEPS` pattern) so shaping cannot be farmed post-success.

## Reset Events

Same event set as shirt_pick (conveyor collider swap, cloth prestartup,
scene/arm/gripper reset); the cloth reset lays the shirt flat at the belt
edge `(0.75, 0.30)` — to be replaced by grasped-state initialization from the
shirt_present terminal bank.

## Simulation Parameters

Identical to shirt_pick — `configure_cloth_sim()` profile (60 Hz, 24 PBD
iterations, self-collision on, enlarged GPU buffers, `replicate_physics=False`).
See the [shirt_pick table](../shirt_pick/README.md#simulation-parameters).

## Planned Work (stub → trainable task)

Full staged plan: [doc/TODO.md — Master-Thesis Goal Tasks](../../../../../../../../doc/TODO.md).
Headlines:

1. **Init grasped from the shirt_present terminal-state bank** (stub picks up
   from the belt edge — a different, easier-to-hack MDP).
2. **Real reward structure** — port shirt_place's graded `release_event` +
   anti-hover design + dt-scaled weights.
3. **Bin-layout randomization** so nearest ≠ correct generalizes beyond the
   three fixed showcase drums.
4. **TossingBot-style throw** (release-velocity conditioning) if the fixed
   drums prove outside comfortable placing reach — check the UR5e's envelope
   first (drum_recyclable at 0.6 m lateral offset).
5. Finger-tip offset calibration for the UR/Kinova EE frames (shared grasp
   machinery was calibrated on the tensegrity frame).

## Running

```bash
# from src/tensegrity_pick, conda env env_isaaclab

# Smoke tests (verified 2026-07-03)
python scripts/zero_agent.py   --task=Template-Shirt-Distribute-UR5e-F140-v0   --num_envs 4 --headless
python scripts/random_agent.py --task=Template-Shirt-Distribute-Kinova-F140-v0 --num_envs 4 --headless

# Train (once the real MDP is implemented)
python scripts/skrl/train.py --task=Template-Shirt-Distribute-Kinova-F140-v0 --num_envs 128 --headless
```

## Related

- [Optimization tracking](../../../../../../../../doc/reports/shirt_distribute_optimization_tracking.md)
- [Research report (pipeline plan)](../../../../../../../../doc/reports/RESEARCH_cloth_sorting_pipeline.md)
- [shirt_present](../shirt_present/README.md) — pipeline task 2 (produces this task's initial states)
- [shirt_place](../shirt_place/README.md) — the validated release-into-drum reference (reward design donor)
- [shirt_sort](../shirt_sort/README.md) — legacy template, superseded by this task
- [Shared scene](../shared/cloth_sorting_scene_cfg.py) · [Shared env base](../shared/cloth_sorting_env.py)

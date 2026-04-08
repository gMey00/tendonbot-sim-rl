# Cube Sort Task

> **⚠️ Work in Progress** — Stage 1 (single green cube on active conveyor).
> The agent reliably grasps and transports cubes but has not yet learned the
> release phase.  Reward rebalancing is implemented but awaits a full training
> run.

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

Cube **sorting** with the 5-DOF tensegrity manipulator and Robotiq 2F gripper.
The robot must pick target (green) cubes from a **moving conveyor belt**,
transport them to a drum beside the belt, and place them inside — while ignoring
distractor (red) cubes.  Up to 16 labelled cubes (8 green + 8 red) are managed
as a `RigidObjectCollection`.

![Scene Setup](./figures/scene_setup.png)

## Table of Contents

- [Goal](#goal)
- [Variants](#variants)
- [Scene](#scene)
- [Controlled Joints](#controlled-joints)
- [Actions (6 dims)](#actions-6-dims)
- [Observations (policy group)](#observations-policy-group)
- [Rewards](#rewards)
- [Terminations](#terminations)
- [Curriculum](#curriculum)
- [Reset Events](#reset-events)
- [Simulation Parameters](#simulation-parameters)
- [Training](#training)
- [Design Decisions & Lessons Learned](#design-decisions--lessons-learned)
- [Running](#running)
- [Training Results](#training-results)
- [Related](#related)

## Goal

Train an RL agent to sort cubes from a moving conveyor into a target drum:

1. **Reach** the nearest green (target) cube on the belt.
2. **Grasp** it with the Robotiq 2F gripper.
3. **Lift** it above the belt surface.
4. **Transport** it laterally toward the target drum (with urgency — belt moves).
5. **Release** the cube above the drum opening.
6. **Re-orient** back toward the belt for the next cube.

Success per cube: placement bonus when the cube rests inside the drum cylinder.
Missed cube penalty when a green cube falls off the belt end.  Episodes run the
full 8 s duration without early success termination — the per-step and event
rewards accumulate over the remaining steps.

## Variants

| Environment ID | Description |
|---|---|
| `Template-Tensegrity-Cube-Sort-v0` | Standard training (8 192 envs) |
| `Template-Tensegrity-Cube-Sort-Play-v0` | Evaluation (50 envs) |

Both use `TensegrityCubeSortEnv` (custom `ManagerBasedRLEnv` with event-based
rewards for placement, miss, and red-grab).

## Scene

Extends `ProjBaseSceneCfg` with a `RigidObjectCollection` of 16 labelled cubes.

| Element | Details |
|---|---|
| Robot | `TENS_5DOF_GRIPPER_CFG` at (0.15, 0.0, 2.30) m — ceiling-mounted |
| Green cubes | 8 × 0.05 m side, 0.05 kg, label 0 (target) |
| Red cubes | 8 × 0.05 m side, 0.05 kg, label 1 (distractor) |
| Target drum | Plastic drum at (0.15, ~0.85, 0.0) m (0.547 m diameter, 0.88 m tall) |
| Conveyor | Dual belt (4 m total), surface at 0.80 m, **active** (belt speed randomised) |
| Env spacing | 5.0 m |

### Cube Spawn Randomisation (at reset)

Stage 1: only 1 green cube is active, 0 red cubes (rest parked at (100, 100, 1)).

| Axis | Range |
|---|---|
| X | −1.50 … −0.30 m (CONVEYOR_START_X + 1.50 … + 2.70) |
| Y | −0.20 … +0.20 m |
| Z | belt + 0.03 … belt + 0.05 m (= 0.83 … 0.85 m) |

### Belt Speed Randomisation

| Parameter | Value |
|---|---|
| Speed range | 0.2 … 0.5 m/s (sampled per episode) |
| Direction | +X (downstream) |
| Applied to | All cubes on belt surface (height tolerance ±0.10 m) |

### Drum Success Geometry

| Parameter | Value |
|---|---|
| Cylinder radius | 0.2735 m |
| Cylinder height | 0.30 m |
| Centre | drum root position |

## Controlled Joints

| Joint | Type | Limits | Action Clip |
|---|---|---|---|
| `base_y_joint` | Prismatic | ±0.5 m | (−0.5, 0.5) |
| `base_z_joint` | Prismatic | −0.5 … 0.0 m | (−0.5, 0.0) |
| `elbow_joint` | Revolute | ±1.2217 rad | (−1.2217, 1.2217) |
| `wrist_y_joint` | Revolute | ±0.8727 rad | (−0.8727, 0.8727) |
| `wrist_x_joint` | Revolute | ±0.8727 rad | (−0.8727, 0.8727) |
| `finger_joint` | Revolute | — | Binary (open=0.0, close=0.7854) |

## Actions (6 dims)

| Group | Dims | Type | Scale |
|---|---|---|---|
| `base_delta` | 2 | Joint position delta | 0.50 |
| `arm_delta` | 3 | Joint position delta | 1.0 |
| `gripper_action` | 1 | Binary joint position | open=0.0, close=0.7854 |

## Observations (policy group)

| Term | Dim | Description |
|---|---|---|
| `joint_pos_rel` | 6 | Relative joint positions (controlled joints incl. finger) |
| `joint_vel_rel` | 6 | Relative joint velocities |
| `ee_pos_w` | 3 | Grasp-centre world position |
| `ee_vel_w` | 3 | Grasp-centre linear velocity |
| `nearest_target_rel` | 3 | Nearest green cube position relative to grasp centre |
| `nearest_distractor_rel` | 3 | Nearest red cube position relative to grasp centre |
| `fingertip_target_rel` | 3 | Dynamic fingertip → nearest green cube (closure-dependent) |
| `fingertip_distractor_rel` | 3 | Dynamic fingertip → nearest red cube (closure-dependent) |
| `gripper_closure` | 1 | Normalised finger closure [0=open, 1=closed] |
| `gripper_torque` | 1 | Normalised finger torque [0=free, 1=at effort limit] |
| `target_cube_vel` | 3 | Nearest green cube linear velocity |
| `distractor_cube_vel` | 3 | Nearest red cube linear velocity |
| `drum_rel` | 3 | Drum position relative to grasp centre |
| `belt_speed` | 1 | Current belt speed (m/s) |
| `time_remaining` | 1 | Normalised time remaining in episode |
| `placed_count` | 1 | Number of target cubes placed so far |
| `missed_count` | 1 | Number of target cubes missed so far |
| `actions` | 6 | Previous actions |

Total: **51**.

## Rewards

### Reward pipeline

```mermaid
flowchart TD
  A["1a. Reach coarse: cube_ee_distance\n(fingertip → nearest green, std=2.0)"] --> Af["1b. Reach fine\n(std=0.5)"]
  Af --> B["2. Grasp: cube_grasp_reward\n(closure × proximity, std=0.08)"]
  B --> C["3. Lift: cube_is_lifted\n(binary, velocity-gated)"]
  C --> D["3b. Height: cube_height_bonus\n(smooth gradient, max 0.30 m)"]
  C --> E["4a. Transport coarse: approach_target_tanh\n(urgency-weighted, std=1.0)"]
  L[was_grasped latch] --> E
  E --> F["4b. Transport fine\n(std=0.20)"]
  L --> F
  F --> G["5. Release: release_above_target\n(open gripper above drum)"]
  L --> G
  G --> H["6. Re-orient: reorient_to_belt\n(return to belt, std=2.0)"]
  I["Event: placement bonus\n(50 + scaling)"] --> M[Total reward]
  J["Event: all-complete bonus\n(100)"] --> M
  K["Event: cube missed penalty\n(−8)"] --> M
  P["Event: red grabbed penalty\n(−5)"] --> M
  R["Regularisation:\naction_rate / joint_vel /\nbelt_contact / joint_torque /\nbase_velocity / cube_off_conveyor"] --> M
  A & Af & B & C & D & E & F & G & H --> M
  N["Utility:\narm_utilization"] --> M
```

### Task rewards

| Term | Weight | Function | Gate |
|---|---|---|---|
| `reaching_object` | +2.0 | `exp(−d_fingertip→nearest / 2.0)` | — |
| `reaching_object_fine` | +5.0 | `exp(−d_fingertip→nearest / 0.5)` | — |
| `grasping` | +3.0 | `closure × exp(−d / 0.08)` | — |
| `lifting_object` | +5.0 | Binary: cube > belt+0.06, near gripper, closed, vel < 1 m/s | — |
| `height_bonus` | +5.0 | Smooth: `min(Δz, 0.30) / 0.30`, vel < 1 m/s | — |
| `goal_tracking` | +30.0 | `1 − tanh(d_xy / 1.0)` × urgency(α=4, β=0.5) | `was_grasped` |
| `goal_tracking_fine` | +5.0 | `1 − tanh(d_xy / 0.20)` × urgency | `was_grasped` |
| `release` | +25.0 | `(1−closure) × in_xy × above_rim` | `was_grasped` |
| `reorient` | +2.0 | `exp(−d_ee→belt / 2.0)`, open gripper only | post-place |

### Event-based rewards (in env class, not dt-scaled)

| Event | Value | Condition |
|---|---|---|
| Placement bonus | +50 + scaling × placed_count | Green cube enters drum |
| All-complete bonus | +100 | All active green cubes placed |
| Cube missed | −8 | Green cube falls off belt end |
| Red grabbed | −5 | Red cube detected as grasped |

### Regularisation & utility

| Term | Weight | Notes |
|---|---|---|
| `action_rate` | −1×10⁻⁴ → −2×10⁻³ | Ramped by curriculum at 200k steps |
| `joint_vel` | −1×10⁻⁴ → −2×10⁻³ | Ramped by curriculum at 200k steps |
| `belt_contact` | −10.0 | EE depth below belt surface |
| `joint_torque` | −0.05 | Arm joint effort fraction |
| `base_velocity` | −1.5 | Prefer arm over base movement |
| `arm_utilization` | +0.5 | Reward arm joint activity |
| `cube_off_conveyor` | −5.0 | Cubes outside belt Y ∈ [−0.4, 0.4] or below z = 0.70 |

## Terminations

No early success termination — episodes always run the full 8 s.

| Term | Type | Condition |
|---|---|---|
| `time_out` | Truncation | Episode length exceeded (8.0 s / 400 steps) |
| `all_cubes_passed` | Termination | All active cubes past belt end + 0.35 m |
| `joint_vel_diverged` | Truncation | Any controlled joint velocity > 100 rad/s |
| `belt_collision` | Truncation | EE penetrates > 0.20 m below belt surface |

## Curriculum

| Step Threshold | Change |
|---|---|
| 200 000 | `action_rate` weight: −1×10⁻⁴ → −2×10⁻³ |
| 200 000 | `joint_vel` weight: −1×10⁻⁴ → −2×10⁻³ |

No red cube curriculum yet — Stage 1 trains with green cubes only.  Future
stages will add red cubes and increase the active count.

## Reset Events

| Event | Details |
|---|---|
| `reset_all` | Full scene reset to defaults |
| `reset_arm` | Base + arm joints offset by ±0.10 rad; velocities zeroed |
| `reset_gripper` | `finger_joint` reset to 0.0 (fully open) |
| `reset_cubes` | 1 green cube in spawn box; 15 cubes parked at (100, 100, 1) |
| `sample_belt_speed` | Belt speed sampled uniformly from [0.2, 0.5] m/s |
| `apply_conveyor` | Continuous interval event — pushes on-belt cubes along +X |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Physics dt | 0.01 s (100 Hz) |
| Decimation | 2 (control at 50 Hz) |
| Episode length | 8.0 s (400 control steps) |
| PhysX solver | TGS (type 1) |
| Bounce threshold | 0.2 m/s |
| Stabilisation | Enabled |
| Friction correlation distance | 0.00625 m |
| `gpu_max_rigid_contact_count` | 2²² (4 194 304) |
| `gpu_max_rigid_patch_count` | 2²⁰ (1 048 576) |
| `gpu_collision_stack_size` | 2²⁸ (268 435 456) |
| `gpu_found_lost_aggregate_pairs_capacity` | 4 194 304 |
| `gpu_total_aggregate_pairs_capacity` | 65 536 |
| Default `num_envs` | 8 192 (train) / 50 (play) |

## Training

Training converges at approximately 250k steps (PPO via SKRL with 8 192
parallel environments).  The `skrl_ppo_cfg.yaml` is configured for 300 000
timesteps.

### PPO Hyperparameters (SKRL)

| Parameter | Value |
|---|---|
| Rollout length | 64 steps |
| Discount γ | 0.995 |
| GAE λ | 0.95 |
| Learning rate | 3 × 10⁻⁴ (KL-adaptive, threshold 0.008) |
| Epochs per update | 8 |
| Mini-batches | 4 |
| Entropy coefficient | 0.03 |
| Reward shaper scale | 0.5 |
| Min log-std | −1.0 |
| Timesteps | 300 000 |
| Checkpoint interval | 5 000 steps |
| Time-limit bootstrap | Enabled |

## Design Decisions & Lessons Learned

### No early success termination

Same rationale as the cube place task: without time to accumulate per-step
rewards after success, the agent has no incentive to release.  The 8 s fixed
episode length provides ample time for multi-cube sorting in future stages.

### Multi-resolution reaching

A single `exp(−d/0.1)` reaching reward saturated to zero for cubes beyond
~0.5 m.  Splitting into coarse (std=2.0, weight=2) + fine (std=0.5, weight=5)
provides gradient at any distance while sharpening near the target.

### Exponential vs tanh proximity

`1 − tanh(d/std)` is S-shaped and wastes gradient budget in the middle range.
`exp(−d/std)` provides stronger gradient near the target while still reaching
zero at large distances.  Changed for reach, grasp, and reorient functions.
Transport intentionally keeps `tanh` because the urgency multiplier needs the
bounded [0, 1] output.

### Transport must dominate lift + height

If lift (5) + height (5) yields more per-step reward than transport (30 × tanh),
the agent learns to hold the cube high instead of moving toward the drum.  The
transport weight must be large enough that lateral progress is always more
valuable than vertical hold.

### Spawn geometry tuning

Initial spawn range x = (−2.9, −1.2) placed cubes too far upstream — by the
time the robot reached them, they had moved to the belt end.  Narrowed to
(−1.5, −0.3) to match the robot's workspace, improving grasp rate from 0% to
84%.

### Belt speed calibration

Initial belt speed of (0.1, 0.3) m/s was too slow to create meaningful
urgency.  Increased to (0.2, 0.5) m/s — cubes now transit the reachable zone
in ~3–4 s, forcing the agent to act quickly.

### Exploration preservation

`min_log_std = −2.5` caused exploration collapse — the policy converged to a
narrow action distribution before discovering the grasp strategy.  Raising to
−1.0 with entropy coefficient 0.03 maintains sufficient exploration through
the critical learning phases.

### PhysX buffer sizing

16 cubes × 8192 envs with collection-based physics requires significantly
larger PhysX buffers than the cube place task (2 cubes × 8192 envs).  Buffer
overflows manifest as silent crashes or physics instability.

## Running

```bash
cd src/tensegrity_pick

# Standard training
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/train.py \
    --task Template-Tensegrity-Cube-Sort-v0 --headless

# Play latest checkpoint
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/play.py \
    --task Template-Tensegrity-Cube-Sort-Play-v0 --num_envs 10
```

> **Note:** Isaac Sim 5.1 RC requires `CARB_ASSERT_MODE=ignore` set before
> launching to avoid mutex crashes in `carb.events`.

## Training Results

> **⚠️ Work in Progress** — training is ongoing.  The results below are from
> the best run so far; the release phase has not yet converged.

Results from training run `2026-03-08_04-40-18` (PPO, 8 192 envs, 253k / 300k
steps).  Plots generated with `scripts/plot_sort_training_results.py`.

### Total Episode Reward

The total reward curve shows the complete learning trajectory.  The min/max
envelope reveals the spread across the environment population.  The
regularisation ramp at 200k steps is marked.  The agent reaches ~150 mean
episode return — dominated by transport and reaching rewards.

![Total Reward](figures/01_total_reward.png)

### Task Success

Grasp rate (left axis) and mean targets placed / missed (right axis) track the
key task milestones.  The agent learns to grasp within the first 30k steps and
achieves ~84% grasp rate by 250k.  However, the mean targets placed remains
near zero — the agent has not yet learned to release cubes into the drum.

![Task Success](figures/02_task_success.png)

### Sequential Skill Acquisition

Each reward component activates in sequence, revealing the learning order:
reach → grasp → lift → height → transport.  The transport reward (goal tracking)
shows the largest magnitude, confirming the design goal that lateral progress
dominates holding.  The release reward remains near zero throughout, indicating
the next training challenge.

![Reward Decomposition](figures/03_reward_decomposition.png)

### Penalties & Regularisation

Penalty evolution over training.  The cube-off-conveyor penalty increases as
the agent learns to interact with objects on the belt edge.  The action rate
and joint velocity penalties ramp up over the regularisation curriculum from
200k steps onward, smoothing the policy's motor commands.

![Penalties](figures/04_penalties.png)

### Policy Diagnostics

Policy standard deviation, surrogate loss, value loss, and learning rate
(KL-adaptive) over training.  The standard deviation decreases monotonically,
indicating growing exploitation.  The adaptive learning rate responds to
KL-divergence, dropping when the policy changes too rapidly.

![Policy Diagnostics](figures/05_policy_diagnostics.png)

### Converged Reward Breakdown

Final performance averaged over the last 10% of training.  Transport (goal
tracking) dominates as the agent reliably moves cubes toward the drum.  The
release and reorient components are near zero — the primary remaining
challenge.  The converged reward weights were rebalanced (placement 50,
completion 100, release 25) to incentivise placing; this has not yet been
fully evaluated in a training run.

![Converged Breakdown](figures/06_converged_breakdown.png)

### Regenerating Plots

```bash
cd src/tensegrity_pick
conda run --no-capture-output -n env_isaaclab python3 scripts/plot_sort_training_results.py
# Or specify a run:
conda run --no-capture-output -n env_isaaclab python3 scripts/plot_sort_training_results.py \
    --run 2026-03-08_04-40-18_ppo_torch
```

## Related

- [Robot specification](../../../../../../../../res/Tensegrity/README.md) — kinematic chain, joint constraints
- [Cube Place task](../cube_place/README.md) — single-cube pick-and-place (predecessor)
- [Extension overview](../../../../../../README.md) — all registered tasks and scripts

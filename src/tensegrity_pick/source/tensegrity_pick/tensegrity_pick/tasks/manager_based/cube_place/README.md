# Cube Place Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

Cube pick-and-place with the 5-DOF tensegrity manipulator and Robotiq 2F
gripper.  The robot must grasp a green cube from the conveyor belt, transport
it to a target drum, and release it inside — while avoiding the red distractor
cube that is introduced mid-training via curriculum.

![Task Scene](figures/scene_setup.png)

## Table of Contents

- [Goal](#goal)
- [Variants](#variants)
- [Scene](#scene)
- [Controlled Joints](#controlled-joints)
- [Actions](#actions)
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

Train an RL agent to perform a full pick → transport → place sequence:

1. **Reach** the nearest cube on the conveyor belt.
2. **Grasp** it with the Robotiq 2F gripper (closure × proximity reward).
3. **Lift** it above the belt surface (velocity-gated to reject bounces).
4. **Transport** it laterally toward the target drum.
5. **Release** the cube above the drum opening.
6. **Success** accumulates per-step reward while the cube rests in the drum.

Success is defined as the green cube's centre being inside the cylindrical
drum volume (radius = 0.2735 m, height = 0.30 m).  Episodes run the full
5 s duration without early success termination — the per-step success reward
(weight=100) accumulates over remaining steps and dominates holding rewards.

## Variants

| Environment ID | Robot | Actuation | Config | Log directory |
|---|---|---|---|---|
| `Template-Tensegrity-Cube-Place-v0` | Tensegrity 5-DOF | PD (joint pos) | `config/tensegrity/` | `logs/skrl/cube_place/` |
| `Template-Tensegrity-Cube-Place-Play-v0` | Tensegrity 5-DOF | PD (joint pos) | `config/tensegrity/` | — |
| `Template-Tensegrity-Cube-Place-Tendon-v0` | Tensegrity 5-DOF | Tendon tensions | `config/tensegrity_tendon/` | `logs/skrl/cube_place/tensegrity_tendon/` |
| `Template-Tensegrity-Cube-Place-Tendon-Play-v0` | Tensegrity 5-DOF | Tendon tensions | `config/tensegrity_tendon/` | — |
| `Template-Tensegrity-Cube-Place-Physical-Tendon-v0` | Tensegrity 5-DOF | Body-force tendons | `config/tensegrity_tendon/` | `logs/skrl/cube_place/tensegrity_tendon/` |
| `Template-Tensegrity-Cube-Place-Physical-Tendon-Play-v0` | Tensegrity 5-DOF | Body-force tendons | `config/tensegrity_tendon/` | — |
| `Template-UR10e-Cube-Place-v0` | UR10e 6-DOF | PD (joint pos) | `config/ur10e/` | `logs/skrl/cube_place/ur10e/` |
| `Template-UR10e-Cube-Place-Play-v0` | UR10e 6-DOF | PD (joint pos) | `config/ur10e/` | — |
| `Template-Kinova-Cube-Place-v0` | Kinova Gen3 7-DOF | PD (joint pos) | `config/kinova/` | `logs/skrl/cube_place/kinova/` |
| `Template-Kinova-Cube-Place-Play-v0` | Kinova Gen3 7-DOF | PD (joint pos) | `config/kinova/` | — |

All variants use `TensegrityPlaceEnv` as the gymnasium entry point
(custom `ManagerBasedRLEnv` subclass with physics-based grasp latch).

Robot-specific configurations live under `config/<robot>/`:

```
cube_place/
├── place_env_cfg.py          # Base MDP + Tensegrity PD env configs
├── place_scene_cfg.py        # Scene with cubes and drum
├── place_env.py              # TensegrityPlaceEnv (was_grasped latch)
├── mdp/                      # Rewards, observations, events
└── config/
    ├── tensegrity/           # Tensegrity PD variant
    │   ├── __init__.py       # gym.register() calls
    │   ├── joint_pos_env_cfg.py
    │   └── agents/skrl_ppo_cfg.yaml
    ├── tensegrity_tendon/    # Tensegrity tendon variant
    │   ├── __init__.py
    │   ├── joint_pos_env_cfg.py
    │   ├── joint_pos_env_cfg_physical.py
    │   └── agents/skrl_ppo_cfg.yaml
    ├── ur10e/                # UR10e 6-DOF variant
    │   ├── __init__.py
    │   ├── joint_pos_env_cfg.py
    │   └── agents/skrl_ppo_cfg.yaml
    └── kinova/               # Kinova Gen3 7-DOF variant
        ├── __init__.py
        ├── joint_pos_env_cfg.py
        └── agents/skrl_ppo_cfg.yaml
```

## Scene

Extends `ProjBaseSceneCfg` with two rigid-body cubes.

| Element | Details |
|---|---|
| Robot | `TENS_5DOF_GRIPPER_CFG` at (0.15, 0.0, 2.30) m |
| Green cube | 0.05 m side, 0.05 kg, spawned in spawn box |
| Red cube | 0.05 m side, 0.05 kg, parked until curriculum activates it |
| Target drum | Plastic drum at (0.15, 0.85, 0.0) m (0.547 m diameter, 0.88 m tall) |
| Conveyor | Dual belt (4 m total), surface at 0.80 m, **inactive** (no motion) |
| Env spacing | 5.0 m |

### Cube Spawn Randomisation (at reset)

| Axis | Range |
|---|---|
| X | 0.10 … 0.20 m (relative to env origin) |
| Y | −0.10 … +0.10 m |
| Z | belt + 0.03 … belt + 0.05 m (= 0.83 … 0.85 m) |

Only the green cube is placed in the spawn box initially.  The red cube is
parked at (100, 100, 1) and activated through curriculum after 125 000 steps.
In play mode the curriculum threshold is set to 0 so both cubes are always
visible.

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

## Actions

### PD variant (6 dims)

| Group | Dims | Type | Scale |
|---|---|---|---|
| `base_delta` | 2 | Joint position delta | 0.50 |
| `arm_delta` | 3 | Joint position delta | 1.0 |
| `gripper_action` | 1 | Binary joint position | open=0.0, close=0.7854 |

### Tendon variant (8 dims)

| Group | Dims | Type | Details |
|---|---|---|---|
| `base_delta` | 2 | Joint position delta | scale=0.50 |
| `arm_tendon` | 5 | Tendon tensions | max_tension=500 N |
| `gripper_action` | 1 | Binary joint position | open=0.0, close=0.7854 |

## Observations (policy group)

| Term | Dim | Description |
|---|---|---|
| `joint_pos_rel` | 6 | Relative joint positions (controlled joints incl. finger) |
| `joint_vel_rel` | 6 | Relative joint velocities |
| `ee_pos_w` | 3 | Grasp-centre world position |
| `ee_vel_w` | 3 | Grasp-centre linear velocity |
| `green_rel` | 3 | Green cube position relative to grasp centre |
| `red_rel` | 3 | Red cube position relative to grasp centre (zeros when parked) |
| `fingertip_green_rel` | 3 | Dynamic fingertip → green cube (closure-dependent) |
| `fingertip_red_rel` | 3 | Dynamic fingertip → red cube (closure-dependent) |
| `gripper_closure` | 1 | Normalised finger closure [0=open, 1=closed] |
| `gripper_torque` | 1 | Normalised finger torque [0=free, 1=at effort limit] |
| `green_cube_vel` | 3 | Green cube linear velocity |
| `red_cube_vel` | 3 | Red cube linear velocity |
| `drum_rel` | 3 | Drum position relative to grasp centre |
| `actions` | 6 or 8 | Previous actions |
| `was_placed` | 1 | Binary flag: cube successfully placed in drum |

Total (PD): **48**.  Total (tendon): **50**.

## Rewards

### Reward pipeline

```mermaid
flowchart TD
  A["1. Reach: object_ee_distance\n(fingertip → nearest cube)"] --> B["2. Grasp: grasp_reward\n(closure × proximity)"]
  B --> C["3. Lift: object_is_lifted\n(binary, velocity-gated)"]
  C --> D["3b. Height: cube_height_bonus\n(smooth gradient)"]
  C --> E["4. Transport: goal_tracking\n(coarse std=1.0)"]
  L[was_grasped latch] --> E
  E --> F["4b. Fine: goal_tracking_fine\n(std=0.20)"]
  L --> F
  F --> G["5. Release: release_above_target\n(open gripper above drum)"]
  L --> G
  G --> H["6. Success: green_in_target\n(per-step, weight=100)"]
  L --> H
  I[Red cube curriculum] --> J["Penalty: red_in_target"]
  K["Regularisation:\naction_rate / joint_vel /\nbelt_contact / joint_torque /\nbase_velocity / cube_off_conveyor"] --> M[Total reward]
  A & B & C & D & E & F & G & H & J --> M
  N["Utility:\narm_utilization"] --> M
```

### Task rewards

| Term | Weight | Function | Gate |
|---|---|---|---|
| `reaching_object` | +2.0 | `1 − tanh(d_fingertip→nearest / 2.0)` — coarse, long-range | `!grasp_active ∧ !was_placed` |
| `reaching_object_fine` | +5.0 | `1 − tanh(d_fingertip→nearest / 0.5)` — mid-range | `!grasp_active ∧ !was_placed` |
| `grasping` | +3.0 | `closure × (1 − tanh(d / 0.08))` | — |
| `lifting_object` | +5.0 | Binary: cube > belt+0.06, near gripper, closed, velocity < 1 m/s | — |
| `height_bonus` | +5.0 | Smooth: `min(Δz, 0.30) / 0.30`, velocity < 1 m/s | — |
| `goal_tracking` | +40.0 | `1 − tanh(d_xy_green→drum / 1.0)`, lift_threshold=0.02 | `was_grasped ∧ !was_placed` |
| `goal_tracking_fine` | +10.0 | `1 − tanh(d_xy_green→drum / 0.20)`, lift_threshold=0.02 | `was_grasped ∧ !was_placed` |
| `release` | +25.0 | `(1−closure) × in_xy × above_rim` | `was_grasped ∧ !was_placed` |
| `green_in_target` | +100.0 | Green cube inside drum cylinder | `was_grasped` |
| `return_to_neutral` | +25.0 | Joint distance to default position (inverted) | `was_placed` |
| `red_in_target` | −12.0 | Red cube inside drum (penalty) | red active |

### Regularisation & utility

| Term | Weight | Notes |
|---|---|---|
| `action_rate` | −1×10⁻⁴ → −2×10⁻³ | Ramped by curriculum at 150k steps |
| `joint_vel` | −1×10⁻⁴ → −2×10⁻³ | Ramped by curriculum at 150k steps |
| `belt_contact` | −10.0 | EE depth below belt surface |
| `joint_torque` | −0.025 | Arm joint effort fraction |
| `base_velocity` | −0.75 | Prefer arm over base movement |
| `arm_utilization` | +0.25 | Reward arm joint activity |
| `cube_off_conveyor` | −5.0 | Cubes outside belt Y∈[−0.4, 0.4] or below z=0.70 |

### Metrics (tiny weight, for TensorBoard)

| Term | Weight | Tracks | TensorBoard tag |
|---|---|---|---|
| `metric_place_success` | +0.01 | Per-step: cube inside drum (→ "Time in Drum %") | `Episode_Reward/metric_place_success` |
| `metric_grasp_rate` | +0.01 | Per-step: cube grasped and lifted | `Episode_Reward/metric_grasp_rate` |
| `metric_ee_distance` | −0.01 | EE-to-green L2 distance | `Episode_Reward/metric_ee_distance` |
| — | — | Per-episode: ≥1 grasp this episode (→ "Grasp Rate %") | `Metrics/grasp_rate` |
| — | — | Per-episode: ≥1 step with cube in drum (→ "Place Success Rate %") | `Metrics/place_success_rate` |
| — | — | Mean episode length in control steps | `Metrics/mean_episode_length` |

## Terminations

No early success termination — episodes always run the full 5 s.

| Term | Type | Condition |
|---|---|---|
| `time_out` | Truncation | Episode length exceeded (5.0 s / 250 steps) |
| `joint_vel_diverged` | Truncation | Any controlled joint velocity > 100 rad/s |
| `belt_collision` | Truncation | EE penetrates > 0.20 m below belt surface |

## Curriculum

| Step Threshold | Change |
|---|---|
| 125 000 | Red cube activated (moved from parking to spawn box) |
| 150 000 | `action_rate` weight: −1×10⁻⁴ → −2×10⁻³ |
| 150 000 | `joint_vel` weight: −1×10⁻⁴ → −2×10⁻³ |

## Reset Events

| Event | Details |
|---|---|
| `reset_all` | Full scene reset to defaults |
| `reset_arm` | Base + arm joints offset by ±0.10 rad; velocities zeroed |
| `reset_gripper` | `finger_joint` reset to 0.0 (fully open) |
| `reset_cubes` | Green cube randomised in spawn box; red parked or in spawn box (after curriculum) |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Physics dt | 0.01 s (100 Hz) |
| Decimation | 2 (control at 50 Hz) |
| Episode length | 5.0 s (250 control steps) |
| PhysX solver | TGS (type 1) |
| Bounce threshold | 0.2 m/s |
| Stabilisation | Enabled |
| Friction correlation distance | 0.00625 m |
| Default num_envs | 8 192 (PD & tendon) / 50 (play) |

## Training

Training converges at approximately 20–30k steps (PPO via SKRL with 8 192
parallel environments).  The `skrl_ppo_cfg.yaml` is configured for 300 000
timesteps.  The reward plateau is reached very early; the remaining steps
refine the policy without significant gain.

### Latest results (Tensegrity PD, 2026-04-05)

| Metric | Value |
|---|---|
| Place success rate | **90.6%** |
| Grasp rate | **98.7%** |
| Red on conveyor | **98.0%** |
| Return-to-neutral | **3.21** ep. reward |
| Total reward (final) | ≈ 333 |
| Training time | ~4.5 h @ 1024 envs |

![Total Reward](figures/tensegrity/01_total_reward.png)
![Task Success](figures/tensegrity/02_task_success.png)
![Reward Decomposition](figures/tensegrity/03_reward_decomposition.png)
![Penalties](figures/tensegrity/04_penalties.png)
![Policy Diagnostics](figures/tensegrity/05_policy_diagnostics.png)
![Converged Breakdown](figures/tensegrity/06_converged_breakdown.png)
![Episode Length](figures/tensegrity/07_episode_length.png)

## Design Decisions & Lessons Learned

### No early success termination

With success termination, the robot gets one step of `green_in_target` reward
(100 × 0.02 = 2.0) before the episode ends.  Meanwhile, holding the cube above
the drum yields ~0.8/step × remaining steps ≈ 80+ total.  **Holding dominates
release, so the robot never drops the cube.**  Removing success termination lets
the per-step success reward accumulate (2.0/step × 79 remaining ≈ 158), making
release clearly optimal.

### Transport must dominate lift + height

If lift (8) + height (8) yields more per-step reward than transport (50 × tanh),
the agent learns to hold the cube high instead of moving toward the drum.  The
transport weight must be large enough that lateral progress is always more
valuable than vertical hold.  A low lift threshold (belt + 0.02 m) keeps the
transport reward active even when the cube dips during arm extension.

### Velocity gate rejects bounces

Without velocity gating, the robot discovers an exploit: push the cube into the
belt to bounce it upward, then catch it mid-air.  The bounce satisfies
lift + height + proximity gates instantaneously.  Adding `max_velocity=1.0`
to lift and height rewards filters out bounced cubes (velocity > 1 m/s at
bounce peak) while passing genuine controlled lifts.

### Dynamic fingertip for reach reward

Using the fixed grasp-centre offset for the reach reward incentivises lateral
approach (the offset is between the finger pads at a fixed height).  Switching
to `_dynamic_finger_tip_w` — which interpolates between open-tip and closed-tip
positions based on actual gripper closure — naturally produces a top-down
approach since the open fingertips are closer to the cube from above.

### Multi-cube reach (nearest active cube)

The reach and grasp rewards target the nearest active cube rather than only the
green cube.  This ensures the agent learns to navigate around the red cube when
it appears, and the proximity gradient always pulls toward the closest object.

### Gripper torque as observation, not detection

Torque-based grasp detection was investigated but rejected: implicit actuator
torques have transient spikes during fast closure, no clean separation from
movement-induced dynamics, and Isaac Lab has no reference implementations.
Instead, `gripper_torque` is provided as an observation so the policy can learn
to correlate torque residual with grasp state, while reward gating uses the
more robust closure + proximity + lift combination.

### Red cube curriculum timing

Introducing the red distractor too early (e.g. 30k steps) causes a ~60% reward
crash that destabilises learning.  At 125k steps the green-only policy has
mastered reach→grasp→lift→transport→place and can absorb the distractor
gracefully.

### `was_grasped` latch sequencing

Isaac Lab resets terminated environments inside `super().step()` before
returning.  The `was_grasped` latch update must mask out terminated/timed-out
environments to prevent the latch from being spuriously re-set on freshly
reset states (where spawn geometry can satisfy the grasp detector).

## Running

```bash
cd src/tensegrity_pick

# Tensegrity PD training
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/train.py \
    --task Template-Tensegrity-Cube-Place-v0 --headless

# Tensegrity tendon-driven training
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/train.py \
    --task Template-Tensegrity-Cube-Place-Tendon-v0 --headless

# UR10e training
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/train.py \
    --task Template-UR10e-Cube-Place-v0 --headless

# Kinova Gen3 training
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/train.py \
    --task Template-Kinova-Cube-Place-v0 --headless

# Play latest checkpoint (replace task ID for other variants)
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/play.py \
    --task Template-Tensegrity-Cube-Place-Play-v0 --num_envs 10
```

## Training Results

Results from training run `2026-04-05_10-54-14` (PPO, 1024 envs, 300k steps), Tensegrity PD variant.
Plots generated with `scripts/plot_place_training_results.py --variant tensegrity`.

### Key Performance Numbers

| Metric | Converged Value | Notes |
|---|---|---|
| **Place Success Rate** | **90.6 %** | % of episodes where cube entered drum ≥1 step (at 300k) |
| **Grasp Rate** | **98.7 %** | % of episodes with ≥1 successful grasp-and-lift |
| **Red on Conveyor** | **98.0 %** | % of episodes where red cube stays on belt |
| **Return-to-Neutral** | **3.21** ep. reward | Arm retracts safely after placement |
| **Reward Plateau** | **333** | Mean total episode return at convergence |
| **Spawn Width** | **±0.30 m** | 75% of conveyor width (via curriculum) |
| **Training Duration** | **300k steps** | ~4.5 h wall-clock with 1024 envs |

### Total Episode Reward

The total reward curve shows the complete learning trajectory.  The min/max
envelope reveals the spread across the environment population.  The spawn
curriculum widens to ±0.30 by 100k steps, and the red distractor cube is
introduced at 125k steps.  The regularisation ramp also begins at 150k steps.
The reward climbs to a plateau of **≈ 333** at convergence.

![Total Reward](figures/tensegrity/01_total_reward.png)

### Task Success

All three success metrics on a common percentage axis:

- **Grasp Rate** — fraction of episodes with at least one successful grasp-and-lift event; reaches ~99 % by 75k steps.
- **Place Success Rate** — fraction of episodes where the cube entered the drum at least once; climbs from 0% to ~90% between 100k–250k steps.
- **Red on Conveyor** — fraction of episodes where the red cube stays on the belt; remains >98% throughout training.

The red cube introduction at 125k steps causes a brief perturbation as the agent adapts to the distractor.

![Task Success](figures/tensegrity/02_task_success.png)

### Sequential Skill Acquisition

Each reward component activates in sequence, revealing the learning order:
reach → grasp → lift → transport → release → success → return.  The transport
reward (goal tracking) shows the largest magnitude during the approach phase.
The success reward (green in target) rises steeply once the agent masters
the full pipeline.  After placement, the return-to-neutral reward activates
to guide the arm safely away from the drum.

![Reward Decomposition](figures/tensegrity/03_reward_decomposition.png)

### Penalties & Regularisation

Penalty evolution over training.  The cube-off-conveyor penalty increases
as the agent learns more aggressive manipulation.  Base velocity penalties
remain stable.  The action rate and joint velocity penalties ramp up over
the regularisation curriculum from 150k steps onward, smoothing the policy's
motor commands.

![Penalties](figures/tensegrity/04_penalties.png)

### Policy Diagnostics

Policy standard deviation, surrogate loss, value loss, and learning rate
(KL-adaptive) over training.  The standard deviation decreases monotonically,
indicating growing exploitation.  The value loss spikes briefly when the red
cube curriculum activates (new dynamics to model) then settles.  The adaptive
learning rate responds to KL-divergence, dropping when the policy changes too
rapidly.

![Policy Diagnostics](figures/tensegrity/05_policy_diagnostics.png)

### Converged Reward Breakdown

Final performance averaged over the last 10% of training.  The success reward
(green in target) dominates as intended, confirming that release-and-accumulate
is the optimal strategy.  The return-to-neutral reward is the second-largest
positive component, showing the arm actively retracts after placement.  The
cube-off-conveyor penalty is the largest negative component.

![Converged Breakdown](figures/tensegrity/06_converged_breakdown.png)

### Average Episode Length

Mean control steps per episode over training.  Early in training, frequent
early terminations (joint velocity divergence, belt collisions) keep episodes
short (~80–115 steps).  As the policy stabilises the episode length converges
to **≈ 248 steps** — effectively full 250-step episodes — indicating that
early-termination events become negligible at convergence.

![Episode Length](figures/tensegrity/07_episode_length.png)

---

## Training Results — Tensegrity Tendon (J^T)

Results from training run `2026-04-05_15-32-10` (PPO, 1024 envs, 300k steps).
Plots generated with `scripts/plot_place_training_results.py --variant tensegrity_tendon`.

### Key Performance Numbers (Tendon)

| Metric | Converged Value | vs PD |
|---|---|---|
| **Place Success Rate** | **93.4 %** | +2.8 % |
| **Grasp Rate** | **94.7 %** | −4.0 % |
| **Red on Conveyor** | **95.2 %** | −2.8 % |
| **Return-to-Neutral** | **3.27** ep. reward | Similar |
| **Reward Plateau** | **333** | Similar |
| **Training Duration** | **300k steps** | ~6.9 h wall-clock |

The tendon variant replaces the arm's 3 implicit PD joint drives with 5 tendon
efforts via a constant Jacobian-transpose mapping.  Despite the more complex
action space (8 vs 6 dims), the tendon model converges faster than PD and
achieves higher place success.  Place success reached 92.6% by 73k steps —
before the red cube curriculum activated.

![Total Reward](figures/tensegrity_tendon/01_total_reward.png)
![Task Success](figures/tensegrity_tendon/02_task_success.png)
![Reward Decomposition](figures/tensegrity_tendon/03_reward_decomposition.png)
![Penalties](figures/tensegrity_tendon/04_penalties.png)
![Policy Diagnostics](figures/tensegrity_tendon/05_policy_diagnostics.png)
![Converged Breakdown](figures/tensegrity_tendon/06_converged_breakdown.png)
![Episode Length](figures/tensegrity_tendon/07_episode_length.png)

---

## Training Results — Tensegrity Physical Tendon (body-force)

Results from training run `2026-04-05_22-43-20` (PPO, 1024 envs, 300k steps).
Plots generated with `scripts/plot_place_training_results.py --variant tensegrity_physical_tendon`.

### Key Performance Numbers (Physical Tendon)

| Metric | Final Value | Notes |
|---|---|---|
| **Place Success Rate** | **0.0 %** | Did not learn placement |
| **Grasp Rate** | **0.5 %** | Did not learn reliable grasping |
| **Total Reward** | **58.9** | Reaching rewards only |
| **Training Duration** | **300k steps** | ~4.5 h wall-clock |

The physical tendon variant replaces the single elbow joint with a four-bar
antiparallelogram linkage driven by body-force cable tendons at the physical
attachment points.  The reaching phase learned normally (arm approaches cube),
but the policy could not bridge the gap from reaching to grasping within 300k
steps.  The much more nonlinear cable-force dynamics make fine motor control
significantly harder than the J^T tendon model.

![Total Reward](figures/tensegrity_physical_tendon/01_total_reward.png)
![Task Success](figures/tensegrity_physical_tendon/02_task_success.png)
![Reward Decomposition](figures/tensegrity_physical_tendon/03_reward_decomposition.png)
![Penalties](figures/tensegrity_physical_tendon/04_penalties.png)
![Policy Diagnostics](figures/tensegrity_physical_tendon/05_policy_diagnostics.png)
![Converged Breakdown](figures/tensegrity_physical_tendon/06_converged_breakdown.png)
![Episode Length](figures/tensegrity_physical_tendon/07_episode_length.png)

---

### Variant Comparison Summary

| Metric | PD | Tendon (J^T) | Physical Tendon |
|---|---|---|---|
| **Place success** | 90.6% | **93.4%** | 0.0% |
| **Grasp rate** | **98.7%** | 94.7% | 0.5% |
| **Red on conveyor** | **98.0%** | 95.2% | 97.8% |
| **Return-to-neutral** | 3.21 | 3.27 | 0.00 |
| **Total reward** | 333 | 333 | 59 |

### Regenerating Plots

```bash
cd src/tensegrity_pick
# Tensegrity PD variant (default)
conda run --no-capture-output -n env_isaaclab python3 scripts/plot_place_training_results.py \
    --variant tensegrity
# Or specify a run:
conda run --no-capture-output -n env_isaaclab python3 scripts/plot_place_training_results.py \
    --variant tensegrity --run 2026-04-05_10-54-14_ppo_torch

# Other variants
conda run --no-capture-output -n env_isaaclab python3 scripts/plot_place_training_results.py \
    --variant tensegrity_tendon
conda run --no-capture-output -n env_isaaclab python3 scripts/plot_place_training_results.py \
    --variant tensegrity_physical_tendon
conda run --no-capture-output -n env_isaaclab python3 scripts/plot_place_training_results.py \
    --variant ur10e
conda run --no-capture-output -n env_isaaclab python3 scripts/plot_place_training_results.py \
    --variant kinova
```

Figures are saved to `figures/<variant>/` (e.g., `figures/tensegrity/01_total_reward.png`).
Seven figures are generated per run: 01–06 (reward/policy diagnostics) plus 07 (episode length).

## Related

- [GEOMETRY.md](GEOMETRY.md) — empirical gripper geometry measurements
- [Robot specification](../../../../../../../../res/Tensegrity/README.md) — kinematic chain, joint constraints, tendon geometry
- [Reach task](../reach/README.md) — EE pose tracking task
- [Extension overview](../../../../../../README.md) — all registered tasks and scripts

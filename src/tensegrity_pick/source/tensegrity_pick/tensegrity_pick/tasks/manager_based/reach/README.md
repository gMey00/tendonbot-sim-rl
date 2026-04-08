# Reach Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

End-effector pose tracking for all active robot configurations based on the IsaacLab sample task.  The robot
must move `TARGET_LINK` to randomly sampled target poses that are guaranteed
reachable by construction.

![Task Scene](figures/scene_setup.png)

## Table of Contents

- [Reach Task](#reach-task)
  - [Table of Contents](#table-of-contents)
  - [Goal](#goal)
  - [Variants](#variants)
  - [Directory Structure](#directory-structure)
  - [Scene](#scene)
  - [Controlled Joints](#controlled-joints)
    - [Tensegrity / Tensegrity Tendon (5 DOF)](#tensegrity--tensegrity-tendon-5-dof)
    - [Tensegrity Physical Tendon (7 DOF)](#tensegrity-physical-tendon-7-dof)
    - [UR10e (6 DOF)](#ur10e-6-dof)
    - [Kinova Gen3 (7 DOF)](#kinova-gen3-7-dof)
  - [Actions](#actions)
    - [PD variants](#pd-variants)
    - [Tendon variant](#tendon-variant)
    - [Physical Tendon variant](#physical-tendon-variant)
  - [Observations (policy group)](#observations-policy-group)
  - [Command Generator — FK-Sampled Pose](#command-generator--fk-sampled-pose)
  - [Rewards](#rewards)
    - [Task rewards](#task-rewards)
    - [Success metrics (logging only)](#success-metrics-logging-only)
    - [Regularisation](#regularisation)
  - [Terminations](#terminations)
  - [Curriculum](#curriculum)
  - [Reset Events](#reset-events)
  - [Simulation Parameters](#simulation-parameters)
  - [PPO Hyperparameters](#ppo-hyperparameters)
  - [Training](#training)
  - [Running](#running)
  - [Training Results](#training-results)
    - [Training Figures](#training-figures)
    - [Regenerating Plots and Reports](#regenerating-plots-and-reports)
  - [Related](#related)

## Goal

Teach an RL agent to track arbitrary end-effector poses (position **and**
orientation) as quickly and smoothly as possible.  Targets are produced by a
custom **FK-Sampled Pose Command** generator that samples random joint
positions within the robot's limits, computes forward kinematics through the
physics engine, and stores the resulting pose as the command.  This guarantees
every target is reachable.

## Variants

| Environment ID | Robot | DOF | Actuation | Config |
|---|---|---|---|---|
| `Template-Reach-Tensegrity-v0` | Tensegrity 5-DOF | 5 | PD (joint position) | `config/tensegrity/` |
| `Template-Reach-Tensegrity-Play-v0` | Tensegrity 5-DOF | 5 | PD (eval) | `config/tensegrity/` |
| `Template-Reach-Tensegrity-Tendon-v0` | Tensegrity 5-DOF | 5 | Tendon | `config/tensegrity_tendon/` |
| `Template-Reach-Tensegrity-Tendon-Play-v0` | Tensegrity 5-DOF | 5 | Tendon (eval) | `config/tensegrity_tendon/` |
| `Template-Reach-Tensegrity-Physical-Tendon-v0` | Tensegrity 5-DOF (physical) | 7 | Physical Tendon | `config/tensegrity_tendon/` |
| `Template-Reach-Tensegrity-Physical-Tendon-Play-v0` | Tensegrity 5-DOF (physical) | 7 | Physical Tendon (eval) | `config/tensegrity_tendon/` |
| `Template-Reach-UR10e-v0` | UR10e + Robotiq 2F-140 | 6 | PD | `config/ur10e/` |
| `Template-Reach-UR10e-Play-v0` | UR10e + Robotiq 2F-140 | 6 | PD (eval) | `config/ur10e/` |
| `Template-Reach-Kinova-v0` | Kinova Gen3 + Robotiq 2F-140 | 7 | PD | `config/kinova/` |
| `Template-Reach-Kinova-Play-v0` | Kinova Gen3 + Robotiq 2F-140 | 7 | PD (eval) | `config/kinova/` |

All variants use the base `ManagerBasedRLEnv` as the gymnasium entry point
(no custom env subclass needed for reach).

### Variant Specifications

Detailed hardware, actuator, observation/action space, and training
configuration for each robot:

- [Physical Elbow Tensegrity](config/tensegrity_tendon/physical_elbow_spec.md)
- [Kinova Gen3](config/kinova/kinova_spec.md)
- [UR10e](config/ur10e/ur10e_spec.md)

## Directory Structure

```
reach/
├── reach_env_cfg.py              # Base MDP config (shared by all variants)
├── mdp/
│   ├── fk_sampled_pose_command.py
│   └── rewards.py
├── config/
│   ├── tensegrity/
│   │   ├── joint_pos_env_cfg.py  # TensegrityReachEnvCfg
│   │   └── agents/skrl_ppo_cfg.yaml
│   ├── tensegrity_tendon/
│   │   ├── joint_pos_env_cfg.py          # TensegrityReachTendonEnvCfg
│   │   ├── joint_pos_env_cfg_physical.py # TensegrityReachPhysicalTendonEnvCfg
│   │   └── agents/
│   │       ├── skrl_ppo_cfg.yaml
│   │       └── skrl_ppo_cfg_physical.yaml
│   ├── ur10e/
│   │   ├── joint_pos_env_cfg.py  # UR10eReachEnvCfg
│   │   └── agents/skrl_ppo_cfg.yaml
│   └── kinova/
│       ├── joint_pos_env_cfg.py  # KinovaReachEnvCfg
│       └── agents/skrl_ppo_cfg.yaml
├── figures/
│   ├── scene_setup.png
│   ├── tensegrity/               # Training plots for tensegrity PD
│   ├── tensegrity_tendon/              # Training plots for tensegrity tendon
│   ├── tensegrity_physical_tendon/     # Training plots for tensegrity physical tendon
│   ├── ur10e/                          # Training plots for UR10e
│   └── kinova/                   # Training plots for Kinova
└── reports/
    ├── tensegrity/
    ├── tensegrity_tendon/
    ├── tensegrity_physical_tendon/
    ├── ur10e/
    └── kinova/
```

## Scene

Inherited from `ProjBaseSceneCfg` (ground plane, dome light, dual conveyor
belts, target drum).

| Element | Details |
|---|---|
| Env spacing | 5.0 m |
| Conveyor | Dual belt (4 m total), surface at 0.80 m, **inactive** |
| Target drum | Plastic drum at (0.15, 0.85, 0.0) m |

Robot mounting positions vary per variant (see `proj_base_scene_cfg.py`).

## Controlled Joints

### Tensegrity / Tensegrity Tendon (5 DOF)

| Joint | Type |
|---|---|
| `base_y_joint` | Prismatic |
| `base_z_joint` | Prismatic |
| `elbow_joint` | Revolute |
| `wrist_y_joint` | Revolute |
| `wrist_x_joint` | Revolute |

Target link: `tool_link_0`

### Tensegrity Physical Tendon (7 DOF)

Uses a 4-bar antiparallelogram linkage instead of a single `elbow_joint`.
`coupler_right_joint` is excluded from the articulation tree (loop-closure
constraint).

| Joint | Type |
|---|---|
| `base_y_joint` | Prismatic |
| `base_z_joint` | Prismatic |
| `rod_left_joint` | Revolute |
| `rod_right_joint` | Revolute |
| `coupler_left_joint` | Revolute |
| `wrist_y_joint` | Revolute |
| `wrist_x_joint` | Revolute |

Target link: `tool_link_0`

### UR10e (6 DOF)

| Joint |
|---|
| `shoulder_pan_joint` |
| `shoulder_lift_joint` |
| `elbow_joint` |
| `wrist_1_joint` |
| `wrist_2_joint` |
| `wrist_3_joint` |

Target link: `robotiq_base_link`

### Kinova Gen3 (7 DOF)

| Joint |
|---|
| `joint_1` … `joint_7` |

Target link: `end_effector_link`

The gripper is **not** controlled in the reach task.

## Actions

### PD variants

| Variant | Term | Dims | Type | Scale |
|---|---|---|---|---|
| Tensegrity | `arm_action` | 5 | Joint position delta | 0.5 |
| UR10e | `arm_action` | 6 | Joint position delta | 0.125 |
| Kinova | `arm_action` | 7 | Joint position delta | 0.125 |

### Tendon variant

| Term | Dims | Type | Details |
|---|---|---|---|
| `base_action` | 2 | Joint position delta | `base_y_joint`, `base_z_joint`, scale=0.5 |
| `arm_tendon` | 5 | Tendon tensions | max_tension=500 N, Jacobian transpose from URDF |

### Physical Tendon variant

| Term | Dims | Type | Details |
|---|---|---|---|
| `base_action` | 2 | Joint position to limits | `base_y_joint`, `base_z_joint`, maps [-1, 1] → joint limits |
| `arm_tendon` | 5 | Physical tendon efforts | max_tension=500 N, body-force tendons at physical attachment points |

Elbow actuation uses 2 body-force tendons applied at physical cable
attachment points on `root_link` and `forearm_link`.  Wrist actuation uses
3 tendons mapped via a constant Jacobian transpose.

All PD actions use `use_default_offset=True`.

## Observations (policy group)

All observation terms are concatenated into a single vector.

| Term | Dim | Description |
|---|---|---|
| `joint_pos` | N | Relative joint positions (±0.01 uniform noise) |
| `joint_vel` | N | Relative joint velocities (±0.01 uniform noise) |
| `pose_command` | 7 | FK-sampled target pose (x, y, z, qw, qx, qy, qz) in root frame |
| `actions` | M | Previous actions |

Where N = DOF count and M = action dimension for that variant.

| Variant | N | M | Total |
|---|---|---|---|
| Tensegrity PD | 5 | 5 | 22 |
| Tensegrity Tendon | 5 | 7 | 24 |
| Tensegrity Physical Tendon | 7 | 7 | 28 |
| UR10e | 6 | 6 | 25 |
| Kinova | 7 | 7 | 28 |

`enable_corruption = True` during training, disabled during play.

## Command Generator — FK-Sampled Pose

| Parameter | Value |
|---|---|
| Resampling interval | 4.0 s (fixed) |
| Success threshold | 0.05 m (position error) |
| Sampling method | Uniform random in `[joint_lower, joint_upper]` → FK |
| Debug visualisation | Frame markers for goal + current EE pose |

## Rewards

### Task rewards

| Term | Weight | Function |
|---|---|---|
| `end_effector_position_tracking` | −0.2 | L2 position error (unbounded penalty) |
| `end_effector_position_tracking_fine_grained` | +0.1 | `1 − tanh(d / 0.1)` — medium-range dense reward |
| `end_effector_orientation_tracking` | −0.1 | Quaternion error magnitude |

### Success metrics (logging only)

These terms have near-zero weight (1×10⁻⁶) so they appear in TensorBoard
without affecting the reward signal.

| Term | Threshold | Function |
|---|---|---|
| `position_reached` | 0.02 m (2 cm) | Binary 1.0 when position error < threshold |
| `orientation_reached` | 0.1 rad (5.7°) | Binary 1.0 when orientation error < threshold |
| `pose_reached` | both | Binary 1.0 when both position **and** orientation thresholds are met |

### Regularisation

| Term | Initial Weight | Final Weight | Notes |
|---|---|---|---|
| `action_rate` | −0.0001 | −0.005 | L2 action delta (curriculum ramp over 4 500 steps) |
| `joint_vel` | −0.0001 | −0.001 | L2 joint velocity (curriculum ramp over 4 500 steps) |

## Terminations

| Term | Type | Condition |
|---|---|---|
| `time_out` | Truncation | Episode length exceeded (12.0 s / 360 steps) |
| `joint_vel_diverged` | Truncation | Any joint velocity exceeds 100 rad/s (physics divergence guard) |

## Curriculum

| Step Threshold | Change |
|---|---|
| 0 → 4 500 | `action_rate` weight ramps from −0.0001 to −0.005 |
| 0 → 4 500 | `joint_vel` weight ramps from −0.0001 to −0.001 |

## Reset Events

| Event | Details |
|---|---|
| `reset_robot_joints` | Controlled joints scaled to 75 %–125 % of defaults; velocities zeroed |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Physics dt | 1/60 s ≈ 16.67 ms |
| Decimation | 2 (control at 30 Hz) |
| Episode length | 12.0 s (360 control steps) |
| Default num_envs | 4 096 (train) / 50 (play) |

## PPO Hyperparameters

All variants share the same PPO configuration (see `config/<variant>/agents/skrl_ppo_cfg.yaml`):

| Parameter | Value | Notes |
|---|---|---|
| `models.separate` | `False` | Shared policy / value backbone |
| `policy.min_log_std` | `−20.0` | Effectively unclamped exploration |
| `policy.initial_log_std` | `0.0` | Start with std ≈ 1 |
| `policy.layers` | `[64, 64]` | Compact network, ELU activations |
| `value.layers` | `[64, 64]` | Matches policy network size |
| `rollouts` | `24` | Rollout horizon per update |
| `learning_epochs` | `5` | Gradient steps per rollout |
| `mini_batches` | `4` | Mini-batch splits per epoch |
| `learning_rate` | `1.0e-03` | KL-adaptive scheduler (threshold 0.01) |
| `discount_factor` | `0.99` | — |
| `lambda` (GAE) | `0.95` | — |
| `write_interval` | `auto` | Eliminates TensorBoard I/O bottleneck |
| `timesteps` | 48 000 (tensegrity, tendon, physical tendon) / 96 000 (UR10e, Kinova) | Industrial arms need longer training |

## Training

Training uses SKRL PPO with 4 096 parallel environments.
See `config/<variant>/agents/skrl_ppo_cfg.yaml` for the full hyperparameter set.

Use `train_reach.sh` to train one or more variants in sequence and
auto-generate plots and reports:

```bash
cd src/tensegrity_pick

# All variants
./scripts/train_reach.sh

# Specific variants
./scripts/train_reach.sh tensegrity ur10e

# Plot / report only (no training)
./scripts/train_reach.sh --skip-train
```

## Running

```bash
cd src/tensegrity_pick

# Training (replace <VARIANT_ID> with an ID from the Variants table)
conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/train.py \
    --task Template-Reach-Tensegrity-v0 --headless

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/train.py \
    --task Template-Reach-Tensegrity-Tendon-v0 --headless

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/train.py \
    --task Template-Reach-Tensegrity-Physical-Tendon-v0 --headless

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/train.py \
    --task Template-Reach-UR10e-v0 --headless

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/train.py \
    --task Template-Reach-Kinova-v0 --headless

# Play latest checkpoint
conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/play.py \
    --task Template-Reach-Tensegrity-Play-v0 --num_envs 10
```

## Training Results

| Variant | Steps | Total Reward | Pos. Error | Orient. Error | Fine-Grained | Report | Date |
|---|---|---|---|---|---|---|---|
| Tensegrity PD | 48k | +0.77 | −0.006 | −0.009 | 0.087 | [Report](reports/tensegrity/reach_results_tensegrity_2026-04-06_17-53-45_ppo_torch.md) | 2026-04-06 |
| Tensegrity Tendon | 48k | +0.75 | −0.007 | −0.012 | 0.085 | [Report](reports/tensegrity_tendon/reach_results_tensegrity_tendon_2026-04-06_18-19-18_ppo_torch.md) | 2026-04-06 |
| Tensegrity Physical Tendon | 48k | −1.64 | −0.055 | −0.092 | 0.025 | [Report](reports/tensegrity_physical_tendon/reach_results_tensegrity_physical_tendon_2026-04-06_18-44-32_ppo_torch.md) | 2026-04-06 |
| UR10e | 96k | −0.21 | −0.051 | −0.021 | 0.056 | [Report](reports/ur10e/reach_results_ur10e_2026-03-30_07-37-24_ppo_torch.md) | 2026-03-30 |
| Kinova | 96k | +0.62 | −0.014 | −0.019 | 0.088 | [Report](reports/kinova/reach_results_kinova_2026-03-30_06-29-39_ppo_torch.md) | 2026-03-30 |

**Note:** Tensegrity PD and Tendon both improved after Iteration 11 (penetrable
scene, margin 0.01, FK coupling).  The Physical Tendon variant regressed
significantly — the combination of antiparallelogram coupling enforcement and
margin reduction from 0.25 → 0.01 made the goal distribution much harder for
this variant.  Further investigation needed (longer training, intermediate
margin, etc.).

### Training Figures

| Variant | Figures |
|---|---|
| Tensegrity PD | [figures/tensegrity/](figures/tensegrity/) |
| Tensegrity Tendon | [figures/tensegrity_tendon/](figures/tensegrity_tendon/) |
| Tensegrity Physical Tendon | [figures/tensegrity_physical_tendon/](figures/tensegrity_physical_tendon/) |
| UR10e | [figures/ur10e/](figures/ur10e/) |
| Kinova | [figures/kinova/](figures/kinova/) |

### Regenerating Plots and Reports

```bash
cd src/tensegrity_pick

# All variants (uses latest run per variant automatically)
conda run --no-capture-output -n env_isaaclab \
    python3 scripts/plot_reach_training_results.py --variant tensegrity

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/plot_reach_training_results.py --variant tensegrity_tendon

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/plot_reach_training_results.py --variant tensegrity_physical_tendon

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/plot_reach_training_results.py --variant ur10e

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/plot_reach_training_results.py --variant kinova

# Or regenerate all at once via the pipeline script
./scripts/train_reach.sh --skip-train
```

## Related

- [Robot specification](../../../../../../../../res/Tensegrity/README.md) — kinematic chain, joint constraints, tendon geometry
- [Tendon simulation](../../../../../../../../doc/tendon_simulation.md) — physics model and validation
- [Cube place task](../cube_place/README.md) — cube pick-and-place task
- [Cube sort task](../cube_sort/README.md) — cube sorting on active conveyor
- [Extension overview](../../../../../../README.md) — all registered tasks and scripts

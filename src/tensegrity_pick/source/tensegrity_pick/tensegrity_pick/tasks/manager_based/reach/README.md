# Reach Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

End-effector pose tracking for all active robot configurations based on the IsaacLab sample task.  The robot
must move `TARGET_LINK` to randomly sampled target poses that are guaranteed
reachable by construction.

![Task Scene](figures/scene_setup.png)

## Table of Contents

- [Goal](#goal)
- [Variants](#variants)
- [Directory Structure](#directory-structure)
- [Scene](#scene)
- [Controlled Joints](#controlled-joints)
- [Actions](#actions)
- [Observations (policy group)](#observations-policy-group)
- [Command Generator — FK-Sampled Pose](#command-generator--fk-sampled-pose)
- [Rewards](#rewards)
- [Terminations](#terminations)
- [Curriculum](#curriculum)
- [Reset Events](#reset-events)
- [Simulation Parameters](#simulation-parameters)
- [Training](#training)
- [Running](#running)
- [Training Results](#training-results)
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
| `Template-Reach-UR10e-v0` | UR10e + Robotiq 2F-140 | 6 | PD | `config/ur10e/` |
| `Template-Reach-UR10e-Play-v0` | UR10e + Robotiq 2F-140 | 6 | PD (eval) | `config/ur10e/` |
| `Template-Reach-Kinova-v0` | Kinova Gen3 + Robotiq 2F-140 | 7 | PD | `config/kinova/` |
| `Template-Reach-Kinova-Play-v0` | Kinova Gen3 + Robotiq 2F-140 | 7 | PD (eval) | `config/kinova/` |

All variants use the base `ManagerBasedRLEnv` as the gymnasium entry point
(no custom env subclass needed for reach).

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
│   │   ├── joint_pos_env_cfg.py  # TensegrityReachTendonEnvCfg
│   │   └── agents/skrl_ppo_cfg.yaml
│   ├── ur10e/
│   │   ├── joint_pos_env_cfg.py  # UR10eReachEnvCfg
│   │   └── agents/skrl_ppo_cfg.yaml
│   └── kinova/
│       ├── joint_pos_env_cfg.py  # KinovaReachEnvCfg
│       └── agents/skrl_ppo_cfg.yaml
├── figures/
│   ├── scene_setup.png
│   ├── tensegrity/               # Training plots for tensegrity PD
│   ├── tensegrity_tendon/        # Training plots for tensegrity tendon
│   ├── ur10e/                    # Training plots for UR10e
│   └── kinova/                   # Training plots for Kinova
└── reports/
    ├── tensegrity/
    ├── tensegrity_tendon/
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

All PD actions use `use_default_offset=True`.

## Observations (policy group)

All observation terms are concatenated into a single vector.

| Term | Dim | Description |
|---|---|---|
| `joint_pos` | N | Relative joint positions (±0.002 uniform noise) |
| `joint_vel` | N | Relative joint velocities (±0.002 uniform noise) |
| `pose_command` | 7 | FK-sampled target pose (x, y, z, qw, qx, qy, qz) in root frame |
| `actions` | M | Previous actions |

Where N = DOF count and M = action dimension for that variant.

| Variant | N | M | Total |
|---|---|---|---|
| Tensegrity PD | 5 | 5 | 22 |
| Tensegrity Tendon | 5 | 7 | 24 |
| UR10e | 6 | 6 | 25 |
| Kinova | 7 | 7 | 28 |

`enable_corruption = True` during training, disabled during play.

## Command Generator — FK-Sampled Pose

| Parameter | Value |
|---|---|
| Resampling interval | 2.0 s (fixed) |
| Success threshold | 0.02 m (position error) |
| Sampling method | Uniform random in `[joint_lower, joint_upper]` → FK |
| Debug visualisation | Frame markers for goal + current EE pose |

## Rewards

### Task rewards

| Term | Weight | Function |
|---|---|---|
| `end_effector_position_tracking` | −0.2 | L2 position error |
| `end_effector_position_tracking_fine_grained` | +0.1 | `1 − tanh(d / 0.1)` |
| `end_effector_orientation_tracking` | −0.1 | Quaternion error magnitude |
| `end_effector_orientation_tracking_fine_grained` | 0.0 *(disabled)* | `1 − tanh(e / 0.2)` |
| `pose_goal_reached` | 0.0 *(disabled)* | Binary: 1.0 when `d < 0.03` m |

### Regularisation

| Term | Initial Weight | Final Weight | Notes |
|---|---|---|---|
| `action_rate` | −0.0001 | −0.005 | L2 action delta (curriculum ramp) |
| `joint_vel` | −0.0001 | −0.001 | L2 joint velocity (curriculum ramp) |

## Terminations

| Term | Type | Condition |
|---|---|---|
| `time_out` | Truncation | Episode length exceeded (12.0 s / 360 steps) |

## Curriculum

| Step Threshold | Change |
|---|---|
| 0 → 4 500 | `action_rate` weight ramps from −0.0001 to −0.005 |
| 0 → 4 500 | `joint_vel` weight ramps from −0.0001 to −0.001 |

## Reset Events

| Event | Details |
|---|---|
| `reset_robot_joints` | Controlled joints scaled to 50 %–150 % of defaults; velocities zeroed |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Physics dt | 1/60 s ≈ 16.67 ms |
| Decimation | 2 (control at 30 Hz) |
| Episode length | 12.0 s (360 control steps) |
| Default num_envs | 4 096 (train) / 50 (play) |

## Training

Training is configured for 24 000 timesteps (PPO via SKRL, 4 096 parallel
environments).  See `config/<variant>/agents/skrl_ppo_cfg.yaml` for the full
hyperparameter set.

Use `train_reach.sh` to train one or more variants in sequence and
auto-generate plots and reports:

```bash
cd src/tensegrity_pick

# All 4 variants
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

| Variant | Latest Report | Date |
|---|---|---|
| Tensegrity PD | [Report](reports/tensegrity/reach_results_tensegrity_2026-03-14_23-42-14_ppo_torch.md) | 2026-03-15 |
| Tensegrity Tendon | [Report](reports/tensegrity_tendon/reach_results_tensegrity_tendon_2026-03-15_00-14-43_ppo_torch.md) | 2026-03-15 |
| UR10e | [Report](reports/ur10e/reach_results_ur10e_2026-03-13_23-47-59_ppo_torch.md) | 2026-03-15 |
| Kinova | [Report](reports/kinova/reach_results_kinova_2026-03-14_00-26-45_ppo_torch.md) | 2026-03-15 |

### Training Figures

| Variant | Figures |
|---|---|
| Tensegrity PD | [figures/tensegrity/](figures/tensegrity/) |
| Tensegrity Tendon | [figures/tensegrity_tendon/](figures/tensegrity_tendon/) |
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

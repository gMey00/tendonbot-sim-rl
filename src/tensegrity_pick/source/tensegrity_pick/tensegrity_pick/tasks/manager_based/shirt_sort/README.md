# Shirt Sort Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

Shirt sorting with the 5-DOF tensegrity manipulator and Robotiq 2F gripper on
an active conveyor belt.  The robot must selectively grasp target shirts (e.g.
clean) from the moving belt, transport them to a target drum, and release them
inside — while ignoring distractor shirts (e.g. dirty) that share the same
belt.

> **Status: Template** — cloth simulation integration is pending.  The scene
> and environment configurations contain placeholder definitions that will be
> completed once deformable object support is integrated.

![Task Scene](figures/scene_setup.png)

## Table of Contents

- [Goal](#goal)
- [Variants](#variants)
- [Scene](#scene)
- [Controlled Joints](#controlled-joints)
- [Actions (6 dims)](#actions-6-dims)
- [Observations](#observations)
- [Rewards](#rewards)
- [Terminations](#terminations)
- [Simulation Parameters](#simulation-parameters)
- [Training](#training)
- [Running](#running)
- [Related](#related)

## Goal

Train an RL agent to perform label-based sorting of deformable cloth objects
on a moving conveyor:

1. **Reach** the nearest target shirt on the moving belt.
2. **Grasp** it with the Robotiq 2F gripper.
3. **Lift** it above the belt surface.
4. **Transport** it laterally toward the target drum.
5. **Release** the shirt above the drum opening.
6. **Success** accumulates per-step reward while the shirt rests in the drum.

## Variants

| Environment ID | Description |
|---|---|
| `Template-Tensegrity-Shirt-Sort-v0` | Standard training (4 096 envs) |

## Scene

Extends `ProjBaseSceneCfg` with pools of cloth-simulated T-shirts on a moving
conveyor.

| Element | Details |
|---|---|
| Robot | `TENS_5DOF_GRIPPER_CFG` at (0.15, 0.0, 2.30) m |
| Shirts | TODO: DeformableObjectCfg pools (clean vs dirty) |
| Target drum | Plastic drum at (0.15, 0.85, 0.0) m |
| Conveyor | Dual belt (4 m total), surface at 0.80 m, **active** |
| Env spacing | 4.0 m |

### Label System

Shirts will use the same integer label system as the cube sort task,
enabling label-based sorting with vectorised boolean masks.

| Label | Category | Count |
|---|---|---|
| 0 | Target (e.g. clean) | TODO |
| 1 | Distractor (e.g. dirty) | TODO |

## Controlled Joints

| Joint | Type | Limits | Action Clip |
|---|---|---|---|
| `base_y_joint` | Prismatic | ±0.5 m | (−0.5, 0.5) |
| `base_z_joint` | Prismatic | −0.5 … 0.0 m | (−0.5, 0.0) |
| `elbow_joint` | Revolute | ±1.5 rad | (−1.5, 1.5) |
| `wrist_y_joint` | Revolute | ±0.8 rad | (−0.8, 0.8) |
| `wrist_x_joint` | Revolute | ±0.8 rad | (−0.8, 0.8) |
| `finger_joint` | Revolute | — | Binary (open=0.0, close=0.7854) |

## Actions (6 dims)

| Group | Dims | Type | Scale |
|---|---|---|---|
| `base_delta` | 2 | Joint position delta | 0.50 |
| `arm_delta` | 3 | Joint position delta | 1.0 |
| `gripper` | 1 | Binary joint position | open=0.0, close=0.7854 |

## Observations

<!-- TODO: Define observation terms once cloth simulation is integrated -->

## Rewards

<!-- TODO: Define reward terms once cloth simulation is integrated -->

The reward pipeline will follow the same 6-phase structure as the cube sort
task (reach → grasp → lift → transport → release → success), adapted for
deformable object manipulation with label-based sorting.

## Terminations

| Term | Type | Condition |
|---|---|---|
| `time_out` | Truncation | Episode length exceeded (8.0 s) |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Physics dt | 0.01 s (100 Hz) |
| Decimation | 2 (control at 50 Hz) |
| Episode length | 8.0 s |
| Default num_envs | 4 096 |

## Training

<!-- TODO: Add training results once cloth simulation is integrated -->

## Running

```bash
cd src/tensegrity_pick

# Training
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/train.py \
    --task Template-Tensegrity-Shirt-Sort-v0 --headless
```

## Related

- [Cube sort task](../cube_sort/README.md) — rigid-body cube sorting (reference implementation)
- [Shirt place task](../shirt_place/README.md) — single-shirt pick-and-place
- [Robot specification](../../../../../../../../res/Tensegrity/README.md) — kinematic chain, joint constraints, tendon geometry
- [Extension overview](../../../../../../README.md) — all registered tasks and scripts

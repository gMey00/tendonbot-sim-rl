# Kinova Gen3 — Variant Specification

> Task: `Template-Reach-Kinova-v0`
> Config: `joint_pos_env_cfg.py`

## Robot

| Property | Value |
|---|---|
| Model | Kinova Gen3 7-DOF + Robotiq 2F-140 |
| DOF | 7 |
| Controlled joints | `joint_1` – `joint_7` |
| End effector | `end_effector_link` |
| USD | `res/KinovaGen3/KinovaGen3_Robotiq2F140.usd` |
| Mount | Table, (0.15, 0.0, 1.40) m |
| Gripper | Robotiq 2F-140 (locked for reach) |
| Self-collisions | Disabled |
| Gravity | Disabled |

## Actuators

| Group | Joints | K | D | τ_max | ω_max |
|---|---|---|---|---|---|
| `shoulder` | joint_1, joint_2 | 5000 | 220 | 39.0 N·m | 1.40 rad/s |
| `elbow` | joint_3, joint_4 | 2500 | 160 | 39.0 N·m | 1.40 rad/s |
| `wrist` | joint_5, joint_6, joint_7 | 2500 | 160 | 9.0 N·m | 1.22 rad/s |

## Action Space (dim = 7)

| Group | Type | Dim | Notes |
|---|---|---|---|
| `arm_action` | EMAJointPositionToLimits (α = 0.2) | 7 | All arm joints |

## Observation Space (dim = 28)

| Term | Dim |
|---|---|
| joint_pos (relative) | 7 |
| joint_vel (relative) | 7 |
| pose_command (pos + quat) | 7 |
| last_action | 7 |

## Training Configuration

| Parameter | Value |
|---|---|
| Network | [64, 64] ELU (shared) |
| Timesteps | 48 000 |
| FK margin | 0.40 |
| Rollouts | 24 |
| Learning rate | 1e-3 (KLAdaptive) |
| Entropy coefficient | 0.01 |
| Joint limit fallback | ±π from defaults (continuous joints) |
| Fine-grained reward σ | 0.5 |
| Reset range | ±0.125 (offset) |

## Notes

Joints 1, 3, 5, 7 are continuous (±2π in USD).  A fallback range of 6.2832 rad
clamps these to ±π from the default position for feasible FK sampling and
action mapping.

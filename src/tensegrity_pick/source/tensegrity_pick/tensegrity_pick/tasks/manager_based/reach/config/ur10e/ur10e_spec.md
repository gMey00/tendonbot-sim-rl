# UR10e — Variant Specification

> Task: `Template-Reach-UR10e-v0`
> Config: `joint_pos_env_cfg.py`

## Robot

| Property | Value |
|---|---|
| Model | Universal Robots UR10e 6-DOF + Robotiq 2F-140 |
| DOF | 6 |
| Controlled joints | `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint` |
| End effector | `robotiq_base_link` |
| USD | `{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd` (variant `Gripper: Robotiq_2f_140`) |
| Mount | Table, (0.15, 0.0, 1.40) m |
| Gripper | Robotiq 2F-140 (locked for reach) |
| Gravity | Disabled |

## Default Joint Positions

| Joint | Default [rad] |
|---|---|
| shoulder_pan | 0.0 |
| shoulder_lift | −π/2 |
| elbow | +π/2 |
| wrist_1 | −π/2 |
| wrist_2 | +π/2 |
| wrist_3 | 0.0 |

## Actuators

| Group | Joints | K | D |
|---|---|---|---|
| `shoulder` | shoulder_pan, shoulder_lift | 1320.0 | 72.66 |
| `elbow` | elbow | 600.0 | 34.64 |
| `wrist` | wrist_1, wrist_2, wrist_3 | 216.0 | 29.39 |

Effort and velocity limits are taken from the USD defaults.

## Action Space (dim = 6)

| Group | Type | Dim | Notes |
|---|---|---|---|
| `arm_action` | EMAJointPositionToLimits (α = 0.2) | 6 | All arm joints |

## Observation Space (dim = 25)

| Term | Dim |
|---|---|
| joint_pos (relative) | 6 |
| joint_vel (relative) | 6 |
| pose_command (pos + quat) | 7 |
| last_action | 6 |

## Training Configuration

| Parameter | Value |
|---|---|
| Network | [64, 64] ELU (shared) |
| Timesteps | 48 000 |
| FK margin | 0.20 |
| Rollouts | 24 |
| Learning rate | 1e-3 (KLAdaptive) |
| Entropy coefficient | 0.01 |
| Joint limit clamping | max_range = 3.0 (±1.5 rad from defaults) |
| Fine-grained reward σ | 0.5 |
| Reset range | ±0.125 (offset) |

## Notes

All joints in the UR10e USD have ±2π limits.  `clamp_infinite_joint_limits`
with `max_range=3.0` restricts each joint to ±1.5 rad around the default
position.  This narrows the FK workspace and provides a finer action mapping.

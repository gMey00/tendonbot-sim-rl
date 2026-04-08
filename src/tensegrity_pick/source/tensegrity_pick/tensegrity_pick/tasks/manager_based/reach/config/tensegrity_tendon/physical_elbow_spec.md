# Physical Elbow Tensegrity — Variant Specification

> Task: `Template-Reach-Tensegrity-Physical-Tendon-v0`
> Config: `joint_pos_env_cfg_physical.py`

## Robot

| Property | Value |
|---|---|
| Model | 5-DOF tensegrity manipulator with antiparallelogram 4-bar elbow |
| Effective DOF | 5 (2 base + 1 elbow + 2 wrist) |
| Controlled joints (7) | `base_y_joint`, `base_z_joint`, `rod_left_joint`, `rod_right_joint`, `coupler_left_joint`, `wrist_y_joint`, `wrist_x_joint` |
| End effector | `tool_link_0` |
| USD | `res/Tensegrity/fivedof_manipulator/fivedof_linear_base_physical_robotiq2f140.usd` |
| Mount | Ceiling, (0.15, 0.0, 2.30) m |
| Gripper | Robotiq 2F-140 (locked for reach) |

## 4-Bar Linkage

The elbow is an antiparallelogram mechanism with 1 kinematic DOF.  The free
parameter is `rod_left_joint`; the remaining two joints are coupled:

    rod_right = coupler_left = φ + θ₀

where θ = rod_left + θ₀, φ = θ + 2·atan2(−k_e·cos θ, l_e − k_e·sin θ),
l_e = 0.150 m, k_e = 0.060 m, θ₀ = arcsin(k_e / l_e) ≈ 0.4115 rad.

Joint limits are asymmetric:
- `rod_left_joint`: [−0.6886, +0.5332] rad
- `rod_right_joint`: [−0.5332, +0.6886] rad
- `coupler_left_joint`: [−0.6886, +0.6886] rad

## Actuators

| Group | Joints | Type | K | D | τ_max | ω_max |
|---|---|---|---|---|---|---|
| `base_y` | base_y | ImplicitActuator | 8000 | 800 | 300 N·m | 5.0 rad/s |
| `base_z` | base_z | ImplicitActuator | 8000 | 800 | 200 N·m | 5.0 rad/s |
| `linkage` | rod_*, coupler_left | IdealPDActuator (K=0, D=0) | 0 | 0 | 35 N·m | 2.5 rad/s |
| `wrist` | wrist_y, wrist_x | IdealPDActuator (K=0, D=0) | 0 | 0 | 3.5 N·m | 9.0 rad/s |

## Action Space (dim = 7)

| Group | Type | Dim | Notes |
|---|---|---|---|
| `base_action` | JointPositionToLimits | 2 | base_y, base_z |
| `arm_tendon` | PhysicalTendonEffort | 5 | 2 elbow + 3 wrist tendons |

Actions in [−1, 1] are mapped to cable tensions [0, 500] N.

- **Elbow tendons (T0, T1):** body forces at physical attachment points
  - Root link offsets: [0, ±0.0725, −0.34] m
  - Forearm link offsets: [0, ±0.0725, −0.02] m
- **Wrist tendons (T2–T4):** constant J^T mapping to joint efforts
  - J^T: [[−0.0173, 0.0, +0.0173], [+0.010, −0.020, +0.010]]

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
| FK margin | 0.35 |
| Joint coupling | ON (antiparallelogram closure) |
| Rollouts | 24 |
| Learning rate | 1e-3 (KLAdaptive) |
| Entropy coefficient | 0.01 |

## Results (Iteration 12)

| Metric | Value |
|---|---|
| Total reward | −0.32 |
| Position tracking | −0.032 |
| Orientation tracking | −0.046 |
| Fine-grained position | +0.066 |
| Policy σ | 0.064 |
| Position reached (< 5 cm) | 0 % |

The physical variant converges with lower absolute reward than the PD (+0.77)
and Tendon (+0.75) variants due to inherent limitations of body-force actuation
through the 4-bar linkage.  Cable forces applied at physical attachment points
have lower positioning precision than direct position or Jacobian-transpose
effort commands.

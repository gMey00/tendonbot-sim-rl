# Physical Elbow Tensegrity — Variant Specification

> Tasks: `Template-Reach-Tensegrity-Physical-Tendon-v0` (direct tension control)
>        `Template-Reach-Tensegrity-Physical-Hierarchical-v0` (inner PID→tension loop)
> Config: `joint_pos_env_cfg_physical.py`
> Rework change log: [physical_variant_fix_report.md](../../../../../../../../../../doc/reports/physical_variant_fix_report.md) (2026-07-09)

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
| Physics | **120 Hz**, decimation 4 (control 30 Hz); `*_awake.usd` (per-body sleep zeroed); `enable_external_forces_every_iteration` |

## 4-Bar Linkage

The elbow is an antiparallelogram mechanism with 1 kinematic DOF.  The free
parameter is `rod_left_joint`; the remaining two joints are coupled:

    rod_right = coupler_left = φ + θ₀

where θ = rod_left + θ₀, φ = θ + 2·atan2(−k_e·cos θ, l_e − k_e·sin θ),
l_e = 0.150 m, k_e = 0.060 m, θ₀ = arcsin(k_e / l_e) ≈ 0.4115 rad.
(Verified numerically against a continuation solve, 2026-07-09.)

Joint limits are asymmetric (map to exactly ±70° elbow):
- `rod_left_joint`: [−0.6886, +0.5332] rad
- `rod_right_joint`: [−0.5332, +0.6886] rad
- `coupler_left_joint`: [−0.6886, +0.6886] rad

The **effective elbow angle** is the forearm body's twist about the upper-arm
X axis and equals `rod_left + rod_right`; it is measured from body
quaternions (`compute_lower_arm_angle_and_rate`), never from a single joint.

**Linkage-integrity penalty** `linkage_broken` (reward term, −1/step):
closure-anchor gap > 3 cm (elastic solver drift under full 480 N stays ≲ 1 cm
at 120 Hz) OR parallelogram branch flip (|elbow − (rod_L + rod_R)| > 0.3 rad).
A penalty rather than a termination — early termination under the
net-negative reach reward is itself an exploitable suicide exit.

## Actuators

| Group | Joints | Type | K | D | τ_max | ω_max |
|---|---|---|---|---|---|---|
| `base_y` | base_y | ImplicitActuator | 8000 | 800 | 300 N·m | 5.0 rad/s |
| `base_z` | base_z | ImplicitActuator | 8000 | 800 | 200 N·m | 5.0 rad/s |
| `linkage` | rod_*, coupler_left | IdealPDActuator (K=0, D=0) | 0 | 0 | 35 N·m | 2.5 rad/s |
| `wrist` | wrist_y, wrist_x | IdealPDActuator (K=0, D=0) | 0 | 0 | 3.5 N·m | 9.0 rad/s |

## Cable model (both variants)

- **Per-tendon tension saturation** `[480, 480, 80, 80, 80]` N
  (elbow after 3:1 pulley, wrist direct-drive; Klein 2023 §3.2.5/§3.2.6).
- **Cable length limits** between the attachment points, from the ±70°
  workspace geometry: **[0.0913, 0.2577] m** (0.1775 m at 0°).
  Wind-up stop (active tension → 0 over 1 cm above min length) +
  spring–damper stretch stop (20 kN/m, 200 N·s/m, cap 1 kN) beyond max.
- **Motor/spool lag**: 50 ms first-order filter on commanded tensions/efforts.
- **Elbow tendons (T0, T1):** body forces at the physical attachment points
  - Root link offsets: [0, ±0.0725, −0.34] m
  - Forearm link offsets: [0, ±0.0725, −0.02] m
- **Wrist tendons (T2–T4):** constant J^T mapping to joint efforts
  - J^T: [[−0.0173, 0.0, +0.0173], [+0.010, −0.020, +0.010]]

## Action Spaces

**Direct (`…-Physical-Tendon-v0`, dim = 7):**

| Group | Type | Dim | Notes |
|---|---|---|---|
| `base_action` | JointPositionToLimits | 2 | base_y, base_z |
| `arm_tendon` | PhysicalTendonEffort | 5 | [−1, 1] → [0, max] N per tendon |

**Hierarchical (`…-Physical-Hierarchical-v0`, dim = 5):**

| Group | Type | Dim | Notes |
|---|---|---|---|
| `base_action` | JointPositionToLimits | 2 | base_y, base_z |
| `arm_tendon` | HierarchicalPhysicalTendon | 3 | set-points: elbow **±60°**, wrist_y/x **±40°** (achievable workspace; slew-limited 1.2/5/5 rad/s) |

The hierarchical inner loop runs at the physics rate (120 Hz): PID
(elbow kp=75, ki=6, kd=3; wrist kp=10, ki=1.5, kd=0.6 — gains from the
step-response validation study), gravity compensation (elbow m·g·l ≈ 2.8 N·m,
calibrated to the measured 5-DOF static balance), block-wise torque→tension
distribution (elbow antagonistic pair, pre-tension 6 N; wrist pseudo-inverse
with null-space smoothing, pre-tension 5 N), then the same body-force channel
as the direct variant.

## Observation Space (dim = 30 direct / 28 hierarchical)

| Term | Dim | Notes |
|---|---|---|
| joint_pos (relative) | 4 | base_y, base_z, wrist_y, wrist_x only |
| joint_vel (relative) | 4 | 〃 |
| pose_command (pos + quat) | 7 | |
| last_action | 7 / 5 | per action space |
| lower_arm_angle | 1 | forearm body twist about upper-arm X |
| lower_arm_ang_vel | 1 | 〃 rate |
| cable_lengths | 2 | elbow cables, between attachment points |
| cable_length_rates | 2 | positive = extending |
| applied_tensions | 2 | lag-filtered, stop-adjusted elbow tensions (scale 1/480) |

## Training Configuration (2026-07 retrain)

| Parameter | Value |
|---|---|
| Network | [64, 64] ELU (shared) |
| Timesteps | 48 000 |
| FK margin | 0.01 (targets from the hidden PD FK reference robot) |
| Rollouts | 48 |
| Learning rate | 1e-3 (KLAdaptive) |
| Entropy coefficient | 0.01 |

## Results

Historical (before the 2026-07 rework): direct tension control converged to
reward −0.35 (150 k steps) and 31.4 % ± 14.4 % evaluation success; policies
exploited the loop-closure joint ("break the linkage, steer with the base").
Current results: see the updated
[reach_evaluation_findings.md](../../../../../../../../../../doc/reports/reach_evaluation_findings.md)
and the [tracking log](../../../../../../../../../../doc/reports/reach_optimization_tracking.md)
(iteration 16).

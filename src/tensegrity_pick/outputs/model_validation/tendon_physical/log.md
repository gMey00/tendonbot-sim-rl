# Model Validation Log — TENDON_PHYSICAL Mode

**Generated:** 2026-04-04 13:49:43
**Isaac Sim version:** 5.1.0  |  **Physics:** PhysX GPU

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Physics dt | 0.00833 s  (120 Hz) |
| Render interval | 2 steps |
| Step hold | 2.5 s |
| Return hold | 1.0 s |
| Warm-up | 0.5 s |

## Robot Configuration

| Parameter | Value |
|-----------|-------|
| Variant | physical |
| Model | TENS_3DOF_PHYSICAL_TENDON_CFG (physical four-bar linkage) |
| USD | tensegrity_threedof_arm_physical.usd |
| Actuator type | IdealPDActuatorCfg (K=0, D=0 — effort passthrough) |
| Control law | PID → body forces (elbow) + J^T (wrist) |
| Elbow PID | kp=75.0, ki=6.0, kd=3.0 |
| Wrist PID | kp=10.0, ki=1.5, kd=0.6 |
| Jacobian | 3×5 constant (zero-config approximation, Klein §3.2.4) |
| Mount position | (0.0, 0.0, 1.0) m |
| Num envs | 1 |

## Klein (2023) Target Criteria

| Criterion | Target |
|-----------|--------|
| Damping ratio ζ | 1.0–1.5 (near-critical) |
| Settling time | 0.1–0.3 s |
| Overshoot | < 5.0 % |

## Step Response Metrics

| Joint | Step [°] | Rise [ms] | Overshoot | Settle [ms] | SS Err [°] | RMSE [°] | NRMSE [%] | Pass |
|-------|---------|-----------|-----------|-------------|-----------|---------|-----------|------|
| elbow_physical | 20 | 91.7 | 0.1% | 133.3 | 0.13 | 2.55 | 12.7 | ✓ |
| elbow_physical | 30 | 108.3 | 0.5% | 166.7 | 0.12 | 3.96 | 13.1 | ✓ |
| elbow_physical | 40 | 116.7 | 1.4% | 183.3 | 0.47 | 5.68 | 14.0 | ✓ |
| wrist_y_joint | 10 | 83.3 | 0.3% | 150.0 | 0.02 | 0.97 | 9.7 | ✓ |
| wrist_y_joint | 20 | 100.0 | —% | 183.3 | 0.01 | 2.49 | 12.4 | ✓ |
| wrist_y_joint | 30 | 141.7 | —% | 216.7 | 0.04 | 4.47 | 14.9 | ✓ |
| wrist_x_joint | 10 | 83.3 | —% | 283.3 | 0.12 | 1.01 | 10.1 | ✓ |
| wrist_x_joint | 20 | 100.0 | —% | 191.7 | 0.15 | 2.52 | 12.6 | ✓ |
| wrist_x_joint | 30 | 133.3 | —% | 225.0 | 0.17 | 4.48 | 14.9 | ✓ |

## Key Findings

- **elbow_physical**: avg NRMSE = 13.3 %, avg settle = 161 ms, 3/3 trials passing Klein criteria
- **wrist_y_joint**: avg NRMSE = 12.3 %, avg settle = 183 ms, 3/3 trials passing Klein criteria
- **wrist_x_joint**: avg NRMSE = 12.5 %, avg settle = 233 ms, 3/3 trials passing Klein criteria

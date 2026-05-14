# Model Validation Log — TENDON Mode

**Generated:** 2026-04-04 13:48:21
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
| Variant | elbow_approx |
| Model | TENS_3DOF_TENDON_CFG (IdealPDActuatorCfg, K=0, D=0) |
| USD | threedof_manipulator.usd |
| Actuator type | IdealPDActuatorCfg (K=0, D=0 — effort passthrough) |
| Control law | PID position ctrl + gravity comp + Jacobian transpose |
| Elbow PID | kp=50.0, ki=4.0, kd=2.0 |
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
| elbow_joint | 20 | 100.0 | 0.0% | 141.7 | 0.03 | 2.58 | 12.9 | ✓ |
| elbow_joint | 30 | 125.0 | 0.1% | 191.7 | 0.00 | 4.52 | 15.0 | ✓ |
| elbow_joint | 40 | 166.7 | 0.3% | 233.3 | 0.08 | 6.83 | 17.0 | ✓ |
| wrist_y_joint | 10 | 83.3 | 0.1% | 150.0 | 0.01 | 0.97 | 9.7 | ✓ |
| wrist_y_joint | 20 | 100.0 | —% | 183.3 | 0.04 | 2.48 | 12.4 | ✓ |
| wrist_y_joint | 30 | 133.3 | —% | 225.0 | 0.08 | 4.43 | 14.8 | ✓ |
| wrist_x_joint | 10 | 75.0 | 0.2% | 158.3 | 0.01 | 0.97 | 9.7 | ✓ |
| wrist_x_joint | 20 | 108.3 | —% | 183.3 | 0.03 | 2.48 | 12.4 | ✓ |
| wrist_x_joint | 30 | 141.7 | —% | 225.0 | 0.06 | 4.43 | 14.8 | ✓ |

## NRMSE vs. Klein (2023) Gazebo Reference

| Joint | Step [°] | Klein NRMSE [%] | Isaac Sim NRMSE [%] | Δ |
|-------|---------|----------------|---------------------|---|
| elbow_joint | 20 | 39.1 | 12.9 | -26.2 |
| elbow_joint | 30 | 11.7 | 15.0 | +3.3 |
| elbow_joint | 40 | 12.1 | 17.0 | +4.9 |
| wrist_y_joint | 10 | 19.2 | 9.7 | -9.5 |
| wrist_y_joint | 20 | 11.6 | 12.4 | +0.8 |
| wrist_y_joint | 30 | 12.5 | 14.8 | +2.3 |
| wrist_x_joint | 10 | 25.5 | 9.7 | -15.8 |
| wrist_x_joint | 20 | 9.9 | 12.4 | +2.5 |
| wrist_x_joint | 30 | 11.2 | 14.8 | +3.6 |

## Key Findings

- **elbow_joint**: avg NRMSE = 15.0 %, avg settle = 189 ms, 3/3 trials passing Klein criteria
- **wrist_y_joint**: avg NRMSE = 12.3 %, avg settle = 186 ms, 3/3 trials passing Klein criteria
- **wrist_x_joint**: avg NRMSE = 12.3 %, avg settle = 189 ms, 3/3 trials passing Klein criteria

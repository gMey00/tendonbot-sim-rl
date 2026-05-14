# Model Validation Log — PD Mode

**Generated:** 2026-04-04 00:53:37
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
| Model | TENS_3DOF_CFG (3-DOF arm only) |
| USD | threedof_manipulator.usd |
| Actuator type | ImplicitActuatorCfg (PhysX PD position drives) |
| Arm stiffness | K = 400.0 N·m/rad |
| Arm damping | D = 20.0 N·m·s/rad  (validated by tune_pd_gains.py) |
| Effort limit | 40.0 N·m |
| Velocity limit | 2.0 rad/s |
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
| elbow_joint | 20 | 58.3 | —% | 108.3 | 0.33 | 1.87 | 9.5 | ✓ |
| elbow_joint | 30 | 58.3 | —% | 116.7 | 0.48 | 2.82 | 9.5 | ✓ |
| elbow_joint | 40 | 58.3 | —% | 116.7 | 0.61 | 3.76 | 9.5 | ✓ |
| wrist_y_joint | 10 | 50.0 | —% | 100.0 | 0.01 | 0.83 | 8.3 | ✓ |
| wrist_y_joint | 20 | 50.0 | —% | 100.0 | 0.01 | 1.66 | 8.3 | ✓ |
| wrist_y_joint | 30 | 50.0 | —% | 100.0 | 0.02 | 2.49 | 8.3 | ✓ |
| wrist_x_joint | 10 | 50.0 | —% | 100.0 | 0.01 | 0.83 | 8.3 | ✓ |
| wrist_x_joint | 20 | 50.0 | —% | 100.0 | 0.01 | 1.66 | 8.3 | ✓ |
| wrist_x_joint | 30 | 50.0 | —% | 100.0 | 0.02 | 2.49 | 8.3 | ✓ |

## PD Gain Sweep — Elbow Joint

| D [N·m·s/rad] | Avg Settle [ms] | Best |
|--------------|----------------|------|
| 15 | 69.4 ◄ BEST | ✓ |
| 20 | 113.9 |  |
| 25 | 158.3 |  |
| 30 | 191.7 |  |
| 60 | 408.3 |  |
| 120 | 863.9 |  |

## Key Findings

- **elbow_joint**: avg NRMSE = 9.5 %, avg settle = 114 ms, 3/3 trials passing Klein criteria
- **wrist_y_joint**: avg NRMSE = 8.3 %, avg settle = 100 ms, 3/3 trials passing Klein criteria
- **wrist_x_joint**: avg NRMSE = 8.3 %, avg settle = 100 ms, 3/3 trials passing Klein criteria

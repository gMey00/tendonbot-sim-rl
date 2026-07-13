# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_f140_ik` |
| Run ID | `2026-07-13_02-43-55_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.0995 |
| **Mean position error** | **8.5 cm** |
| **Position reached (< 5 cm)** | **66 % of steps** |
| Orientation reached (< 0.3 rad) | 44 % of steps |
| Pose reached (both) | 36 % of steps |
| Position tracking reward | -0.0169 |
| Orientation tracking reward | -0.0447 |
| Position fine-grained reward | 0.057938 |
| Action rate penalty | -0.009127 |
| Joint velocity penalty | -0.003406 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.3108 |
| Learning rate (final) | 1.72e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_f140_ik/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_f140_ik/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_f140_ik/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_f140_ik/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_f140_ik/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_f140_ik/06_episode_length.png)

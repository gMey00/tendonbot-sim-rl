# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_f140` |
| Run ID | `2026-07-13_02-16-36_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.2439 |
| **Mean position error** | **5.3 cm** |
| **Position reached (< 5 cm)** | **86 % of steps** |
| Orientation reached (< 0.3 rad) | 88 % of steps |
| Pose reached (both) | 82 % of steps |
| Position tracking reward | -0.0106 |
| Orientation tracking reward | -0.0183 |
| Position fine-grained reward | 0.072962 |
| Action rate penalty | -0.000718 |
| Joint velocity penalty | -0.002796 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.0634 |
| Learning rate (final) | 2.57e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_f140/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_f140/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_f140/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_f140/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_f140/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_f140/06_episode_length.png)

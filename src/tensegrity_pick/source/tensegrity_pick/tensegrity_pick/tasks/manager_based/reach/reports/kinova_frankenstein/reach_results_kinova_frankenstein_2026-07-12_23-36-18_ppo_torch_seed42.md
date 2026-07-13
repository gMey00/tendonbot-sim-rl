# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_frankenstein` |
| Run ID | `2026-07-12_23-36-18_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.0705 |
| **Mean position error** | **4.4 cm** |
| **Position reached (< 5 cm)** | **87 % of steps** |
| Orientation reached (< 0.3 rad) | 53 % of steps |
| Pose reached (both) | 50 % of steps |
| Position tracking reward | -0.0089 |
| Orientation tracking reward | -0.0355 |
| Position fine-grained reward | 0.072031 |
| Action rate penalty | -0.009768 |
| Joint velocity penalty | -0.006228 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2520 |
| Learning rate (final) | 6.86e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova_frankenstein/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova_frankenstein/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova_frankenstein/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova_frankenstein/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova_frankenstein/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova_frankenstein/06_episode_length.png)

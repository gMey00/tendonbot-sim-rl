# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_frankenstein_ik` |
| Run ID | `2026-07-13_04-59-17_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.2740 |
| **Mean position error** | **12.8 cm** |
| **Position reached (< 5 cm)** | **30 % of steps** |
| Orientation reached (< 0.3 rad) | 35 % of steps |
| Pose reached (both) | 14 % of steps |
| Position tracking reward | -0.0256 |
| Orientation tracking reward | -0.0470 |
| Position fine-grained reward | 0.036187 |
| Action rate penalty | -0.009103 |
| Joint velocity penalty | -0.002274 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.3426 |
| Learning rate (final) | 2.26e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_frankenstein_ik/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_frankenstein_ik/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_frankenstein_ik/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_frankenstein_ik/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_frankenstein_ik/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_frankenstein_ik/06_episode_length.png)

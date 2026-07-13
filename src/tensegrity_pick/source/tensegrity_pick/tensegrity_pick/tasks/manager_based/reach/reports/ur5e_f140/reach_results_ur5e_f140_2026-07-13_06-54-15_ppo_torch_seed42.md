# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_f140` |
| Run ID | `2026-07-13_06-54-15_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.3367 |
| **Mean position error** | **3.1 cm** |
| **Position reached (< 5 cm)** | **93 % of steps** |
| Orientation reached (< 0.3 rad) | 93 % of steps |
| Pose reached (both) | 90 % of steps |
| Position tracking reward | -0.0063 |
| Orientation tracking reward | -0.0144 |
| Position fine-grained reward | 0.080781 |
| Action rate penalty | -0.000478 |
| Joint velocity penalty | -0.003603 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.0464 |
| Learning rate (final) | 1.81e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_f140/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_f140/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_f140/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_f140/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_f140/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_f140/06_episode_length.png)

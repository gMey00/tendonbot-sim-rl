# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_frankenstein_ikabs` |
| Run ID | `2026-07-13_10-13-27_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.0571 |
| **Mean position error** | **5.3 cm** |
| **Position reached (< 5 cm)** | **78 % of steps** |
| Orientation reached (< 0.3 rad) | 52 % of steps |
| Pose reached (both) | 46 % of steps |
| Position tracking reward | -0.0107 |
| Orientation tracking reward | -0.0355 |
| Position fine-grained reward | 0.067670 |
| Action rate penalty | -0.008330 |
| Joint velocity penalty | -0.004700 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2885 |
| Learning rate (final) | 1.06e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_frankenstein_ikabs/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_frankenstein_ikabs/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_frankenstein_ikabs/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_frankenstein_ikabs/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_frankenstein_ikabs/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_frankenstein_ikabs/06_episode_length.png)

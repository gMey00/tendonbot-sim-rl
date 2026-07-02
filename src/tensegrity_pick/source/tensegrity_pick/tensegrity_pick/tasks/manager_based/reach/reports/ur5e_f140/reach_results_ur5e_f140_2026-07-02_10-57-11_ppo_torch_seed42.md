# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_f140` |
| Run ID | `2026-07-02_10-57-11_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.5214 |
| **Mean position error** | **1.8 cm** |
| **Position reached (< 5 cm)** | **96 % of steps** |
| Orientation reached (< 0.3 rad) | 0 % of steps |
| Pose reached (both) | 0 % of steps |
| Position tracking reward | -0.0036 |
| Orientation tracking reward | -0.1631 |
| Position fine-grained reward | 0.086446 |
| Action rate penalty | -0.002665 |
| Joint velocity penalty | -0.004255 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.1049 |
| Learning rate (final) | 1.11e-04 |

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

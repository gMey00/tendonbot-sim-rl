# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_f140_osc` |
| Run ID | `2026-07-13_03-41-02_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.4523 |
| **Mean position error** | **12.4 cm** |
| **Position reached (< 5 cm)** | **24 % of steps** |
| Orientation reached (< 0.3 rad) | 27 % of steps |
| Pose reached (both) | 8 % of steps |
| Position tracking reward | -0.0249 |
| Orientation tracking reward | -0.0483 |
| Position fine-grained reward | 0.032649 |
| Action rate penalty | -0.027753 |
| Joint velocity penalty | -0.006820 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.4639 |
| Learning rate (final) | 8.78e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_f140_osc/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_f140_osc/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_f140_osc/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_f140_osc/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_f140_osc/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_f140_osc/06_episode_length.png)

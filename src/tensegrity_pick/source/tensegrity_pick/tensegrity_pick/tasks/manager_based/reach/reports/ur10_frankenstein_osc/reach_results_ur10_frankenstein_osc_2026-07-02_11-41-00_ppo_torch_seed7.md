# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_frankenstein_osc` |
| Run ID | `2026-07-02_11-41-00_ppo_torch_seed7` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.7059 |
| **Mean position error** | **4.1 cm** |
| **Position reached (< 5 cm)** | **90 % of steps** |
| Orientation reached (< 0.3 rad) | 0 % of steps |
| Pose reached (both) | 0 % of steps |
| Position tracking reward | -0.0083 |
| Orientation tracking reward | -0.1663 |
| Position fine-grained reward | 0.079313 |
| Action rate penalty | -0.016309 |
| Joint velocity penalty | -0.006501 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2782 |
| Learning rate (final) | 2.44e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_frankenstein_osc/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_frankenstein_osc/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_frankenstein_osc/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_frankenstein_osc/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_frankenstein_osc/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_frankenstein_osc/06_episode_length.png)

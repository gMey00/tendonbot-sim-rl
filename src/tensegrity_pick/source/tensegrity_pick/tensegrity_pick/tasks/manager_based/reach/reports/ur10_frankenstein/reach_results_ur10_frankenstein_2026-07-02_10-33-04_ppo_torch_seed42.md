# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_frankenstein` |
| Run ID | `2026-07-02_10-33-04_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.2473 |
| **Mean position error** | **4.7 cm** |
| **Position reached (< 5 cm)** | **88 % of steps** |
| Orientation reached (< 0.3 rad) | 5 % of steps |
| Pose reached (both) | 5 % of steps |
| Position tracking reward | -0.0094 |
| Orientation tracking reward | -0.0814 |
| Position fine-grained reward | 0.073161 |
| Action rate penalty | -0.015487 |
| Joint velocity penalty | -0.007981 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2667 |
| Learning rate (final) | 1.65e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_frankenstein/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_frankenstein/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_frankenstein/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_frankenstein/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_frankenstein/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_frankenstein/06_episode_length.png)

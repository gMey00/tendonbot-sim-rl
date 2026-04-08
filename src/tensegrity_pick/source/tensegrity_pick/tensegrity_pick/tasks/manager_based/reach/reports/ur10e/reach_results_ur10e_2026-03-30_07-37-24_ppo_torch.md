# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10e` |
| Run ID | `2026-03-30_07-37-24_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.2129 |
| Position tracking reward | -0.0509 |
| Orientation tracking reward | -0.0213 |
| Position fine-grained reward | 0.056286 |
| Position reached (< 5cm) | 0.0000 |
| Orientation reached (< 0.3 rad) | 0.0000 |
| Pose reached (both) | 0.0000 |
| Action rate penalty | -0.001221 |
| Joint velocity penalty | -0.001276 |
| Policy std deviation | 0.0615 |
| Learning rate (final) | 2.67e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10e/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10e/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10e/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10e/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10e/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10e/06_episode_length.png)

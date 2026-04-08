# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova` |
| Run ID | `2026-03-30_06-29-39_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.6211 |
| Position tracking reward | -0.0139 |
| Orientation tracking reward | -0.0192 |
| Position fine-grained reward | 0.088315 |
| Position reached (< 5cm) | 0.0000 |
| Orientation reached (< 0.3 rad) | 0.0000 |
| Pose reached (both) | 0.0000 |
| Action rate penalty | -0.001406 |
| Joint velocity penalty | -0.001919 |
| Policy std deviation | 0.0323 |
| Learning rate (final) | 1.61e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova/06_episode_length.png)

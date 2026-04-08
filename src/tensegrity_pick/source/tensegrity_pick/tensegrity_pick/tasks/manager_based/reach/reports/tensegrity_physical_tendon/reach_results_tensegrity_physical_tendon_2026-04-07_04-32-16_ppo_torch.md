# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `tensegrity_physical_tendon` |
| Run ID | `2026-04-07_04-32-16_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.3257 |
| Position tracking reward | -0.0309 |
| Orientation tracking reward | -0.0473 |
| Position fine-grained reward | 0.067486 |
| Position reached (< 5cm) | 0.0000 |
| Orientation reached (< 0.3 rad) | 0.0000 |
| Pose reached (both) | 0.0000 |
| Action rate penalty | -0.007772 |
| Joint velocity penalty | -0.009405 |
| Policy std deviation | 0.0552 |
| Learning rate (final) | 1.65e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/tensegrity_physical_tendon/01_total_reward.png)

### Task Success

![Task Success](../../figures/tensegrity_physical_tendon/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/tensegrity_physical_tendon/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/tensegrity_physical_tendon/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/tensegrity_physical_tendon/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/tensegrity_physical_tendon/06_episode_length.png)

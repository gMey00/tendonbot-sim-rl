# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `tensegrity_physical_tendon` |
| Run ID | `2026-04-07_05-29-15_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -1.1929 |
| Position tracking reward | -0.0597 |
| Orientation tracking reward | -0.0390 |
| Position fine-grained reward | 0.001239 |
| Position reached (< 5cm) | 0.0000 |
| Orientation reached (< 0.3 rad) | 0.0000 |
| Pose reached (both) | 0.0000 |
| Action rate penalty | -0.001833 |
| Joint velocity penalty | -0.000949 |
| Policy std deviation | 0.1030 |
| Learning rate (final) | 1.19e-04 |

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

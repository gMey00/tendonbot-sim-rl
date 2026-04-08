# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `tensegrity_physical_tendon` |
| Run ID | `2026-04-06_22-36-34_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -3.5896 |
| Position tracking reward | -0.1430 |
| Orientation tracking reward | -0.1703 |
| Position fine-grained reward | 0.000923 |
| Position reached (< 5cm) | 0.0000 |
| Orientation reached (< 0.3 rad) | 0.0000 |
| Pose reached (both) | 0.0000 |
| Action rate penalty | -0.001316 |
| Joint velocity penalty | -0.002316 |
| Policy std deviation | 0.9395 |
| Learning rate (final) | 5.88e-03 |

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

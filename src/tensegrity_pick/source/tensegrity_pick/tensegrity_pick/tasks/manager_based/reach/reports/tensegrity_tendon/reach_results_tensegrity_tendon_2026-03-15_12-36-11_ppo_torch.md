# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `tensegrity_tendon` |
| Run ID | `2026-03-15_12-36-11_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -3.4886 |
| Position tracking reward | -0.0873 |
| Orientation tracking reward | -0.0773 |
| Position fine-grained reward | 0.001544 |
| Position proximity reward | 0.000085 |
| Goal reached reward | 0.000736 |
| Action rate penalty | -0.065423 |
| Joint velocity penalty | -0.055021 |
| Policy std deviation | 0.2384 |
| Learning rate (final) | 1.31e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/tensegrity_tendon/01_total_reward.png)

### Task Success

![Task Success](../../figures/tensegrity_tendon/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/tensegrity_tendon/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/tensegrity_tendon/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/tensegrity_tendon/05_converged_breakdown.png)

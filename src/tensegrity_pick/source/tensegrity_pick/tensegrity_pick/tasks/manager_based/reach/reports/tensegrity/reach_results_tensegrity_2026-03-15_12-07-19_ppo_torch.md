# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `tensegrity` |
| Run ID | `2026-03-15_12-07-19_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -6.8094 |
| Position tracking reward | -0.4976 |
| Orientation tracking reward | -0.0017 |
| Position fine-grained reward | 0.000022 |
| Position proximity reward | 0.000001 |
| Goal reached reward | 0.000008 |
| Action rate penalty | -0.051891 |
| Joint velocity penalty | -0.024788 |
| Policy std deviation | 0.5400 |
| Learning rate (final) | 4.82e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/tensegrity/01_total_reward.png)

### Task Success

![Task Success](../../figures/tensegrity/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/tensegrity/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/tensegrity/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/tensegrity/05_converged_breakdown.png)

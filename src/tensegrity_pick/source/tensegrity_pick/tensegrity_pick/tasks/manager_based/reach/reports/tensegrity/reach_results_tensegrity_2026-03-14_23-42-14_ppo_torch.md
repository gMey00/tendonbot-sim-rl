# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `tensegrity` |
| Run ID | `2026-03-14_23-42-14_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -4419591643.4286 |
| Position tracking reward | -13.0188 |
| Orientation tracking reward | -0.2585 |
| Position fine-grained reward | 0.001558 |
| Orientation fine-grained reward | 0.001614 |
| Pose goal reached reward | 0.000006 |
| Action rate penalty | -0.130782 |
| Joint velocity penalty | -775735414603.776001 |
| Policy std deviation | 1.6144 |
| Learning rate (final) | 3.94e-03 |

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

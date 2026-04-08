# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `tensegrity` |
| Run ID | `2026-04-06_17-53-45_ppo_torch` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.7739 |
| Position tracking reward | -0.0061 |
| Orientation tracking reward | -0.0092 |
| Position fine-grained reward | 0.086736 |
| Position reached (< 5cm) | 0.0000 |
| Orientation reached (< 0.3 rad) | 0.0000 |
| Pose reached (both) | 0.0000 |
| Action rate penalty | -0.001194 |
| Joint velocity penalty | -0.006255 |
| Policy std deviation | 0.0206 |
| Learning rate (final) | 1.40e-04 |

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

### Episode Length

![Episode Length](../../figures/tensegrity/06_episode_length.png)

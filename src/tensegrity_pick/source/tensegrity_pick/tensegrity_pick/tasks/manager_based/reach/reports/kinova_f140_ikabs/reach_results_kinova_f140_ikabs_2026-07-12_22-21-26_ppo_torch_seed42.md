# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_f140_ikabs` |
| Run ID | `2026-07-12_22-21-26_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.2038 |
| **Mean position error** | **2.8 cm** |
| **Position reached (< 5 cm)** | **92 % of steps** |
| Orientation reached (< 0.3 rad) | 46 % of steps |
| Pose reached (both) | 45 % of steps |
| Position tracking reward | -0.0056 |
| Orientation tracking reward | -0.0383 |
| Position fine-grained reward | 0.083199 |
| Action rate penalty | -0.003047 |
| Joint velocity penalty | -0.002059 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.1494 |
| Learning rate (final) | 2.43e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova_f140_ikabs/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova_f140_ikabs/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova_f140_ikabs/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova_f140_ikabs/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova_f140_ikabs/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova_f140_ikabs/06_episode_length.png)

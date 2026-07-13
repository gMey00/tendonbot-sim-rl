# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_f140_ik` |
| Run ID | `2026-07-12_21-51-17_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.1509 |
| **Mean position error** | **3.8 cm** |
| **Position reached (< 5 cm)** | **85 % of steps** |
| Orientation reached (< 0.3 rad) | 51 % of steps |
| Pose reached (both) | 48 % of steps |
| Position tracking reward | -0.0076 |
| Orientation tracking reward | -0.0366 |
| Position fine-grained reward | 0.074939 |
| Action rate penalty | -0.004597 |
| Joint velocity penalty | -0.001176 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2149 |
| Learning rate (final) | 1.88e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova_f140_ik/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova_f140_ik/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova_f140_ik/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova_f140_ik/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova_f140_ik/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova_f140_ik/06_episode_length.png)

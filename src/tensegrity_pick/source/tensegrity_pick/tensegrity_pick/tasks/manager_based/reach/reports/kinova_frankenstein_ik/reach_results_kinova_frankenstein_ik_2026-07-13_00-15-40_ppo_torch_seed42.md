# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_frankenstein_ik` |
| Run ID | `2026-07-13_00-15-40_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.0080 |
| **Mean position error** | **5.8 cm** |
| **Position reached (< 5 cm)** | **72 % of steps** |
| Orientation reached (< 0.3 rad) | 40 % of steps |
| Pose reached (both) | 35 % of steps |
| Position tracking reward | -0.0117 |
| Orientation tracking reward | -0.0464 |
| Position fine-grained reward | 0.062924 |
| Action rate penalty | -0.004927 |
| Joint velocity penalty | -0.001162 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2354 |
| Learning rate (final) | 1.51e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova_frankenstein_ik/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova_frankenstein_ik/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova_frankenstein_ik/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova_frankenstein_ik/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova_frankenstein_ik/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova_frankenstein_ik/06_episode_length.png)

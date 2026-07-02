# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_frankenstein_osc` |
| Run ID | `2026-07-02_10-33-04_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.3470 |
| **Mean position error** | **3.0 cm** |
| **Position reached (< 5 cm)** | **90 % of steps** |
| Orientation reached (< 0.3 rad) | 0 % of steps |
| Pose reached (both) | 0 % of steps |
| Position tracking reward | -0.0061 |
| Orientation tracking reward | -0.1066 |
| Position fine-grained reward | 0.074988 |
| Action rate penalty | -0.013711 |
| Joint velocity penalty | -0.007087 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2869 |
| Learning rate (final) | 4.71e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova_frankenstein_osc/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova_frankenstein_osc/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova_frankenstein_osc/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova_frankenstein_osc/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova_frankenstein_osc/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova_frankenstein_osc/06_episode_length.png)

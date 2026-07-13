# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_frankenstein_osc` |
| Run ID | `2026-07-13_01-21-29_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.1226 |
| **Mean position error** | **3.3 cm** |
| **Position reached (< 5 cm)** | **88 % of steps** |
| Orientation reached (< 0.3 rad) | 12 % of steps |
| Pose reached (both) | 12 % of steps |
| Position tracking reward | -0.0066 |
| Orientation tracking reward | -0.0587 |
| Position fine-grained reward | 0.074942 |
| Action rate penalty | -0.026897 |
| Joint velocity penalty | -0.004774 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.4284 |
| Learning rate (final) | 8.56e-04 |

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

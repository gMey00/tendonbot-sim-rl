# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_frankenstein_osc` |
| Run ID | `2026-07-13_10-41-26_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.4855 |
| **Mean position error** | **12.1 cm** |
| **Position reached (< 5 cm)** | **21 % of steps** |
| Orientation reached (< 0.3 rad) | 25 % of steps |
| Pose reached (both) | 6 % of steps |
| Position tracking reward | -0.0243 |
| Orientation tracking reward | -0.0508 |
| Position fine-grained reward | 0.032054 |
| Action rate penalty | -0.030655 |
| Joint velocity penalty | -0.006434 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.4802 |
| Learning rate (final) | 8.78e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_frankenstein_osc/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_frankenstein_osc/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_frankenstein_osc/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_frankenstein_osc/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_frankenstein_osc/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_frankenstein_osc/06_episode_length.png)

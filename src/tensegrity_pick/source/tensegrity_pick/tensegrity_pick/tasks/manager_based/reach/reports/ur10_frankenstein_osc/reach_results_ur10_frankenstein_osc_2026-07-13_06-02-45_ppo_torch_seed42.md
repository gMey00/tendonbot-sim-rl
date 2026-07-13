# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_frankenstein_osc` |
| Run ID | `2026-07-13_06-02-45_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.6914 |
| **Mean position error** | **18.4 cm** |
| **Position reached (< 5 cm)** | **7 % of steps** |
| Orientation reached (< 0.3 rad) | 29 % of steps |
| Pose reached (both) | 3 % of steps |
| Position tracking reward | -0.0367 |
| Orientation tracking reward | -0.0512 |
| Position fine-grained reward | 0.018811 |
| Action rate penalty | -0.035619 |
| Joint velocity penalty | -0.005907 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.5238 |
| Learning rate (final) | 9.38e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_frankenstein_osc/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_frankenstein_osc/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_frankenstein_osc/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_frankenstein_osc/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_frankenstein_osc/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_frankenstein_osc/06_episode_length.png)

# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_frankenstein_ik` |
| Run ID | `2026-07-13_09-36-56_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.0048 |
| **Mean position error** | **6.4 cm** |
| **Position reached (< 5 cm)** | **74 % of steps** |
| Orientation reached (< 0.3 rad) | 48 % of steps |
| Pose reached (both) | 43 % of steps |
| Position tracking reward | -0.0128 |
| Orientation tracking reward | -0.0415 |
| Position fine-grained reward | 0.063228 |
| Action rate penalty | -0.007800 |
| Joint velocity penalty | -0.003243 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2896 |
| Learning rate (final) | 1.70e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_frankenstein_ik/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_frankenstein_ik/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_frankenstein_ik/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_frankenstein_ik/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_frankenstein_ik/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_frankenstein_ik/06_episode_length.png)

# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_frankenstein` |
| Run ID | `2026-07-13_09-09-17_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.1211 |
| **Mean position error** | **4.5 cm** |
| **Position reached (< 5 cm)** | **86 % of steps** |
| Orientation reached (< 0.3 rad) | 68 % of steps |
| Pose reached (both) | 66 % of steps |
| Position tracking reward | -0.0090 |
| Orientation tracking reward | -0.0291 |
| Position fine-grained reward | 0.072109 |
| Action rate penalty | -0.003417 |
| Joint velocity penalty | -0.011565 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.1223 |
| Learning rate (final) | 2.02e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_frankenstein/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_frankenstein/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_frankenstein/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_frankenstein/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_frankenstein/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_frankenstein/06_episode_length.png)

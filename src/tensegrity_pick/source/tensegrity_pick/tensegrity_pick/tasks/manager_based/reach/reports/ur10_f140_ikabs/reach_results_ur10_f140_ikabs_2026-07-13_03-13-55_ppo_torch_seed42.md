# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_f140_ikabs` |
| Run ID | `2026-07-13_03-13-55_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.2889 |
| **Mean position error** | **13.3 cm** |
| **Position reached (< 5 cm)** | **34 % of steps** |
| Orientation reached (< 0.3 rad) | 40 % of steps |
| Pose reached (both) | 17 % of steps |
| Position tracking reward | -0.0266 |
| Orientation tracking reward | -0.0418 |
| Position fine-grained reward | 0.036833 |
| Action rate penalty | -0.010245 |
| Joint velocity penalty | -0.005987 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.3302 |
| Learning rate (final) | 1.54e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_f140_ikabs/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_f140_ikabs/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_f140_ikabs/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_f140_ikabs/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_f140_ikabs/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_f140_ikabs/06_episode_length.png)

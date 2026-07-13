# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_f140_ikabs` |
| Run ID | `2026-07-13_07-52-45_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.0273 |
| **Mean position error** | **5.8 cm** |
| **Position reached (< 5 cm)** | **78 % of steps** |
| Orientation reached (< 0.3 rad) | 53 % of steps |
| Pose reached (both) | 48 % of steps |
| Position tracking reward | -0.0115 |
| Orientation tracking reward | -0.0366 |
| Position fine-grained reward | 0.066148 |
| Action rate penalty | -0.007436 |
| Joint velocity penalty | -0.006096 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2713 |
| Learning rate (final) | 1.29e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_f140_ikabs/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_f140_ikabs/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_f140_ikabs/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_f140_ikabs/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_f140_ikabs/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_f140_ikabs/06_episode_length.png)

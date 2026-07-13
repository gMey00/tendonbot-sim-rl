# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_f140_ik` |
| Run ID | `2026-07-13_07-21-50_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.0384 |
| **Mean position error** | **6.5 cm** |
| **Position reached (< 5 cm)** | **67 % of steps** |
| Orientation reached (< 0.3 rad) | 40 % of steps |
| Pose reached (both) | 33 % of steps |
| Position tracking reward | -0.0129 |
| Orientation tracking reward | -0.0432 |
| Position fine-grained reward | 0.059517 |
| Action rate penalty | -0.006805 |
| Joint velocity penalty | -0.003239 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2755 |
| Learning rate (final) | 2.05e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_f140_ik/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_f140_ik/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_f140_ik/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_f140_ik/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_f140_ik/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_f140_ik/06_episode_length.png)

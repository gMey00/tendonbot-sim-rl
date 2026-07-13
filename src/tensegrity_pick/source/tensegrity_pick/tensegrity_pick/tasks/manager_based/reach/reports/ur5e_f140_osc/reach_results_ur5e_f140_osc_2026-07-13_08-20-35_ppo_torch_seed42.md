# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_f140_osc` |
| Run ID | `2026-07-13_08-20-35_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.3855 |
| **Mean position error** | **11.0 cm** |
| **Position reached (< 5 cm)** | **26 % of steps** |
| Orientation reached (< 0.3 rad) | 33 % of steps |
| Pose reached (both) | 11 % of steps |
| Position tracking reward | -0.0220 |
| Orientation tracking reward | -0.0454 |
| Position fine-grained reward | 0.035011 |
| Action rate penalty | -0.025904 |
| Joint velocity penalty | -0.005191 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.4417 |
| Learning rate (final) | 8.78e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_f140_osc/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_f140_osc/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_f140_osc/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_f140_osc/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_f140_osc/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_f140_osc/06_episode_length.png)

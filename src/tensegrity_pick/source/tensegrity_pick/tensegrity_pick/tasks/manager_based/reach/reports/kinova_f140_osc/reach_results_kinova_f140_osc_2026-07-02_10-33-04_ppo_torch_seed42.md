# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_f140_osc` |
| Run ID | `2026-07-02_10-33-04_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.1775 |
| **Mean position error** | **3.7 cm** |
| **Position reached (< 5 cm)** | **87 % of steps** |
| Orientation reached (< 0.3 rad) | 9 % of steps |
| Pose reached (both) | 8 % of steps |
| Position tracking reward | -0.0074 |
| Orientation tracking reward | -0.0750 |
| Position fine-grained reward | 0.074721 |
| Action rate penalty | -0.016356 |
| Joint velocity penalty | -0.005801 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.3220 |
| Learning rate (final) | 4.35e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova_f140_osc/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova_f140_osc/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova_f140_osc/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova_f140_osc/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova_f140_osc/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova_f140_osc/06_episode_length.png)

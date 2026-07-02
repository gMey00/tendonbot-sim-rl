# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_f140_ik` |
| Run ID | `2026-07-02_10-33-04_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.0586 |
| **Mean position error** | **4.0 cm** |
| **Position reached (< 5 cm)** | **89 % of steps** |
| Orientation reached (< 0.3 rad) | 26 % of steps |
| Pose reached (both) | 26 % of steps |
| Position tracking reward | -0.0081 |
| Orientation tracking reward | -0.0670 |
| Position fine-grained reward | 0.075783 |
| Action rate penalty | -0.004129 |
| Joint velocity penalty | -0.005592 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2057 |
| Learning rate (final) | 8.62e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova_f140_ik/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova_f140_ik/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova_f140_ik/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova_f140_ik/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova_f140_ik/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova_f140_ik/06_episode_length.png)

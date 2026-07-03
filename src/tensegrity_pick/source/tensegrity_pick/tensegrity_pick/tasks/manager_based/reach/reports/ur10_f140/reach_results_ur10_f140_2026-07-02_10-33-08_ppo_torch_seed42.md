# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_f140` |
| Run ID | `2026-07-02_10-33-08_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.5669 |
| **Mean position error** | **3.2 cm** |
| **Position reached (< 5 cm)** | **91 % of steps** |
| Orientation reached (< 0.3 rad) | 0 % of steps |
| Pose reached (both) | 0 % of steps |
| Position tracking reward | -0.0064 |
| Orientation tracking reward | -0.1622 |
| Position fine-grained reward | 0.082337 |
| Action rate penalty | -0.003673 |
| Joint velocity penalty | -0.004609 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.1165 |
| Learning rate (final) | 2.30e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_f140/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_f140/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_f140/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_f140/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_f140/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_f140/06_episode_length.png)

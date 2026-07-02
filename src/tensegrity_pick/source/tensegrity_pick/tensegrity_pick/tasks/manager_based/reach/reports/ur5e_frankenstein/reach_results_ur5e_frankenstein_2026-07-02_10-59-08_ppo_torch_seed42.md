# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_frankenstein` |
| Run ID | `2026-07-02_10-59-08_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.2227 |
| **Mean position error** | **3.7 cm** |
| **Position reached (< 5 cm)** | **87 % of steps** |
| Orientation reached (< 0.3 rad) | 0 % of steps |
| Pose reached (both) | 0 % of steps |
| Position tracking reward | -0.0074 |
| Orientation tracking reward | -0.0829 |
| Position fine-grained reward | 0.074279 |
| Action rate penalty | -0.010785 |
| Joint velocity penalty | -0.010640 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.2732 |
| Learning rate (final) | 1.44e-04 |

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

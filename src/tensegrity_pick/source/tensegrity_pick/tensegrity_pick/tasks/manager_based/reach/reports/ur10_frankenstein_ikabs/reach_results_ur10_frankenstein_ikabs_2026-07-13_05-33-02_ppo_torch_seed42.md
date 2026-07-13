# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur10_frankenstein_ikabs` |
| Run ID | `2026-07-13_05-33-02_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.2045 |
| **Mean position error** | **10.5 cm** |
| **Position reached (< 5 cm)** | **50 % of steps** |
| Orientation reached (< 0.3 rad) | 36 % of steps |
| Pose reached (both) | 22 % of steps |
| Position tracking reward | -0.0210 |
| Orientation tracking reward | -0.0439 |
| Position fine-grained reward | 0.047877 |
| Action rate penalty | -0.013248 |
| Joint velocity penalty | -0.005460 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.3907 |
| Learning rate (final) | 1.34e-03 |

## Figures

### Total Reward

![Total Reward](../../figures/ur10_frankenstein_ikabs/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur10_frankenstein_ikabs/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur10_frankenstein_ikabs/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur10_frankenstein_ikabs/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur10_frankenstein_ikabs/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur10_frankenstein_ikabs/06_episode_length.png)

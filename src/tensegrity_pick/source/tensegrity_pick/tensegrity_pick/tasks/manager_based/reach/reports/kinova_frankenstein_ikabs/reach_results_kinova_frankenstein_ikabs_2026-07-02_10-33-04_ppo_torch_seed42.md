# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `kinova_frankenstein_ikabs` |
| Run ID | `2026-07-02_10-33-04_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | 0.1465 |
| **Mean position error** | **2.4 cm** |
| **Position reached (< 5 cm)** | **90 % of steps** |
| Orientation reached (< 0.3 rad) | 40 % of steps |
| Pose reached (both) | 40 % of steps |
| Position tracking reward | -0.0048 |
| Orientation tracking reward | -0.0455 |
| Position fine-grained reward | 0.080390 |
| Action rate penalty | -0.000994 |
| Joint velocity penalty | -0.004634 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.0711 |
| Learning rate (final) | 4.64e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/kinova_frankenstein_ikabs/01_total_reward.png)

### Task Success

![Task Success](../../figures/kinova_frankenstein_ikabs/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/kinova_frankenstein_ikabs/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/kinova_frankenstein_ikabs/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/kinova_frankenstein_ikabs/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/kinova_frankenstein_ikabs/06_episode_length.png)

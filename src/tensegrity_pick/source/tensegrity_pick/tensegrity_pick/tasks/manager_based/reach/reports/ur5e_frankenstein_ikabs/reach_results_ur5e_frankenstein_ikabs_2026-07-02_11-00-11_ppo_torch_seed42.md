# Reach Results Report

## Run Metadata

| Field | Value |
|---|---|
| Variant | `ur5e_frankenstein_ikabs` |
| Run ID | `2026-07-02_11-00-11_ppo_torch_seed42` |

## Converged Metrics (last 10% of training)

| Metric | Value |
|---|---|
| Total episode reward (mean) | -0.1027 |
| **Mean position error** | **3.8 cm** |
| **Position reached (< 5 cm)** | **87 % of steps** |
| Orientation reached (< 0.3 rad) | 0 % of steps |
| Pose reached (both) | 0 % of steps |
| Position tracking reward | -0.0077 |
| Orientation tracking reward | -0.0768 |
| Position fine-grained reward | 0.075941 |
| Action rate penalty | -0.001465 |
| Joint velocity penalty | -0.007653 |
| Mean episode length | 180.0 steps |
| Policy std deviation | 0.0912 |
| Learning rate (final) | 2.97e-04 |

## Figures

### Total Reward

![Total Reward](../../figures/ur5e_frankenstein_ikabs/01_total_reward.png)

### Task Success

![Task Success](../../figures/ur5e_frankenstein_ikabs/02_task_success.png)

### Reward Decomposition

![Reward Decomposition](../../figures/ur5e_frankenstein_ikabs/03_reward_decomposition.png)

### Policy Diagnostics

![Policy Diagnostics](../../figures/ur5e_frankenstein_ikabs/04_policy_diagnostics.png)

### Converged Breakdown

![Converged Breakdown](../../figures/ur5e_frankenstein_ikabs/05_converged_breakdown.png)

### Episode Length

![Episode Length](../../figures/ur5e_frankenstein_ikabs/06_episode_length.png)

# Reinforcement Learning

Notes on the RL training pipeline used in this project.

---

## Framework

This project uses [skrl](https://skrl.readthedocs.io/) with **PPO** (Proximal Policy
Optimization) as the primary training algorithm.  Training runs are launched via Isaac Lab's
integration scripts:

```bash
python scripts/skrl/train.py --task=<TASK> --headless
```

Configuration files for each task live under
`source/tensegrity_pick/tensegrity_pick/tasks/manager_based/<task>/agents/skrl_ppo_cfg.yaml`.

---

## Evaluating a Learned Policy

```bash
# Play with a trained checkpoint (use the -Play- variant for eval settings)
python scripts/skrl/play.py --task=<TASK>-Play-v0 --num_envs=10
```

### Metrics

Training progress is logged to **TensorBoard**:

```bash
tensorboard --logdir logs/skrl/
```

Key metrics to monitor:

| Metric | Meaning |
|--------|---------|
| `Reward / Total reward (mean)` | Average episodic return across all envs |
| `Reward / Total reward (max)` | Best episodic return (convergence indicator) |
| `Loss / Policy loss` | PPO surrogate loss |
| `Loss / Value loss` | Critic MSE loss |

### Evaluate Learning Function

When comparing PD-driven vs. tendon-driven environments, train both variants with identical
hyperparameters and compare the reward curves and final evaluation performance on the
corresponding `-Play-v0` environments.
- Tool: [TensorBoard](https://www.tensorflow.org/tensorboard)
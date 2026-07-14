# Optimization Tracking

[← Back to reports](../README.md)

Per-task **optimization tracking logs** — the dated, iteration-by-iteration
change history for each task's reward structure, MDP design, PPO
hyperparameters, and training pipeline. Each entry records *what* was changed,
*why*, and the observed training outcome (the repo's "project memory" for a
task).

These are the *process* records; the final seed-aggregated outcome reports live
in [`../task_evaluations/`](../task_evaluations/README.md).

## Logs

| Log | Task | Description |
|-----|------|-------------|
| [`reach_optimization_tracking.md`](reach_optimization_tracking.md) | Reach | Full change history (iterations 0–22) incl. NaN-divergence fixes and the physical-variant / action-space studies |
| [`cube_place_optimization_tracking.md`](cube_place_optimization_tracking.md) | Cube Place | Reward and pipeline optimization for the tensegrity cube-place task |
| [`cube_sort_optimization_tracking.md`](cube_sort_optimization_tracking.md) | Cube Sort | MDP-redesign rework log (R1–R7); 2G+1R validated, 4G+2R scaling |
| [`shirt_pick_optimization_tracking.md`](shirt_pick_optimization_tracking.md) | Shirt Pick | Pipeline task 1 — retriever grasp from the moving belt |
| [`shirt_place_optimization_tracking.md`](shirt_place_optimization_tracking.md) | Shirt Place | Pick→place variant — deterministic drum placement (final: 100% place, ~93% cloth-in-drum) |
| [`shirt_present_optimization_tracking.md`](shirt_present_optimization_tracking.md) | Shirt Present | Pipeline task 2 — second-arm stretch/presentation for inspection |
| [`shirt_distribute_optimization_tracking.md`](shirt_distribute_optimization_tracking.md) | Shirt Distribute | Pipeline task 3 — goal-conditioned bin sorting |

## Related

- [Task evaluation reports](../task_evaluations/README.md) — the aggregated outcomes these logs lead to
- [Project TODO / roadmap](../../TODO.md)

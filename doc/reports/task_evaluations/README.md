# Task Evaluations

[← Back to reports](../README.md)

Comprehensive, **seed-aggregated** evaluation reports for the trained tensegrity
tasks. Each report trains multiple PPO seeds per variant, evaluates every agent
against control baselines (zero-action, uniform random) over many
seeds × parallel-environments × episodes, and reports aggregate success/error
statistics, per-variant deep dives, figures, and reproducibility pointers.

These are the *outcome* reports (how well the final policies perform); the
iteration-by-iteration history that produced them lives in
[`../tracking/`](../tracking/README.md).

## Reports

| Report | Description |
|--------|-------------|
| [`reach_evaluation_findings.md`](reach_evaluation_findings.md) | Reach task — seed-aggregated training/evaluation across the tensegrity variants (PD / tendon / physical-tendon) plus baselines; includes the 2026-07 physical-variant rework status |
| [`cube_place_evaluation_findings.md`](cube_place_evaluation_findings.md) | Cube Place task — the place-task analogue of the reach report (5 seeds × 3 variants, 15 PPO runs) filling the single-seed gap left by the project thesis |

## Related

- [Optimization tracking logs](../tracking/README.md) — the change history behind these results
- [`../physical_variant_fix_report.md`](../physical_variant_fix_report.md) — physical-tendon reach rework referenced by the reach evaluation

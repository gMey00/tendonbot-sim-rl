# Reports

[← Back to documentation index](../README.md)

Per-task training/evaluation reports, optimization change logs, and pipeline
studies for the tensegrity-manipulator and cloth-sorting work. Reports are the
authoritative, dated record behind the results tables in the task READMEs and
the thesis chapters.

## Layout

| Path | Contents |
|------|----------|
| [`task_evaluations/`](task_evaluations/README.md) | Comprehensive, seed-aggregated evaluation reports (trained policy vs. baselines) |
| [`tracking/`](tracking/README.md) | Per-task optimization tracking logs — dated iterations with what/why/outcome |
| *(this folder)* | Standalone pipeline / benchmark / de-risk studies (listed below) |
| `figures/`, `data/` | Supporting artifacts (plots and raw CSV/JSON/benchmark records) referenced by the reports |
| `tmp/` | Scratch — superseded research briefs and drafts, not part of the curated hierarchy |

## Standalone reports

| Report | Description |
|--------|-------------|
| [`benchmark_performance_report.md`](benchmark_performance_report.md) | Computational performance of the GPU-parallel cloth-manipulation RL pipeline (Isaac Lab / PhysX 5, `skrl` PPO); cross-GPU comparison of RTX A6000 vs. RTX PRO 6000 Blackwell — MA benchmarking chapter |
| [`present_heuristics_study.md`](present_heuristics_study.md) | Robot-free grasp-heuristics study for shirt presentation: which grasp points/strategies maximise camera-plane visibility (success threshold + regrasp-target calibration for `shirt_present`) |
| [`cloth_stage0_physics_derisk.md`](cloth_stage0_physics_derisk.md) | Stage-0 physics de-risk for the cloth-sorting pipeline: PBD particle-cloth stability and attachment-slot behaviour, prerequisite to the shirt-pick implementation |
| [`physical_variant_fix_report.md`](physical_variant_fix_report.md) | Rework of the physical (body-force, antiparallelogram) tendon reach variant to a trainable state (2026-07-09) |

## Related

- [Documentation index](../README.md)
- [Project TODO / roadmap](../TODO.md)
- [Source code & tasks](../../src/tensegrity_pick/README.md) — each task README links its own tracking log and evaluation report

# Benchmark results (raw data)

Machine-tagged, reproducible benchmark artifacts produced by
[`src/tensegrity_pick/scripts/benchmarking/`](../../../../src/tensegrity_pick/scripts/benchmarking/).
Analysed in [`../../benchmark_performance_report.md`](../../benchmark_performance_report.md).

## Layout

```
<spec>/<machine>_<timestamp>/
  spec.yaml            # exact spec used
  manifest.jsonl       # expanded points (array index = line)
  environment.json     # hardware/software/git provenance at run time
  points/point_<i>.json  # per-point RESULT record (lossless)
  points/point_<i>.log   # per-point stdout+stderr (PhysX overflow grep source)
  results.jsonl        # aggregated, physics-gated records (source of truth)
  results.csv          # flattened scalar columns for quick inspection
```

## Runs in this directory (RTX A6000 workstation)

| Spec | Points | Valid | Notes |
|---|---:|---:|---|
| `rigid_baseline` | 27 | 27 | reach UR5e, N=16…4096, cloth-free reference |
| `action_space_cost` | 24 | 24 | joint / IK-rel / IK-abs / OSC × N |
| `robot_dof_matrix` | 48 | 48 | 8 morphologies × N (incl. tensegrity wrist/tendon) |
| `env_count_cloth_quick` | 18 | 15 | PBD cloth knee; **N=512 (3 pts) invalid — PhysX overflow** |
| `optimization_levers` | 12 | 12 | decimation × PBD solver iterations, N=64 cloth |

`*_BROKEN_*` directories are archived invalid runs kept for provenance (see the
report's methodology notes) — a first `optimization_levers` run in which the
solver-iteration override was clobbered by the task's `__post_init__` before the
harness fix. Excluded from analysis.

## Regenerate

```bash
# figures  → doc/reports/figures/benchmarks/
python3 src/tensegrity_pick/scripts/benchmarking/plot_benchmarks.py --auto
# validity report (physics gate, overflow, seed CV)
python3 src/tensegrity_pick/scripts/benchmarking/validate_results.py --auto
```

## Cross-GPU (Alex RTX PRO 6000)

`TODO(alex)` — not yet collected. Submit the same specs with
`--dispatch slurm --stage`; results land under
`<spec>/alex_<timestamp>/` and merge directly (same schema, machine-tagged).

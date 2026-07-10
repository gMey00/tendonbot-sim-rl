# Benchmarking suite (`scripts/benchmarking/`)

Reproducible performance benchmarking for the cloth-sorting RL pipeline, on the
**A6000 workstation** and the **Alex RTX PRO 6000** cluster. It measures simulator
throughput, GPU/host memory, power/energy, and per-step latency across the axes
the thesis compares: env count, cloth resolution, action space, robot morphology,
and DOF count.

> **Skeleton status.** This is the implementation scaffold for the study designed
> in [`doc/reports/RESEARCH_PROMPT_benchmarking_suite.md`](../../../../doc/reports/RESEARCH_PROMPT_benchmarking_suite.md).
> The core harness, metric collectors, matrix runner, SLURM dispatch, and provenance
> capture are functional. Parameter grids and the final metric set carry
> `TODO(research)` markers to be closed once the research report returns. Specs that
> need config-path or asset prerequisites are marked **EXPERIMENTAL** in their header.

## Layout

| File | Role |
|---|---|
| `bench_core.py` | Atomic harness: benchmark **one** `(task, num_envs, overrides)` point in one process. Emits a single `RESULT_JSON,{…}` line. |
| `bench_metrics.py` | Metric collectors (no Isaac imports): GPU telemetry sampler (`pynvml`→`nvidia-smi`), torch memory counters, host RAM/CPU, per-step latency. |
| `bench_env.py` | Reproducibility capture: GPU/driver/CUDA/torch/isaac/skrl versions, git provenance, machine profile, SLURM context. |
| `run_matrix.py` | Expand a spec YAML → manifest → dispatch (local loop \| SLURM array) → aggregate to JSONL + CSV; folds the per-point PhysX-overflow grep into the physics-validity gate. |
| `plot_benchmarks.py` | Thesis-ready figures (Okabe-Ito CVD-safe palette, PDF+PNG) from `results.jsonl`. Run with **`/usr/bin/python3`** (needs matplotlib, not in the Isaac env): `python3 plot_benchmarks.py --auto`. |
| `validate_results.py` | Validity report: physics gate, overflow, errored points, seed-dispersion CV (>15 % warned). `python3 validate_results.py --auto`. |
| `specs/*.yaml` | Experiment definitions, one per research question. |

Results, figures, and the thesis-ready analysis live under
[`doc/reports/`](../../../../doc/reports/): `benchmark_performance_report.md`,
`figures/benchmarks/`, `data/benchmarks/`.

Why one process per point: Isaac Sim cannot rebuild its stage in-process, so each
configuration is a fresh Python process — the same constraint the original
`model_validation/bench_cloth_env_count.py` worked around. This suite generalises
that script (any task, config-driven overrides, richer metrics, provenance).

## Quickstart

```bash
cd src/tensegrity_pick
conda activate env_isaaclab            # sets PYTHONPATH; do NOT unset it

# 1. Inspect what a spec expands to (writes a manifest, runs nothing):
python scripts/benchmarking/run_matrix.py specs/env_count_sweep.yaml --dispatch expand

# 2. Run it on this workstation (A6000):
python scripts/benchmarking/run_matrix.py specs/env_count_sweep.yaml --dispatch local

# 3. One-off single point (no spec):
python scripts/benchmarking/bench_core.py --headless \
    --task Template-Tensegrity-Shirt-Place-v0 --num_envs 64
```

Paths in `run_matrix.py` are relative to `src/tensegrity_pick`; run it from there.

## Running on Alex (SLURM)

```bash
# from a login node, repo checked out, .config/env_vars.sh present
python scripts/benchmarking/run_matrix.py specs/env_count_sweep.yaml \
    --dispatch slurm --time 04:00:00 --max-parallel 8 --stage --dry-run   # preview
# drop --dry-run to submit; array index = point index, mirrors tools/train_alex.sh
# env (gcc/conda/proxy/gres). --stage copies res/ to node-local $TMPDIR.

# after the array finishes, aggregate:
python scripts/benchmarking/run_matrix.py --dispatch collect \
    --run-dir doc/reports/data/benchmarks/env_count_sweep/alex_<timestamp>
```

## Outputs

Per run: `doc/reports/data/benchmarks/<spec>/<machine>_<timestamp>/`
```
spec.yaml           # exact spec used
manifest.jsonl      # one expanded point per line (array index = line)
environment.json    # provenance snapshot at expansion time
points/point_<i>.json   # per-point RESULT record (lossless)
results.jsonl       # aggregated records (source of truth)
results.csv         # flattened scalar columns for quick inspection / plotting
```
Machine + timestamp in the path means A6000 and Alex results never collide and are
directly comparable (each row also carries the full `environment` provenance blob).

## Specs

| Spec | Question | Runnable now? |
|---|---|---|
| `env_count_sweep.yaml` | A — throughput knee vs. env count | ✅ |
| `action_space_cost.yaml` | C7 — joint/IK-Rel/IK-Abs/OSC cost | ✅ |
| `robot_dof_matrix.yaml` | C8/C9 — arms, DOF, tensegrity-wrist/tendon overhead | ✅ |
| `optimization_levers.yaml` | E — decimation × PBD solver iters ablation | ✅ (paths verified) |
| `vertex_x_envcount.yaml` | B — cloth resolution × env count | ⚠️ path verified; still needs authored lo/hi-res USD assets |

### Spec format

```yaml
name: my_experiment
base:                       # applied to every point
  task: Template-...-v0
  num_envs: 64              # default; usually overridden by an axis
  warmup_steps: 30
  bench_steps: 240
  seed: 0
  overrides: {}             # dotted env_cfg paths applied to every point
  cloth_cfg_ref: "pkg.mod:SINGLETON"   # needed only if you sweep cloth.* keys
axes:                       # Cartesian product across these
  num_envs: [32, 64, 128]   # 'task'/'num_envs'/'seed' are special top-level keys
  sim.dt: [0.0166, 0.0083]  # any other key = dotted env_cfg override path
  cloth.usd_path: [a, b]    # 'cloth.<path>' = override on the cloth cfg singleton
repeats: 2                  # each repeat uses seed = base.seed + repeat
points: []                  # optional explicit extra points (override dicts)
filters: {exclude: []}      # drop points matching all key/values of a rule
```

Overrides are config-driven on purpose: `dt`, decimation, and PhysX buffer sizes
are swept **without code changes** as dotted `env_cfg` paths, reading the project's
own registered task configs. An unknown path raises (fails loudly, never benchmarks
the wrong thing); each record stores requested + read-back `applied_overrides`.

**Cloth params are special.** The cloth USD path and PBD solver iterations live on
a *module-level singleton* (e.g. `SHIRT_CLOTH_CFG`) that the env reads directly —
`parse_env_cfg` only deep-copies it into the spawn event, so a normal `env_cfg`
override would silently no-op. Sweep them with `cloth.<path>` axes + a
`base.cloth_cfg_ref: "module:attr"`; `bench_core` mutates that singleton *before*
`parse_env_cfg`, keeping the spawned mesh and the `ClothObject` view consistent.
Verified ref for shirt_place:
`tensegrity_pick.tasks.manager_based.shirt_place.shirt_place_scene_cfg:SHIRT_CLOTH_CFG`.

Verified reachable levers (Template-Tensegrity-Shirt-Place-v0): `decimation`,
`sim.dt`, `sim.physx.solver_type`, `sim.physx.gpu_collision_stack_size`,
`sim.physx.gpu_max_particle_contacts` (env_cfg); `pbd_params.solver_position_iterations`,
`usd_path` (cloth singleton).

## Metrics collected (baseline)

throughput (`steps_per_s`, `env_steps_per_s`), per-step latency (`step_ms_p50/p95/max`),
memory (`cuda_alloc/reserved/max_*`, `driver_used_gb`, `gpu_mem_used_mb_max`, host RAM),
GPU telemetry (`gpu_power_w_mean/max`, `gpu_util_mean`, `gpu_sm_clock`, `gpu_temp`),
`gpu_energy_j` + `energy_j_per_Mstep`, construction/first-reset/reset time,
obs/action dims, cloth particle counts, and a physics-sanity flag.

`TODO(research)`: SM-occupancy, energy convention (J vs Wh, per-Mstep normalisation),
and any cloth-fidelity-vs-speed metric will be finalised from the research report.

## Notes / gotchas

- Run inside `env_isaaclab` (conda activation sets `PYTHONPATH`; do not unset it).
- `pynvml` is not installed by default → telemetry falls back to `nvidia-smi`
  polling. `pip install nvidia-ml-py` for lower-overhead in-process sampling.
- Per-step `torch.cuda.synchronize()` in the measured window gives an honest
  latency distribution but adds a small fixed overhead — kept identical across all
  points so comparisons stay fair. (An un-synced throughput-only mode can be added
  if the research report wants the two reported separately.)
- The existing `model_validation/bench_cloth_env_count.{py,sh}` is superseded by
  `env_count_sweep.yaml`; keep it until this suite is validated against it.

# Model Validation Scripts

Validates the tensegrity arm simulation against Klein (2023) by running step-response
tests and comparing timing metrics (rise time, settling time, overshoot, NRMSE) to the
physical robot and Gazebo reference results.

## Scripts

| Script | Requires IsaacLab | Description |
|--------|:-----------------:|-------------|
| `run_validation.sh` | ✓ | **Pipeline script** — runs all steps sequentially |
| `run_step_response_pd.py` | ✓ | Step responses for the PD arm (ImplicitActuator, K=400, D=20) |
| `run_step_response_tendon.py` | ✓ | Step responses for the tendon arm (PID + Jacobian); supports `--variant elbow_approx` (default) and `--variant physical` |
| `run_step_response_base.py` | ✓ | Step responses for the prismatic base joints |
| `plot_validation.py` | ✗ | Reads NPZ data files and generates all plots (needs `matplotlib` from conda env) |
| `common.py` | ✗ | Shared constants, control math, metrics, NPZ I/O |

All data-generation scripts write compressed `.npz` files to
`outputs/model_validation/{pd,tendon,base}/data/` and print a per-trial
summary to stdout.  Run `plot_validation.py` afterwards to generate
PNG plots in `outputs/model_validation/plots/`.

---

## Standard Workflow

> **Recommended: use the pipeline script** — it runs all steps, shows live output,
> and writes per-step log files automatically:
>
> ```bash
> cd /home/robot/studentische-arbeiten/src/tensegrity_pick
> bash scripts/model_validation/run_validation.sh
> ```
>
> With optional damping sweeps:
> ```bash
> bash scripts/model_validation/run_validation.sh --damping-sweep
> ```
>
> Run a single step only:
> ```bash
> bash scripts/model_validation/run_validation.sh --pd-only
> bash scripts/model_validation/run_validation.sh --tendon-only           # elbow_approx
> bash scripts/model_validation/run_validation.sh --tendon-physical-only  # physical linkage
> bash scripts/model_validation/run_validation.sh --base-only
> bash scripts/model_validation/run_validation.sh --plots-only
> ```

### Running scripts individually

> ⚠️ **Output-buffering note:** Isaac Sim scripts buffer stdout by default.
> Always use `conda run --no-capture-output` and `python3 -u` (unbuffered)
> together, otherwise you will see no output for the entire runtime (~5–15 min).

All commands are run from the **`src/tensegrity_pick`** directory.

```bash
cd /home/robot/studentische-arbeiten/src/tensegrity_pick
```

### Step 1 — PD arm validation

```bash
conda run --no-capture-output -n env_isaaclab python3 -u \
    scripts/model_validation/run_step_response_pd.py --headless
```

Runs all 3 arm joints (elbow, wrist_y, wrist_x) at their standard step
amplitudes (20°/30°/40° for elbow, 10°/20°/30° for wrists).
Pass/fail printed per trial.

#### Optional: include damping sweep (re-runs elbow at D = 120, 60, 30, 25, 20, 15)

```bash
conda run --no-capture-output -n env_isaaclab python3 -u \
    scripts/model_validation/run_step_response_pd.py --headless --damping_sweep
```

### Step 2 — Tendon arm validation (elbow_approx)

```bash
conda run --no-capture-output -n env_isaaclab python3 -u \
    scripts/model_validation/run_step_response_tendon.py --headless --variant elbow_approx
```

Uses PID + Jacobian transpose control.  Reports NRMSE alongside Klein
(2023) Table 4.2 reference values.

### Step 2b — Tendon arm validation (physical linkage)

```bash
conda run --no-capture-output -n env_isaaclab python3 -u \
    scripts/model_validation/run_step_response_tendon.py --headless --variant physical
```

Same PID controller and wrist J^T; elbow uses body-force tendons on the
antiparallelogram linkage.  Data written to `tendon_physical/data/`.

### Step 3 — Base joint validation

```bash
conda run --no-capture-output -n env_isaaclab python3 -u \
    scripts/model_validation/run_step_response_base.py --headless
```

Tests base_y (50/100/200 mm) and base_z (30/60/100 mm) with the full
arm + gripper suspended.

#### Optional: include base damping sweep

```bash
conda run --no-capture-output -n env_isaaclab python3 -u \
    scripts/model_validation/run_step_response_base.py --headless --damping_sweep
```

### Step 4 — Generate plots

`matplotlib` is installed inside the conda env, so use `conda run` here too.

```bash
conda run --no-capture-output -n env_isaaclab python3 \
    scripts/model_validation/plot_validation.py
```

Plots are written to `outputs/model_validation/plots/`.

#### Selective plot regeneration

```bash
# PD plots only
conda run --no-capture-output -n env_isaaclab python3 \
    scripts/model_validation/plot_validation.py --no_tendon --no_comparison --no_sweep

# Tendon plots only
conda run --no-capture-output -n env_isaaclab python3 \
    scripts/model_validation/plot_validation.py --no_pd --no_comparison --no_sweep

# Skip damping-sweep plot (if no sweep data was generated)
conda run --no-capture-output -n env_isaaclab python3 \
    scripts/model_validation/plot_validation.py --no_sweep
```

---

## Output Files

```
outputs/model_validation/
├── pd/
│   ├── data/
│   │   ├── elbow_joint__20deg.npz          # standard trial
│   │   ├── elbow_joint__D60__20deg.npz     # damping-sweep trial (tagged with D value)
│   │   └── ...
│   └── log.md                              # run summary written by the script
├── tendon/
│   ├── data/
│   │   └── *.npz
│   └── log.md
├── tendon_physical/
│   ├── data/
│   │   └── *.npz                           # physical model trials
│   └── log.md
├── base/
│   └── data/
│       └── *.npz
└── plots/
    ├── step_response_pd_elbow_joint.png
    ├── step_response_pd_wrist_y_joint.png
    ├── step_response_pd_wrist_x_joint.png
    ├── step_response_tendon_elbow_joint.png
    ├── step_response_tendon_wrist_y_joint.png
    ├── step_response_tendon_wrist_x_joint.png
    ├── step_response_tendon_physical_elbow_physical.png
    ├── step_response_tendon_physical_wrist_y_joint.png
    ├── step_response_tendon_physical_wrist_x_joint.png
    ├── metrics_table_elbow_joint.png
    ├── metrics_table_wrist_y_joint.png
    ├── metrics_table_wrist_x_joint.png
    ├── nrmse_comparison.png
    ├── settling_time_comparison.png
    └── damping_sweep_elbow.png
```

---

## Pass Criteria (Klein 2023 §4.1)

| Metric | Target |
|--------|--------|
| Settling time | 100–300 ms |
| Overshoot | < 5 % |
| Damping ratio ζ | 1.0–1.5 (near-critical) |
| NRMSE (wrist) | ≲ 25 % (matches or beats Klein's Gazebo reference) |
| NRMSE (elbow) | ≲ 40 % (matches or beats Klein's Gazebo reference) |

---

## Notes

- The PD validation script uses inflated effort/velocity limits (400 N·m /
  10 rad/s) so the ImplicitActuator PD dynamics are visible without
  saturation.  **Production limits** are lower (elbow: 40 N·m / 1.0 rad/s,
  wrist: 10 N·m / 0.5 rad/s) and applied in the RL environment configs.
- The tendon validation applies production limits (via `IdealPDActuatorCfg`
  effort passthrough) and is therefore directly representative of RL
  training behaviour.
- `common.py` can be imported on any machine without Isaac Sim — useful for
  offline metric inspection or plotting.

## Related Docs

- [`doc/model_validation_report.md`](../../../../doc/model_validation_report.md) — validation results and analysis
- [`doc/pd_tuning_results.md`](../../../../doc/pd_tuning_results.md) — PD gain sweep results
- [`doc/tendon_simulation.md`](../../../../doc/tendon_simulation.md) — tendon model architecture
- [`res/Tensegrity/README.md`](../../../../res/Tensegrity/README.md) — physical robot specs and actuator limits

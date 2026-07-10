# `scripts/` — Tensegrity-Pick utility scripts

Standalone tooling for the tensegrity-arm / PBD-cloth Isaac Lab project:
training launchers, checkpoint evaluation, result plotting, model & cloth
validation, robot diagnostics, procedural asset generation, and offscreen
rendering. Scripts are grouped by purpose into the subdirectories below.

## Running the scripts

Most scripts import Isaac Sim / Isaac Lab and **must** be launched through the
`env_isaaclab` conda environment, from the extension root
(`src/tensegrity_pick`) so that relative paths (`logs/`, `res/`, `outputs/`)
resolve:

```bash
cd src/tensegrity_pick
source /home/robot/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
unset VIRTUAL_ENV                       # don't let the project .venv shadow conda
env PYTHONUNBUFFERED=1 "$CONDA_PREFIX/bin/python" -u scripts/<subdir>/<script>.py --headless
```

> Use `conda activate` (not `conda run`) — the activation hooks put `isaacsim`
> and `pxr` on `PYTHONPATH`. Do **not** `unset PYTHONPATH`. Isaac startup is
> ~1–3 min warm (cloth tasks longer); prefer running in the background.

The **`plotting/`** scripts are pure Python (matplotlib + tensorboard only) and
can be run without the Isaac Sim PYTHONPATH, though the conda env still provides
the dependencies. They pull the shared FAPS plot style from
[`.config/plot_config.py`](../../../.config/plot_config.py).

---

## Directories

| Directory | Purpose |
|-----------|---------|
| [`agents/`](agents/) | Non-trained rollout agents (zero / random) for env sanity checks and baselines |
| [`asset_generation/`](asset_generation/) | Generate cached cloth initial-state banks (`res/Props/Cloth/banks/*.pt`) |
| [`diagnostics/`](diagnostics/) | Robot / actuator inspection, PD tuning, step-response, env listing |
| [`plotting/`](plotting/) | Training- and evaluation-result figures + task-README reports |
| [`training/`](training/) | End-to-end training / evaluation pipeline drivers (shell) |
| [`model_validation/`](model_validation/) | Arm step-response validation vs Klein (2023) **and** the PBD/XPBD cloth validation suite — see its [README](model_validation/README.md) |
| [`workspace_analysis/`](workspace_analysis/) | 5-DOF reachable-workspace sampling & visualisation (+ notebook) |
| [`rendering/`](rendering/) | Offscreen policy-rollout rendering, incl. Slurm/container helpers — see its [README](rendering/README.md) |
| [`skrl/`](skrl/) | skrl (PPO) train / play / evaluate entry points for every registered task |

---

## `agents/`

| Script | Description |
|--------|-------------|
| `zero_agent.py` | Roll out any registered task with all-zero actions (physics / plumbing sanity check, lower-bound baseline). |
| `random_agent.py` | Roll out any registered task with uniformly random actions (baseline reference). |

## `asset_generation/`

| Script | Description |
|--------|-------------|
| `generate_crumpled_bank.py` | Generate the cached **crumpled**-shirt state bank (Task-1 initial states) → `res/Props/Cloth/banks/tshirt_crumpled_bank.pt`. |
| `generate_hanging_bank.py` | Generate the cached **hanging**-shirt state bank (Task-2/3 initial states) → `res/Props/Cloth/banks/tshirt_hanging_bank.pt`. |

## `diagnostics/`

| Script | Description |
|--------|-------------|
| `verify_actuation.py` | Verify robot actuation: joint exploration + a hardcoded pick-and-place. |
| `verify_robot_reach.py` | Verify robot articulation setup for the reach tasks. |
| `measure_positions.py` | Measure actual body positions in the tensegrity place environment. |
| `tune_pd_gains.py` | PD gain tuning sweep for the tensegrity robot. |
| `step_response_test.py` | Standalone step-response test for the tendon-driven arm (writes plots + CSV). |
| `list_envs.py` | List all registered Gym task / environment IDs. |

## `plotting/` (pure Python)

| Script | Description |
|--------|-------------|
| `analyze_training.py` | Quick TF-events metrics analyzer — prints key training scalars for the latest (or given) run. |
| `plot_reach_training_results.py` | Reach-task figures + README report (`--variant`, `--root`). Driven by `training/train_reach.sh`. |
| `plot_place_training_results.py` | Cube-Place training figures + README report. |
| `plot_place_eval_results.py` | Aggregate cube-place evaluation JSONs (`logs/skrl/cube_place/eval`) into tables + figures. |
| `plot_shirt_training_results.py` | Shirt-Place training figures + README report (reads events via `model_validation/read_tfevents.py`). |
| `plot_shirt_pick_training_results.py` | Shirt-Pick training figures + README report. |
| `plot_sort_training_results.py` | Cube-Sort training figures + README report. |

## `training/` (shell drivers)

| Script | Description |
|--------|-------------|
| `train_reach.sh` | Train reach-task variants (`tensegrity`, `…_tendon`, `ur10e`, `kinova`) then regenerate plots + reports. `--skip-train` for plot-only. |
| `run_place_eval_matrix.sh` | Run the full cube-place evaluation matrix (checkpoint / zero / random × 5 seeds × 3 variants). |

## `model_validation/`

Arm step-response validation against Klein (2023) plus the cloth-fidelity suite.
Key entry points (full details in [`model_validation/README.md`](model_validation/README.md)):

| Script | Description |
|--------|-------------|
| `run_validation.sh` | Pipeline: runs all arm step-response steps sequentially. |
| `run_step_response_pd.py` / `run_step_response_tendon.py` / `run_step_response_base.py` | Step responses for the PD arm / tendon arm / prismatic base. |
| `plot_validation.py` | Plot the step-response `.npz` data (pure Python). |
| `run_shirt_validation.py` | **Cloth validation** — shirt drop, drape, and robot grasp/lift (PBD or XPBD backend). `--record <dir>` writes an MP4 + key PNGs. |
| `validate_shirt_place.py` | Headless validation of the registered `shirt_place` task (belt collision, rewards, success metric). `--record` supported. |
| `render_shirt_check.py` | Minimal "is the shirt visible?" render diagnostic. |
| `test_cloth_metrics.py` | Offline unit test for `shared/cloth_metrics.py` (no Isaac Sim). |
| `baseline_shirt_pick.py`, `diag_*.py`, `probe_present_pose.py`, `check_*.py`, `bench_cloth_env_count.*`, `test_*.py` | Task baselines, Stage-0 de-risk benchmarks, and targeted diagnostics. |
| `_recorder.py`, `common.py`, `read_tfevents.py` | Shared helpers (frame recorder, validation math, tfevents reader). |
| `present_heuristics/` | Presentation-pose heuristic study (grasp-pair sweeps + plots). |

## `workspace_analysis/`

| Script | Description |
|--------|-------------|
| `workspace_sample.py` | Sample the 5-DOF tensegrity robot's reachable workspace. |
| `workspace_visualize.py` | Visualise the sampled workspace (slices / 3-D). |
| `workspace_config.py`, `workspace_analysis_helper.py` | Shared configuration and helper utilities. |
| `workspace_analysis.ipynb` | Interactive workspace-analysis notebook. |

## `rendering/`

Offscreen rendering of trained policy rollouts, including Slurm/Singularity
container helpers. See [`rendering/README.md`](rendering/README.md).

| Script | Description |
|--------|-------------|
| `render_play.py` | Render a trained checkpoint rollout to video (Isaac Lab play + camera). |
| `render_reach_workstation.sh` / `render_reach_alex.sh` | Local / cluster render launchers for the reach task. |
| `sync_checkpoints.sh` | Sync checkpoints from the training node for rendering. |
| `container/` | Singularity image definition + Vulkan probe / render sbatch jobs. |

## `skrl/`

skrl (PPO) entry points shared by every registered task.

| Script | Description |
|--------|-------------|
| `train.py` | Train a policy with skrl PPO on a `--task`. |
| `play.py` | Play / visualise a trained skrl checkpoint. |
| `evaluate_reach.py` / `evaluate_place.py` | Deterministic checkpoint evaluation (metrics → JSON) for reach / cube-place. |
| `evaluate_shirt_pick.py` | Deterministic evaluation of shirt-pick checkpoints (mean actions). |
| `_eval_diag.py` | Evaluation diagnostics helper. |

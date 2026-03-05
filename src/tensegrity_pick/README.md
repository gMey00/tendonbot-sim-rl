# Tensegrity Pick — Isaac Lab RL Tasks

[← Back to project root](../../README.md)

## Overview

Reinforcement-learning tasks for a ceiling-mounted **5-DOF tensegrity manipulator with gripper**, built as an external Isaac Lab extension.
Each task reuses the same base scene (ground plane, conveyor belt, plastic drum, dome light) and robot configuration.

The robot is available in two actuation modes:

- **PD-driven** — Standard joint-position control via `IdealPDActuator`.
- **Tendon-driven** — A custom `TendonEffortAction` maps 5 cable tensions to 3 arm-joint
  torques via the Jacobian transpose, while base and gripper remain PD-controlled.

See [`doc/tendon_simulation.md`](../../doc/tendon_simulation.md) for the full tendon
simulation documentation.

### Task Documentation

| Task | README |
|------|--------|
| Reach | [tensegrity_reach/README.md](source/tensegrity_pick/tensegrity_pick/tasks/manager_based/tensegrity_reach/README.md) |
| Place | [tensegrity_place/README.md](source/tensegrity_pick/tensegrity_pick/tasks/manager_based/tensegrity_place/README.md) |

### Related

- [Robot specification](../../res/Tensegrity/README.md) — kinematic chain, joint constraints, tendon geometry
- [Test suite](../../test/README.md) — pytest tests for actuators, configs, and environments
- [Documentation index](../../doc/README.md) — guides and literature

### Registered Tasks

| Gym ID | Drive | Description |
|--------|-------|-------------|
| `Template-Tensegrity-Pick-v0` | PD | **Cube Sorting** — 8 green + 8 red cubes on a moving conveyor. Pick green cubes and place them into the target drum while ignoring red. |
| `Template-Tensegrity-Reach-v0` | PD | **Reach** — Move the end-effector to a random target position and orientation. No gripper action. |
| `Template-Tensegrity-Reach-Play-v0` | PD | Reach (play/eval variant, 50 envs, no observation noise). |
| `Template-Tensegrity-Reach-Tendon-v0` | Tendon | **Reach (Tendon)** — Same task as Reach but the arm is tendon-driven. Direct comparison with PD variant. |
| `Template-Tensegrity-Reach-Tendon-Play-v0` | Tendon | Reach Tendon (play/eval variant, 50 envs). |
| `Template-Tensegrity-Place-v0` | PD | **Place** — 1 green + 1 red cube below the robot, conveyor inactive. Place the green cube into the drum. Curriculum: green-only → green + red. |
| `Template-Tensegrity-Place-Play-v0` | PD | Place (play/eval variant, 50 envs). |
| `Template-Tensegrity-Place-Tendon-v0` | Tendon | **Place (Tendon)** — Same task as Place but the arm is tendon-driven. |
| `Template-Tensegrity-Place-Tendon-Play-v0` | Tendon | Place Tendon (play/eval variant, 50 envs). |

> New tasks are auto-discovered by `import_packages` — just add a new sub-package under
> `tasks/manager_based/` with an `__init__.py` that calls `gym.register(...)`.

---

## Installation

1. Install Isaac Lab (conda recommended):
   <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>

2. Install this extension in editable mode:

    ```bash
    cd src/tensegrity_pick
    python -m pip install -e source/tensegrity_pick
    ```

3. Verify:

    ```bash
    python scripts/list_envs.py
    ```

---

## Quick Reference

All commands below assume you are inside `src/tensegrity_pick/` and the `env_isaaclab` conda environment is active.
Replace `<TASK>` with any Gym ID from the table above.

### List available environments

```bash
python scripts/list_envs.py
```

### Test with dummy agents

```bash
# Zero-action agent (verify scene loads and steps without errors)
python scripts/zero_agent.py --task=<TASK> --num_envs=10

# Random-action agent (verify all joints move)
python scripts/random_agent.py --task=<TASK> --num_envs=10
```

### Train

```bash
# Headless training (default num_envs from env config)
python scripts/skrl/train.py --task=<TASK> --headless

# Override number of parallel environments
python scripts/skrl/train.py --task=<TASK> --headless --num_envs=2048
```

### Evaluate / Play

```bash
# Play with a trained checkpoint (use the -Play- variant for eval settings)
python scripts/skrl/play.py --task=<TASK_PLAY> --num_envs=10
```

### Task-specific examples

```bash
# --- Cube Sorting (Pick) ---
python scripts/zero_agent.py   --task=Template-Tensegrity-Pick-v0  --num_envs=10
python scripts/skrl/train.py   --task=Template-Tensegrity-Pick-v0  --headless
python scripts/skrl/play.py    --task=Template-Tensegrity-Pick-v0  --num_envs=10

# --- Reach (PD) ---
python scripts/zero_agent.py   --task=Template-Tensegrity-Reach-v0      --num_envs=10
python scripts/skrl/train.py   --task=Template-Tensegrity-Reach-v0      --headless
python scripts/skrl/play.py    --task=Template-Tensegrity-Reach-Play-v0 --num_envs=10

# --- Reach (Tendon) ---
python scripts/zero_agent.py   --task=Template-Tensegrity-Reach-Tendon-v0      --num_envs=10
python scripts/skrl/train.py   --task=Template-Tensegrity-Reach-Tendon-v0      --headless
python scripts/skrl/play.py    --task=Template-Tensegrity-Reach-Tendon-Play-v0 --num_envs=10

# --- Place (PD) ---
python scripts/zero_agent.py   --task=Template-Tensegrity-Place-v0      --num_envs=10
python scripts/skrl/train.py   --task=Template-Tensegrity-Place-v0      --headless
python scripts/skrl/play.py    --task=Template-Tensegrity-Place-Play-v0 --num_envs=10

# --- Place (Tendon) ---
python scripts/zero_agent.py   --task=Template-Tensegrity-Place-Tendon-v0      --num_envs=10
python scripts/skrl/train.py   --task=Template-Tensegrity-Place-Tendon-v0      --headless
python scripts/skrl/play.py    --task=Template-Tensegrity-Place-Tendon-Play-v0 --num_envs=10
```

### Step Response Validation

Replicates the PID step-response test from Klein (2023, §3.5 / §4.2):

```bash
cd /home/robot/Isaac/IsaacLab
./isaaclab.sh -p /path/to/scripts/step_response_test.py \
    --headless --num-envs 1 --output-dir ./step_response_results
```

Outputs: per-joint time-series plots, metrics tables, NRMSE comparison vs. Gazebo,
and a CSV with numerical results.
See [`doc/tendon_simulation.md`](../../doc/tendon_simulation.md) for details.

---

## Project Structure

```
source/tensegrity_pick/tensegrity_pick/
├── __init__.py                          # Auto-imports tasks + UI extension
├── robots/                              # Centralised robot definitions
│   ├── __init__.py                      #   Exports all configs + TendonEffortAction
│   ├── tensegrity_robot_cfg.py          #   TENS_3DOF_CFG, TENS_5DOF_GRIPPER_CFG (PD)
│   ├── tendon_actuator.py               #   TendonEffortAction + TendonEffortActionCfg
│   └── tendon_robot_cfg.py              #   TENS_3DOF_TENDON_CFG, TENS_5DOF_GRIPPER_TENDON_CFG
├── tasks/
│   ├── __init__.py                      # import_packages auto-discovery
│   └── manager_based/
│       ├── tensegrity_pick/             # Cube Sorting task
│       │   ├── __init__.py              #   gym.register(Pick-v0)
│       │   ├── tensegrity_pick_env_cfg.py
│       │   ├── proj_base_scene_cfg.py   #   Shared base scene
│       │   ├── cube_sorting_scene_cfg.py
│       │   ├── tensegrity_robot_cfg.py  #   Backward-compat shim → robots/
│       │   ├── mdp/                     #   Rewards, observations, events
│       │   └── agents/                  #   skrl PPO config
│       ├── tensegrity_reach/            # Reach task
│       │   ├── __init__.py              #   gym.register(Reach, Reach-Play, Reach-Tendon, ...)
│       │   ├── tensegrity_reach_env_cfg.py          # PD-driven
│       │   ├── tensegrity_reach_tendon_env_cfg.py   # Tendon-driven
│       │   ├── mdp/                     #   FK sampling, position/orientation rewards
│       │   └── agents/                  #   skrl PPO config
│       └── tensegrity_place/            # Place task
│           ├── __init__.py              #   gym.register(Place, Place-Play, Place-Tendon, ...)
│           ├── tensegrity_place_env_cfg.py           # PD + tendon variants
│           ├── place_scene_cfg.py       #   1 green + 1 red cube scene
│           ├── place_env.py             #   TensegrityPlaceEnv (grasp tracking)
│           ├── mdp/                     #   Rewards, curriculum, tendon_actions shim
│           └── agents/                  #   skrl PPO config
scripts/
├── list_envs.py                         # List all registered environments
├── zero_agent.py                        # Zero-action smoke test
├── random_agent.py                      # Random-action smoke test
├── step_response_test.py                # Klein (2023) step response validation
├── skrl/                                # Training and evaluation scripts
│   ├── train.py
│   └── play.py
└── workspace_analysis/                  # Workspace reachability analysis
```

### Adding a new task

1. Create `tasks/manager_based/<your_task>/` with `__init__.py`, env config, mdp, and agents.
2. In `__init__.py`, call `gym.register(id="Template-Tensegrity-<YourTask>-v0", ...)`.
3. The task is automatically discovered — no other files need editing.

---

## IDE Setup (Optional)

Run VSCode Tasks (`Ctrl+Shift+P` → `Tasks: Run Task` → `setup_python_env`) and provide the Isaac Sim path.
This creates `.vscode/.python.env` for Pylance indexing.

## Code Formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing

Add to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/tensegrity_pick"
    ]
}
```

### Pylance Crash

Exclude unused Omniverse packages in `.vscode/settings.json` under `"python.analysis.extraPaths"`:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"
"<path-to-isaac-sim>/extscache/omni.kit.*"
"<path-to-isaac-sim>/extscache/omni.graph.*"
"<path-to-isaac-sim>/extscache/omni.services.*"
```
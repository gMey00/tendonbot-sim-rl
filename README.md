# Simulation and Reinforcement Learning for Autonomous Robotics with Isaac Sim

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76b900?logo=nvidia)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-latest-76b900?logo=nvidia)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue)](https://opensource.org/licenses/BSD-3-Clause)
[![RL Framework](https://img.shields.io/badge/RL-skrl%20%7C%20PPO-orange)](https://skrl.readthedocs.io/)

> Reinforcement-learning environments for a **tendon-driven tensegrity manipulator** and a
> **multi-robot textile-sorting pipeline** in NVIDIA Isaac Sim / Isaac Lab. The project spans
> a Project Thesis (simulation + RL validation of the tendon-driven arm) and a Master Thesis
> (multi-agent cloth-sorting: pick → present → distribute).

---

## Table of Contents

- [Simulation and Reinforcement Learning for Autonomous Robotics with Isaac Sim](#simulation-and-reinforcement-learning-for-autonomous-robotics-with-isaac-sim)
  - [Table of Contents](#table-of-contents)
  - [🚀 About](#-about)
  - [🧩 Tasks](#-tasks)
    - [PA foundation tasks (single tensegrity robot)](#pa-foundation-tasks-single-tensegrity-robot)
    - [MA cloth-sorting pipeline (Robot 1 + Robot 2, multi-stage)](#ma-cloth-sorting-pipeline-robot-1--robot-2-multi-stage)
  - [📁 Repository Structure](#-repository-structure)
  - [🔧 Prerequisites](#-prerequisites)
  - [📝 Getting Started](#-getting-started)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Step Response Validation](#step-response-validation)
  - [📚 Documentation](#-documentation)
    - [Theses](#theses)
  - [🗓️ Organisational](#️-organisational)

---

## 🚀 About

**Project Thesis (PA, submitted):** Simulative construction and RL-based control of a 5-DOF
tendon-driven tensegrity manipulator — a 2-DOF linear ceiling base plus a 3-DOF tendon-driven
arm (elbow + 2-DOF wrist) actuated by 5 steel cables mapped to joint torques via the Jacobian
transpose. The PA validated the simulated robot against the PID step-response and workspace
methodology of Klein (2023) and trained PPO policies for **Reach**, **Cube Place**, and
**Cube Sort**, comparing PD-driven, tendon-driven, and physical-tendon actuation. See
[`project_thesis.md`](doc/Theses/project_thesis.md) for the full research question and partial goals.

**Master Thesis (MA, in progress):** Extends the validated pick-and-place pipeline into a
**multi-agent textile sorting system**. **Robot 1** (the tensegrity picker) retrieves a
T-shirt from a moving conveyor; **Robot 2** — a Kinova Gen3 or UR5e arm with a Robotiq 2F-140
gripper, mounted on a second pedestal — regrasps and stretches the garment taut for
camera-based condition inspection, then sorts it into a condition-specific bin. The MA also
runs a controlled **tendon-wrist transfer experiment**: splicing the compliant 2-DOF
tensegrity wrist onto conventional industrial arms (UR5e/UR10/Kinova) to test whether it
improves reach/place performance over their native rigid wrists. See
[`master_thesis.md`](doc/Theses/master_thesis.md) for the full research question and the
PG‑1…PG‑7 partial-goal breakdown (conveyor sorting, cloth simulation, second-arm integration,
wrist transfer, stretching/sorting primitives, damage classification, multi-agent
coordination, sim-to-real prep, evaluation).

**Current pipeline status:** the single-agent building blocks (cloth simulation, T-shirt pick,
second-arm stretching/presentation, goal-conditioned bin sorting) each have a trained or
in-progress RL policy, chained today through cached terminal-state snapshots rather than live
multi-agent hand-off. Camera-based damage classification and joint multi-agent training
(PG-4/PG-5) are the next open partial goals — see [`Organisation.md`](Organisation.md) for
milestone-level tracking.

---

## 🧩 Tasks

Tasks are organised into two groups: the **PA foundation tasks** (single tensegrity robot,
validated in the Project Thesis) and the **MA cloth-sorting pipeline** (multi-robot, built for
the Master Thesis). Each task is a separate Isaac Lab extension package under
[`src/tensegrity_pick`](src/tensegrity_pick/README.md) with its own registered Gym
environments, README, and training results — see the linked READMEs for the full variant list,
reward design, and results tables. In total **80 Gym environments** are registered across the
8 task families (list them all with `python scripts/list_envs.py`).

### PA foundation tasks (single tensegrity robot)

| Task | Robot variants | Status | Gym IDs | Docs |
|------|-----------------|--------|:-------:|------|
| **Reach** | Tensegrity (PD / Tendon / Physical-Tendon) + a 6-arm × 4-action-space comparison grid (UR5e/UR10/Kinova × F140/Frankenstein wrist × Joint/IK-Rel/IK-Abs/OSC) | ✅ Trained, full results matrix | 54 | [reach/README.md](src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/reach/README.md) |
| **Cube Place** | Tensegrity (PD / Tendon / Physical-Tendon) | ✅ Trained (PD 90.6%, Tendon 93.4% place success) | 6 | [cube_place/README.md](src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_place/README.md) |
| **Cube Sort** | Tensegrity (PD / Tendon / Physical-Tendon) | 🔄 Rework in progress (2G+1R validated; 4G+2R scaling blocked) | 5 | [cube_sort/README.md](src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/README.md) |

The **Reach** comparison grid is also the testbed for the MA tendon-wrist transfer
experiment (PG-3.2): "Frankenstein" variants splice the tensegrity wrist onto each
industrial arm for direct comparison against its native rigid wrist ("F140").

### MA cloth-sorting pipeline (Robot 1 + Robot 2, multi-stage)

| Stage | Task | Robot(s) | Status | Gym IDs | Docs |
|---|------|----------|--------|:---:|------|
| 1. Retrieve | **Shirt Pick** | Tensegrity | ✅ Trained & deployable | 2 | [shirt_pick/README.md](src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_pick/README.md) |
| — | **Shirt Place** *(pick→place variant)* | Tensegrity (PD / Physical-Tendon) | ✅ Trained & deployable | 4 | [shirt_place/README.md](src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_place/README.md) |
| 2. Present | **Shirt Present** | Kinova Gen3-F140 / UR5e-F140 | ✅ Trained & validated (present-rate 0.927) | 4 | [shirt_present/README.md](src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_present/README.md) |
| 3. Distribute | **Shirt Distribute** | Kinova Gen3-F140 / UR5e-F140 | 🧪 Stub (goal-conditioning works; reward is placeholder shaping) | 4 | [shirt_distribute/README.md](src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_distribute/README.md) |
| — | **Shirt Sort** *(legacy, superseded)* | Tensegrity | 🗄️ Template — replaced by the Pick → Present → Distribute pipeline above | 1 | [shirt_sort/README.md](src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_sort/README.md) |

Play/eval variants (`*-Play-v0`) are registered for every task above except **Shirt Sort**.

---

## 📁 Repository Structure

```
studentische-arbeiten/
├── doc/                    Documentation (setup guides, literature, tendon simulation, thesis writing)
│   ├── reports/            Task evaluation reports, optimization tracking logs, and pipeline/benchmark studies
│   └── Theses/             Project & Master thesis proposals, Typst sources, build system
├── res/                    Simulation assets (USD scenes, robot models, meshes, textures)
│   ├── Props/              Environment props (drum, conveyor, T-shirt cloth)
│   ├── Scenes/             Pre-built USD scenes
│   ├── Tensegrity/         Tensegrity robot (URDF, USD, configs)
│   └── UR10/, UR5E/, KinovaGen3/   Industrial-arm reference assets (PG-6 baselines, Robot 2)
├── samples/                Reference Isaac Lab projects (Cartpole, UR10 Reach)
├── src/
│   └── tensegrity_pick/    Main Isaac Lab extension (tasks, robot configs, scripts)
├── test/                   Test suite (pytest — actuator math, config validation, env smoke tests)
├── tools/                  Installation scripts and the live training-monitor dashboard
├── reports/                Thesis chapter review tooling (methodology_review)
└── renders/                Rendered validation clips (e.g. cloth grasp/drape)
```

> See each folder's `README.md` for details.

---

## 🔧 Prerequisites

| Component | Version / Notes |
|-----------|-----------------|
| **OS** | Ubuntu 22.04 LTS (64-bit) |
| **GPU** | NVIDIA RTX (A6000 recommended; ≥ 8 GB VRAM minimum) |
| **NVIDIA Driver** | ≥ 535.xx |
| **Isaac Sim** | 5.1.0 |
| **Isaac Lab** | Latest main branch |
| **Conda** | Miniconda or Anaconda |
| **Python** | 3.11 (managed by conda environment `env_isaaclab`) |

---

## 📝 Getting Started

```bash
# 1. Clone the repository (HTTPS — SSH is blocked on FAPS network)
git clone -b project/tendonbot-sim-rl \
  https://git.faps.uni-erlangen.de/alschlosser/studentische-arbeiten.git
cd studentische-arbeiten

# 2. Configure local environment variables
source .config/env_vars.sh

# 3. Install Isaac Sim + Isaac Lab (if not already)
cd tools && bash install_IsaacLab.sh && cd ..

# 4. Activate the conda environment
conda activate env_isaaclab

# 5. Install the tensegrity_pick extension
cd src/tensegrity_pick
python -m pip install -e source/tensegrity_pick

# 6. Verify all environments are registered
python scripts/list_envs.py

# 7. Run a quick smoke test
python scripts/zero_agent.py --task=Template-Reach-Tensegrity-v0 --num_envs=2 --headless
```

### Training

```bash
# Headless PPO training (default: env-specific num_envs)
python scripts/skrl/train.py --task=Template-Reach-Tensegrity-v0 --headless

# Override environment count
python scripts/skrl/train.py --task=Template-Reach-Tensegrity-v0 --headless --num_envs=4096
```

### Evaluation

```bash
python scripts/skrl/play.py --task=Template-Reach-Tensegrity-Play-v0 --num_envs=10
```

### Step Response Validation

```bash
cd /home/robot/Isaac/IsaacLab
./isaaclab.sh -p /path/to/src/tensegrity_pick/scripts/diagnostics/step_response_test.py \
    --headless --num-envs 1 --output-dir ./step_response_results
```

> See [`doc/Tensegrity_robot/tendon_simulation.md`](doc/Tensegrity_robot/tendon_simulation.md) for full details on the
> tendon simulation, validation methodology, and reference data from Klein (2023).

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`doc/README.md`](doc/README.md) | Documentation index |
| [`doc/nv_isaac.md`](doc/nv_isaac.md) | NVIDIA Isaac Sim / Isaac Lab resources and links |
| [`doc/Alex_cluster/README.md`](doc/Alex_cluster/README.md) | Alex NHR@FAU GPGPU cluster: quickstart, RTX PRO 6000 hardware, Tier3 application |
| [`doc/Tensegrity_robot/README.md`](doc/Tensegrity_robot/README.md) | Tensegrity robot documentation: kinematics, tendon simulation, model validation, workspace analysis |
| [`doc/Tensegrity_robot/tendon_simulation.md`](doc/Tensegrity_robot/tendon_simulation.md) | Tendon simulation: physics, architecture, validation |
| [`doc/reports/README.md`](doc/reports/README.md) | Task evaluation reports, optimization tracking logs, and pipeline/benchmark studies (Reach, Cube Place/Sort, Shirt pipeline) |
| [`doc/remote_desktop_setup.md`](doc/remote_desktop_setup.md) | Remote desktop setup (Tailscale + RustDesk) |
| [`doc/reinforcement_learning.md`](doc/reinforcement_learning.md) | Reinforcement learning notes |
| [`doc/robot_gripper_comparison.md`](doc/robot_gripper_comparison.md) | Robot and gripper comparison datasheet (UR10e, UR10, Kinova Gen3, tensegrity) |
| [`doc/workflow_guide.md`](doc/workflow_guide.md) | Development workflow for Manipulator RL tasks from scratch |
| [`doc/Literatur/README.md`](doc/Literatur/README.md) | Annotated bibliography: tendon robots, tensegrity, linkage mechanisms, grasping, simulation, RL, cloth manipulation, sim-to-real, control & kinematics, workspace analysis |
| [`res/README.md`](res/README.md) | Simulation assets overview |
| [`res/Tensegrity/README.md`](res/Tensegrity/README.md) | Robot specification: kinematic chain, joint limits, tendon geometry |
| [`src/tensegrity_pick/README.md`](src/tensegrity_pick/README.md) | Isaac Lab extension: tasks, scripts, project structure |
| [`tools/README.md`](tools/README.md) | Installation scripts, USD builder, training monitor |
| [`test/README.md`](test/README.md) | Test suite: actuator math, config validation, environment smoke tests |

### Theses

| Thesis | Status | Overview | Guideline | Document |
|--------|--------|----------|-----------|----------|
| Project Thesis (PA) | ✅ Submitted | [project_thesis.md](doc/Theses/project_thesis.md) | [Guideline](doc/Theses/project_thesis/guideline/PA-Guideline.pdf) | [Thesis](doc/Theses/project_thesis/thesis/PA-Thesis.pdf) |
| Master Thesis (MA) | 🔄 WIP | [master_thesis.md](doc/Theses/master_thesis.md) | [Guideline](doc/Theses/master_thesis/guideline/MA-Guideline.pdf) | [Thesis](doc/Theses/master_thesis/thesis/MA-Thesis.pdf) |

Both theses are written in Typst; see [`doc/Theses/BUILD_SYSTEM.md`](doc/Theses/BUILD_SYSTEM.md)
for the build layout and `make` targets, and
[`doc/Theses/FAPS_FORMATTING.md`](doc/Theses/FAPS_FORMATTING.md) for the FAPS formatting spec.
The [`reports/methodology_review/`](reports/methodology_review/README.md) framework holds the
parallelised critical review of the PA methodology chapter.

---

## 🗓️ Organisational

👉 [Open the project calendar](https://kalender.digital/8a3e2b7f89063ab12eeb)

See [`Organisation.md`](Organisation.md) for meeting notes and project milestones.

[Back to top](#top)

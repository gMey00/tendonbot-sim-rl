# Simulation and Reinforcement Learning for Autonomous Robotics with Isaac Sim

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76b900?logo=nvidia)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-latest-76b900?logo=nvidia)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue)](https://opensource.org/licenses/BSD-3-Clause)
[![RL Framework](https://img.shields.io/badge/RL-skrl%20%7C%20PPO-orange)](https://skrl.readthedocs.io/)

> Reinforcement-learning environments for a ceiling-mounted **tensegrity manipulator** in
> NVIDIA Isaac Sim / Isaac Lab.  The project provides PD-driven and **tendon-driven** robot
> configurations, multiple RL tasks (reach, place, pick), and a physics-validation pipeline
> based on the controller and test methodology from Klein (2023).

---

## Table of Contents

- [Simulation and Reinforcement Learning for Autonomous Robotics with Isaac Sim](#simulation-and-reinforcement-learning-for-autonomous-robotics-with-isaac-sim)
  - [Table of Contents](#table-of-contents)
  - [🚀 About](#-about)
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

**Project:** Simulation and reinforcement learning for a 5-DOF tensegrity robot arm
with a parallel gripper, trained inside NVIDIA Isaac Sim using the Isaac Lab framework
and the skrl reinforcement-learning library.

**Robot:** The manipulator consists of a 2-DOF linear base (Y/Z translation) and a
3-DOF tendon-driven arm (elbow + 2-DOF wrist).  The arm is actuated by 5 steel cables
(2 antagonistic for the elbow, 3 at 120° for the wrist) whose tensions map to joint
torques via the Jacobian transpose.

**Tasks:**

| Gym ID | Description |
|--------|-------------|
| `Template-Reach-Tensegrity-v0` | Move end-effector to a random 6-DOF target pose |
| `Template-Reach-Tensegrity-Tendon-v0` | Reach with tendon-driven arm (J^T mapping) |
| `Template-Reach-Tensegrity-Physical-Tendon-v0` | Reach with physical tendon model (body-force elbow) |
| `Template-Reach-UR10e-v0` | Reach with UR10e + Robotiq 2F-140 (PG-6 baseline) |
| `Template-Reach-Kinova-v0` | Reach with Kinova Gen3 + Robotiq 2F-140 (PG-6 baseline) |
| `Template-Tensegrity-Cube-Sort-v0` | Cube sorting — pick green cubes off a conveyor, ignore red |
| `Template-Tensegrity-Cube-Place-v0` | Place a green cube into a drum (with curriculum) |
| `Template-Tensegrity-Cube-Place-Tendon-v0` | Cube Place with tendon-driven arm |
| `Template-Tensegrity-Cube-Place-Physical-Tendon-v0` | Cube Place with physical tendon model |
| `Template-UR10e-Cube-Place-v0` | Cube Place with UR10e + Robotiq 2F-140 (PG-6 baseline) |
| `Template-Kinova-Cube-Place-v0` | Cube Place with Kinova Gen3 + Robotiq 2F-140 (PG-6 baseline) |
| `Template-Tensegrity-Shirt-Place-v0` | *(template)* Shirt place — cloth simulation |
| `Template-Tensegrity-Shirt-Place-Physical-Tendon-v0` | *(template)* Shirt place with physical tendon model |
| `Template-Tensegrity-Shirt-Sort-v0` | *(template)* Shirt sort — cloth simulation |

Play/eval variants (`*-Play-v0`) are registered for all applicable tasks.

---

## 📁 Repository Structure

```
studentische-arbeiten/
├── doc/                    Documentation (setup guides, literature, tendon simulation)
├── res/                    Simulation assets (USD scenes, robot models, meshes, textures)
│   ├── Props/              Environment props (drum, t-shirt cloth)
│   ├── Scenes/             Pre-built USD scenes
│   ├── Tensegrity/         Tensegrity robot (URDF, USD, configs)
│   └── UR10/               UR10 reference assets
├── samples/                Reference Isaac Lab projects (Cartpole, UR10 Reach)
├── src/
│   └── tensegrity_pick/    Main Isaac Lab extension (tasks, robot configs, scripts)
├── test/                   Test suite (pytest — actuator math, config validation, env smoke tests)
└── tools/                  Installation and setup scripts
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
python scripts/diagnostics/list_envs.py

# 7. Run a quick smoke test
python scripts/agents/zero_agent.py --task=Template-Reach-Tensegrity-v0 --num_envs=2 --headless
```

### Training

```bash
# Headless PPO training (default: 2000 parallel environments)
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
| [`doc/remote_desktop_setup.md`](doc/remote_desktop_setup.md) | Remote desktop setup (Tailscale + RustDesk) |
| [`doc/reinforcement_learning.md`](doc/reinforcement_learning.md) | Reinforcement learning notes |
| [`doc/robot_gripper_comparison.md`](doc/robot_gripper_comparison.md) | Robot and gripper comparison datasheet (UR10e, UR10, Kinova Gen3, tensegrity) |
| [`doc/workflow_guide.md`](doc/workflow_guide.md) | Development workflow for Manipulator RL tasks from scratch |
| [`doc/Literatur/README.md`](doc/Literatur/README.md) | Annotated bibliography: tendon robots, tensegrity, linkage mechanisms, grasping, simulation, RL, cloth manipulation, sim-to-real, control & kinematics, workspace analysis |
| [`res/Tensegrity/README.md`](res/Tensegrity/README.md) | Robot specification: kinematic chain, joint limits, tendon geometry |
| [`src/tensegrity_pick/README.md`](src/tensegrity_pick/README.md) | Isaac Lab extension: tasks, scripts, project structure |
| [`tools/README.md`](tools/README.md) | Installation scripts, USD builder, training monitor |
| [`test/README.md`](test/README.md) | Test suite: actuator math, config validation, environment smoke tests |

### Theses

| Thesis | Status | Overview | Guideline | Document |
|--------|--------|----------|-----------|----------|
| Project Thesis (PA) | ✅ Submitted | [project_thesis.md](doc/Theses/project_thesis.md) | [Guideline](doc/Theses/project_thesis/guideline/PA-Guideline.pdf) | [Thesis](doc/Theses/project_thesis/thesis/PA-Thesis.pdf) |
| Master Thesis (MA) | 🔄 WIP | [master_thesis.md](doc/Theses/master_thesis.md) | [Guideline](doc/Theses/master_thesis/guideline/MA-Guideline.pdf) | [Thesis](doc/Theses/master_thesis/thesis/MA-Thesis.pdf) |

---

## 🗓️ Organisational

👉 [Open the project calendar](https://kalender.digital/8a3e2b7f89063ab12eeb)

See [`Organisation.md`](Organisation.md) for meeting notes and project milestones.

[Back to top](#top)
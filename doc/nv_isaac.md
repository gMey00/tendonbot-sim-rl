# NVIDIA Isaac

## Isaac Sim

### Introduction

NVIDIA Isaac Sim is a GPU-accelerated robotics simulator built on the Omniverse platform.
It provides photorealistic rendering, PhysX-based rigid-body and articulation dynamics,
and turnkey sensor models (LiDAR, RGB-D, IMU).  This project uses Isaac Sim **5.1.0**
as the physics backend for all tensegrity robot simulations.

### Sources
- [Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html#)
- [Nvidia DLI Courses](https://www.nvidia.com/en-us/learn/learning-path/robotics/)
  - [Getting Started: Simulating Your First Robot in Isaac Sim](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-27+V1)
  - [Ingesting Robot Assets and Simulating Your Robot in Isaac Sim](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-29+V1)
  - [Synthetic Data Generation for Perception Model Training in Isaac Sim](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-30+V1)
  - [Developing Robots With Software-in-the-Loop (SIL) In Isaac Sim](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-31+V1)
---

## Isaac Lab

### Introduction

Isaac Lab is a modular framework for robot learning built on top of Isaac Sim.  It provides
standardized APIs for defining environments, actuator models, observation/action managers,
reward functions, and curriculum strategies.  This project implements all RL tasks as
Isaac Lab *manager-based* environments using the `ManagerBasedRLEnv` class, trained with
the [skrl](https://skrl.readthedocs.io/) PPO implementation.

### Sources
- [Documentation](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [NVIDIA Robot Learning](https://www.nvidia.com/en-us/use-cases/robot-learning/)
- [Nvidia DLI Courses](https://www.nvidia.com/en-us/learn/learning-path/robotics/)
  - [An Introduction to Robot Learning and Isaac Lab](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-36+V1)
  - [Train Your First Robot in Isaac Lab](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-46+V1)
  - [Train Your Second Robot in Isaac Lab](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-47+V1)
  - [Bring Your Own Robot: Integrating Assets Into Isaac Lab (Comind Soon)](https://www.nvidia.com/en-us/learn/learning-path/robotics/)
  - [Transferring Robot Learning Policies From Simulation to Reality](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-28+V1)
- [Core Paper (always cite)](https://isaac-orbit.github.io)
- [Comparison of supported RL Frameworks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_frameworks.html)
- [Reinforcement Learning Scripts](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
---

## OpenUSD

### Introduction

OpenUSD (Universal Scene Description) is the scene representation format used by Isaac Sim.
All robot models, environments, and props in this project are stored as `.usd` / `.usdc`
files.  Robot articulations are imported from URDF via Isaac Sim's URDF importer and then
configured with physics materials, collision meshes, and joint drive parameters in USD.

### Sources
- [Nvidia DLI Courses](https://www.nvidia.com/en-us/learn/learning-path/openusd/)
# Grasping & Manipulation

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Grasping Theory

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Birglen2008 | Birglen, L., Laliberté, T. & Gosselin, C. M. | *Underactuated Robotic Hands* | 2008 | 🟡 | 🟢 | 🟢 | Comprehensive monograph on underactuated grasping: taxonomy, kinematic analysis, force distribution, and design synthesis. Core reference for parallel-jaw gripper mechanics. |

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Nguyen1988 | Nguyen, V.-D. | "Constructing Force-Closure Grasps" | 1988 | 🔴 | 🟢 | 🟢 | Formal conditions for force-closure grasps. Foundational theory for evaluating grasp stability in simulation. |
| Ferrari1992 | Ferrari, C. & Canny, J. | "Planning Optimal Grasps" | 1992 | 🔴 | 🟢 | 🟢 | Introduces the ε-metric (largest inscribed ball in the wrench space) for grasp quality evaluation. Standard reference for grasp quality. [DOI](https://doi.org/10.1109/ROBOT.1992.220116) |
| Bicchi2000 | Bicchi, A. & Kumar, V. | "Robotic Grasping and Contact: A Review" | 2000 | 🔴 | 🟢 | 🟢 | Comprehensive review of contact models, force-closure conditions, and dexterous manipulation theory. Survey covering form- and force-closure distinctions. |
| Roa2015 | Roa, M. A. & Suárez, R. | "Grasp Quality Measures: Review and Performance" | 2015 | 🔴 | 🟢 | 🟢 | Systematic review and empirical comparison of grasp quality metrics. Context for evaluating simulated grasps. |

---

## Gripper Design

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Laliberte2002 | Laliberté, T., Birglen, L. & Gosselin, C. M. | "Underactuation in Robotic Grasping Hands" | 2002 | 🟡 | 🟢 | 🟢 | Design principles for underactuated mechanical grippers with self-adaptive finger mechanisms. Background for Robotiq gripper design lineage. |
| Montambault1998 | Montambault, S. & Gosselin, C. M. | "Analysis of Underactuated Mechanical Grippers" | 1998 | 🔴 | 🟡 | 🟢 | Early analysis of cable-driven underactuated grippers with passive adaptation. Precursor to modern adaptive grippers. |
| Demers2014 | Demers, L.-A. A. | "Adaptive Parallel Gripper" (US Patent 8,651,539 B2, Robotiq Inc.) | 2014 | 🟡 | 🟡 | 🟢 | Patent for the Robotiq adaptive gripper mechanism. Documents the coupled four-bar linkage design with travel limiter enabling inner/outer pinch modes. |
| Gomes2020 | Gomes, D. F. et al. | "Generation and Assessment of Grasps for an Underactuated Gripper" | 2020 | 🟡 | 🟡 | 🟢 | Detailed kinematic and force analysis of the Robotiq 2F-85 gripper. Relevant for simulation model accuracy. |

---

## RL for Grasping

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Popov2017 | Popov, I., Heess, N., Lillicrap, T. et al. | "Data-efficient Deep Reinforcement Learning for Dexterous Manipulation" | 2017 | 🟡 | 🟢 | 🟢 | Composite shaping rewards (reach→grasp→stack phases) for Jaco arm stacking. Key reference for sequential reward gating approach. |
| Zeng2018 | Zeng, A. et al. | "Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning" | 2018 | 🔴 | 🟡 | 🟢 | Self-supervised learning of push-grasp synergies. Context for action primitive coordination in manipulation. |
| Alkhatib2019 | Alkhatib, R. et al. | "RL-Based Adaptive Gripper Control" | 2019 | 🟡 | 🟡 | 🟡 | Reinforcement learning applied to adaptive gripper control. Context for RL-controlled grasping. |
| Kasaei2021 | Kasaei, S. H. et al. | "Efficient Object Grasping Through Reinforcement Learning" | 2021 | 🟡 | 🟡 | 🟢 | Efficient RL-based grasping with reduced sample complexity. Relevant for grasping policy design. |
| Zheng2024 | Zheng, Z. et al. | "Gripper Action Space RL for Parallel-Jaw Manipulation" | 2024 | 🟢 | 🟢 | 🟢 | Analysis of gripper action space representations in RL: binary vs. continuous vs. position-controlled. Directly relevant to the Robotiq gripper RL action design. |

---

## Benchmarks & Environments

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| James2019 | James, S. et al. | "RLBench: The Robot Learning Benchmark" | 2019 | 🔴 | 🟡 | 🟢 | 100-task benchmark for robot learning with CoppeliaSim. Context for manipulation benchmark comparison. |
| Tao2024 | Tao, S. et al. | "ManiSkill3: GPU-Parallelized Robotics Simulation and Benchmark for Generalizable Manipulation Skills" | 2024 | 🟡 | 🟡 | 🟢 | GPU-parallelized manipulation benchmark with diverse environments and tasks. Alternative framework reference. |
| Zakka2022 | Zakka, K. et al. | "MuJoCo Menagerie: A Collection of High-Quality Simulation Models" | 2022 | 🟡 | 🟡 | 🟡 | Curated collection of simulation models including grippers; provides validated Robotiq models for MuJoCo. |

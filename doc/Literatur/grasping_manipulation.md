# Grasping & Manipulation

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Grasping Theory

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Bicchi2000](sources/grasping_manipulation/Papers/Bicchi2000.pdf) | Bicchi, A. & Kumar, V. | "Robotic Grasping and Contact: A Review" | 2000 | 🔴 | 🟢 | 🟢 | Comprehensive review of contact models, force-closure conditions, and dexterous manipulation theory. Survey covering form- and force-closure distinctions. |

---

## Gripper Design

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Montambault1998 | Montambault, S. & Gosselin, C. M. | "Analysis of Underactuated Mechanical Grippers" | 1998 | 🔴 | 🟡 | 🟢 | Early analysis of cable-driven underactuated grippers with passive adaptation. Precursor to modern adaptive grippers. |
| Demers2014 | Demers, L.-A. A. | "Adaptive Parallel Gripper" (US Patent 8,651,539 B2, Robotiq Inc.) | 2014 | 🟡 | 🟡 | 🟢 | Patent for the Robotiq adaptive gripper mechanism. Documents the coupled four-bar linkage design with travel limiter enabling inner/outer pinch modes. |

---

## RL for Grasping

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Popov2017](sources/grasping_manipulation/Papers/Popov2017.pdf) | Popov, I., Heess, N., Lillicrap, T. et al. | "Data-efficient Deep Reinforcement Learning for Dexterous Manipulation" | 2017 | 🟡 | 🟢 | 🟢 | Composite shaping rewards (reach→grasp→stack phases) for Jaco arm stacking. Key reference for sequential reward gating approach. |
| [Zeng2018](sources/grasping_manipulation/Papers/Zeng2018.pdf) | Zeng, A. et al. | "Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning" | 2018 | 🔴 | 🟡 | 🟢 | Self-supervised learning of push-grasp synergies. Context for action primitive coordination in manipulation. |
| Alkhatib2019 | Alkhatib, R. et al. | "RL-Based Adaptive Gripper Control" | 2019 | 🟡 | 🟡 | 🟡 | Reinforcement learning applied to adaptive gripper control. Context for RL-controlled grasping. |
| [Kasaei2021](sources/grasping_manipulation/Papers/Kasaei2021.pdf) | Kasaei, S. H. et al. | "Efficient Object Grasping Through Reinforcement Learning" | 2021 | 🟡 | 🟡 | 🟢 | Efficient RL-based grasping with reduced sample complexity. Relevant for grasping policy design. |
| Zheng2024 | Zheng, Z. et al. | "Gripper Action Space RL for Parallel-Jaw Manipulation" | 2024 | 🟢 | 🟢 | 🟢 | Analysis of gripper action space representations in RL: binary vs. continuous vs. position-controlled. Directly relevant to the Robotiq gripper RL action design. |
| [Kasaei2023MVGrasp](sources/grasping_manipulation/Papers/Kasaei2023MVGrasp.pdf) | Kasaei, S. H. et al. | "MVGrasp — Real-time Multi-View 3D Object Grasping in Highly Cluttered Environments" | 2023 | 🟡 | 🟡 | 🟢 | Multi-view 3D grasp learning in clutter. Strong comparator for vision-based grasp policies. [DOI](https://doi.org/10.1016/j.robot.2022.104313) |
| [Hu2025tactile](sources/grasping_manipulation/Papers/Hu2025tactile.pdf) | Hu, S. et al. | "Tactile-based Reinforcement Learning for Adaptive Grasping under Observation Uncertainties" | 2025 | 🟡 | 🟡 | 🟢 | Tactile-feedback-driven RL for adaptive grasping. Relevant for sensor-augmented grasp policies. [arXiv](https://arxiv.org/abs/2505.16167) |
| [Singh2025E2E](sources/grasping_manipulation/Papers/Singh2025E2E.pdf) | Singh, R. et al. | "End-to-end RL Improves Dexterous Grasping Policies" | 2025 | 🟡 | 🟢 | 🟢 | End-to-end deep RL outperforms staged pipelines for dexterous grasping. Direct comparator for thesis architecture choice. [arXiv](https://arxiv.org/abs/2509.16434) |
| [Yonemaru2025](sources/grasping_manipulation/Papers/Yonemaru2025.pdf) | Yonemaru, S. et al. | "Learning to Group and Grasp Multiple Objects" | 2025 | 🔴 | 🟡 | 🟢 | Learns to group then grasp multiple objects in one motion. Context for multi-object grasping policies. [arXiv](https://arxiv.org/abs/2502.08452) |
| [Lyu2025](sources/grasping_manipulation/Papers/Lyu2025.pdf) | Lyu, S. et al. | "Dynamic Grasping Based on Reinforcement Learning and Differentiable MPC" | 2025 | 🟡 | 🟡 | 🟢 | RL combined with differentiable MPC for dynamic grasping. Recent hybrid model-based/learning approach. [DOI](https://doi.org/10.1109/TASE.2025.3576419) |
| [Liao2025flipping](sources/grasping_manipulation/Papers/Liao2025flipping.pdf) | Liao, Y. et al. | "Flipping Manipulation with a Two-Fingered Parallel-Jaw Gripper" | 2025 | 🔴 | 🟡 | 🟢 | RL-based flipping manipulation primitive on a parallel-jaw gripper. Relevant for non-prehensile primitive design. [arXiv](https://arxiv.org/abs/2503.21847) |
| [OudeVrielink2021](sources/grasping_manipulation/Papers/OudeVrielink2021.pdf) | Oude Vrielink, T. | "Learning to Grasp Objects in Highly Cluttered Environments using Deep Convolutional Neural Networks" | 2021 | 🔴 | 🟡 | 🟡 | Bachelor's thesis on CNN-based grasp learning in clutter. Methodological reference for vision-based grasping pipelines. |

---

## Benchmarks & Environments

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [James2019](sources/grasping_manipulation/Papers/James2019.pdf) | James, S. et al. | "RLBench: The Robot Learning Benchmark" | 2019 | 🔴 | 🟡 | 🟢 | 100-task benchmark for robot learning with CoppeliaSim. Context for manipulation benchmark comparison. |
| [Luo2024FMB](sources/grasping_manipulation/Papers/Luo2024FMB.pdf) | Luo, J. et al. | "FMB: A Functional Manipulation Benchmark for Generalizable Robotic Learning" | 2024 | 🟡 | 🟡 | 🟢 | Functional manipulation benchmark with peg-in-hole and assembly tasks. Comparator benchmark for generalization. [DOI](https://doi.org/10.1177/02783649241276017) |

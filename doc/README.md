# Documentation

[← Back to project root](../README.md)

Project documentation for the tensegrity robot simulation and reinforcement learning project.

## Contents

| Document | Description |
|----------|-------------|
| [nv_isaac.md](nv_isaac.md) | NVIDIA Isaac Sim and Isaac Lab — overview, links, and learning resources |
| [Alex_cluster/](Alex_cluster/README.md) | Alex NHR@FAU GPGPU cluster: quickstart, RTX PRO 6000 hardware/environment, Tier3 application |
| [Tensegrity_robot/](Tensegrity_robot/README.md) | Tensegrity robot: kinematics, tendon simulation, model validation, workspace analysis |
| [Tensegrity_robot/tendon_simulation.md](Tensegrity_robot/tendon_simulation.md) | Tendon-driven simulation: physical model, software architecture, validation |
| [remote_desktop_setup.md](remote_desktop_setup.md) | Secure remote desktop setup with Tailscale and RustDesk |
| [reinforcement_learning.md](reinforcement_learning.md) | Reinforcement learning notes and evaluation methodology |
| [robot_gripper_comparison.md](robot_gripper_comparison.md) | Robot and gripper comparison datasheet (UR10e, UR10, Kinova Gen3, tensegrity) |
| [open_questions.md](open_questions.md) | Open questions and research gaps |
| [TODO.md](TODO.md) | Project TODO list |
| [project_thesis.md](Theses/project_thesis.md) | Project thesis proposal, research question, goals, and literature |
| [master_thesis.md](Theses/master_thesis.md) | Master thesis proposal, research question, goals, and literature |
| [workflow_guide.md](workflow_guide.md) | Development workflow for Manipulator RL Tasks from scratch |
| [workflow_guide_de.md](workflow_guide_de.md) | Entwicklungs-Workflow für Manipulator-RL-Aufgaben von Grund auf |
| [antiparallelogram_kinematics.ipynb](Tensegrity_robot/antiparallelogram_kinematics.ipynb) | Interactive antiparallelogram linkage kinematics notebook |
| [wrist_kinematics.ipynb](Tensegrity_robot/wrist_kinematics.ipynb) | Interactive wrist kinematics notebook |

### Literature

Annotated bibliographies organized by topic, located in [`Literatur/`](Literatur/):

| File | Topic |
|------|-------|
| [tendon_robots.md](Literatur/tendon_robots.md) | Tendon-driven / cable-driven robots |
| [tensegrity_robots.md](Literatur/tensegrity_robots.md) | Tensegrity robots & compliant mechanisms |
| [linkage_mechanisms.md](Literatur/linkage_mechanisms.md) | Linkage mechanisms & closed-loop kinematics |
| [grasping_manipulation.md](Literatur/grasping_manipulation.md) | Grasping & manipulation |
| [simulation.md](Literatur/simulation.md) | Simulation & physics engines |
| [reinforcement_learning.md](Literatur/reinforcement_learning.md) | Reinforcement learning |
| [cloth_manipulation.md](Literatur/cloth_manipulation.md) | Cloth & deformable object manipulation |
| [sim_to_real.md](Literatur/sim_to_real.md) | Sim-to-real transfer |
| [control_kinematics.md](Literatur/control_kinematics.md) | Robot control & kinematics |
| [workspace_analysis.md](Literatur/workspace_analysis.md) | Workspace analysis |

## Related

- [Robot specification](../res/Tensegrity/README.md) — kinematic chain, joint constraints, tendon geometry
- [Source code & tasks](../src/tensegrity_pick/README.md) — Isaac Lab extension, registered environments
- [Test suite](../test/README.md) — pytest tests for actuators, configs, and environments
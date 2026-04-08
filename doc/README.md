# Documentation

[← Back to project root](../README.md)

Project documentation for the tensegrity robot simulation and reinforcement learning project.

## Contents

| Document | Description |
|----------|-------------|
| [nv_isaac.md](nv_isaac.md) | NVIDIA Isaac Sim and Isaac Lab — overview, links, and learning resources |
| [tendon_simulation.md](tendon_simulation.md) | Tendon-driven simulation: physical model, software architecture, validation |
| [remote_desktop_setup.md](remote_desktop_setup.md) | Secure remote desktop setup with Tailscale and RustDesk |
| [reinforcement_learning.md](reinforcement_learning.md) | Reinforcement learning notes and evaluation methodology |
| [robot_gripper_comparison.md](robot_gripper_comparison.md) | Robot and gripper comparison datasheet (UR10e, UR10, Kinova Gen3, tensegrity) |
| [open_questions.md](open_questions.md) | Open questions and research gaps |
| [TODO.md](TODO.md) | Project TODO list |
| [project_thesis.md](project_thesis.md) | Project thesis proposal, research question, goals, and literature |
| [master_thesis.md](master_thesis.md) | Master thesis proposal, research question, goals, and literature |
| [workflow_guide.md](workflow_guide.md) | Development workflow for Manipulator RL Tasks from scratch |
| [workflow_guide_de.md](workflow_guide_de.md) | Entwicklungs-Workflow für Manipulator RL-Aufgaben von Grunddauf |
| [antiparallelogram_kinematics.ipynb](antiparallelogram_kinematics.ipynb) | Interactive antiparallelogram linkage kinematics notebook |
| [wrist_kinematics.ipynb](wrist_kinematics.ipynb) | Interactive wrist kinematics notebook |

### Literature

Annotated bibliographies organized by topic, located in [`literatur/`](literatur/):

| File | Topic |
|------|-------|
| [reinforcement_learning.md](literatur/reinforcement_learning.md) | Reinforcement learning and robotics |
| [simulation.md](literatur/simulation.md) | Simulation frameworks and benchmarks |
| [control_kinematics.md](literatur/control_kinematics.md) | Control and kinematics of robotic systems |
| [tendon_robots.md](literatur/tendon_robots.md) | Tendon-driven / cable-driven robots |
| [cloth_manipulation.md](literatur/cloth_manipulation.md) | Cloth and deformable object manipulation |
| [sim_to_real.md](literatur/sim_to_real.md) | Sim-to-real transfer |
| [workspace_analysis.md](literatur/workspace_analysis.md) | Workspace analysis |

## Related

- [Robot specification](../res/Tensegrity/README.md) — kinematic chain, joint constraints, tendon geometry
- [Source code & tasks](../src/tensegrity_pick/README.md) — Isaac Lab extension, registered environments
- [Test suite](../test/README.md) — pytest tests for actuators, configs, and environments
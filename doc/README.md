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
| [project_thesis.md](project_thesis.md) | Project thesis proposal, research question, goals, and literature |
| [master_thesis.md](master_thesis.md) | Master thesis proposal, research question, goals, and literature |
| [workflow_guide.md](workflow_guide.md) | Development workflow for Manipulator RL Tasks from scratch |
| [workflow_guide_de.md](workflow_guide_de.md) | Entwicklungs-Workflow für Manipulator RL-Aufgaben von Grunddauf |

### Literature

Annotated bibliographies organized by topic, located in [`literatur/`](literatur/):

| File | Topic |
|------|-------|
| [learning.md](literatur/learning.md) | Reinforcement learning and robotics |
| [sim_env.md](literatur/sim_env.md) | Simulation frameworks and benchmarks |
| [control.md](literatur/control.md) | Control of robotic systems |
| [tendon_robot.md](literatur/tendon_robot.md) | Tendon-driven / cable-driven robots |

## Related

- [Robot specification](../res/Tensegrity/README.md) — kinematic chain, joint constraints, tendon geometry
- [Source code & tasks](../src/tensegrity_pick/README.md) — Isaac Lab extension, registered environments
- [Test suite](../test/README.md) — pytest tests for actuators, configs, and environments
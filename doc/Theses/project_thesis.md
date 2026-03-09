# Project Thesis (Projektarbeit, 10 ECTS)

## Title

**Simulative Construction and Verification of a Tendon-Driven Robot for Conveyor-Based Pick-and-Place Tasks Using Isaac Lab**

## Clickbait Buzzword Title Options

- "Building the Future of Soft Robotics: AI-Driven Simulation and Digital-Twin Verification of a Tendon-Actuated Robot Arm in NVIDIA Isaac Lab"  
- "From CAD to Autonomous Agent: Sim-to-Real-Ready Tendon Robot Design, Physics Simulation, and Reinforcement Learning for Next-Gen Industrial Pick-and-Place"  

## Research Question

How can a custom tendon-driven manipulator be faithfully modeled, validated, and controlled via reinforcement learning in NVIDIA Isaac Sim to perform conveyor-based pick-and-place tasks, and does the tendon-driven wrist mechanism transfer beneficially to a standard industrial robot arm?

## Overarching Goal

Develop and validate a high-fidelity simulation of a 5-DOF tendon-driven robot in NVIDIA Isaac Sim 5.1.0, demonstrate that learned RL policies can control the robot for reach and place tasks, and evaluate the transferability of the tendon-driven wrist to a conventional robot arm.

## Partial Goals

### PG-1: Simulation Environment Setup
Establish a reproducible Isaac Sim scene comprising a ground plane, conveyor belt, plastic drum, and representative objects, with configurable lighting and randomized initial conditions. The scene serves as the shared foundation for all subsequent RL tasks.

### PG-2: Robot Model Integration
Construct the full 5-DOF manipulator in simulation: a 2-DOF linear base (prismatic Y/Z), a 3-DOF tendon-driven arm (elbow + 2-DOF wrist), and a Robotiq 2F-140 gripper. The robot is assembled modularly via Isaac Sim's Robot Assembler from separate USD/URDF components.

### PG-3: Robot Simulation Validation
Validate the simulated robot through:
- **Step-response testing** against the PID controller methodology of Klein (2023), comparing settling time, overshoot, and steady-state error across all joints.
- **Monte Carlo workspace analysis** to characterize reachable volume, Yoshikawa manipulability, and inverse condition number; compare against Klein's (2023) task geometry.

### PG-4: Reach Task (RL)
Formulate and train a PPO-based RL policy (via skrl) for the reach task: move the end-effector to a random 6-DOF target pose. Implement both PD-driven and tendon-driven variants for direct comparison. This validates the RL pipeline and observation/action space design.

### PG-5: Cube Place Task (RL)
Formulate and train a PPO policy for the cube place task: pick a green cube and place it into a drum, with a curriculum that progresses from green-only to green-plus-red (colour-selective placement). Implement both PD and tendon-driven variants.

### PG-6: Tendon Wrist Transfer Experiment
Isolate the 2-DOF tendon-driven wrist mechanism and apply it to a standard industrial robot arm (e.g., UR10) in Isaac Sim. Conduct a controlled comparison of reach accuracy and place success rate between the native rigid wrist and the tendon wrist, using identical RL hyperparameters, to evaluate whether the compliant wrist design improves manipulation capability.

### PG-7: Documentation and Outlook
Document all simulation configurations, RL hyperparameters, and results for full reproducibility. Identify conveyor-based sorting and deformable cloth manipulation as the natural extensions for subsequent work (master thesis).

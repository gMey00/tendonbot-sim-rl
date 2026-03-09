# Organisation

[← Back to project root](README.md)

Project management and organisational information.

## Calendar

👉 [Open the project calendar](https://kalender.digital/8a3e2b7f89063ab12eeb)

---

## Project Timeline

**Project Thesis deadline:** 01.07.2026  
**Master Thesis deadline:** ~01.09.2026  
**Today:** 09.03.2026

---

## Milestone 1 — Foundation and Environment Setup ✅

> **Status:** Done · **Period:** Jul 2025 – Nov 2025

- [x] Literature review: simulation environments, RL, tendon-driven robots
- [x] Isaac Sim 5.1.0 installation and conda environment setup
- [x] Remote desktop configuration for GPU workstation
- [x] Isaac Lab introductory tutorials (Cartpole, UR10 reaching)
- [x] Initial project structure and repository setup

## Milestone 2 — Robot Import and Scene Construction ✅

> **Status:** Done · **Period:** Nov 2025 – Feb 2026

- [x] 3-DoF tensegrity arm: URDF conversion, USD articulation
- [x] 5-DoF robot assembly: 2-DoF linear base + 3-DoF arm + Robotiq 2F-140 gripper
- [x] Base simulation scene: ground plane, conveyor belt, drum, dome lighting
- [x] T-shirt cloth asset (PBD particle mesh, no self-collision yet)
- [x] Isaac Sim physics showcase scene for visual verification
- [x] URDF models for wrist components (with and without IMU)

## Milestone 3 — Control Validation and Workspace Analysis ✅

> **Status:** Done · **Period:** Feb 2026 – early Mar 2026

- [x] PD gain tuning per joint (Isaac Sim Gain Tuner)
- [x] Step-response testing and validation against Klein (2023)
- [x] Robot actuation verification script
- [x] FK-based Monte Carlo workspace sampling (200k samples, 4096 envs)
- [x] Workspace visualization: 3D scatter plots, 2D cross-section heatmaps
- [x] Geometry documentation for place environment

## Milestone 4 — RL Task Development (Reach, Place, Sort) ✅

> **Status:** Done · **Period:** Mar 2026

- [x] Reach task: FK-sampled end-effector targets, PD-driven
- [x] Place task: green cube → drum with colour-discrimination curriculum
- [x] Cube sort task: 8 green + 8 red cubes, selective sorting
- [x] Tendon simulation: custom `TendonEffortAction`, Jacobian-transpose mapping
- [x] Tendon-driven variants of reach and place tasks
- [x] Shirt pick-and-place and sorting task templates

## Milestone 5 — Training, Evaluation, and PA Writing (Ch. 2) 🔄

> **Status:** In progress · **Deadline:** 31.03.2026

### Coding

- [ ] PPO training runs: PD vs. tendon comparison (reach task)
- [ ] PPO training runs: PD vs. tendon comparison (place task)
- [ ] Cube sort training and evaluation
- [ ] Training result plotting and analysis scripts
- [ ] Reproducibility table: software versions, hardware specs, hyperparameters
- [ ] UR10 + native rigid wrist: reach and place baselines
- [ ] UR10 + tendon wrist replacement: reach and place comparison

### Writing — Project Thesis

- [ ] **Ch. 1 Introduction: (Draft only)** motivation, problem statement, objectives, thesis outline
- [ ] **Ch. 2 Background:** §2.1 Reinforcement Learning for Robotic Manipulation
- [ ] **Ch. 2 Background:** §2.2 Physics Simulation in Isaac Sim 5.1
- [ ] **Ch. 2 Background:** §2.3 NVIDIA Isaac Sim and IsaacLab Architecture
- [ ] **Ch. 2 Background:** §2.4 Tendon-Driven Robot Kinematics and Modeling
- [ ] **Ch. 2 Background:** §2.5 Policy Learning Pipeline
- [ ] **Ch. 2 Background:** §2.6 Prior Work
- [ ] **Ch. 2 Background:** §2.7 Robot and Scene Description Formats

## Milestone 6 — PA Writing and Submission 🔲

> **Status:** Planned · **Deadline:** 30.04.2026

### Coding

- [ ] Statistical comparison of wrist transfer results
- [ ] Workspace analysis comparison (with/without 2-DoF base, vs. Klein 2023)

### Writing — Project Thesis

- [ ] **Ch. 3 Research Gap:** critical analysis, gap, refined problem definition
- [ ] **Ch. 4 Methodology:** §4.1–4.3 approach overview, environment setup, robot integration
- [ ] **Ch. 4 Methodology:** §4.4–4.7 gain tuning, workspace analysis, RL setup, wrist transfer
- [ ] **Ch. 5 Results:** validation, workspace, reach, place, wrist transfer results
- [ ] **Ch. 6 Discussion:** evaluation, transferability, limitations, practical implications
- [ ] **Ch. 7 Conclusion:** summary, outlook to master thesis
- [ ] **Ch. 1 Introduction:** motivation, problem statement, objectives, thesis outline
- [ ] Figures: all diagrams, plots, keyframe sequences finalized
- [ ] Bibliography: verify all citations, clean .bib file
- [ ] Internal draft review with supervisor
- [ ] Final revision and formatting check
- [ ] Print and bind two copies; prepare CD/DVD with digital appendix
- [ ] **Submit Project Thesis**

---

## Milestone 8 — Master Thesis: Sorting Task and Cloth Integration 🔲

> **Status:** Planned · **Deadline:** 15.05.2026

### Coding

- [ ] Extend cube sorting to selective T-shirt retrieval on moving conveyor
- [ ] T-shirt cloth model: PBD/VBD material parameter tuning (stretch, bending, friction)
- [ ] Cloth–articulation interaction: gripper grasping cloth particles
- [ ] Cloth behavior validation: draping and stretching tests
- [ ] Domain randomization: belt speed, object placement, friction, cloth parameters
- [ ] Performance benchmarking: training FPS with cloth vs. rigid objects

### Writing — Master Thesis

- [ ] **Ch. 1 Introduction:** motivation, problem statement, objectives, thesis outline
- [ ] **Ch. 2 Background:** §2.1 Foundation summary (reference PA results)
- [ ] **Ch. 2 Background:** §2.2 Cloth and Deformable Object Simulation

## Milestone 9 — Master Thesis: Multi-Agent System and Classification 🔲

> **Status:** Planned · **Deadline:** 15.06.2026

### Coding

- [ ] Second robot arm: selection, kinematic config, scene placement
- [ ] Stretching primitive: grasp cloth edge, extend to flat configuration
- [ ] Camera mounting: simulated camera, resolution, field-of-view design
- [ ] Damage classifier: CNN (ResNet-18) trained on synthetic stretched-cloth images
- [ ] Procedural damage textures and visual domain randomization
- [ ] Multi-agent RL formulation: MAPPO/IPPO, shared vs. independent rewards
- [ ] Curriculum: single-agent sub-tasks → coordinated full pipeline

### Writing — Master Thesis

- [ ] **Ch. 2 Background:** §2.3 Multi-Agent Reinforcement Learning
- [ ] **Ch. 2 Background:** §2.4 Vision-Based Classification for Robotic Sorting
- [ ] **Ch. 2 Background:** §2.5 Prior Work in Robotic Cloth Manipulation
- [ ] **Ch. 2 Background:** §2.6 Sim-to-Real Transfer Strategies
- [ ] **Ch. 3 Research Gap:** critical analysis, gap, refined research question
- [ ] **Ch. 4 Methodology:** §4.1–4.5 approach, sorting, cloth, second arm, classification

## Milestone 10 — Master Thesis: Sim-to-Real Prep, Results, and Final Writing 🔲

> **Status:** Planned · **Deadline:** 10.08.2026 (draft) · 01.09.2026 (submission)

### Coding

- [ ] End-to-end multi-agent evaluation: sorting accuracy per damage class, cycle time
- [ ] Ablation studies: without stretching, without classification, single vs. multi-agent
- [ ] Domain randomization sweep and robustness analysis
- [ ] ROS 2 architecture: joint state pub/sub, policy inference node, camera pipeline
- [ ] Sim-to-real readiness assessment: latency, message integrity validation

### Writing — Master Thesis

- [ ] **Ch. 4 Methodology:** §4.6–4.7 multi-agent RL formulation, sim-to-real preparation
- [ ] **Ch. 5 Results:** sorting, cloth sim, multi-agent pipeline, classification, ablations, sim-to-real
- [ ] **Ch. 6 Discussion:** evaluation, comparison with related work, limitations, practical implications
- [ ] **Ch. 7 Conclusion:** summary, outlook (physical deployment, real-world fine-tuning)
- [ ] Figures: all diagrams, confusion matrices, keyframe sequences finalized
- [ ] Bibliography: verify all citations, clean .bib file
- [ ] Internal draft review with supervisor
- [ ] Final revision and formatting check
- [ ] Print and bind two copies; prepare CD/DVD with digital appendix
- [ ] **Submit Master Thesis**


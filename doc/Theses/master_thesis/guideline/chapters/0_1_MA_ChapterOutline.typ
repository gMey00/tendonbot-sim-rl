// MA Chapter Outline — Planning reference for each thesis chapter and section
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *

#heading(numbering: none)[MA Chapter Outline] <ch:ma_chapter_outline>

// ═══════════════════════════════════════════════════════
// Chapter 1 — ~5 pages
// ═══════════════════════════════════════════════════════
== Introduction

=== Motivation and Context

- Industrial textile recycling requires automated sorting by garment condition (intact, minor damage, major damage, unusable)
- Current approaches rely on manual inspection or single-robot rigid-object pipelines; neither scales to deformable textiles on conveyors
- Multi-agent robotic systems combining pick, stretch, inspect, and sort offer a path to full automation
- This thesis builds on the simulation environment and validated tendon-driven robot from the preceding project thesis (Meyer, 2026)
- Central challenge: coordinating two robots under RL for deformable cloth manipulation with vision-based classification

=== Problem Statement and Objectives

- Thesis problem: developing a multi-agent RL system that coordinates a tendon-driven picker and a collaborative sorting arm for condition-based textile sorting in simulation
- Research objectives:
  + Selective sorting task: retrieve T-shirts from mixed conveyor waste
  + Deformable cloth simulation: integrate T-shirt cloth physics (PBD/VBD)
  + Second robot arm: stretch garments for camera-based inspection
  + Damage classification: vision-based condition assessment
  + Multi-agent coordination: joint policies for pick-stretch-classify-sort
  + Sim-to-real preparation: domain randomization, ROS 2 integration pathway
- Scope: simulation-based development and evaluation; physical deployment is prepared but not executed

=== Thesis Outline

- Brief summary of chapters 2–7 for reader orientation
- Explicitly reference the project thesis as the foundation (simulation, robot model, basic RL tasks)

// ═══════════════════════════════════════════════════════
// Chapter 2 — ~20–25 pages
// ═══════════════════════════════════════════════════════
== Background

=== Summary of Foundation (Project Thesis)

- Brief recap of the tendon-driven robot model, simulation environment, and RL pipeline established in the project thesis (Meyer, 2026)
- Reference rather than repeat: direct the reader to the project thesis for full details
- Summarize key results: validated robot model, reach/place task performance, workspace analysis findings
- State what is taken as given for the master thesis

=== Cloth and Deformable Object Simulation

- Why deformable objects break rigid-body assumptions
- Common cloth modeling approaches: mass-spring, particle-based (PBD), continuum (FEM/MPM)
- Position-Based Dynamics (PBD) in PhysX: particles, constraints, Gauss--Seidel projection, XPBD compliance
- Vertex Block Descent (VBD) / Newton Thin Deformables: implicit energy minimization, vertex-level Newton updates
- Comparison: PBD vs. VBD for cloth in RL workflows (speed, stability, stiffness fidelity)
- Isaac Sim cloth integration: particle cloth API, material parameters, collision with articulations

=== Multi-Agent Reinforcement Learning

- Extension from single-agent MDP to multi-agent: Dec-POMDP, centralized training / decentralized execution (CTDE)
- Multi-agent PPO variants: independent PPO (IPPO), multi-agent PPO (MAPPO)
- Shared vs. independent reward structures; cooperative vs. mixed objectives
- Communication and coordination mechanisms in multi-agent RL
- Prior work: multi-robot manipulation (e.g., bi-manual grasping, handover tasks)

=== Vision-Based Classification for Robotic Sorting

- Visual inspection in manufacturing and recycling contexts
- CNN-based defect detection and classification (ResNet, EfficientNet architectures)
- Sim-to-real for visual classifiers: synthetic data generation, domain randomization for textures/lighting/backgrounds
- Integration of perception with robot control: observation pipelines in IsaacLab

=== Prior Work in Robotic Cloth Manipulation

- Robotic cloth folding, unfolding, smoothing: key benchmarks (GarmentBench, GarmentLab, Seita et al.)
- RL approaches for deformable manipulation: reward design challenges, representation learning for cloth state
- Sim-to-real for cloth tasks: domain randomization of material properties, teacher-student distillation
- Multi-robot cloth handling: bi-manual stretching, cooperative folding
- Gap: no existing work combines tendon-driven robots, multi-agent RL, and condition-based sorting

=== Sim-to-Real Transfer Strategies

- Domain randomization: which parameters and how aggressively
- System identification and parameter calibration
- Teacher-student policy distillation
- ROS 2 as the deployment middleware: joint state interfaces, camera topics, action servers
- Prior sim-to-real demonstrations for manipulation tasks

// ═══════════════════════════════════════════════════════
// Chapter 3 — ~3–4 pages
// ═══════════════════════════════════════════════════════
== Problem Statement and Research Gap

=== Critical Analysis of State of the Art

- Cloth manipulation studied primarily with rigid-joint robots; tendon-driven compliance advantages unexplored for deformable objects
- Multi-agent RL applied to rigid object manipulation but not to cloth sorting pipelines
- Vision-based classification exists for quality inspection but not integrated into a multi-agent pick-stretch-sort loop
- Sim-to-real for multi-agent deformable manipulation is an open problem

=== Research Gap and Justification

- Key gap: no existing system combines multi-agent RL, tendon-driven manipulation, deformable cloth physics, and condition-based classification in a single pipeline
- Scientific contribution: multi-agent MDP formulation for condition-based textile sorting; evaluation of cooperative policies for pick-stretch-classify-sort
- Practical contribution: complete simulation framework for textile recycling automation; ROS 2 integration pathway

=== Refined Research Question

- How can a multi-agent RL system coordinate a tendon-driven picker and a collaborative sorting arm to selectively retrieve, inspect, and sort textiles by damage condition, and what strategies enable robust sim-to-real transfer?
- Expected contributions: multi-agent sorting pipeline, cloth sim integration, classification integration, sim-to-real preparation

// ═══════════════════════════════════════════════════════
// Chapter 4 — ~20–25 pages
// ═══════════════════════════════════════════════════════
== Methodology

=== Overview of Approach

- System architecture diagram: conveyor → tensegrity picker → handoff → sorting arm → stretch → camera → classify → sort into bins
- Development stages: (1) sorting task, (2) cloth integration, (3) second arm, (4) classification, (5) multi-agent, (6) sim-to-real prep

=== Sorting Task (Single-Agent Extension)

- Extend cube sorting to selective retrieval: T-shirt vs. non-target objects on moving conveyor
- MDP formulation: observation (joint states, conveyor state, object labels/features), action (joint targets + gripper), reward (correct pick, correct ignore, penalty for false pick/miss)
- Training with domain randomization (belt speed, object placement, friction)

=== Cloth Simulation Integration

- T-shirt cloth model: particle mesh, material parameters (mass, stretch stiffness, bending stiffness, friction, damping)
- PBD vs. VBD selection and justification for training throughput
- Cloth-articulation interaction: gripper grasping cloth particles, collision handling
- Validation of cloth behavior: qualitative draping tests, quantitative stretch tests
- Domain randomization of cloth parameters

=== Second Robot Arm Integration

- Arm selection and kinematic configuration
- Scene placement: workspace complementarity with tensegrity picker
- Stretching primitive: grasp cloth edge, extend to flat configuration
- Camera mounting and field-of-view design

=== Camera-Based Damage Classification

- Simulated camera setup: resolution, positioning, synthetic image generation
- Damage classes: intact, minor damage, major damage, unusable
- Classifier architecture: CNN (e.g., ResNet-18) trained on synthetic stretched-cloth images
- Training data generation: procedural damage textures, domain randomization (lighting, background, cloth color)
- Integration as observation in the RL loop or as a post-stretch evaluation module

=== Multi-Agent RL Formulation

- Agent decomposition: Agent 1 (tensegrity picker), Agent 2 (sorting arm)
- Observation spaces per agent; shared state components
- Action spaces per agent
- Reward structure: cooperative (shared sorting reward), individual shaping rewards per agent
- Coordination protocol: sequential (pick → handoff → stretch → classify → sort) or learned timing
- Algorithm: MAPPO or IPPO via skrl; centralized critic with decentralized actors
- Curriculum: single-agent sub-tasks → coordinated full pipeline

=== Sim-to-Real Transfer Preparation

- Domain randomization strategy: systematic parameter sweep and sensitivity analysis
- ROS 2 architecture: joint state publisher/subscriber, policy inference node, camera pipeline, safety constraints
- Action-space compatibility: simulation actions → ROS 2 joint trajectory commands
- Latency and timing considerations for real-time execution
- Gap analysis: what remains to be addressed for physical deployment

// ═══════════════════════════════════════════════════════
// Chapter 5 — ~15–20 pages
// ═══════════════════════════════════════════════════════
== Results

=== Sorting Task Results (Single Agent)

- Sorting accuracy: T-shirt correctly retrieved, non-targets correctly ignored
- False-pick rate, missed-pick rate
- Learning curves; comparison with project thesis cube sort baseline
- Effect of domain randomization on robustness

=== Cloth Simulation Results

- Cloth behavior validation: draping, stretching, grasping fidelity
- Performance impact: training FPS with cloth vs. rigid objects
- Stability under domain randomization of cloth parameters

=== Multi-Agent Sorting Pipeline Results

- End-to-end sorting accuracy by damage class
- Cycle time: pick to sort completion
- Coordination efficiency: handoff success rate, idle time
- Learning curves for multi-agent vs. single-agent baselines
- Qualitative policy behavior: keyframe sequences of the full pipeline

=== Damage Classification Results

- Classification accuracy per damage class (confusion matrix)
- Effect of cloth stretching quality on classification accuracy
- Robustness under visual domain randomization

=== Ablation Studies

- Without stretching: classification accuracy on unstretched cloth
- Without classification: sorting by pick heuristics only
- Single-agent vs. multi-agent: performance comparison
- Shared vs. independent reward: coordination quality

=== Sim-to-Real Readiness Assessment

- Robustness metrics under maximal domain randomization
- ROS 2 interface validation in simulation (latency, message integrity)
- Identified gaps and risk assessment for physical deployment

// ═══════════════════════════════════════════════════════
// Chapter 6 — ~5–7 pages
// ═══════════════════════════════════════════════════════
== Discussion

=== Evaluation of Outcomes

- Achievement of each thesis objective
- Multi-agent vs. single-agent: when is coordination worth the complexity?
- Cloth simulation fidelity: sufficient for policy learning? What artifacts affect results?

=== Comparison with Related Work

- Comparison with GarmentBench/GarmentLab benchmarks
- Comparison with single-robot cloth manipulation approaches
- Novelty: first multi-agent tendon-driven cloth sorting pipeline

=== Limitations

- Cloth simulation approximations (PBD/VBD vs. real fabric behavior)
- Synthetic damage textures vs. real-world garment damage
- No physical deployment; sim-to-real gap remains
- Classification limited to synthetic training data
- Computational cost of multi-agent training with cloth

=== Practical Implications

- Applicability to textile recycling industry
- Modularity: individual pipeline components (classifier, sorting policy) can be updated independently
- ROS 2 integration pathway enables incremental physical deployment

// ═══════════════════════════════════════════════════════
// Chapter 7 — ~3 pages
// ═══════════════════════════════════════════════════════
== Conclusion

=== Conclusion

- Summary of the complete multi-agent textile sorting pipeline
- Key quantitative achievements: sorting accuracy, classification accuracy, cycle time
- Answer the research question

=== Outlook

- Physical deployment via ROS 2 on real hardware
- Real-world fine-tuning: sim-to-real domain adaptation, real cloth training data
- More garment types and damage categories
- Higher-level task planning for continuous conveyor operation
- Integration with upstream waste detection and downstream recycling processes
- More realistic tendon model incorporating cable dynamics

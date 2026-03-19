# Master Thesis (Masterarbeit, 30 ECTS)

## Title

**Multi-Agent Reinforcement Learning for Condition-Based Textile Sorting with a Tendon-Driven and Collaborative Robot System**

## Clickbait Buzzword Title Options

- "Autonomous Textile Recycling at Scale: Multi-Agent Deep RL, AI-Powered Damage Classification, and Cloth-Aware Collaborative Robotics for Sustainable Smart Manufacturing" (tech-stack-forward, recruiterbait)
- "Closing the Loop on Fast Fashion: Embodied AI, Multi-Robot Coordination, and Vision-Based Condition Assessment for Intelligent Garment Sorting in Industry 4.0" (sustainability, linkedin push)

## Research Question

### Main Question  

How can a multi-agent reinforcement learning system coordinate a tendon-driven pick robot and a collaborative sorting arm to selectively retrieve, inspect, and sort textiles by damage condition on a simulated conveyor?

### Connected Questions

- Does the tendon-driven wrist mechanism transfer beneficially to a conventional robot arm?  
- What strategies enable robust sim-to-real transfer for deformable object manipulation?  

## Overarching Goal

Design and implement a complete multi-agent textile sorting pipeline in NVIDIA Isaac Sim, where a tendon-driven robot selectively retrieves clothing from a conveyor, a second robot arm stretches garments for camera-based damage classification, and items are sorted into condition-specific bins.
Evaluate the transferability and taskspecific utility of the tendon-driven wrist to a conventional robot arm.
Validate the approach through quantitative evaluation and prepare the integration pathway for physical deployment via ROS 2.

## Partial Goals

### PG-1: Conveyor-Based Sorting Task (Single Agent)

Extend the validated pick-and-place pipeline from the project thesis to a full sorting task: selective retrieval of target items (T-shirts) from a mixed waste stream on a moving conveyor while ignoring non-target objects. Train PPO policies with selectivity encoded in the reward function and evaluate sorting accuracy, false-pick rate, and missed-pick rate.

### PG-2: Cloth Simulation Integration

Integrate deformable cloth simulation into the Isaac Sim environment using PhysX particle cloth (PBD) or Newton thin deformables (VBD). Model T-shirt objects with realistic material properties (mass, bending stiffness, friction, damping) and validate cloth behavior through qualitative and quantitative benchmarks. Apply domain randomization over cloth parameters.

### PG-3.1: Second Robot Arm Integration

Add a second collaborative robot arm to the simulation scene for cloth stretching and presentation. Define the arm's kinematic configuration, placement, and workspace to complement the tensegrity picker. Implement the necessary control interfaces and observation/action spaces for the second arm, ensuring compatibility with the multi-agent RL framework. Validate the arm's ability to reach and manipulate garments within the shared workspace.

### PG-3.2: Tendon Wrist Transfer Experiment

Isolate the 2-DOF tendon-driven wrist mechanism and apply it to a standard industrial robot arm (e.g., Kinova Gen3, UR10e/UR5e) in Isaac Sim. Conduct a controlled comparison of reach accuracy and place success rate between the native rigid wrist and the tendon wrist, using identical RL hyperparameters, to evaluate whether the compliant wrist design improves manipulation capability.

### PG-3.3: Stretching and Sorting Primitive Implementation

Implement the stretching primitive: the second arm grasps a garment edge and extends it to a flat, camera-visible configuration.  
Implement the sorting primitive: the second arm picks the inspected garment and places it into one of several bins based on the classification result.  
Define the necessary observation and action spaces for these primitives, and validate their execution in isolation before integrating into the multi-agent pipeline.

### PG-4: Camera-Based Damage Classification

Integrate a simulated camera system for garment inspection. Develop or integrate a classification model that assesses textile damage degree (e.g., intact, minor damage, major damage, unusable) from the stretched cloth image. Train or fine-tune the classifier on synthetic data with domain randomization.

### PG-5: Multi-Agent Coordination

Formulate the full pipeline as a multi-agent MDP: the tensegrity robot picks and hands off, the sorting arm stretches, inspects, and sorts. Design the multi-agent observation and action spaces, inter-agent communication or shared reward structure, and coordination protocol. Train coordinated policies and evaluate end-to-end sorting performance.

### PG-6: Sim-to-Real Transfer Preparation

Assess transfer robustness through systematic domain randomization (conveyor speed, cloth properties, sensor noise, lighting). Define the ROS 2 integration architecture for physical deployment: joint state publishers, policy inference nodes, camera pipelines, and safety constraints. Identify and document remaining gaps for real-world operation.

### PG-7: Evaluation and Benchmarking

Define and apply comprehensive evaluation metrics: sorting accuracy by damage class, cycle time, false/missed classification rates, cloth coverage during inspection, and multi-agent coordination efficiency. Compare against single-agent baselines and ablation studies (e.g., without stretching, without classification).

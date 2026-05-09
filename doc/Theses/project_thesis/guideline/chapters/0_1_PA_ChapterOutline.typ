// PA Chapter Outline - Updated to match restructured PA thesis
// Last updated: 2026-04-03 (Typst overhaul)
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *

= Introduction

== Motivation and Context

- Tendon-driven robots: compliant design, reduced distal inertia, complex actuation modeling
- Reinforcement learning as a data-driven alternative to analytical control
- GPU-accelerated simulation (Isaac Sim + IsaacLab) as the enabling platform
- Frame the PA as the simulation foundation of a larger textile-sorting project at FAPS

== Problem Statement and Objectives

- *Objective 1:* Validated tendon-driven robot model in Isaac Sim
- *Objective 2:* Progressive RL-based manipulation (reach then cube place), comparing PD vs tendon actuation
- *Objective 3:* Documented infrastructure foundation for the master thesis
- Scope exclusion: no cloth, no sorting, no sim-to-real, no wrist transfer (deferred to MA)

== Thesis Outline

- Brief chapter summary for reader orientation

= Background

== Reinforcement Learning for Robotic Manipulation (2.1)
== Physics Simulation in Isaac Sim 5.1 (2.2)
== NVIDIA Isaac Sim and IsaacLab Architecture (2.3)
== Tendon-Driven Robot Kinematics and Modeling (2.4)

= State of the Art and Research Gap

== GPU-Accelerated RL for Manipulation (3.1)
== Tendon- and Cable-Driven Robot Simulation (3.2)
== Workspace Analysis Methods (3.3)
== Identified Research Gap (3.4)
== Success Criteria (3.5)

= Methodology

== Overview of Approach (4.1)
== Physical Robot Design (4.2)
=== Kinematic Chain and Degrees of Freedom (4.2.1)
=== Antiparallelogram Elbow Mechanism (4.2.2)
=== Cable-Driven Wrist (4.2.3)
=== Drive System and Tendon Routing (4.2.4)
== Simulation Model Construction (4.3)
=== Mesh Processing Pipeline (4.3.1)
=== Articulation and Physics Configuration (4.3.2)
=== Disc Approximation Variant (4.3.3)
=== Physical Antiparallelogram Variant (4.3.4)
=== Five-DoF Assembly (4.3.5)
== Tendon Actuation in Simulation (4.4)
=== Jacobian-Transpose Torque Mapping (4.4.1)
=== Physical Body-Force Tendon Model (4.4.2)
=== Actuation Mode Comparison (4.4.3)
=== Motor Specifications and Simulation Parameters (4.4.4)
== Model Validation (4.5)
=== Step-Response Testing Protocol (4.5.1)
=== Monte Carlo Workspace Analysis (4.5.2)
=== PD Gain Tuning (4.5.3)
== Reinforcement Learning Task Formulation (4.6)
=== Simulation Environment (4.6.1)
=== Reach Task (4.6.2)
=== Cube Place Task (4.6.3)
=== Cube Sort Task — Outlook (4.6.4)
== Training Infrastructure (4.7)
=== PPO Configuration (4.7.1)
=== Simulation Parameters (4.7.2)
=== Software and Hardware (4.7.3)

= Results

== Model Validation Results (5.1)
== Reach Task Results (5.2)
== Cube Place Results (5.3)
== Cube Sort Preliminary (5.4)
== Summary of Key Results (5.5)

= Discussion

== Evaluation of Outcomes (6.1)
== Comparison with Related Work (6.2)
== Limitations of the Study (6.3)
== Practical Implications (6.4)
== Recommendations and Lessons Learned (6.5)

= Conclusion

== Conclusion (7.1)
== Outlook (7.2)

= Appendices

- Appendix A: Simulation and Control Parameters
- Appendix B: PPO Hyperparameters
- Appendix C: Cube Place Reward Terms

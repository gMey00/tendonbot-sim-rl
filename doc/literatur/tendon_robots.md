# Tendon-Driven & Cable Robots

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Cable-Driven Parallel Robots

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Pott2018 | Pott, A. | *Cable-Driven Parallel Robots: Theory and Application* | 2018 | 🟡 | 🟢 | 🟢 | Comprehensive monograph covering workspace analysis, tension distribution, stiffness modelling, and trajectory planning for CDPRs. Provides the formal framework for kinetostatic duality between cable tensions and end-effector wrenches. [Springer](https://link.springer.com/book/10.1007/978-3-319-76138-1) |
| Bruckmann2013 | Bruckmann, T. & Pott, A. (Eds.) | *Cable-Driven Parallel Robots* (Mechanisms and Machine Science, Vol. 12) | 2013 | 🔴 | 🟡 | 🟢 | Edited volume collecting state-of-the-art CDPR contributions from leading research groups. Covers design, modelling, control, and applications — broad context for the tendon-driven approach. [Springer](https://link.springer.com/book/10.1007/978-3-642-31988-4) |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Borgstrom2009 | Borgstrom, P. H. et al. | "Rapid Computation of Optimally Safe Tension Distributions for Parallel Cable-Driven Robots" | 2009 | 🟡 | 🟢 | 🟢 | Efficient algorithms for computing cable tension distributions satisfying positive-tension constraints while minimising total cable force. Applicable to the 5-tendon / 3-DOF mapping in the tensegrity manipulator. |
| Oh2005 | Oh, S.-R. & Agrawal, S. K. | "Cable Suspended Planar Robots with Redundant Cables: Controllers with Positive Tensions" | 2005 | 🟡 | 🟢 | 🟢 | Theoretical basis for controlling cable-driven robots with redundant cables under positive-tension constraints. The pseudo-inverse tension distribution approach used in step-response tests follows this line of work. |
| Dhakate2025 | Dhakate, R. et al. | "CaRoSaC: A Reinforcement Learning-Based Kinematic Control of Cable-Driven Parallel Robots by Addressing Cable Sag Through Simulation" | 2025 | 🟡 | 🟢 | 🟢 | RL-based kinematic control of CDPRs addressing cable sag in simulation. Directly relevant as a prior-work comparison for RL-controlled cable robots. [DOI](https://doi.org/10.1109/LRA.2025.3564225) |
| Garrido2024 | Garrido Campos, J. et al. | "Integrated Simulation and Control Environment for CDPRs" | 2024 | 🟡 | 🟡 | 🟢 | Integrated simulation and control environment for cable-driven parallel robots, relevant as a comparable simulation approach. [DOI](https://doi.org/10.1080/0951192X.2024.2389640) |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Walter2023 | Walter, J. et al. | "Tensegrity-Inspired Joint Can Protect from Impacts by Isolating" | 2023 | 🟢 | 🟢 | 🟢 | Introduces the tensegrity-inspired joint design used in this project's manipulator. Core reference for the mechanical concept. [DOI](https://doi.org/10.1007/978-3-031-32322-5_28) |
| Walter2022 | Walter, J. et al. | "Bringing the New Class of CDPRs to Industrial CPS Based on RAMI 4.0" | 2022 | 🔴 | 🟡 | 🟢 | Contextualises cable-driven parallel robots within Industry 4.0 cyber-physical systems. Background for the industrial motivation. [DOI](https://doi.org/10.1109/ICPS51978.2022.9816877) |

---

## Tendon-Driven Continuum Robots

### Review Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Rao2021 / Lilge2021 | Rao, P. et al. / Lilge, S. & Burgner-Kahrs, J. | "How to Model Tendon-Driven Continuum Robots and Benchmark Modelling Performance" | 2021 | 🟡 | 🟢 | 🟢 | Benchmarks modelling approaches (constant-curvature, Cosserat rod, etc.) for tendon-driven continuum robots. Key context for understanding the limitations of simplified kinematic models. [DOI](https://doi.org/10.3389/frobt.2020.630245) |
| Webster2010 | Webster III, R. J. & Jones, B. A. | "Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review" | 2010 | 🔴 | 🟢 | 🟢 | Foundational review of constant-curvature continuum robot modelling. Provides the kinematic modelling context used to justify the piecewise-constant approach. [DOI](https://doi.org/10.1177/0278364910368147) |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Nemoto2022 | Nemoto, T. et al. | "Highly Dynamic 2-DOF Cable-Driven Robotic Wrist Based on a Novel Topology" | 2022 | 🟢 | 🟢 | 🟢 | Presents the 2-DOF cable-driven wrist topology directly used in the tensegrity manipulator. Core hardware reference for the wrist mechanism. [DOI](https://doi.org/10.1109/LRA.2022.3159816) |
| Miyasaka2020 | Miyasaka, M. et al. | "Modeling Cable-Driven Robot With Hysteresis and Cable–Pulley Network Friction" | 2020 | 🟡 | 🟡 | 🟢 | Models hysteresis and friction in cable-pulley networks. Relevant for understanding sim-to-real discrepancies in tendon force transmission. [DOI](https://doi.org/10.1109/TMECH.2020.2973428) |
| Kato2016 | Kato, T. et al. | "Tendon-driven continuum robot for neuroendoscopy: validation of extended kinematic mapping for hysteresis operation" | 2016 | 🔴 | 🟡 | 🟢 | Validates kinematic mapping with hysteresis compensation for tendon-driven continuum robots in medical applications. Background for hysteresis modelling. [DOI](https://doi.org/10.1007/s11548-015-1310-2) |
| Saito2022 | Saito, N. & Morimoto, J. | "Characterization of Continuum Robot Arms Under Reinforcement Learning and Derived Improvements" | 2022 | 🟡 | 🟢 | 🟢 | Characterises continuum robot arms under RL control. Directly relevant as a prior-work comparison for RL-controlled tendon robots. [DOI](https://doi.org/10.3389/frobt.2022.895388) |
| Shahid2023 | Shahid, A. A. et al. | "Curriculum-Reinforcement Learning on Simulation Platform of Tendon-Driven High-DOF Underactuated Manipulator" | 2023 | 🟢 | 🟢 | 🟢 | Curriculum-based RL for tendon-driven underactuated manipulators. One of the closest prior works to this project's approach. [DOI](https://doi.org/10.3389/frobt.2023.1066518) |
| Cao2017 | Cao, K. et al. | "Workspace Analysis of Tendon-Driven Continuum Robots Based on Mechanical Interference Identification" | 2017 | 🔴 | 🟡 | 🟢 | Workspace analysis for TDCRs accounting for mechanical interference. Background for workspace analysis methodology. [DOI](https://doi.org/10.1115/1.4036395) |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Nemoto2019 | Nemoto, T., Niiyama, R. & Iwata, H. | "Design and Analysis of a Compact Tendon-Driven Rotary Joint with Application to a Robotic Arm" | 2019 | 🟡 | 🟢 | 🟢 | Foundational control principle for tendon-driven orientation regulation used in the tensegrity manipulator's wrist mechanism. Cited as [84] in Klein (2023). |

### Theses

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Klein2023 | Klein, M. | *Arbeitsraumanalyse, Simulation und Bewegungsplanung eines seilgetriebenen robotischen Manipulators* | 2023 | 🟢 | 🟢 | 🟢 | **Primary reference.** Describes the tensegrity manipulator's kinematics, cable-tension distribution, PID control with gravity compensation, and step-response validation in Gazebo. The Isaac Sim implementation replicates control gains, test methodology, and NRMSE metrics from this thesis. |

### Research Contributions

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Mukherjee | Mukherjee, S., Kamal, A. & Hirai, S. | Gravity compensation: τ_a = m·g·l_c·sin(θ) | — | 🟢 | 🟡 | 🟡 | Provides the gravity compensation formula used in the PID controller for step-response validation ([85] in Klein 2023, Eq. 3.2). |

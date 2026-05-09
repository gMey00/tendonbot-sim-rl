# Tendon-Driven & Cable Robots

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Cable-Driven Parallel Robots

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Pott2018](sources/tendon_robots/Books/Pott2018.pdf) | Pott, A. | *Cable-Driven Parallel Robots: Theory and Application* | 2018 | 🟡 | 🟢 | 🟢 | Comprehensive monograph covering workspace analysis, tension distribution, stiffness modelling, and trajectory planning for CDPRs. Provides the formal framework for kinetostatic duality between cable tensions and end-effector wrenches. [Springer](https://link.springer.com/book/10.1007/978-3-319-76138-1) |
| Bruckmann2013 | Bruckmann, T. & Pott, A. (Eds.) | *Cable-Driven Parallel Robots* (Mechanisms and Machine Science, Vol. 12) | 2013 | 🔴 | 🟡 | 🟢 | Edited volume collecting state-of-the-art CDPR contributions from leading research groups. Covers design, modelling, control, and applications — broad context for the tendon-driven approach. [Springer](https://link.springer.com/book/10.1007/978-3-642-31988-4) |
| [Lau2025CableCon](sources/tendon_robots/Books/Lau2025CableCon.epub) | Lau, D. et al. (Eds.) | *Cable-Driven Parallel Robots (CableCon 2025)* — Mechanisms and Machine Science | 2025 | 🔴 | 🟡 | 🟢 | Proceedings of the 7th International Conference on Cable-Driven Parallel Robots (CableCon 2025). Latest compilation of CDPR research on design, modelling, control, and applications. [Springer](https://link.springer.com/book/10.1007/978-3-031-84914-3) |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Borgstrom2009](sources/tendon_robots/Papers/Borgstrom2009.pdf) | Borgstrom, P. H. et al. | "Rapid Computation of Optimally Safe Tension Distributions for Parallel Cable-Driven Robots" | 2009 | 🟡 | 🟢 | 🟢 | Efficient algorithms for computing cable tension distributions satisfying positive-tension constraints while minimising total cable force. Applicable to the 5-tendon / 3-DOF mapping in the tensegrity manipulator. |
| [Dhakate2025](sources/tendon_robots/Papers/Dhakate2025.pdf) | Dhakate, R. et al. | "CaRoSaC: A Reinforcement Learning-Based Kinematic Control of Cable-Driven Parallel Robots by Addressing Cable Sag Through Simulation" | 2025 | 🟡 | 🟢 | 🟢 | RL-based kinematic control of CDPRs addressing cable sag in simulation. Directly relevant as a prior-work comparison for RL-controlled cable robots. [DOI](https://doi.org/10.1109/LRA.2025.3564225) |
| [Garrido2024](sources/tendon_robots/Papers/Garrido2024.pdf) | Garrido Campos, J. et al. | "Integrated Simulation and Control Environment for CDPRs" | 2024 | 🟡 | 🟡 | 🟢 | Integrated simulation and control environment for cable-driven parallel robots, relevant as a comparable simulation approach. [DOI](https://doi.org/10.1080/0951192X.2024.2389640) |

### Foundational Works

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Ming1994a | Ming, A. & Higuchi, T. | "Study on Multiple DOF Positioning Mechanism Using Wires (Part 1) — Concept, Design and Control" | 1994 | 🔴 | 🟢 | 🟢 | First m vs. n classification of cable robots: incompletely, completely, and redundantly restrained. |
| Ming1994b | Ming, A. & Higuchi, T. | "Study on Multiple DOF Positioning Mechanism Using Wires (Part 2) — Development of a Planar Completely Restrained Positioning Mechanism" | 1994 | 🔴 | 🟡 | 🟢 | Planar m = n + 1 prototype with force equilibrium equations for completely restrained positioning. |
| [Verhoeven2004](sources/tendon_robots/Papers/Verhoeven2004.pdf) | Verhoeven, R. | *Analysis of the Workspace of Tendon-based Stewart Platforms* | 2004 | 🟡 | 🟢 | 🟢 | Origin of the "structure matrix" terminology. IRPM/CRPM/RRPM classification and controllable workspace definition. |

### Structure Matrix & Force Distribution

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Bruckmann2008a](sources/tendon_robots/Papers/Bruckmann2008a.pdf) | Bruckmann, T., Mikelsons, L., Brandt, T., Hiller, M. & Schramm, D. | "Wire Robots Part I: Kinematics, Analysis & Design" | 2008 | 🟡 | 🟢 | 🟢 | Explicit naming of the structure matrix; establishes J^T equivalence for cable robots. |
| [Bruckmann2008b](sources/tendon_robots/Papers/Bruckmann2008b.pdf) | Bruckmann, T., Mikelsons, L., Brandt, T., Hiller, M. & Schramm, D. | "Wire Robots Part II: Dynamics, Control & Application" | 2008 | 🟡 | 🟢 | 🟢 | A^T · t = w equation, dynamics, and force distribution methods (barycentric, LP, optimization). |
| [Pott2009](sources/tendon_robots/Papers/Pott2009.pdf) | Pott, A., Bruckmann, T. & Mikelsons, L. | "Closed-Form Force Distribution for Parallel Wire Robots" | 2009 | 🟡 | 🟢 | 🟢 | Non-iterative algorithm for force distribution with continuity guarantees. IPAnema prototype validation. |
| [Gosselin2011](sources/tendon_robots/Papers/Gosselin2011.pdf) | Gosselin, C. & Grenier, M. | "On the Determination of the Force Distribution in Overconstrained Cable-Driven Parallel Mechanisms" | 2011 | 🟡 | 🟢 | 🟢 | Wrench matrix terminology; p-norm optimization (4-norm) for force distribution via univariate polynomial. |
| Gosselin2008 | Gosselin, C. | "On the Determination of the Force Distribution in Overconstrained Cable-Driven Parallel Mechanisms" | 2008 | 🔴 | 🟡 | 🟢 | Conference version of Gosselin & Grenier (2011). Early wrench matrix force distribution formulation. |

### Workspace Analysis (CDPM)

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Bosscher2006](sources/tendon_robots/Papers/Bosscher2006.pdf) | Bosscher, P., Riechel, A. T. & Ebert-Uphoff, I. | "Wrench-Feasible Workspace Generation for Cable-Driven Robots" | 2006 | 🟡 | 🟢 | 🟢 | Wrench-feasible workspace (WFW) definition via available wrench set (zonotope) with bounded tension constraints. |
| [GouttefardeGosselin2006](sources/tendon_robots/Papers/Gouttefarde2006.pdf) | Gouttefarde, M. & Gosselin, C. M. | "Analysis of the Wrench-Closure Workspace of Planar Parallel Cable-Driven Mechanisms" | 2006 | 🔴 | 🟢 | 🟢 | Wrench-closure workspace formulation for planar cable-driven mechanisms. Companion to Bosscher 2006; precursor to Gouttefarde 2011. Published in IEEE Transactions on Robotics. |
| [Gouttefarde2011](sources/tendon_robots/Papers/Gouttefarde2011.pdf) | Gouttefarde, M., Daney, D. & Merlet, J.-P. | "Interval-Analysis-Based Determination of the Wrench-Feasible Workspace of Parallel Cable-Driven Robots" | 2011 | 🔴 | 🟡 | 🟢 | Interval arithmetic for WFW computation accounting for geometric uncertainties. |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Walter2023](sources/tendon_robots/Papers/Walter2023.pdf) | Walter, J. et al. | "Tensegrity-Inspired Joint Can Protect from Impacts by Isolating" | 2023 | 🟢 | 🟢 | 🟢 | Introduces the tensegrity-inspired joint design used in this project's manipulator. Core reference for the mechanical concept. [DOI](https://doi.org/10.1007/978-3-031-32322-5_28) |
| [Khosravi2011](sources/tendon_robots/Papers/Khosravi2011.pdf) | Khosravi, M. A. & Taghirad, H. D. | "Dynamic Analysis and Control of Cable-Driven Robots with Elastic Cables" | 2011 | 🔴 | 🟡 | 🟢 | Analyses cable elasticity effects on dynamics and proposes a robust controller. Relevant for modelling cable compliance in tension distribution. [DOI](https://doi.org/10.1109/CCECE.2011.6030586) |

---

## Tendon-Driven Continuum Robots

### Review Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Rao2021](sources/tendon_robots/Reviews/Rao2021.pdf) | Rao, P., Peyron, Q., Lilge, S. & Burgner-Kahrs, J. | "How to Model Tendon-Driven Continuum Robots and Benchmark Modelling Performance" | 2021 | 🟡 | 🟢 | 🟢 | Benchmarks modelling approaches (constant-curvature, Cosserat rod, etc.) for tendon-driven continuum robots. Key context for understanding the limitations of simplified kinematic models. [DOI](https://doi.org/10.3389/frobt.2020.630245) |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Nemoto2022](sources/tendon_robots/Papers/Nemoto2022.pdf) | Nemoto, T. et al. | "Highly Dynamic 2-DOF Cable-Driven Robotic Wrist Based on a Novel Topology" | 2022 | 🟢 | 🟢 | 🟢 | Presents the 2-DOF cable-driven wrist topology directly used in the tensegrity manipulator. Core hardware reference for the wrist mechanism. [DOI](https://doi.org/10.1109/LRA.2022.3159816) |
| [Miyasaka2020](sources/tendon_robots/Papers/Miyasaka2020.pdf) | Miyasaka, M. et al. | "Modeling Cable-Driven Robot With Hysteresis and Cable–Pulley Network Friction" | 2020 | 🟡 | 🟡 | 🟢 | Models hysteresis and friction in cable-pulley networks. Relevant for understanding sim-to-real discrepancies in tendon force transmission. [DOI](https://doi.org/10.1109/TMECH.2020.2973428) |
| [Kato2016](sources/tendon_robots/Papers/Kato2016.pdf) | Kato, T. et al. | "Tendon-driven continuum robot for neuroendoscopy: validation of extended kinematic mapping for hysteresis operation" | 2016 | 🔴 | 🟡 | 🟢 | Validates kinematic mapping with hysteresis compensation for tendon-driven continuum robots in medical applications. Background for hysteresis modelling. [DOI](https://doi.org/10.1007/s11548-015-1310-2) |
| [Saito2022](sources/tendon_robots/Papers/Saito2022.pdf) | Saito, N. & Morimoto, J. | "Characterization of Continuum Robot Arms Under Reinforcement Learning and Derived Improvements" | 2022 | 🟡 | 🟢 | 🟢 | Characterises continuum robot arms under RL control. Directly relevant as a prior-work comparison for RL-controlled tendon robots. [DOI](https://doi.org/10.3389/frobt.2022.895388) |
| Shahid2023 | Shahid, A. A. et al. | "Curriculum-Reinforcement Learning on Simulation Platform of Tendon-Driven High-DOF Underactuated Manipulator" | 2023 | 🟢 | 🟢 | 🟢 | Curriculum-based RL for tendon-driven underactuated manipulators. One of the closest prior works to this project's approach. [DOI](https://doi.org/10.3389/frobt.2023.1066518) |
| Cao2017 | Cao, K. et al. | "Workspace Analysis of Tendon-Driven Continuum Robots Based on Mechanical Interference Identification" | 2017 | 🔴 | 🟡 | 🟢 | Workspace analysis for TDCRs accounting for mechanical interference. Background for workspace analysis methodology. [DOI](https://doi.org/10.1115/1.4036395) |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Nemoto2019 | Nemoto, T., Niiyama, R. & Iwata, H. | "Design and Analysis of a Compact Tendon-Driven Rotary Joint with Application to a Robotic Arm" | 2019 | 🟡 | 🟢 | 🟢 | Foundational control principle for tendon-driven orientation regulation used in the tensegrity manipulator's wrist mechanism. Cited as [84] in Klein (2023). |

### Theses

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Klein2023](sources/tendon_robots/Theses/Klein2023.pdf) | Klein, M. | *Arbeitsraumanalyse, Simulation und Bewegungsplanung eines seilgetriebenen robotischen Manipulators* | 2023 | 🟢 | 🟢 | 🟢 | **Primary reference.** Describes the tensegrity manipulator's kinematics, cable-tension distribution, PID control with gravity compensation, and step-response validation in Gazebo. The Isaac Sim implementation replicates control gains, test methodology, and NRMSE metrics from this thesis. |

### Research Contributions

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Mukherjee](sources/tendon_robots/Papers/Mukherjee.pdf) | Mukherjee, S., Kamal, A. & Hirai, S. | Gravity compensation: τ_a = m·g·l_c·sin(θ) | — | 🟢 | 🟡 | 🟡 | Provides the gravity compensation formula used in the PID controller for step-response validation ([85] in Klein 2023, Eq. 3.2). |

---

## Cable-Driven Wrist Mechanisms

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Jiang2019](sources/tendon_robots/Papers/Jiang2019.pdf) | Jiang, H., Xiao, C., Li, J., Zhong, Z., Zhang, T. & Guan, Y. | "Design and Modeling of a 2-DOF Cable-Driven Parallel Wrist Mechanism" | 2019 | 🔴 | 🟡 | 🟢 | 2-DOF hemispherical cable-driven wrist with no singular points and 1.07 × 10⁶ Nmm/rad stiffness. |
| [Tsumaki2018](sources/tendon_robots/Papers/Tsumaki2018.pdf) | Tsumaki, Y., Suzuki, Y., Sasaki, N., Obara, E. & Kanazawa, S. | "A 7-DOF Wire-Driven Lightweight Arm with Wide Wrist Motion Range" | 2018 | 🔴 | 🟡 | 🟢 | 7-DOF arm with a 3-DOF parallel-wire wrist and novel wire arrangements. |

---

## RL for Tendon-Driven Robots

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Or2023](sources/tendon_robots/Papers/Or2023.pdf) | Or, K., Wu, K. Y., Nakano, K., Ikeda, M., Ando, M., Kuniyoshi, Y. & Niiyama, R. | "Curriculum-reinforcement learning on simulation platform of tendon-driven high-degree of freedom underactuated manipulator" | 2023 | 🟢 | 🟢 | 🟢 | Robostrich 18-DOF tendon arm trained with SAC and curriculum learning in MuJoCo. Key prior work for tendon-driven RL manipulation. |
| [Guist2024](sources/tendon_robots/Papers/Guist2024.pdf) | Guist, S., Schneider, J., Ma, H. et al. | "Safe & Accurate at Speed with Tendons: A Robot Arm for Exploring Dynamic Motion" | 2024 | 🟡 | 🟢 | 🟢 | PAMY2 4-DOF pneumatic tendon arm trained over 25 days of real RL for dynamic table tennis. Demonstrates real-world tendon RL feasibility. |
| [Liu2024](sources/tendon_robots/Papers/Liu2024.pdf) | Liu, Y., Luo, Z., Qian, S., Wang, S. & Wu, Z. | "Deep Reinforcement Learning and Decoupling PID Control of a Humanoid Cable-Driven Hybrid Robot" | 2024 | 🟡 | 🟢 | 🟢 | Cable-driven humanoid arm with decoupling PID + TD3, achieving 77.58 % error reduction. Hybrid RL/classical control. |
| [Yuryev2026](sources/tendon_robots/Papers/Yuryev2026.pdf) | Yuryev, V. & Hughes, J. | "Tendon Force Modeling for Sim2Real Transfer of Reinforcement Learning Policies for Tendon-Driven Robots" | 2026 | 🟡 | 🟢 | 🟢 | Transformer-based tendon force model reducing sim-to-real gap by 41 % and improving pose tracking by 50 %. |
| [Elstner2024](sources/tendon_robots/Papers/Elstner2024.pdf) | Elstner, L., Kyrkjebø, E. & Stoelen, M. F. | "Contact-Rich Task Learning on an Articulated Soft Robot Arm Through Simulation" | 2024 | 🟡 | 🟡 | 🟢 | Antagonistic VSA with stretchable tendons in MuJoCo RL. Context for soft-arm RL. |
| [Theodorou2012](sources/tendon_robots/Papers/Theodorou2012.pdf) | Theodorou, E., Buchli, J. & Schaal, S. | "Tendon-Driven Variable Impedance Control Using Reinforcement Learning" | 2012 | 🔴 | 🟡 | 🟢 | PI² RL on ACT hand for antagonist co-contraction learning. Early tendon RL reference. |
| [Li2025robust](sources/tendon_robots/Papers/Li2025robust.pdf) | Li, X. et al. | "Robust Reinforcement Learning Control for Tendon-Driven Robots" | 2025 | 🟡 | 🟢 | 🟢 | Robust RL control approach tailored to tendon-driven manipulators with parameter uncertainty. Direct comparator for the project's RL pipeline. |
| [Maghooli2025](sources/tendon_robots/Papers/Maghooli2025.pdf) | Maghooli, M. et al. | "Adaptive Control of Cable-Driven Robotic Manipulators with Time-Varying Cable Stiffness" | 2025 | 🟡 | 🟡 | 🟢 | Adaptive controller accounting for time-varying cable stiffness in CDPRs. Relevant for handling cable compliance drift in long-running deployments. |

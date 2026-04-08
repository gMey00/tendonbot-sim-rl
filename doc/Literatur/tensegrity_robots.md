# Tensegrity Robots & Compliant Mechanisms

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Foundations & History

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Skelton2009 | Skelton, R. E. & de Oliveira, M. C. | *Tensegrity Systems* | 2009 | 🔴 | 🟢 | 🟢 | Class-k tensegrity taxonomy, equilibrium conditions, stiffness matrices, dynamics, and closed-loop control. Provides the formal mathematical framework for tensegrity analysis. |
| Motro2003 | Motro, R. | *Tensegrity: Structural Systems for the Future* | 2003 | 🔴 | 🟢 | 🟢 | Historical narrative from Ioganson (1921) through Snelson and Fuller. Defines tensegrity as systems of compressed and tensioned components in self-equilibrium. |
| Zhang2015 | Zhang, J. Y. & Ohsaki, M. | *Tensegrity Structures: Form, Stability, and Symmetry* | 2015 | 🔴 | 🟡 | 🟢 | Mathematical form-finding methods, stability analysis, and symmetry properties of tensegrity structures. Rigorous mathematical treatment. |

### Survey Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Sultan2009 | Sultan, C. | "Tensegrity: 60 Years of Art, Science, and Engineering" | 2009 | 🔴 | 🟢 | 🟢 | Comprehensive 77-page survey covering history, form-finding, stability, dynamics, control, and applications of tensegrity structures. |

### Foundational Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Roth1981 | Roth, B. & Whiteley, W. | "Tensegrity Frameworks" | 1981 | 🔴 | 🟡 | 🟢 | First mathematical formalization of tensegrity, establishing rigidity and flexibility criteria for tensegrity frameworks. |
| Connelly1996 | Connelly, R. & Whiteley, W. | "Second-Order Rigidity and Prestress Stability for Tensegrity Frameworks" | 1996 | 🔴 | 🟡 | 🟢 | Establishes the rigidity hierarchy and demonstrates the importance of prestress stability for tensegrity analysis. |
| Connelly1992 | Connelly, R. & Whiteley, W. | "The Stability of Tensegrity Frameworks" | 1992 | 🔴 | 🟡 | 🟢 | Four mathematical rigidity detection concepts for tensegrity structures. |
| Connelly1998 | Connelly, R. & Back, A. | "Mathematics and Tensegrity" | 1998 | 🔴 | 🟡 | 🟢 | Accessible overview of combinatorial rigidity theory applied to tensegrity. |

### Patents & Historical

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Snelson1965 | Snelson, K. D. | "Continuous Tension, Discontinuous Compression Structures" (US Patent 3,169,611) | 1965 | 🔴 | 🟡 | 🟢 | First tensegrity patent, documenting the X-Piece concept invented 1948–49. |
| Fuller1962 | Fuller, R. B. | "Tensile-Integrity Structures" (US Patent 3,063,521) | 1962 | 🔴 | 🟡 | 🟢 | Origin of the "tensegrity" terminology, combining "tensile" and "integrity". |
| Snelson2012 | Snelson, K. | "The Art of Tensegrity" | 2012 | 🔴 | 🟡 | 🟡 | First-person account of the invention of tensegrity from the artistic perspective. |

---

## Tensegrity Robotics Reviews

### Review Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Shah2022 | Shah, D. S., Booth, J. W., Baines, R. L., Wang, K., Vespignani, M., Bekris, K. & Kramer-Bottiglio, R. | "Tensegrity Robotics" | 2022 | 🟡 | 🟢 | 🟢 | Comprehensive review covering simulation tools (NTRT, MuJoCo, Bullet), locomotion, and manipulation applications of tensegrity robots. |
| Liu2022review | Liu, Y., Bi, Q., Yue, X., Wu, J., Yang, B. & Li, Y. | "A Review on Tensegrity Structures-Based Robots" | 2022 | 🟡 | 🟢 | 🟢 | Design, fabrication, modeling, and control of tensegrity robots. Positions tensegrity as bridging rigid and soft robotics. |

---

## Tensegrity Locomotion

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Paul2006 | Paul, C., Valero-Cuevas, F. J. & Lipson, H. | "Design and Control of Tensegrity Robots for Locomotion" | 2006 | 🔴 | 🟢 | 🟢 | Seminal paper on tensegrity locomotion: TR3, TR4 robots with genetically evolved control strategies. |
| Sabelhaus2015 | Sabelhaus, A. P., Bruce, J., Caluwaerts, K. et al. | "System Design and Locomotion of SUPERball, an Untethered Tensegrity Robot" | 2015 | 🔴 | 🟢 | 🟢 | NASA SUPERball 6-bar tensegrity with modular end-caps and impact absorption. Key engineering reference. |
| Caluwaerts2014 | Caluwaerts, K., Despraz, J., Işçen, A., Sabelhaus, A. P., Bruce, J., Schrauwen, B. & SunSpiral, V. | "Design and Control of Compliant Tensegrity Robots through Simulation and Hardware Validation" | 2014 | 🔴 | 🟡 | 🟢 | Controllable compliance, force distribution, and CPG validation for tensegrity robots. |
| Rieffel2018 | Rieffel, J. & Mouret, J.-B. | "Adaptive and Resilient Soft Tensegrity Robots" | 2018 | 🔴 | 🟡 | 🟢 | Structural resilience and behavioral adaptation of soft tensegrity robots post-damage. |

---

## Tensegrity Manipulators

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Lessard2016 | Lessard, S., Castro, D., Asper, W., Chopra, S. D., Baltaxe-Admony, L. B., Teodorescu, M., SunSpiral, V. & Agogino, A. | "A Bio-Inspired Tensegrity Manipulator with Multi-DOF, Structurally Compliant Joints" | 2016 | 🔴 | 🟢 | 🟢 | 4-DOF tensegrity arm using nested tetrahedrons with compliant joints as an alternative to variable stiffness actuators. |
| Fasquelle2020 | Fasquelle, B., Furet, M., Khanna, P., Chablat, D., Chevallereau, C. & Wenger, P. | "A Bio-Inspired 3-DOF Light-Weight Manipulator with Tensegrity X-Joints" | 2020 | 🟡 | 🟢 | 🟢 | 3-DOF X-joint arm hardware from the AVINECK bird-neck project. Directly relevant as a comparable tensegrity manipulator design. |
| Furet2018 | Furet, M., Lettl, M. & Wenger, P. | "Kinematic Analysis of Planar Tensegrity 2-X Manipulators" | 2018 | 🟡 | 🟢 | 🟢 | Symbolic inverse kinematics (up to 4 solutions) and cuspidal manipulation capability of 2-X tensegrity arms. |
| Furet2019 | Furet, M. & Wenger, P. | "Kinetostatic Analysis and Actuation Strategy of a Planar Tensegrity 2-X Manipulator" | 2019 | 🟡 | 🟢 | 🟢 | 2-X planar arm with variable stiffness achieved through antagonistic cables. |
| Fasquelle2019 | Fasquelle, B., Furet, M., Chevallereau, C. & Wenger, P. | "Dynamic Modeling and Control of a Tensegrity Manipulator Mimicking a Bird Neck" | 2019 | 🟡 | 🟢 | 🟢 | Lagrangian dynamics of 3-X arm with computed-torque control. |
| Fasquelle2022 | Fasquelle, B., Khanna, P., Chevallereau, C., Chablat, D., Creusot, D., Jolivet, S., Lemoine, P. & Wenger, P. | "Identification and Control of a 3-X Cable-Driven Manipulator Inspired From the Bird's Neck" | 2022 | 🟡 | 🟢 | 🟢 | Physical prototype validation. Key finding: friction dominates performance; cable elasticity is secondary. |
| Muralidharan2024a | Muralidharan, V., Chevallereau, C. & Wenger, P. | "Kinematics and Workspace of a Spatial 3-DoF Manipulator with Anti-parallelogram Joints" | 2024 | 🟡 | 🟢 | 🟢 | Spatial 3-DOF antiparallelogram manipulator with up to 32 IK solutions. Directly relevant to this project's manipulator. |
| Muralidharan2024b | Muralidharan, V., Wenger, P. & Chevallereau, C. | "Design Considerations and Workspace Computation of 2-X and 2-R Planar Cable-Driven Tensegrity-Inspired Manipulators" | 2024 | 🟡 | 🟢 | 🟢 | Pareto-optimal design analysis comparing X-joint and R-joint variants, including joint limits and spring feasibility. |
| Testard2024 | Testard, N. J. S., Chevallereau, C. & Wenger, P. | "Control Analysis of an Underactuated Bio-Inspired Robot" | 2024 | 🟡 | 🟢 | 🟢 | Stability analysis for underactuated tendon-driven X-joint control. |
| Muralidharan2023 | Muralidharan, V., Testard, N., Chevallereau, C., Abourachid, A. & Wenger, P. | "Variable Stiffness and Antagonist Actuation for Cable-Driven Manipulators Inspired by the Bird Neck" | 2023 | 🟡 | 🟢 | 🟢 | Variable stiffness control via antagonistic cable modulation for bird-neck-inspired arms. |
| Muralidharan2021 | Muralidharan, V. & Wenger, P. | "Optimal Design and Comparative Study of Two Antagonistically Actuated Tensegrity Joints" | 2021 | 🟡 | 🟢 | 🟢 | Systematic comparison of R-joint vs. X-joint variants for tensegrity arms. |
| Lenoir2026 | Lenoir, A., Chevallereau, C., Pitti, A. & Wenger, P. | "Kinetostatic Analysis of a Class of Planar 2-DOF Tensegrity Manipulators" | 2026 | 🟡 | 🟡 | 🟢 | Forthcoming ARK 2026 paper on kinetostatic analysis of tensegrity arms. |
| Fadeyev2019 | Fadeyev, D., Zhakatayev, A., Kuzdeuov, A. & Varol, H. A. | "Generalized Dynamics of Stacked Tensegrity Manipulators" | 2019 | 🔴 | 🟡 | 🟢 | Nonlinear dynamics with damping for stacked tensegrity manipulators; two-stage prototype. |
| Ikemoto2021 | Ikemoto, S., Tsukamoto, K. & Yoshimitsu, Y. | "Development of a Modular Tensegrity Robot Arm Capable of Continuous Bending" | 2021 | 🔴 | 🟡 | 🟢 | Modular stacked tensegrity with pneumatic 20-actuator continuum arm. |
| Ramadoss2023 | Ramadoss, V. et al. | "Design and Modeling Framework for DexTeR: Dexterous Continuum Tensegrity Manipulator" | 2023 | 🔴 | 🟡 | 🟢 | 10-vertebra tensegrity with Euler-Newton dynamics and centralized base actuation. |
| Wang2023humanoid | Wang, B., Zhang, T., Chen, J., Xu, W., Wei, H., Song, Y. & Guan, Y. | "A Modular Cable-Driven Humanoid Arm with Anti-Parallelogram Mechanisms and Bowden Cables" | 2023 | 🟡 | 🟢 | 🟢 | 7-DOF humanoid arm with antiparallelogram joints and modular cable routing via Bowden cables. |
| Wittmeier2013 | Wittmeier, S., Alessandro, C., Bascarevic, N. et al. | "Toward Anthropomimetic Robotics: Development, Simulation, and Control of a Musculoskeletal Torso" | 2013 | 🔴 | 🟡 | 🟢 | ECCEROBOT: bones (compression) + tendons (tension) as biotensegrity, demonstrating morphological computation. |
| Hao2025 | Hao, Z. et al. | "A Controllable-Stiffness Tensegrity Robot Joint for Robust Compliant Manipulation" | 2025 | 🟡 | 🟢 | 🟢 | Tensegrity 2-DOF joint (180 g), no rotary/sliding friction, data-driven stiffness control. |

---

## RL for Tensegrity Robots

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Zhang2017tensRL | Zhang, M., Geng, X., Bruce, J., Caluwaerts, K., Vespignani, M., SunSpiral, V., Abbeel, P. & Levine, S. | "Deep Reinforcement Learning for Tensegrity Robot Locomotion" | 2017 | 🟡 | 🟢 | 🟢 | Mirror Descent Guided Policy Search (MDGPS) for SUPERball 6-strut tensegrity, demonstrating sim-to-real locomotion transfer. |
| Zhang2025tensGNN | Zhang, C. et al. | "Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion" | 2025 | 🟡 | 🟢 | 🟢 | GNN-SAC on 3-bar tensegrity with zero-shot sim-to-real transfer. Morphology-aware policy representation. |
| Surovik2021 | Surovik, D., Wang, K., Vespignani, M., Bruce, J. & Bekris, K. E. | "Adaptive Tensegrity Locomotion: Controlling a Compliant Icosahedron with Symmetry-Reduced Reinforcement Learning" | 2021 | 🔴 | 🟡 | 🟢 | Symmetry-reduced policy search for icosahedron tensegrity locomotion. |
| Chen2024tensGNN | Chen, N., Wang, K., Johnson III, W. R., Kramer-Bottiglio, R., Bekris, K. & Aanjaneya, M. | "Learning Differentiable Tensegrity Dynamics using Graph Neural Networks" | 2024 | 🟡 | 🟡 | 🟢 | GNN-learned dynamics achieving 30 % accuracy improvement over differentiable physics for tensegrity. |
| Wang2023r2s2r | Wang, K., Johnson III, W. R., Lu, S., Huang, X., Booth, J., Kramer-Bottiglio, R., Aanjaneya, M. & Bekris, K. | "Real2Sim2Real Transfer for Control of Cable-driven Robots via a Differentiable Physics Engine" | 2023 | 🟡 | 🟢 | 🟢 | Real2Sim2Real pipeline for 3-bar tensegrity using a differentiable physics engine. |

---

## Safety & Compliant Mechanisms

### Books & Surveys

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Vanderborght2013 | Vanderborght, B., Albu-Schäffer, A., Bicchi, A. et al. | "Variable Impedance Actuators: A Review" | 2013 | 🔴 | 🟢 | 🟢 | Four VIA categories: active impedance, inherent compliance, inertial, and combinations. Key reference for motivating tensegrity compliance. |
| Wolf2016 | Wolf, S., Grioli, G., Eiberger, O. et al. | "Variable Stiffness Actuators: Review on Design and Components" | 2016 | 🔴 | 🟢 | 🟢 | Design-oriented VSA guide covering dual motors, springs, and complexity trade-offs. |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Pratt1995 | Pratt, G. A. & Williamson, M. M. | "Series Elastic Actuators" | 1995 | 🔴 | 🟢 | 🟢 | Series spring between motor and load for shock tolerance and force control. Foundational concept for compliant actuation. |
| Haddadin2009 | Haddadin, S., Albu-Schäffer, A. & Hirzinger, G. | "Requirements for Safe Robots: Measurements, Analysis and New Insights" | 2009 | 🔴 | 🟢 | 🟢 | DLR III safety evaluation showing mass/velocity dominance; weight reduction as primary safety measure. |
| Haddadin2017 | Haddadin, S., De Luca, A. & Albu-Schäffer, A. | "Robot Collisions: A Survey on Detection, Isolation, and Identification" | 2017 | 🔴 | 🟢 | 🟢 | Collision pipeline, generalized momentum observer, and proprioceptive sensor strategies. |
| DeLuca2006 | De Luca, A., Albu-Schäffer, A., Haddadin, S. & Hirzinger, G. | "Collision Detection and Safe Reaction with the DLR-III Lightweight Manipulator Arm" | 2006 | 🔴 | 🟡 | 🟢 | Proprioceptive collision detection based on total energy and motor inertia reduction. |
| AlbuSchaffer2007 | Albu-Schäffer, A., Ott, C. & Hirzinger, G. | "A Unified Passivity-Based Control Framework for Position, Torque and Impedance Control of Flexible Joint Robots" | 2007 | 🔴 | 🟡 | 🟢 | Passivity-based framework for elastic joint robots with motor inertia shaping. |
| Kim2017LIMS | Kim, Y.-J. | "Anthropomorphic Low-Inertia High-Stiffness Manipulator for High-Speed Safe Interaction" | 2017 | 🔴 | 🟢 | 🟢 | LIMS 7-DOF with tension amplification, wrist 2-DOF wire pairs, 94 % motor inertia reduction for safety. |
| Kim2015 | Kim, Y.-J. | "Design of Low Inertia Manipulator with High Stiffness and Strength Using Tension Amplifying Mechanisms" | 2015 | 🔴 | 🟡 | 🟢 | Early LIMS concept: tension amplification for high stiffness-to-inertia ratio. |
| Bicchi2004safety | Bicchi, A. & Tonietti, G. | "Fast and 'Soft-Arm' Tactics: Dealing with the Safety-Performance Trade-Off in Robot Arms Design and Control" | 2004 | 🔴 | 🟢 | 🟢 | Mechanical compliance essential for HRI; Head Injury Criterion analysis demonstrates passive safety outperforms active safety. |

### Standards

| Ref | Source | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|------|-------|--------|-------|---------|
| ISO10218 | ISO | "Robots and Robotic Devices — Safety Requirements for Industrial Robots — Part 1: Robots" (ISO 10218-1:2011, updated 2025) | 2011 | 🔴 | 🟢 | 🟢 | ISO Standard for industrial robot safety requirements. |
| ISO15066 | ISO | "Robots and Robotic Devices — Collaborative Robots" (ISO/TS 15066:2016) | 2016 | 🔴 | 🟢 | 🟢 | ISO Standard for collaborative robot power and force limiting (PFL) with biomechanical injury thresholds. |

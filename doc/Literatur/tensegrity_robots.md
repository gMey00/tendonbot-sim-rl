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
| [Fraternali2025](sources/tensegrity_robots/Books/Fraternali2025.pdf) | Fraternali, F. (Ed.) | *Tensegrity Systems — Recent Advances in Mechanics, Design and Control* | 2025 | 🔴 | 🟡 | 🟢 | Recent edited volume compiling advances in tensegrity mechanics, design, and control. Up-to-date companion to Skelton 2009. |

### Survey Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Sultan2009 | Sultan, C. | "Tensegrity: 60 Years of Art, Science, and Engineering" | 2009 | 🔴 | 🟢 | 🟢 | Comprehensive 77-page survey covering history, form-finding, stability, dynamics, control, and applications of tensegrity structures. |

### Patents & Historical

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Fuller1962](sources/tensegrity_robots/Papers/Fuller1962.pdf) | Fuller, R. B. | "Tensile-Integrity Structures" (US Patent 3,063,521) | 1962 | 🔴 | 🟡 | 🟢 | Origin of the "tensegrity" terminology, combining "tensile" and "integrity". |

---

## Tensegrity Robotics Reviews

### Review Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Shah2022](sources/tensegrity_robots/Reviews/Shah2022.pdf) | Shah, D. S., Booth, J. W., Baines, R. L., Wang, K., Vespignani, M., Bekris, K. & Kramer-Bottiglio, R. | "Tensegrity Robotics" | 2022 | 🟡 | 🟢 | 🟢 | Comprehensive review covering simulation tools (NTRT, MuJoCo, Bullet), locomotion, and manipulation applications of tensegrity robots. |
| [Liu2022review](sources/tensegrity_robots/Reviews/Liu2022review.pdf) | Liu, Y., Bi, Q., Yue, X., Wu, J., Yang, B. & Li, Y. | "A Review on Tensegrity Structures-Based Robots" | 2022 | 🟡 | 🟢 | 🟢 | Design, fabrication, modeling, and control of tensegrity robots. Positions tensegrity as bridging rigid and soft robotics. |
| [Micheletti2022](sources/tensegrity_robots/Reviews/Micheletti2022.pdf) | Micheletti, A. et al. | "Recent Advances in Tensegrity Structures and Mechanisms" | 2022 | 🔴 | 🟡 | 🟢 | Recent-advances review of tensegrity structures and mechanisms. Complementary to Shah 2022 with stronger structural-mechanics focus. |

---

## Tensegrity Manipulators

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Lessard2016](sources/tensegrity_robots/Papers/Lessard2016.pdf) | Lessard, S., Castro, D., Asper, W., Chopra, S. D., Baltaxe-Admony, L. B., Teodorescu, M., SunSpiral, V. & Agogino, A. | "A Bio-Inspired Tensegrity Manipulator with Multi-DOF, Structurally Compliant Joints" | 2016 | 🔴 | 🟢 | 🟢 | 4-DOF tensegrity arm using nested tetrahedrons with compliant joints as an alternative to variable stiffness actuators. |
| [Fasquelle2020](sources/tensegrity_robots/Papers/Fasquelle2020.pdf) | Fasquelle, B., Furet, M., Khanna, P., Chablat, D., Chevallereau, C. & Wenger, P. | "A Bio-Inspired 3-DOF Light-Weight Manipulator with Tensegrity X-Joints" | 2020 | 🟡 | 🟢 | 🟢 | 3-DOF X-joint arm hardware from the AVINECK bird-neck project. Directly relevant as a comparable tensegrity manipulator design. |
| Fasquelle2019 | Fasquelle, B., Furet, M., Chevallereau, C. & Wenger, P. | "Dynamic Modeling and Control of a Tensegrity Manipulator Mimicking a Bird Neck" | 2019 | 🟡 | 🟢 | 🟢 | Lagrangian dynamics of 3-X arm with computed-torque control. |
| [FuretWenger2019](sources/tensegrity_robots/Papers/Furet2019.pdf) | Furet, M. & Wenger, P. | "Kinetostatic Analysis and Actuation Strategy of a Planar Tensegrity 2-X Manipulator" | 2019 | 🟡 | 🟢 | 🟢 | Kinetostatic analysis and antagonistic actuation strategy for a planar 2-X tensegrity arm. Foundational reference for the X-joint actuation pattern reused in this project. [DOI](https://doi.org/10.1115/1.4044209) |
| [MuralidharanWenger2021](sources/tensegrity_robots/Papers/Muralidharan2021.pdf) | Muralidharan, V. & Wenger, P. | "Optimal Design and Comparative Study of Two Antagonistically Actuated Tensegrity Joints" | 2021 | 🟡 | 🟢 | 🟢 | Optimal-design study comparing two antagonistically actuated tensegrity joint topologies (X-joint vs. R-joint). Provides design-space insight directly applicable to the manipulator. [DOI](https://doi.org/10.1016/j.mechmachtheory.2021.104249) |
| Fasquelle2022 | Fasquelle, B., Khanna, P., Chevallereau, C., Chablat, D., Creusot, D., Jolivet, S., Lemoine, P. & Wenger, P. | "Identification and Control of a 3-X Cable-Driven Manipulator Inspired From the Bird's Neck" | 2022 | 🟡 | 🟢 | 🟢 | Physical prototype validation. Key finding: friction dominates performance; cable elasticity is secondary. |
| [Muralidharan2023](sources/tensegrity_robots/Papers/Muralidharan2023.pdf) | Muralidharan, V., Testard, N., Chevallereau, C., Abourachid, A. & Wenger, P. | "Variable Stiffness and Antagonist Actuation for Cable-Driven Manipulators Inspired by the Bird Neck" | 2023 | 🟡 | 🟢 | 🟢 | Variable stiffness control via antagonistic cable modulation for bird-neck-inspired arms. |
| Ramadoss2023 | Ramadoss, V. et al. | "Design and Modeling Framework for DexTeR: Dexterous Continuum Tensegrity Manipulator" | 2023 | 🔴 | 🟡 | 🟢 | 10-vertebra tensegrity with Euler-Newton dynamics and centralized base actuation. |
| [Wang2023humanoid](sources/tensegrity_robots/Papers/Wang2023humanoid.pdf) | Wang, B., Zhang, T., Chen, J., Xu, W., Wei, H., Song, Y. & Guan, Y. | "A Modular Cable-Driven Humanoid Arm with Anti-Parallelogram Mechanisms and Bowden Cables" | 2023 | 🟡 | 🟢 | 🟢 | 7-DOF humanoid arm with antiparallelogram joints and modular cable routing via Bowden cables. |
| [Hao2025](sources/tensegrity_robots/Papers/Hao2025.pdf) | Hao, Z. et al. | "A Controllable-Stiffness Tensegrity Robot Joint for Robust Compliant Manipulation" | 2025 | 🟡 | 🟢 | 🟢 | Tensegrity 2-DOF joint (180 g), no rotary/sliding friction, data-driven stiffness control. |

---

## RL for Tensegrity Robots

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Zhang2017tensRL](sources/tensegrity_robots/Papers/Zhang2017tensRL.pdf) | Zhang, M., Geng, X., Bruce, J., Caluwaerts, K., Vespignani, M., SunSpiral, V., Abbeel, P. & Levine, S. | "Deep Reinforcement Learning for Tensegrity Robot Locomotion" | 2017 | 🟡 | 🟢 | 🟢 | Mirror Descent Guided Policy Search (MDGPS) for SUPERball 6-strut tensegrity, demonstrating sim-to-real locomotion transfer. |
| [Chen2024tensGNN](sources/tensegrity_robots/Papers/Chen2024tensGNN.pdf) | Chen, N., Wang, K., Johnson III, W. R., Kramer-Bottiglio, R., Bekris, K. & Aanjaneya, M. | "Learning Differentiable Tensegrity Dynamics using Graph Neural Networks" | 2024 | 🟡 | 🟡 | 🟢 | GNN-learned dynamics achieving 30 % accuracy improvement over differentiable physics for tensegrity. |

---

## Safety & Compliant Mechanisms

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Pratt1995](sources/tensegrity_robots/Papers/Pratt1995.pdf) | Pratt, G. A. & Williamson, M. M. | "Series Elastic Actuators" | 1995 | 🔴 | 🟢 | 🟢 | Series spring between motor and load for shock tolerance and force control. Foundational concept for compliant actuation. |
| [DeLuca2006](sources/tensegrity_robots/Papers/DeLuca2006.pdf) | De Luca, A., Albu-Schäffer, A., Haddadin, S. & Hirzinger, G. | "Collision Detection and Safe Reaction with the DLR-III Lightweight Manipulator Arm" | 2006 | 🔴 | 🟡 | 🟢 | Proprioceptive collision detection based on total energy and motor inertia reduction. |
| [AlbuSchaffer2007](sources/tensegrity_robots/Papers/AlbuSchaffer2007.pdf) | Albu-Schäffer, A., Ott, C. & Hirzinger, G. | "A Unified Passivity-Based Control Framework for Position, Torque and Impedance Control of Flexible Joint Robots" | 2007 | 🔴 | 🟡 | 🟢 | Passivity-based framework for elastic joint robots with motor inertia shaping. |

### Standards

| Ref | Source | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|------|-------|--------|-------|---------|
| ISO10218 | ISO | "Robots and Robotic Devices — Safety Requirements for Industrial Robots — Part 1: Robots" (ISO 10218-1:2011, updated 2025) | 2011 | 🔴 | 🟢 | 🟢 | ISO Standard for industrial robot safety requirements. |
| ISO15066 | ISO | "Robots and Robotic Devices — Collaborative Robots" (ISO/TS 15066:2016) | 2016 | 🔴 | 🟢 | 🟢 | ISO Standard for collaborative robot power and force limiting (PFL) with biomechanical injury thresholds. |

# Linkage Mechanisms & Closed-Loop Kinematics

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Textbooks & Fundamentals

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| McCarthy2011 | McCarthy, J. M. & Soh, G. S. | *Geometric Design of Linkages* (2nd ed.) | 2011 | 🔴 | 🟢 | 🟢 | Comprehensive textbook on linkage design theory: Grashof criterion, coupler curves, cognate linkages, and spatial mechanisms. Standard reference for four-bar linkage analysis. |
| Uicker2017 | Uicker, J. J., Pennock, G. R. & Shigley, J. E. | *Theory of Machines and Mechanisms* (5th ed.) | 2017 | 🔴 | 🟢 | 🟢 | Standard undergraduate text on mechanism kinematics covering graphical synthesis, analytical techniques, and dynamic analysis. |
| Hartenberg1955 | Hartenberg, R. S. & Denavit, J. | "A Kinematic Notation for Lower Pair Mechanisms Based on Matrices" | 1955 | 🔴 | 🟢 | 🟢 | Introduces the Denavit-Hartenberg convention for describing kinematic chains via homogeneous transformation matrices. Foundational for all serial-chain manipulator descriptions. |

### Seminal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Freudenstein1954 | Freudenstein, F. | "An Analytical Approach to the Design of Four-Link Mechanisms" | 1954 | 🔴 | 🟡 | 🟢 | First analytical (algebraic) method for four-bar synthesis, replacing graphical methods. Historical milestone in mechanism design. |

---

## Antiparallelogram Kinematics

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Dijksman1977 | Dijksman, E. A. | "A Study of Some Properties of the Antiparallelogram Linkage" | 1977 | 🔴 | 🟢 | 🟢 | Definitive kinematic analysis of the antiparallelogram linkage: instantaneous centre of rotation (ICR), bifurcation modes, and coupler curve properties. Core reference for the project's X-joint analysis. |
| Stachel2000 | Stachel, H. | "Flexible Cross-Polytopes in the Euclidean 3-Space" | 2000 | 🔴 | 🟡 | 🟢 | Proves antiparallelogram serves as flexibility element in cross-polytopes; classifies finite-flexibility modes. |
| Bryant2013 | Bryant, R. & Theran, L. | "Rigidity of Antiparallelogram via Algebraic and Graphical Methods" | 2013 | 🔴 | 🟡 | 🟢 | Modern algebraic rigidity analysis of antiparallelogram frameworks. Complements the classical geometric treatment. |
| Muirhead1923 | Muirhead, R. F. | "The Antiparallelogram — A Historical Note on Its Properties" | 1923 | 🔴 | 🔴 | 🟡 | Early historical treatment of antiparallelogram geometry. Background for the evolution of linkage analysis. |

---

## Closed-Loop Simulation

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Tsounis2026 | Tsounis, V., Maloisel, G., Schumacher, C., Grandia, R., Serifi, A., Müller, D., Amevor, C., Widmer, T. & Bächer, M. | "Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies" | 2026 | 🟡 | 🟢 | 🟢 | GPU solver for kinematic loops using ADMM-based maximal-coordinate formulation. Demonstrated on DR Legs biped with 6 nested loops. Directly relevant as a simulation backend for antiparallelogram joints. |
| Tsounis2025 | Tsounis, V., Grandia, R. & Bächer, M. | "On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops" | 2025 | 🟡 | 🟢 | 🟢 | Algorithmic foundations for the Kamino solver: constrained dynamics formulation for closed kinematic chains. |
| Gondokaryono2019 | Gondokaryono, R., Agrawal, A., Munawar, A., Nycz, C. J. & Fischer, G. S. | "An Approach to Modeling Closed-Loop Kinematic Chain Mechanisms, Applied to Simulations of the da Vinci Surgical System" | 2019 | 🟡 | 🟢 | 🟢 | Double four-bar model for da Vinci PSM/ECM using Gazebo/Bullet with mimic joints. Key reference for workaround strategies. |
| Kulkarni2025 | Kulkarni, M., Alexis, K. et al. | "Aerial Gym Simulator" | 2025 | 🔴 | 🟡 | 🟢 | GPU-accelerated simulator handling under/over-actuated complex morphologies. Context for closed-loop simulation in Isaac Sim alternatives. |

---

## Robotics Applications of Four-Bar Mechanisms

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Tomishiro2019 | Tomishiro, K. et al. | "Design of Robot Leg with Variable Reduction Ratio Crossed Four-bar Linkage Mechanism" | 2019 | 🔴 | 🟡 | 🟢 | Variable-ratio crossed four-bar knee joint design with increased reduction ratio for legged robots. |

## Hamon and Aoustin (HamonAoustin2010)
**Citation:** Hamon, A. and Aoustin, Y. (2010). "Cross Four-Bar Linkage for the Knees of a Planar Bipedal Robot." *2010 IEEE-RAS International Conference on Humanoid Robots (Humanoids)*, Nashville, TN, pp. 379–384. IEEE. *(Note: Extended in a 2014 Multibody System Dynamics journal paper with co-author S. Caro)*.
**Summary Highlights:**
- Proposes a crossed (antiparallelogram) four-bar linkage for the knee joint of a planar bipedal robot to replicate the translating Instantaneous Center of Rotation (ICR) of the human knee (guided by cruciate ligaments).
- Compares optimized walking gaits between robots using crossed-linkage knees and standard revolute knees.
- Demonstrates that the crossed-linkage knee performs worse at low velocities but better at higher walking speeds (beneficial for dynamic locomotion).
- Serves as the most direct early precedent for using an antiparallelogram as a robotic joint actuated by tendons, providing explicit biomechanical motivation.

## Yoon et al. (Yoon2021DLRWrist)
**Citation:** Yoon, D., Kang, L., Manzoor, S., and Choi, Y. (2021). "The Improved DLR Wrist: Design and Analysis of 2-Degrees-of-Freedom Rotational Mechanism Using Spatial Antiparallelogram Linkages." *ASME Journal of Mechanical Design*, 143(5), 053303. DOI: 10.1115/1.4048719.
**Summary Highlights:**
- Analyzes and improves the 2-DOF wrist mechanism of the DLR robot arm, which utilizes spatial antiparallelogram linkages.
- Investigates the elliptical rolling motion of the overconstrained antiparallelogram and provides axode analysis of the instantaneous screw axis.
- Shows that an improved mechanism achieves a wider range of motion, reduced parasitic motion, and approximately decoupled output by substituting a small-displacement joint with a flexible hinge.
- Highlights that parasitic motion and kinematic coupling are the primary challenges when extending antiparallelogram joints from planar to spatial (3D) implementations.

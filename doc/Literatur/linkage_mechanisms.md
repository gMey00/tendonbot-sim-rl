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

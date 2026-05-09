# Linkage Mechanisms & Closed-Loop Kinematics

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Textbooks & Fundamentals

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [McCarthy2011](sources/linkage_mechanisms/Books/McCarthy2011.pdf) | McCarthy, J. M. & Soh, G. S. | *Geometric Design of Linkages* (2nd ed.) | 2011 | 🔴 | 🟢 | 🟢 | Comprehensive textbook on linkage design theory: Grashof criterion, coupler curves, cognate linkages, and spatial mechanisms. Standard reference for four-bar linkage analysis. |
| [Uicker2017](sources/linkage_mechanisms/Books/Uicker2017.pdf) | Uicker, J. J., Pennock, G. R. & Shigley, J. E. | *Theory of Machines and Mechanisms* (5th ed.) | 2017 | 🔴 | 🟢 | 🟢 | Standard undergraduate text on mechanism kinematics covering graphical synthesis, analytical techniques, and dynamic analysis. |
| [Hartenberg1955](sources/linkage_mechanisms/Books/Hartenberg1955.pdf) | Hartenberg, R. S. & Denavit, J. | "A Kinematic Notation for Lower Pair Mechanisms Based on Matrices" | 1955 | 🔴 | 🟢 | 🟢 | Introduces the Denavit-Hartenberg convention for describing kinematic chains via homogeneous transformation matrices. Foundational for all serial-chain manipulator descriptions. |

## Antiparallelogram Kinematics

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Dijksman1977](sources/linkage_mechanisms/Papers/Dijksman1977.pdf) | Dijksman, E. A. | "A Study of Some Properties of the Antiparallelogram Linkage" | 1977 | 🔴 | 🟢 | 🟢 | Definitive kinematic analysis of the antiparallelogram linkage: instantaneous centre of rotation (ICR), bifurcation modes, and coupler curve properties. Core reference for the project's X-joint analysis. |
| [Stachel2000](sources/linkage_mechanisms/Papers/Stachel2000.pdf) | Stachel, H. | "Flexible Cross-Polytopes in the Euclidean 3-Space" | 2000 | 🔴 | 🟡 | 🟢 | Proves antiparallelogram serves as flexibility element in cross-polytopes; classifies finite-flexibility modes. |
| [Bryant2013](sources/linkage_mechanisms/Papers/Bryant2013.pdf) | Bryant, R. & Theran, L. | "Rigidity of Antiparallelogram via Algebraic and Graphical Methods" | 2013 | 🔴 | 🟡 | 🟢 | Modern algebraic rigidity analysis of antiparallelogram frameworks. Complements the classical geometric treatment. |
| Muirhead1923 | Muirhead, R. F. | "The Antiparallelogram — A Historical Note on Its Properties" | 1923 | 🔴 | 🔴 | 🟡 | Early historical treatment of antiparallelogram geometry. Background for the evolution of linkage analysis. |
| [Gur2019](sources/linkage_mechanisms/Papers/Gur2019.pdf) | Gur, M. & Mimna, R. | "On the Geometry and Kinematics of the Antiparallelogram Linkage" | 2019 | 🔴 | 🟡 | 🟢 | Detailed geometric and kinematic study of the antiparallelogram linkage. Complements Dijksman 1977 with modern algebraic treatment. |

---

## Closed-Loop Simulation

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Tsounis2025](sources/linkage_mechanisms/Papers/Tsounis2025.pdf) | Tsounis, V., Grandia, R. & Bächer, M. | "On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops" | 2025 | 🟡 | 🟢 | 🟢 | Algorithmic foundations for the Kamino solver: constrained dynamics formulation for closed kinematic chains. |
| Kulkarni2025 | Kulkarni, M., Alexis, K. et al. | "Aerial Gym Simulator" | 2025 | 🔴 | 🟡 | 🟢 | GPU-accelerated simulator handling under/over-actuated complex morphologies. Context for closed-loop simulation in Isaac Sim alternatives. |

---

## Robotics Applications of Four-Bar Mechanisms

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Tomishiro2019](sources/linkage_mechanisms/Papers/Tomishiro2019.pdf) | Tomishiro, K. et al. | "Design of Robot Leg with Variable Reduction Ratio Crossed Four-bar Linkage Mechanism" | 2019 | 🔴 | 🟡 | 🟢 | Variable-ratio crossed four-bar knee joint design with increased reduction ratio for legged robots. |


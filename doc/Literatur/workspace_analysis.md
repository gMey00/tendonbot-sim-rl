# Workspace Analysis

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Monte Carlo Methods

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Rastegar1990](sources/workspace_analysis/Journal_Articles/Rastegar1990.pdf) | Rastegar, J. & Fardanesh, B. | "Manipulation Workspace Analysis Using the Monte Carlo Method" | 1990 | 🟢 | 🟢 | 🟢 | Foundational work on Monte Carlo workspace analysis. Method directly used in the project thesis for reachable volume estimation. [DOI](https://doi.org/10.1016/0094-114X(90)90124-3) |
| [Peidro2017](sources/workspace_analysis/Journal_Articles/Peidro2017.pdf) | Peidró, A. et al. | "An Improved Monte Carlo Method Based on Gaussian Growth to Calculate the Workspace of Robots" | 2017 | 🟡 | 🟡 | 🟢 | Improved MC method with Gaussian sampling for workspace computation. Context for alternative sampling strategies. [DOI](https://doi.org/10.1016/j.engappai.2017.06.009) |

---

## Manipulability Measures

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Yoshikawa1985](sources/workspace_analysis/Journal_Articles/Yoshikawa1985.pdf) | Yoshikawa, T. | "Manipulability of Robotic Mechanisms" | 1985 | 🟢 | 🟢 | 🟢 | Introduces the Yoshikawa manipulability measure used in workspace analysis. Core metric for characterizing the tensegrity manipulator's dexterity. [DOI](https://doi.org/10.1177/027836498500400201) |
| [Vahrenkamp2015](sources/workspace_analysis/Journal_Articles/Vahrenkamp2015.pdf) | Vahrenkamp, N. & Asfour, T. | "Representing the Robot's Workspace Through Constrained Manipulability Analysis" | 2015 | 🔴 | 🟡 | 🟢 | Constrained manipulability analysis. Context for workspace quality metrics beyond volume. [DOI](https://doi.org/10.1007/s10514-014-9388-z) |

---

## Workspace Density

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Dong2013](sources/workspace_analysis/Journal_Articles/Dong2013.pdf) | Dong, H. et al. | "Workspace Density and Inverse Kinematics for Planar Serial Revolute Manipulators" | 2013 | 🟡 | 🟡 | 🟢 | Workspace density analysis for serial manipulators. Background for interpreting Monte Carlo workspace distributions. [DOI](https://doi.org/10.1016/j.mechmachtheory.2013.08.008) |
| [Ding2024](sources/workspace_analysis/Journal_Articles/Ding2024.pdf) | Ding, J. et al. | "Workspace Analysis and Density Distribution for a Cable-Driven Manipulator" | 2024 | 🟡 | 🟡 | 🟢 | Recent workspace-density study on a cable-driven manipulator. Directly comparable to the Monte Carlo workspace analysis used in this project. |

---

## CDPM Workspace Methods

### Journal & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Verhoeven2004WS | Verhoeven, R. | *Analysis of the Workspace of Tendon-based Stewart Platforms* | 2004 | 🟡 | 🟢 | 🟢 | Controllable workspace definition for CDPMs. IRPM/CRPM/RRPM classification. Cross-ref: [tendon_robots.md](tendon_robots.md). |
| [Gouttefarde2006WS](sources/workspace_analysis/Journal_%26_Conference_Papers/Gouttefarde2006WS.pdf) | Gouttefarde, M. & Gosselin, C. M. | "Analysis of the Wrench-Closure Workspace of Planar Parallel Cable-Driven Mechanisms" | 2006 | 🟡 | 🟢 | 🟢 | Wrench-closure workspace (WCW) definition: purely geometric, ensures arbitrary wrench feasibility. Cross-ref: [tendon_robots.md](tendon_robots.md). |
| [Bosscher2006WS](sources/workspace_analysis/Journal_%26_Conference_Papers/Bosscher2006WS.pdf) | Bosscher, P., Riechel, A. T. & Ebert-Uphoff, I. | "Wrench-Feasible Workspace Generation for Cable-Driven Robots" | 2006 | 🟡 | 🟢 | 🟢 | Wrench-feasible workspace (WFW) via available wrench set (zonotope) with bounded tension constraints. Cross-ref: [tendon_robots.md](tendon_robots.md). |
| [Stump2006WS](sources/workspace_analysis/Journal_%26_Conference_Papers/Stump2006WS.pdf) | Stump, E. & Kumar, V. | "Workspaces of Cable-Actuated Parallel Manipulators" | 2006 | 🔴 | 🟡 | 🟢 | Closed-form workspace boundaries via convex analysis for cable-actuated parallel manipulators. |
| [Gouttefarde2011WS](sources/workspace_analysis/Journal_%26_Conference_Papers/Gouttefarde2011WS.pdf) | Gouttefarde, M., Daney, D. & Merlet, J.-P. | "Interval-Analysis-Based Determination of the Wrench-Feasible Workspace of Parallel Cable-Driven Robots" | 2011 | 🔴 | 🟡 | 🟢 | Interval arithmetic for WFW computation with geometric uncertainties. Robust workspace guarantees. |

---

## Toolbox & Software

### Documentation

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| mathworks_generaterobotworkspace | MathWorks | generateRobotWorkspace | 🔴 | 🟡 | 🟡 | MATLAB function for workspace generation. Context for comparison with the custom Python implementation. [Link](https://www.mathworks.com/help/robotics/ref/generaterobotworkspace.html) |

# Robot Control & Kinematics

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Textbooks & Fundamentals

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Craig2005 | Craig, J. J. | *Introduction to Robotics: Mechanics and Control* (3rd ed.) | 2005 | 🟡 | 🟢 | 🟢 | Standard reference on DH conventions, forward/inverse kinematics, Jacobians, trajectory planning, and PID control. Directly relevant to the tensegrity manipulator's workspace analysis and controller design. |
| Siciliano2009 | Siciliano, B. et al. | *Robotics: Modelling, Planning and Control* | 2009 | 🟡 | 🟢 | 🟢 | In-depth treatment of robot dynamics, impedance/force control, and redundancy resolution. Theoretical backdrop for tendon-driven control with over-actuated mapping. [Springer](https://link.springer.com/book/10.1007/978-1-84628-642-1) |
| Spong2020 | Spong, M. W. et al. | *Robot Modeling and Control* (2nd ed.) | 2020 | 🟡 | 🟢 | 🟢 | Lagrangian dynamics, PID and computed-torque control, adaptive control. The gravity-compensation approach (τ_a = m·g·l_c·sin θ) follows directly from this text. |
| Murray1994 | Murray, R. M. et al. | *A Mathematical Introduction to Robotic Manipulation* | 1994 | 🔴 | 🟢 | 🟢 | Rigorous differential-geometric treatment of manipulation using screw theory. The Jacobian-transpose mapping in the tendon simulation follows the wrench-twist duality framework. |

---

## Tension Distribution & Cable Control

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Bicchi2004 | Bicchi, A. & Tonietti, G. | "Fast and 'Soft-Arm' Tactics: Safety-Performance Tradeoff in Robot Arms" | 2004 | 🔴 | 🟢 | 🟢 | Safety-performance tradeoff in compliant robot arms. Motivation for the tensegrity approach's inherent compliance. [DOI](https://doi.org/10.1109/MRA.2004.1310939) |
| Salisbury1982 | Salisbury, J. K. & Craig, J. J. | "Articulated Hands: Force Control and Kinematic Issues" | 1982 | 🔴 | 🟡 | 🟢 | Foundational work on force control in articulated hands. Historical context for tendon-driven force control. [DOI](https://doi.org/10.1177/027836498200100102) |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Reis2015 | Reis, M. et al. | "Modeling and Control of a Multifingered Robot Hand for Object Grasping and Manipulation Tasks" | 2015 | 🔴 | 🟡 | 🟢 | Multifingered hand control. Background for tendon-driven grasping mechanics. [DOI](https://doi.org/10.1109/CDC.2015.7402102) |

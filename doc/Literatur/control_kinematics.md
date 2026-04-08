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
| Tsai1999 | Tsai, L.-W. | *Robot Analysis: The Mechanics of Serial and Parallel Manipulators* | 1999 | 🔴 | 🟢 | 🟢 | Parallel manipulator kinematics/dynamics, Jacobian formulations, and constraint analysis. Reference for serial-parallel hybrid structures. |

---

## Tension Distribution & Cable Control

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Bicchi2004 | Bicchi, A. & Tonietti, G. | "Fast and 'Soft-Arm' Tactics: Safety-Performance Tradeoff in Robot Arms" | 2004 | 🔴 | 🟢 | 🟢 | Safety-performance tradeoff in compliant robot arms. Motivation for the tensegrity approach's inherent compliance. [DOI](https://doi.org/10.1109/MRA.2004.1310939) |
| Salisbury1982 | Salisbury, J. K. & Craig, J. J. | "Articulated Hands: Force Control and Kinematic Issues" | 1982 | 🔴 | 🟡 | 🟢 | Foundational work on force control in articulated hands. Historical context for tendon-driven force control. [DOI](https://doi.org/10.1177/027836498200100102) |
| Lee1988 | Lee, J. J. & Tsai, L.-W. | "The Structural Synthesis of Tendon-Driven Manipulators Having a Pseudotriangular Structure Matrix" | 1988 | 🟡 | 🟢 | 🟢 | Generalized-coordinate Jacobian for tendon-driven systems. Establishes the pseudotriangular structure matrix concept. |
| Kobayashi1998 | Kobayashi, H. et al. | "Tendon-driven Robot Arm with Jacobian-Transpose Control" | 1998 | 🟡 | 🟢 | 🟢 | Jacobian-transpose control for tendon-driven arms. Directly relevant to the project's tendon force mapping approach. |
| Jacobsen1992 | Jacobsen, S. C. et al. | "Design of the UTAH/MIT Dextrous Hand: Tendon Routing and Analysis" | 1992 | 🔴 | 🟡 | 🟢 | Tendon routing topology and force analysis for the Utah/MIT hand. Historical reference for tendon routing design. |
| Abdallah1991 | Abdallah, M. E. et al. | "Variable Structure Control of Cable-Driven Robot Systems" | 1991 | 🔴 | 🟡 | 🟢 | Variable structure (sliding mode) control for cable-driven robots. Alternative control approach for tendon systems. |
| Chang2005 | Chang, W. K. et al. | "Multi-DOF Cable-Driven System: Jacobian and Force Analysis" | 2005 | 🔴 | 🟡 | 🟢 | Jacobian and force analysis for multi-DOF cable-driven systems. Background for redundant cable force distribution. |
| Ma2018 | Ma, R. et al. | "Cable-Actuated Parallel Mechanism: Jacobian and Control" | 2018 | 🔴 | 🟡 | 🟢 | Jacobian formulation and control for cable-actuated parallel mechanisms. Context for CDPM control strategies. |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Reis2015 | Reis, M. et al. | "Modeling and Control of a Multifingered Robot Hand for Object Grasping and Manipulation Tasks" | 2015 | 🔴 | 🟡 | 🟢 | Multifingered hand control. Background for tendon-driven grasping mechanics. [DOI](https://doi.org/10.1109/CDC.2015.7402102) |

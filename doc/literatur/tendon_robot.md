# Annotated Bibliography: Key Works about Tendon-Driven Robots

---

### Nemoto, T., Niiyama, R., & Iwata, H. (2019). "Design and Analysis of a Compact Tendon-Driven Rotary Joint with Application to a Robotic Arm"
**Type:** Conference paper
Proposes a compact tendon-driven rotary joint and analyses its force transmission characteristics.  Cited as [84] in Klein (2023), this work provides the foundational control principle for tendon-driven orientation regulation used in the tensegrity manipulator's wrist mechanism.

---

### Borgstrom, P. H., Jordan, B. L., Sukhatme, G. S., Batalin, M. A., & Kaiser, W. J. (2009). "Rapid Computation of Optimally Safe Tension Distributions for Parallel Cable-Driven Robots." *IEEE Transactions on Robotics*, 25(6).
**Type:** Journal article
Develops efficient algorithms for computing cable tension distributions that satisfy positive-tension constraints while minimising total cable force.  The general framework is applicable to any over-actuated cable-driven system, including the 5-tendon / 3-DOF mapping in our manipulator.

---

### Oh, S.-R. & Agrawal, S. K. (2005). "Cable Suspended Planar Robots with Redundant Cables: Controllers with Positive Tensions." *IEEE Transactions on Robotics*, 21(3).
**Type:** Journal article
Provides the theoretical basis for controlling cable-driven robots with redundant cables under the constraint that all tensions must remain positive.  The pseudo-inverse tension distribution approach used in our step-response test follows this line of work.

---

### Pott, A. (2018). *Cable-Driven Parallel Robots: Theory and Application*
**Type:** Book (Springer)
A comprehensive monograph on cable-driven parallel robots covering workspace analysis, tension distribution, stiffness modelling, and trajectory planning.  Provides the formal framework for understanding the kinetostatic duality between cable tensions and end-effector wrenches.

---

### Bruckmann, T. & Pott, A. (Eds.) (2013). *Cable-Driven Parallel Robots.* Mechanisms and Machine Science, Vol. 12 (Springer).
**Type:** Edited volume
Collects state-of-the-art contributions on cable-driven parallel robots from leading research groups.  Covers design, modelling, control, and applications, providing broad context for the tendon-driven approach adopted in this project.

---

### Mukherjee, S., Kamal, A., & Hirai, S. — Gravity compensation: τ_a = m·g·l_c·sin(θ)
**Type:** Research contribution
Cited as [85] in Klein (2023, Eq. 3.2).  Provides the gravity compensation formula used in the PID controller for the step-response validation, where the gravitational torque on each joint is computed from the link mass, gravity, and the distance from the joint axis to the link's centre of mass.
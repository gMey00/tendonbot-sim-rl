# Annotated Bibliography: Key Works in Control of Robotics

---

### Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control* (3rd ed.)
**Type:** Textbook (Pearson Prentice Hall)
The standard reference on robot kinematics, dynamics, and control.  Covers Denavit–Hartenberg conventions, forward/inverse kinematics, Jacobians, trajectory planning, and PID joint-level control — all directly relevant to the tensegrity manipulator's workspace analysis and controller design.

---

### Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009). *Robotics: Modelling, Planning and Control*
**Type:** Textbook (Springer)
An in-depth treatment of robot dynamics, impedance and force control, and redundancy resolution.  The chapters on operational-space control and constrained systems provide the theoretical backdrop for tendon-driven manipulator control where the tendon Jacobian creates an over-actuated, under-determined mapping.

---

### Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2020). *Robot Modeling and Control* (2nd ed.)
**Type:** Textbook (Wiley)
Covers Lagrangian dynamics, PID and computed-torque control, and adaptive control.  The gravity-compensation approach used in our step-response test (τ_a = m·g·l_c·sin θ) follows directly from the potential-energy-based formulation described here.

---

### Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*
**Type:** Textbook (CRC Press)
Provides a rigorous differential-geometric treatment of robot manipulation, including screw theory, the manipulator Jacobian, and force/velocity duality.  The Jacobian-transpose mapping central to our tendon simulation follows the wrench-twist duality framework presented in this text.

---

### Klein, M. (2023). *Arbeitsraumanalyse, Simulation und Bewegungsplanung eines seilgetriebenen robotischen Manipulators*
**Type:** Master's thesis (FAU Erlangen-Nürnberg)
The primary reference for this project.  Describes the tensegrity manipulator's kinematics, the cable-tension distribution via the structural matrix pseudo-inverse, PID control with gravity compensation, and step-response validation in Gazebo.  Our Isaac Sim implementation replicates the control gains, test methodology, and NRMSE metrics from this thesis.
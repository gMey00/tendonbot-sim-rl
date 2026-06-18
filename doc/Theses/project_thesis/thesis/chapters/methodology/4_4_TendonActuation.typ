// §4.4 Tendon Actuation in Simulation
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table
#import "../../../../shared/formatting/diagrams.typ": tendon-actuation-dataflow

== Tendon Actuation in Simulation <sec:tendon_actuation>

A central contribution of this work is the implementation of tendon-driven actuation within the Isaac Lab framework. The physical robot's five cables must be mapped to joint torques in a way that is compatible with PhysX's reduced-coordinate articulation solver, differentiable for gradient-based #ac("RL") training, and GPU-parallelizable across thousands of environment instances. This section derives the actuation model and describes three implementation variants of increasing physical fidelity.

=== Jacobian-Transpose Torque Mapping <subsec:jt_mapping>

#reg-sym("sym:tau", "sym:T", "sym:Jt", "sym:J_mat", "sym:Tmax", "sym:re", "sym:rw", "sym:rs")

The fundamental relationship between cable tensions and joint torques in a tendon-driven mechanism is given by the Jacobian-transpose mapping~@Verhoeven2004TendonPlatforms:
$ bold(tau) = bold(J)^top (bold(q)) bold(T) $ <eq:jt_general>
where $bold(tau) in bb(R)^n$ is the vector of joint torques, $bold(T) in bb(R)^m$ is the vector of cable tensions ($m = 5$, $n = 3$), and $bold(J)(bold(q)) in bb(R)^(m times n)$ is the tendon Jacobian that maps infinitesimal joint displacements to cable length changes. In general, $bold(J)$ is configuration-dependent since the geometric relationship between cables and joints changes with joint angles.

For the constant-$J^top$ actuation mode, $bold(J)^top$ is evaluated once at the zero configuration ($bold(q) = bold(0)$) and held fixed. The five tendons decompose into two subsystems:

==== Elbow subsystem (2 tendons, 1 joint)
The two antagonistic elbow cables attach symmetrically at the midpoints of the antiparallelogram side links. At zero configuration, each cable has a perpendicular lever arm $r_e = 72.5 "mm"$ with respect to the effective elbow rotation axis. The net elbow torque is:
$ tau_"elbow" = r_e (T_1 - T_2) $ <eq:elbow_torque>
where $T_1$ and $T_2$ are the tensions in the two antagonistic cables. The sign convention is such that $T_1$ produces positive (flexion) torque and $T_2$ produces negative (extension) torque.

==== Wrist subsystem (3 tendons, 2 joints)
The three wrist cables generate torques depending on their geometric attachment to the wrist platform ($r_w = 20 "mm"$). Adjusting for the specific phase offsets ($0 degree, 120 degree, 240 degree$) around the end-effector coordinate frame, the exact mapping implemented in the simulation is:
$ mat(tau_"pitch"; tau_"roll") = mat(
  -r_w sqrt(3)/2, 0, r_w sqrt(3)/2;
  r_w/2, -r_w, r_w/2
) mat(T_3; T_4; T_5) $ <eq:wrist_torque>

The symbols `pitch` and `roll` correspond to the URDF/USD joints `wrist_y_joint` and `wrist_x_joint`, respectively, in the integrated articulation tree.

Combining both subsystems, the full constant $bold(J)^top in bb(R)^(3 times 5)$ matrix matches the zero-configuration simulation property exactly:
$ bold(J)^top = mat(
  r_e, -r_e, 0, 0, 0;
  0, 0, -r_w sqrt(3)/2, 0, r_w sqrt(3)/2;
  0, 0, r_w/2, -r_w, r_w/2;
) = mat(
  0.0725, -0.0725, 0, 0, 0;
  0, 0, -0.0173, 0, 0.0173;
  0, 0, 0.010, -0.020, 0.010;
) $ <eq:jt_full>

@fig:tendon_dataflow illustrates the complete data flow from policy output to physics-engine joint torques.

#faps-figure(
  tendon-actuation-dataflow(),
  caption: [Data flow of the tendon actuation pipeline. The #ac("RL") policy outputs five normalized actions $bold(a)_t in [-1, 1]^5$. These are scaled to cable tensions $bold(T)$ by the maximum tension $T_"max"$. The $bold(J)^top$ mapping converts tensions to elbow and wrist torques, which are applied to the PhysX articulation joints.],
  short-caption: [Tendon actuation pipeline data flow],
) <fig:tendon_dataflow>

==== Architectural Justification vs. Built-in Tendons
Isaac Sim/PhysX 5 provides native "spatial tendon" primitives designed for muscle-like actuation~@physx_articulations_550. However, they were deliberately not used in this architecture for two primary reasons. First, spatial tendons frequently induce numerical instabilities and explosions when simulating stiff cable-driven linkages such as the tensegrity arm mechanism. Second, PhysX excludes native tendon tension signals from its standard `joint_force_report` #ac("API"). This makes it mechanically impossible to extract or penalize individual cable tensions during standard #ac("RL") step loops. The custom `ActionTerm` approach provides direct, observable control over the force space, perfectly suiting standard #ac("RL") reward formulations.

=== Physical Body-Force Tendon Model <subsec:body_force_model>

The physical tendon mode increases fidelity by modeling the elbow tendons as body forces applied at the actual cable attachment points on the antiparallelogram linkage bars, rather than abstracting them as scalar torques through a constant Jacobian. This produces a configuration-dependent torque that naturally captures the nonlinear transmission characteristics of the four-bar mechanism.

==== Elbow body forces
Each of the two elbow tendons is defined by an attachment point on the root link and a corresponding point on the forearm side of the linkage. At each simulation step, the force direction is computed as the unit vector from attachment to anchor, and the force magnitude is set to the cable tension output by the policy. The resulting force is applied through PhysX's body-force API directly on the rigid-body link prim. The net torque about the effective elbow axis depends on the lever arm at the _current_ linkage configuration rather than the zero-configuration constant.

==== Wrist torques
The wrist retains the constant $bold(J)^top$ mapping from @eq:wrist_torque because the cable-driven wrist's deviation from constant behavior over its $plus.minus 50 degree$ range is small. The principal effect is captured by the structure matrix~@Nemoto2022CableWrist.

==== Implementation
The physical tendon model is implemented as a custom `ActionTerm` class in the Isaac~Lab extension. During each environment step, the term:
+ Reads the five cable-tension actions from the policy (clipped to $[0, T_"max"]$).
+ For tendons 1--2 (elbow): retrieves current link poses from the simulation, computes force vectors between attachment points, and applies body forces via the PhysX `apply_body_force` API.
+ For tendons 3--5 (wrist): computes joint torques via @eq:wrist_torque and applies them as joint efforts using Isaac~Lab's `set_joint_effort_target` API.
+ Applies #ac("PD") position control to the two prismatic base joints (which are not tendon-driven).

=== Actuation Mode Comparison <subsec:actuation_comparison>

Three actuation modes are implemented for systematic comparison. @tab:actuation_modes summarizes their characteristics. Each mode represents a different trade-off between computational cost and physical fidelity.

#faps-table(
  table(
    columns: (auto, 1fr, auto, auto),
    table.header([*Mode*], [*Elbow actuation*], [*Wrist actuation*], [*Elbow model*]),
    [PD-driven], [`ImplicitActuator`, position targets], [`ImplicitActuator`], [Disc-approximation],
    [Tendon-driven (constant $bold(J)^top$)], [Constant $bold(J)^top$ ($r_e = 72.5$~mm)], [Constant $bold(J)^top$], [Disc-approximation],
    [Physical tendon (body-force)], [directional force], [Constant $J^top$], [Antiparallelogram],
  ),
  caption: [Comparison of the three actuation modes. The #ac("PD")-driven mode serves as the baseline. The tendon-driven mode introduces the cable-to-joint coupling. The physical tendon mode adds configuration-dependent elbow forces and the four-bar linkage constraint.],
  short-caption: [Comparison of the three actuation modes],
) <tab:actuation_modes>

In the *PD-driven mode*, all five joints (base $Y$, base $Z$, elbow, wrist pitch, wrist roll) use Isaac Sim's built-in `ImplicitActuator` with position targets. The policy outputs joint position deltas $Delta bold(q)$ that are added to the current positions, and the simulator applies torques via:
$ tau_i = K_(p,i) (q_i^* - q_i) - K_(d,i) dot(q)_i $ <eq:pd_control>
This mode ignores the tendon transmission entirely and treats each joint as independently actuated.

In the *tendon-driven mode (constant $J^top$)*, the two base joints remain PD-controlled while the three arm joints are driven through the constant $bold(J)^top$ mapping (@eq:jt_full). The policy outputs seven values: two base position deltas and five cable tensions. This mode tests whether an #ac("RL") policy can learn to coordinate antagonistic tendons to achieve precise joint-space or task-space targets.

In the *physical tendon mode (body-force)*, the elbow model is the full antiparallelogram linkage variant (@subsec:physical_linkage), and elbow forces are applied as body forces at the physical cable attachment points (@subsec:body_force_model). The wrist and base are actuated identically to the tendon-driven mode. This mode provides the highest physical fidelity and the greatest challenge for learning due to the nonlinear, configuration-dependent elbow transmission.

=== Motor Specifications and Simulation Parameters <subsec:motor_specs>

@tab:motor_params lists the motor and cable parameters used in simulation, derived from the Maxon EC60 datasheet and the spool geometry described in @subsec:drive_system. The per-tendon cable saturations and the resulting joint-level effort limits cited in the table are derived in @subsec:drive_system (see @eq:max_tension_elbow, @eq:effort_elbow, and @eq:effort_wrist).

#faps-table(
  table(
    columns: (auto, auto, auto),
    table.header([*Parameter*], [*Value*], [*Source*]),
    [Motor type], [Maxon EC60 Flat #acs("BLDC")], [@MaxonEC60Datasheet],
    [Nominal voltage], [24~V], [@MaxonEC60Datasheet],
    [Max.\ continuous torque], [0.401~Nm], [@MaxonEC60Datasheet],
    [Spool radius $r_s$], [5~mm], [@Klein2023],
    [Elbow pulley ratio], [3:1 (block-and-tackle)], [@Klein2023],
    [Wrist pulley ratio], [1:1 (direct drive)], [@Klein2023],
    [Max.\ cable tension $T_"max"$], [480~N (elbow tendons) / 160~N (wrist tendons)], [@eq:max_tension_elbow],
    [Elbow effort limit], [35.0~Nm], [@eq:effort_elbow],
    [Wrist effort limit (per axis)], [3.5~Nm], [@eq:effort_wrist],
    [Base effort limit], [300~Nm ($Y$) / 200~Nm ($Z$)], [Configured],
  ),
  caption: [Motor and actuation parameters used in the simulation model. All values are derived from the Maxon EC60 specifications and the mechanical design of the tendon routing system.],
  short-caption: [Motor and actuation parameters],
) <tab:motor_params>

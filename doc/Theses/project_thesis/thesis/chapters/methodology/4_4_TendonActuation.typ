// §4.4 Tendon Actuation in Simulation
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table
#import "../../../../shared/formatting/diagrams.typ": tendon-actuation-dataflow

== Tendon Actuation in Simulation <sec:tendon_actuation>

A central contribution of this work is the implementation of tendon-driven actuation within the Isaac Lab framework. The physical robot's five cables must be mapped to joint torques in a way that is compatible with PhysX's reduced-coordinate articulation solver, differentiable for gradient-based #ac("RL") training, and GPU-parallelizable across thousands of environment instances. This section derives the actuation model and describes three implementation variants of increasing physical fidelity.

=== Jacobian-Transpose Torque Mapping <subsec:jt_mapping>

#reg-sym("sym:tau", "sym:T", "sym:Jt", "sym:J_mat")

The fundamental relationship between cable tensions and joint torques in a tendon-driven mechanism is given by the Jacobian-transpose mapping~@Verhoeven2004TendonPlatforms:
$ bold(tau) = bold(J)^top (bold(q)) bold(T) $ <eq:jt_general>
where $bold(tau) in bb(R)^n$ is the vector of joint torques, $bold(T) in bb(R)^m$ is the vector of cable tensions ($m = 5$, $n = 3$), and $bold(J)(bold(q)) in bb(R)^(m times n)$ is the tendon Jacobian that maps infinitesimal joint displacements to cable length changes. In general, $bold(J)$ is configuration-dependent since the geometric relationship between cables and joints changes with joint angles.

For the constant-Jacobian actuation mode, $bold(J)^top$ is evaluated once at the zero configuration ($bold(q) = bold(0)$) and held fixed. The five tendons decompose into two subsystems:

==== Elbow subsystem (2 tendons, 1 joint)
The two antagonistic elbow cables attach symmetrically at the midpoints of the antiparallelogram side links. At zero configuration, each cable has a perpendicular lever arm $r_e = 72.5 "mm"$ with respect to the effective elbow rotation axis. The net elbow torque is:
$ tau_"elbow" = r_e (T_1 - T_2) $ <eq:elbow_torque>
where $T_1$ and $T_2$ are the tensions in the two antagonistic cables. The sign convention is such that $T_1$ produces positive (flexion) torque and $T_2$ produces negative (extension) torque.

==== Wrist subsystem (3 tendons, 2 joints)
The three wrist cables at $120 degree$ spacing produce the two wrist torques via the structure matrix derived in @eq:structure_matrix_explicit:
$ mat(tau_"pitch"; tau_"roll") = r_w mat(
  1, -1/2, -1/2;
  0, sqrt(3)/2, -sqrt(3)/2
) mat(T_3; T_4; T_5) $ <eq:wrist_torque>

Combining both subsystems, the full constant $bold(J)^top in bb(R)^(3 times 5)$ matrix is:
$ bold(J)^top = mat(
  r_e, -r_e, 0, 0, 0;
  0, 0, r_w, -r_w/2, -r_w/2;
  0, 0, 0, r_w sqrt(3)/2, -r_w sqrt(3)/2;
) $ <eq:jt_full>

@fig:tendon_dataflow illustrates the complete data flow from policy output to physics-engine joint torques.

#faps-figure(
  tendon-actuation-dataflow(),
  caption: [Data flow of the tendon actuation pipeline. The #ac("RL") policy outputs five normalized actions $bold(a)_t in [-1, 1]^5$. These are scaled to cable tensions $bold(T)$ by the maximum tension $T_"max"$. The $bold(J)^top$ mapping converts tensions to elbow and wrist torques, which are applied to the PhysX articulation joints.],
  short-caption: [Tendon actuation pipeline data flow],
) <fig:tendon_dataflow>

=== Physical Body-Force Tendon Model <subsec:body_force_model>

The physical tendon mode increases fidelity by modeling the elbow tendons as body forces applied at the actual cable attachment points on the antiparallelogram linkage bars, rather than abstracting them as scalar torques through a constant Jacobian. This produces a configuration-dependent torque that naturally captures the nonlinear transmission characteristics of the four-bar mechanism.

==== Elbow body forces
Each of the two elbow tendons is defined by an attachment point on the root link and a corresponding point on the forearm side of the linkage. At each simulation step, the force direction is computed as the unit vector from attachment to anchor, and the force magnitude is set to the cable tension output by the policy. The resulting force is applied through PhysX's body-force API directly on the rigid-body link prim. The net torque about the effective elbow axis depends on the lever arm at the _current_ linkage configuration rather than the zero-configuration constant.

==== Wrist torques
The wrist retains the constant $bold(J)^top$ mapping from @eq:wrist_torque because the cable-driven wrist's deviation from constant behavior over its $plus.minus 50 degree$ range is small — the principal effect is captured by the structure matrix~@Nemoto2022CableWrist.

==== Implementation
The physical tendon model is implemented as a custom `ActionTerm` class in the IsaacLab extension. During each environment step, the term:
+ Reads the five cable-tension actions from the policy (clipped to $[0, T_"max"]$).
+ For tendons 1--2 (elbow): retrieves current link poses from the simulation, computes force vectors between attachment points, and applies body forces via the PhysX `apply_body_force` API.
+ For tendons 3--5 (wrist): computes joint torques via @eq:wrist_torque and applies them as joint efforts using IsaacLab's `set_joint_effort_target` API.
+ Applies #ac("PD") position control to the two prismatic base joints (which are not tendon-driven).

=== Actuation Mode Comparison <subsec:actuation_comparison>

Three actuation modes are implemented for systematic comparison. @tab:actuation_modes summarizes their characteristics; each mode represents a different trade-off between computational cost and physical fidelity.

#faps-table(
  table(
    columns: (auto, 1fr, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Mode*], [*Elbow actuation*], [*Wrist actuation*], [*Elbow model*]),
    [#ac("PD")-driven], [ImplicitActuator, position targets], [ImplicitActuator], [Disc approx.],
    [Tendon-driven], [Constant $J^top$ ($r_e = 72.5 "mm"$)], [Constant $J^top$], [Disc approx.],
    [Physical tendon], [Body-force at linkage attachment pts.], [Constant $J^top$], [Antiparallelogram],
  ),
  caption: [Comparison of the three actuation modes. The #ac("PD")-driven mode serves as the baseline. The tendon-driven mode introduces the cable-to-joint coupling. The physical tendon mode adds configuration-dependent elbow forces and the four-bar linkage constraint.],
  short-caption: [Comparison of the three actuation modes],
) <tab:actuation_modes>

In the *#ac("PD")-driven mode*, all five joints (base $Y$, base $Z$, elbow, wrist pitch, wrist roll) use Isaac Sim's built-in `ImplicitActuator` with position targets. The policy outputs joint position deltas $Delta bold(q)$ that are added to the current positions, and the simulator applies torques via:
$ tau_i = K_(p,i) (q_i^* - q_i) - K_(d,i) dot(q)_i $ <eq:pd_control>
This mode ignores the tendon transmission entirely and treats each joint as independently actuated.

In the *tendon-driven mode*, the two base joints remain #ac("PD")-controlled while the three arm joints are driven through the constant $bold(J)^top$ mapping (@eq:jt_full). The policy outputs seven values: two base position deltas and five cable tensions. This mode tests whether an #ac("RL") policy can learn to coordinate antagonistic tendons to achieve precise joint-space or task-space targets.

In the *physical tendon mode*, the elbow model is the full antiparallelogram linkage variant (@subsec:physical_linkage), and elbow forces are applied as body forces at the physical cable attachment points (@subsec:body_force_model). The wrist and base are actuated identically to the tendon-driven mode. This mode provides the highest physical fidelity and the greatest challenge for learning due to the nonlinear, configuration-dependent elbow transmission.

=== Motor Specifications and Simulation Parameters <subsec:motor_specs>

@tab:motor_params lists the motor and cable parameters used in simulation, derived from the Maxon EC60 datasheet and the spool geometry described in @subsec:drive_system.

#faps-table(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Parameter*], [*Value*], [*Source*]),
    [Motor type], [Maxon EC60 flat BLDC], [@MaxonEC60Datasheet],
    [Nominal voltage], [24~V], [@MaxonEC60Datasheet],
    [Max.\ continuous torque], [0.444~Nm], [@MaxonEC60Datasheet],
    [Spool radius $r_s$], [5~mm], [@Klein2023],
    [Elbow pulley ratio], [3:1], [@Klein2023],
    [Max.\ cable tension $T_"max"$], [500~N], [Derived],
    [Elbow effort limit], [72.5~Nm], [@eq:effort_elbow],
    [Wrist effort limit (per axis)], [80~Nm], [Derived],
    [Base effort limit (per axis)], [200~Nm], [Configured],
  ),
  caption: [Motor and actuation parameters used in the simulation model. All values are derived from the Maxon EC60 specifications and the mechanical design of the tendon routing system.],
  short-caption: [Motor and actuation parameters],
) <tab:motor_params>

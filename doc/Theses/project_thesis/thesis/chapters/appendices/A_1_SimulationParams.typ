// Appendix A.1 — Simulation and Control Parameters
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table
#reg-sym("sym:theta")

= Appendix: Simulation and Control Parameters <appendix:validation_params>

== Joint Drive Parameters <appendix:drive_params>

@tab:drive_params lists the actuator drive parameters for the tensegrity manipulator's joints.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    table.header([*Joint Group*], [*Stiffness*], [*Damping*], [*Effort Limit*], [*Vel. Limit*]),
    [`base_y`], [8~000], [800], [300~N], [5.0~m/s],
    [`base_z`], [8~000], [800], [200~N], [5.0~m/s],
    [`elbow`], [400], [20], [40~N·m], [1.0~rad/s],
    [`wrist` (both)], [400], [20], [10~N·m], [0.5~rad/s],
  ),
  caption: [Implicit actuator #ac("PD") drive parameters for the tensegrity manipulator.],
  short-caption: [Tensegrity joint drive parameters],
) <tab:drive_params>

== Step-Response #ac("PID") Parameters <appendix:step_pid>

@tab:step_pid lists the #ac("PID") parameters used in the step-response validation tests (@sec:model_validation). Gains are scaled $approx 133 times$ from Klein's originals to match Isaac Sim's effective rotational inertia.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    table.header([*Joint*], [$K_p$], [$K_i$], [$K_d$], [*Saturation (N)*]),
    [`elbow_joint`], [40], [4], [3], [400],
    [`wrist_y_joint`], [10], [2], [0.3], [600],
    [`wrist_x_joint`], [10], [2], [0.3], [600],
  ),
  caption: [#ac("PID") gains and torque saturation for step-response testing.],
  short-caption: [Step-response #ac("PID") gains],
) <tab:step_pid>

The #ac("PID") controllers include feedforward gravity compensation: $tau_g = m_e dot g dot l_c dot sin(theta)$.

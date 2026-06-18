// Appendix A.2 — Simulation and Control Parameters
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table
#reg-sym("sym:theta")

= Appendix: Simulation and Control Parameters <appendix:validation_params>

== Joint Drive Parameters <appendix:drive_params>

@tab:drive_params lists the `ImplicitActuator` #ac("PD") drive parameters configured for the tensegrity manipulator's joints. In the PD-driven actuation mode, these parameters apply to all joints. In the tendon-driven and physical tendon modes, the arm joints (elbow and wrist) are overridden by the tendon actuation model. Only the two prismatic base joints retain #ac("PD") drives in those modes (see @tab:actuation_modes).

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    table.header([*Joint*], [*Stiffness*], [*Damping*], [*Effort Limit*], [*Vel. Limit*]),
    [`base_y_joint`], [8~000], [800], [300~N], [5.0~m/s],
    [`base_z_joint`], [8~000], [800], [200~N], [5.0~m/s],
    [`elbow_joint`], [400], [20], [35~N·m], [1.0~rad/s],
    [`wrist_y_joint`], [400], [20], [3.5~N·m], [0.5~rad/s],
    [`wrist_x_joint`], [400], [20], [3.5~N·m], [0.5~rad/s],
  ),
  caption: [`ImplicitActuator` #ac("PD") drive parameters for the tensegrity manipulator. Effort limits are derived from the motor and tendon specifications (@tab:motor_params, @eq:effort_elbow, @eq:effort_wrist). Base joint effort limits are configured conservatively for the prismatic slide mechanism. Arm joint parameters apply exclusively to the PD-driven actuation mode.],
  short-caption: [Tensegrity joint drive parameters],
) <tab:drive_params>

== Step-Response #ac("PID") Parameters <appendix:step_pid>

@tab:step_pid lists the #ac("PID") gains and cable-tension saturations used in the step-response validation tests (@sec:model_validation). The saturation values refer to per-tendon cable tension clipping applied before the $bold(J)^top$ mapping. 

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    table.header([*Joint*], [$K_p$], [$K_i$], [$K_d$], [*Saturation (N)*], [*Min. Tension (N)*]),
    table.cell(colspan: 6, align: left)[_Disc-approximation variant_],
    [`elbow_joint`], [50], [4], [2], [480], [6],
    [`wrist_y_joint`], [10], [1.5], [0.6], [180], [5],
    [`wrist_x_joint`], [10], [1.5], [0.6], [180], [5],
    table.hline(stroke: 0.4pt),
    table.cell(colspan: 6, align: left)[_Physical tendon variant (elbow only differs)_],
    [`elbow_joint`], [75], [6], [3], [480], [6],
    [`wrist_y_joint`], [10], [1.5], [0.6], [180], [5],
    [`wrist_x_joint`], [10], [1.5], [0.6], [180], [5],
  ),
  caption: [#ac("PID") gains and cable-tension saturations for step-response testing, shown separately for the disc-approximation and physical antiparallelogram linkage variants. Saturation values apply to individual tendon tensions before the $bold(J)^top$ mapping. Minimum tension enforces cable pre-tension on inactive tendons to prevent slack. Wrist gains are identical across both variants.],
  short-caption: [Step-response #ac("PID") gains by variant],
) <tab:step_pid>

The #ac("PID") controllers include feedforward gravity compensation: $tau_g = m_e dot g dot l_c dot sin(theta)$, with $m_e = 2.49 "kg"$ and $l_c = 0.26 "m"$ for the elbow, and $m_e = 0.40 "kg"$ and $l_c = 0.068 "m"$ for the wrist joints.

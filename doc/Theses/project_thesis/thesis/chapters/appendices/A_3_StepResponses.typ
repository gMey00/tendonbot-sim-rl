// Appendix A.7 — Step-Response Time Traces
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure

= Appendix: Step-Response Time Traces <appendix:step_responses>

This appendix collects the full time-trace plots for the step-response validation experiments referenced in @subsec:step_results. Plots are grouped by actuation mode (PD-driven, tendon-driven (constant $bold(J)^top$), physical tendon (body-force)) and ordered Elbow → Wrist-X → Wrist-Y per mode. Within each panel, dashed lines show the commanded setpoint and solid lines the measured joint angle. The final figure covers the two prismatic base joints.

#faps-figure(
  image("../../../assets/figures/appendix/step_pd_grouped.png", width: 100%),
  caption: [PD-driven step responses (`ImplicitActuator`: $K = 400$, $D = 20$). Rise/settle/NRMSE/overshoot values are aggregated into @fig:step_metrics in @subsec:step_results.],
  short-caption: [Arm PD-driven step responses],
) <fig:appendix_step_pd>

#faps-figure(
  image("../../../assets/figures/appendix/step_tendon_grouped.png", width: 100%),
  caption: [Tendon-driven (constant $bold(J)^top$) step responses (elbow: $K_p = 50$, $K_i = 4$, $K_d = 2$ | wrist: $K_p = 10$, $K_i = 1.5$, $K_d = 0.6$). Rise/settle/NRMSE/overshoot values are aggregated into @fig:step_metrics in @subsec:step_results.],
  short-caption: [Arm tendon-driven (constant $bold(J)^top$) step responses],
) <fig:appendix_step_tendon>

#faps-figure(
  image("../../../assets/figures/appendix/step_physical_grouped.png", width: 100%),
  caption: [Physical tendon (body-force) step responses (elbow: $K_p = 75$, $K_i = 6$, $K_d = 3$). Wrist actuation is identical to the tendon-driven (constant $bold(J)^top$) mode (@fig:appendix_step_tendon); only the elbow is replaced by the full four-bar antiparallelogram body-force model. Rise/settle/NRMSE/overshoot values are aggregated into @fig:step_metrics in @subsec:step_results.],
  short-caption: [Arm physical tendon (body-force) step responses],
) <fig:appendix_step_physical>

#faps-figure(
  image("../../../assets/figures/appendix/step_base_grouped.png", width: 100%),
  caption: [Prismatic base-joint step responses ($K = 8000$, $D = 800$). Rise/settle/NRMSE/overshoot values are aggregated into @fig:step_metrics in @subsec:step_results.],
  short-caption: [Prismatic base-joint step responses (Y / Z)],
) <fig:appendix_step_base>

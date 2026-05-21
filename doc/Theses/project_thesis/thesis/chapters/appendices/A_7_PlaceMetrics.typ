// Appendix A.5 — Cube Place Task: Complete Training Metrics
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure

= Appendix: Cube Place Task — Complete Training Metrics <appendix:place_metrics>

This appendix provides complete training curves for the cube place task across all three tensegrity variants. All plots are cut at 300k timesteps (#ac("PD") variant training duration).

== Reward Overview <appendix:place_reward_overview>

#faps-figure(
  image("../../../assets/figures/appendix/place_reward_overview.png", width: 100%),
  caption: [Cube place reward overview. *(a)*~Total reward — solid lines show the per-variant mean, the shaded envelope spans the per-iteration min/max range. *(b)*~Policy standard deviation ($sigma$).],
  short-caption: [Cube place reward overview],
) <fig:appendix_place_rewards>

== Task Metrics <appendix:place_task_metrics>

#faps-figure(
  image("../../../assets/figures/appendix/place_metrics.png", width: 100%),
  caption: [Cube place task metrics. *(a)*~Grasp success rate. *(b)*~Place success rate. *(c)*~Red-cube safety rate (remaining on conveyor). *(d)*~Mean episode length.],
  short-caption: [Cube place task metrics],
) <fig:appendix_place_metrics>

== Task Reward Components — Manipulation Pipeline <appendix:place_manipulation>

#faps-figure(
  image("../../../assets/figures/appendix/place_manipulation_a.png", width: 100%),
  caption: [Cube place manipulation pipeline rewards (part 1/2). *(a)*~Coarse distance shaping toward the cube. *(b)*~Fine-grained reaching shaping near contact. *(c)*~Grasp reward (closure $times$ proximity).],
  short-caption: [Cube place manipulation rewards (reaching/grasping)],
) <fig:appendix_place_manipulation_a>

#faps-figure(
  image("../../../assets/figures/appendix/place_manipulation_b.png", width: 100%),
  caption: [Cube place manipulation pipeline rewards (part 2/2). *(d)*~Lifting binary reward. *(e)*~Height bonus proportional to lift distance.],
  short-caption: [Cube place manipulation rewards (lifting)],
) <fig:appendix_place_manipulation_b>

== Task Reward Components — Placement and Target <appendix:place_placement>

#faps-figure(
  image("../../../assets/figures/appendix/place_placement.png", width: 100%),
  caption: [Cube place placement rewards. *(a)*~Goal-tracking shaping ($1 - tanh(d_"xy"/1.0)$, gated on grasp). *(b)*~Fine goal-tracking ($1 - tanh(d_"xy"/0.20)$). *(c)*~Release reward. *(d)*~Green-cube-in-target-drum bonus.],
  short-caption: [Cube place placement rewards],
) <fig:appendix_place_placement>

== Task Reward Components — Safety and Red-Cube Avoidance <appendix:place_safety>

#faps-figure(
  image("../../../assets/figures/appendix/place_safety.png", width: 100%),
  caption: [Cube place safety / red-cube reward terms. *(a)*~Cube-off-conveyor penalty (activates with the red-cube curriculum at 125k). *(b)*~Red–green separation reward. *(c)*~Red-cube clearance reward (encouraging the policy to stay clear of the red distractor).],
  short-caption: [Cube place safety reward terms],
) <fig:appendix_place_safety>

== Regularization and Training Diagnostics <appendix:place_diagnostics>

#faps-figure(
  image("../../../assets/figures/appendix/place_regularization_a.png", width: 100%),
  caption: [Cube place regularization terms (part 1/2). *(a)*~Action-rate penalty (ramps at 150k). *(b)*~Joint-velocity penalty (ramps at 150k). *(c)*~Joint-torque penalty.],
  short-caption: [Cube place regularization (action/velocity/torque)],
) <fig:appendix_place_regularization_a>

#faps-figure(
  image("../../../assets/figures/appendix/place_regularization_b.png", width: 100%),
  caption: [Cube place regularization terms (part 2/2). *(d)*~Belt-contact penalty. *(e)*~Arm-utilization regulariser discouraging extreme joint excursions.],
  short-caption: [Cube place regularization (belt/utilization)],
) <fig:appendix_place_regularization_b>

#faps-figure(
  image("../../../assets/figures/appendix/place_diagnostics.png", width: 100%),
  caption: [Cube place training diagnostics. *(a)*~Policy loss. *(b)*~Value-function loss. *(c)*~Entropy loss. *(d)*~#acs("KL")-adaptive learning-rate schedule.],
  short-caption: [Cube place training diagnostics],
) <fig:appendix_place_diagnostics>

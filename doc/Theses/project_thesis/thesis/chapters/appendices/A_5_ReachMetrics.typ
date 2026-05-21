// Appendix A.4 — Reach Task: Complete Training Metrics
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure

= Appendix: Reach Task — Complete Training Metrics <appendix:reach_metrics>

This appendix provides complete training curves for the reach task across all three tensegrity variants. All plots are cut at 48k timesteps (#ac("PD") variant training duration).

== Reward Overview <appendix:reach_reward_overview>

#faps-figure(
  image("../../../assets/figures/appendix/reach_reward_overview.png", width: 100%),
  caption: [Reach task reward overview. *(a)*~Total reward (solid lines show the per-variant mean, the shaded envelope spans the per-iteration min/max range). *(b)*~Policy standard deviation ($sigma$).],
  short-caption: [Reach task reward overview],
) <fig:appendix_reach_rewards>

== Task Reward Components <appendix:reach_components>

#faps-figure(
  image("../../../assets/figures/appendix/reach_components.png", width: 100%),
  caption: [Reach task reward components. *(a)*~End-effector position tracking ($1 - tanh(d_"pos")$). *(b)*~Orientation tracking ($1 - tanh(d_"orient")$). *(c)*~Fine-grained position tracking (tight tolerance). *(d)*~Pose-reached bonus.],
  short-caption: [Reach task reward components],
) <fig:appendix_reach_components>

== Regularization and Training Diagnostics <appendix:reach_diagnostics>

#faps-figure(
  image("../../../assets/figures/appendix/reach_diagnostics.png", width: 100%),
  caption: [Reach task regularization and training diagnostics. *(a)*~Action-rate penalty (L2 action delta). *(b)*~Joint-velocity penalty. *(c)*~Policy loss. *(d)*~Value-function loss.],
  short-caption: [Reach task diagnostics],
) <fig:appendix_reach_diagnostics>

#faps-figure(
  image("../../../assets/figures/appendix/reach_diagnostics_extra.png", width: 100%),
  caption: [Reach task additional diagnostics. *(a)*~Entropy loss (exploration incentive). *(b)*~#acs("KL")-adaptive learning-rate schedule.],
  short-caption: [Reach task additional diagnostics],
) <fig:appendix_reach_diagnostics2>

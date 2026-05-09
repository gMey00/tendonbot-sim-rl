// Appendix A.3 — Cube Place Reward Terms
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table

= Appendix: Cube Place Reward Terms <appendix:place_regularization>

@tab:place_full_rewards lists all reward terms used in the cube place task, including regularization and metric-only terms.

#faps-table(
  table(
    columns: (auto, auto, 1fr, auto),
    table.header([*Term*], [*Weight*], [*Description*], [*Gate*]),
    table.hline(),
    table.cell(colspan: 4)[_Task Rewards_],
    [`reaching_object`], [$+2.0$], [$1 - tanh(d_"tip" / 2.0)$ — long-range distance gradient], [`!grasp_active`],
    [`reaching_object_fine`], [$+5.0$], [$1 - tanh(d_"tip" / 0.5)$ — fine approach gradient], [`!grasp_active`],
    [`grasping`], [$+3.0$], [Closure $times$ $(1 - tanh(d_"tip" / 0.08))$], [—],
    [`lifting_object`], [$+5.0$], [Binary: cube above belt $+ 0.06$~m, gated by closure/proximity/velocity], [—],
    [`height_bonus`], [$+5.0$], [$min(Delta z, 0.30) / 0.30$, same gates as `lifting_object`], [—],
    [`goal_tracking`], [$+40.0$], [$1 - tanh(d_"xy" / 1.0)$ — coarse cube-to-drum tracking], [`was_grasped`],
    [`goal_tracking_fine`], [$+10.0$], [$1 - tanh(d_"xy" / 0.20)$ — fine cube-to-drum tracking], [`was_grasped`],
    [`release`], [$+25.0$], [$(1 - "closure") times "in"_"xy" times "above"_"rim"$], [`was_grasped`],
    [`green_in_target`], [$+100.0$], [Per-step bonus while the green cube centre is inside the drum volume], [—],
    [`red_in_target`], [$-12.0$], [Red cube centre inside the drum volume (penalty)], [red active],
    [`red_clearance`], [$+5.0$], [$tanh(d_"red,drum" / 0.4)$ — push red away from drum pre-grasp], [`!was_grasped`],
    [`red_green_separation`], [$+3.0$], [Push red cube away from green cube before grasping], [`!was_grasped`],
    [`return_to_neutral`], [$+25.0$], [$exp(-norm(bold(q) - bold(q)_0)^2 / 1.0)$ once placed], [`was_placed`],
    table.hline(),
    table.cell(colspan: 4)[_Regularisation_],
    [`action_rate`], [$-10^(-4) arrow -2 times 10^(-3)$], [L2 action delta (curriculum ramp at 150k steps)], [—],
    [`joint_vel`], [$-10^(-4) arrow -2 times 10^(-3)$], [L2 controlled-joint velocity (curriculum ramp at 150k steps)], [—],
    [`arm_utilization`], [$+0.25$], [Bonus for arm-joint motion (encourages active manipulation)], [—],
    [`belt_contact`], [$-10.0$], [Finger-tip penetration below belt surface], [—],
    [`joint_torque`], [$-0.025$], [L2 arm-joint torque penalty], [—],
    [`cube_off_conveyor`], [$-5.0$], [Green cube knocked off the belt (red is exempt)], [—],
    table.hline(),
    table.cell(colspan: 4)[_Metric-Only (Diagnostic, Negligible Weight)_],
    [`metric_place_success`], [$+0.01$], [Place-success indicator for TensorBoard], [—],
    [`metric_grasp_rate`], [$+0.01$], [Green-cube grasp indicator for TensorBoard], [—],
    [`metric_ee_distance`], [$-0.01$], [End-effector–to–green distance for TensorBoard], [—],
  ),
  caption: [Complete cube place reward specification, mirroring `RewardsCfg` in `place_env_cfg.py`. Distances $d_"tip"$ and $d_"xy"$ are measured from the dynamic finger-tip point and the cube–drum XY centres, respectively. Gating is enforced inside each reward function: `!grasp_active` and `!was_grasped` deactivate pre-grasp shaping once the cube is held; `was_grasped` enables transport/release rewards; `was_placed` enables the return-to-neutral term. Curriculum-ramped weights are shown as $"initial" arrow "final"$.],
  short-caption: [Cube place reward specification],
) <tab:place_full_rewards>

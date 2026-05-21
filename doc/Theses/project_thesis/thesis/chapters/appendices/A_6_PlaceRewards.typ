// Appendix A.3 — Cube Place Reward Terms
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table

= Appendix: Cube Place Reward Terms <appendix:place_regularization>

// @tab:place_full_rewards lists all reward terms used in the cube place task, including regularization and metric-only terms.

#faps-table(
  table(
    columns: (auto, auto, 1fr, auto),
    table.header([*Term*], [*Weight*], [*Description*], [*Gate*]),
    table.hline(),
    table.cell(colspan: 4)[_Task Rewards_],
    [`reaching_object`], [$+2.0$], [$1 - tanh(d_"tip" / 2.0)$ — long-range distance gradient], [`!grasp_active`],
    [`reaching_object_fine`], [$+5.0$], [$1 - tanh(d_"tip" / 0.5)$ — fine approach gradient], [`!grasp_active`],
    [`grasping`], [$+3.0$], [Closure $times$ $(1 - tanh(d_"tip" / 0.08))$], [—],
    [`lifting_object`], [$+5.0$], [Binary: cube above belt $+ 0.06$~m], [`grasp_active`],
    [`height_bonus`], [$+5.0$], [$min(Delta z, 0.30) / 0.30$], [`grasp_active`],
    [`goal_tracking`], [$+40.0$], [$1 - tanh(d_"xy" / 1.0)$], [`was_grasped`],
    [`goal_tracking_fine`], [$+10.0$], [$1 - tanh(d_"xy" / 0.20)$], [`was_grasped`],
    [`release`], [$+25.0$], [$(1 - "closure") times "in"_"xy" times "above"_"rim"$], [`was_grasped`],
    [`green_in_target`], [$+100.0$], [Per-step bonus], [—],
    [`red_green_separation`], [$+3.0$], [Push red cube away from green cube before grasping], [`!was_grasped`],
    table.hline(),
    table.cell(colspan: 4)[_Regularisation_],
    [`action_rate`], [$-10^(-4) arrow -2 times 10^(-3)$], [L2 action delta], [—],
    [`joint_vel`], [$-10^(-4) arrow -2 times 10^(-3)$], [L2 controlled-joint velocity], [—],
    [`arm_utilization`], [$+0.25$], [Bonus for arm-joint motion], [—],
    [`belt_contact`], [$-10.0$], [Finger-tip below belt], [—],
    [`joint_torque`], [$-0.025$], [L2 arm-joint torque penalty], [—],
    [`cube_off_conveyor`], [$-5.0$], [Cube knocked off the belt], [—],
    table.hline(),
    table.cell(colspan: 4)[_Metric-Only (Diagnostic, Negligible Weight)_],
    [`metric_place_success`], [$+0.01$], [Place-success indicator], [—],
    [`metric_grasp_rate`], [$+0.01$], [Green-cube grasp indicator], [—],
    [`metric_ee_distance`], [$-0.01$], [End-effector–to–green distance], [—],
  ),
  caption: [Complete cube place reward specification, mirroring `RewardsCfg` in `place_env_cfg.py`. Distances $d_"tip"$ and $d_"xy"$ are measured from the dynamic finger-tip point and the cube–drum XY centres, respectively. Gating is enforced inside each reward function: `!grasp_active` and `!was_grasped` deactivate pre-grasp shaping once the cube is held. `was_grasped` enables transport/release rewards. 
  //Curriculum-ramped weights are shown as $"initial" arrow "final"$.
  ],
  short-caption: [Cube place reward specification],
) <tab:place_full_rewards>

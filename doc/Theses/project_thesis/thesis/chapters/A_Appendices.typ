// Appendices
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *
#import "../../../shared/formatting/template.typ": faps-table, faps-figure
#import "../../../shared/formatting/plots.typ": *
#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot, chart

// Reset heading numbering for appendix style
#set heading(numbering: "A.1")
#counter(heading).update(0)

= Appendix: Simulation and Control Parameters <appendix:validation_params>

== Joint Drive Parameters <appendix:drive_params>

@tab:drive_params lists the actuator drive parameters for the tensegrity manipulator's joints.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint Group*], [*Stiffness*], [*Damping*], [*Effort Limit*], [*Vel. Limit*]),
    [`base_y`], [8~000], [800], [300~N], [5.0~m/s],
    [`base_z`], [8~000], [800], [200~N], [5.0~m/s],
    [`elbow`], [400], [20], [40~N·m], [1.0~rad/s],
    [`wrist` (both)], [400], [20], [10~N·m], [0.5~rad/s],
  ),
  caption: [Implicit actuator PD drive parameters for the tensegrity manipulator.],
  short-caption: [Tensegrity joint drive parameters],
) <tab:drive_params>

== Step-Response PID Parameters <appendix:step_pid>

@tab:step_pid lists the PID parameters used in the step-response validation tests (@sec:model_validation). Gains are scaled $approx 133 times$ from Klein's originals to match Isaac Sim's effective rotational inertia.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint*], [$K_p$], [$K_i$], [$K_d$], [*Saturation (N)*]),
    [`elbow_joint`], [40], [4], [3], [400],
    [`wrist_y_joint`], [10], [2], [0.3], [600],
    [`wrist_x_joint`], [10], [2], [0.3], [600],
  ),
  caption: [PID gains and torque saturation for step-response testing.],
  short-caption: [Step-response PID gains],
) <tab:step_pid>

The PID controllers include feedforward gravity compensation: $tau_g = m_e dot g dot l_c dot sin(theta)$.

= Appendix: PPO Hyperparameters <appendix:ppo_hyperparams>

@tab:ppo_full compares the full PPO configurations for the reach and cube place tasks.

#faps-table(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Parameter*], [*Reach*], [*Cube Place*]),
    [Seed], [42], [42],
    [Network architecture], [$[64, 64]$ ELU], [$[256, 128, 64]$ ELU],
    [Shared policy/value], [Yes], [Yes],
    [Rollout horizon], [24 steps], [48 steps],
    [Learning epochs], [5], [5],
    [Mini-batches], [4], [8],
    [Learning rate], [$10^(-3)$], [$10^(-4)$],
    [LR scheduler], [KL-adaptive], [KL-adaptive],
    [KL threshold], [0.01], [0.004],
    [Discount $gamma$], [0.99], [0.99],
    [GAE $lambda$], [0.95], [0.95],
    [Clip ratio $epsilon$], [0.2], [0.2],
    [Value clip], [0.2], [0.2],
    [Entropy scale], [0.01], [0.01],
    [Value loss scale], [1.0], [1.0],
    [Grad norm clip], [1.0], [1.0],
    [Initial log-std range], [$[-20, 2]$], [$[-2.5, 2]$],
    [State preprocessor], [RunningStandardScaler], [RunningStandardScaler],
    [Total timesteps], [48~000], [500~000],
    [`num_envs`], [4~096], [4~096],
  ),
  caption: [Full PPO hyperparameter comparison between reach and cube place tasks.],
  short-caption: [PPO hyperparameter comparison],
) <tab:ppo_full>

= Appendix: Cube Place Reward Terms <appendix:place_regularization>

@tab:place_full_rewards lists all reward terms used in the cube place task, including regularization and metric-only terms.

#faps-table(
  table(
    columns: (auto, auto, 1fr, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Term*], [*Weight*], [*Description*], [*Gate*]),
    table.hline(),
    table.cell(colspan: 4)[_Task Rewards_],
    [`reaching_object`], [$+2.0$], [Fingertip-to-cube distance shaping], [—],
    [`grasping`], [$+5.0$], [Closure $times$ proximity], [—],
    [`lifting_object`], [$+3.0$], [Binary: above belt, near gripper], [—],
    [`height_bonus`], [$+5.0$], [$min(Delta z, 0.30) / 0.30$], [—],
    [`object_velocity_penalty`], [$+5.0$], [Penalize excessive cube motion], [—],
    [`goal_tracking`], [$+40.0$], [$1 - tanh(d_"xy" / 1.0)$], [`was_grasped`],
    [`goal_tracking_fine`], [$+10.0$], [$1 - tanh(d_"xy" / 0.20)$], [`was_grasped`],
    [`release`], [$+25.0$], [$(1 - "cl.") times "in"_"xy" times "above"$], [`was_grasped`],
    [`green_in_target`], [$+100.0$], [Cube center inside drum volume], [`was_grasped`],
    [`red_in_target`], [$-12.0$], [Red cube in drum (penalty)], [red active],
    table.hline(),
    table.cell(colspan: 4)[_Regularization_],
    [`object_stay_bonus`], [$+3.0$], [Bonus for cube remaining on belt], [—],
    [`action_rate`], [$-10^(-4) arrow -2 times 10^(-3)$], [L2 action delta (curriculum)], [—],
    [`joint_vel`], [$-10^(-4) arrow -2 times 10^(-3)$], [L2 joint velocity (curriculum)], [—],
    [`fingertip_orientation`], [$+0.5$], [Fingertip alignment shaping], [—],
    [`belt_contact`], [$-10.0$], [Finger tips below belt surface], [—],
    [`joint_torque`], [$-0.05$], [L2 torque penalty], [—],
    [`cube_off_conveyor`], [$-5.0$], [Cube knocked off belt], [—],
    table.hline(),
    table.cell(colspan: 4)[_Metric-Only (Small Weight)_],
    [`cube_height_info`], [$+0.01$], [Cube height above belt], [—],
    [`gripper_state_info`], [$+0.01$], [Gripper closure tracking], [—],
    [`collision_penalty_info`], [$-0.01$], [Self-collision monitoring], [—],
  ),
  caption: [Complete cube place reward specification. Curriculum terms ramp weights at 200k training steps.],
  short-caption: [Cube place reward specification],
) <tab:place_full_rewards>

// ── RL Appendix shared setup ────────────────────────────────────────────────
#let rl-base = "../../../shared/data/rl_training/"

// X-axis cutoff values — last PD timestep per task
#let reach-xmax = 48000
#let place-xmax = 300000
#let sort-xmax  = 20000

// Subplot height constants
#let stacked-h = 3.0    // cm — height for stacked subplots
#let reward-h  = 5.0    // cm — height for combined reward overview
#let reward-pw = 11     // cm — reward plot width (leaves room for right legend)

// ── Legend helpers ──────────────────────────────────────────────────────────
#let legend-line(color, label, dash: none, thick: 1pt) = {
  let s = if dash == none {
    (paint: color, thickness: thick)
  } else {
    (paint: color, thickness: thick, dash: dash)
  }
  box(width: 14pt, baseline: 2pt, line(length: 14pt, stroke: s))
  h(3pt)
  label
}

// Shared 3-variant legend row
#let variant-legend = align(center, text(size: 8pt, stack(dir: ltr, spacing: 20pt,
  legend-line(plot-blue, [PD]),
  legend-line(plot-green, [Tendon]),
  legend-line(plot-purple, [Physical]),
)))

// Right-side legend for combined reward plots (3 variants × 3 statistics)
#let reward-side-legend = align(horizon, text(size: 7.5pt, stack(dir: ttb, spacing: 3pt,
  text(weight: "bold", size: 7pt, [Variant]),
  legend-line(plot-blue, [PD], thick: 1.2pt),
  legend-line(plot-green, [Tendon], thick: 1.2pt),
  legend-line(plot-purple, [Physical], thick: 1.2pt),
  v(4pt),
  text(weight: "bold", size: 7pt, [Statistic]),
  legend-line(luma(80), [Mean], thick: 1.2pt),
  legend-line(luma(80), [Max], dash: "dashed", thick: 0.6pt),
  legend-line(luma(80), [Min], dash: "dotted", thick: 0.6pt),
)))

// Right-side legend for single-variant combined reward (cube sort)
#let reward-side-legend-single = align(horizon, text(size: 7.5pt, stack(dir: ttb, spacing: 3pt,
  legend-line(plot-blue, [Mean], thick: 1.2pt),
  legend-line(plot-blue, [Max], dash: "dashed", thick: 0.6pt),
  legend-line(plot-blue, [Min], dash: "dotted", thick: 0.6pt),
)))

// ── Stacked full-width subplot (3 variants, no built-in legend) ─────────────
#let stacked-plot-3v(
  title: none,
  pd-data, tn-data, ph-data,
  y-label: none,
  y-min: auto, y-max: auto,
  x-max: auto,
  show-x-label: true,
  width: plot-full-width,
  height: stacked-h,
) = {
  import draw: *
  plot.plot(
    size: (width, height),
    x-label: if show-x-label { [Timesteps] } else { none },
    y-label: y-label,
    x-min: 0, x-max: x-max,
    y-min: y-min, y-max: y-max,
    x-grid: true, y-grid: true,
    legend: none,
    {
      plot.add(pd-data, style: (stroke: (paint: plot-blue, thickness: 1.0pt)))
      plot.add(tn-data, style: (stroke: (paint: plot-green, thickness: 1.0pt)))
      plot.add(ph-data, style: (stroke: (paint: plot-purple, thickness: 1.0pt)))
    }
  )
  if title != none {
    content((width / 2, height + 0.3), text(size: 8pt, weight: "bold", title))
  }
}

// ── Single-variant stacked subplot (for cube sort) ──────────────────────────
#let stacked-plot-1v(
  title: none,
  data,
  color: plot-blue,
  y-label: none,
  y-min: auto, y-max: auto,
  x-max: auto,
  show-x-label: true,
  width: plot-full-width,
  height: stacked-h,
) = {
  import draw: *
  plot.plot(
    size: (width, height),
    x-label: if show-x-label { [Timesteps] } else { none },
    y-label: y-label,
    x-min: 0, x-max: x-max,
    y-min: y-min, y-max: y-max,
    x-grid: true, y-grid: true,
    legend: none,
    {
      plot.add(data, style: (stroke: (paint: color, thickness: 1.0pt)))
    }
  )
  if title != none {
    content((width / 2, height + 0.3), text(size: 8pt, weight: "bold", title))
  }
}

// ── Combined reward overview plot (3 variants × mean/max/min) ───────────────
#let combined-reward-plot(
  mean-pd, mean-tn, mean-ph,
  max-pd, max-tn, max-ph,
  min-pd, min-tn, min-ph,
  x-max: auto,
  y-min: auto, y-max: auto,
) = {
  import draw: *
  plot.plot(
    size: (reward-pw, reward-h),
    x-label: [Timesteps], y-label: [Reward],
    x-min: 0, x-max: x-max,
    y-min: y-min, y-max: y-max,
    x-grid: true, y-grid: true,
    legend: none,
    {
      // Mean (solid, thick)
      plot.add(mean-pd, style: (stroke: (paint: plot-blue,   thickness: 1.2pt)))
      plot.add(mean-tn, style: (stroke: (paint: plot-green,  thickness: 1.2pt)))
      plot.add(mean-ph, style: (stroke: (paint: plot-purple, thickness: 1.2pt)))
      // Max (dashed, thin)
      plot.add(max-pd, style: (stroke: (paint: plot-blue,   thickness: 0.6pt, dash: "dashed")))
      plot.add(max-tn, style: (stroke: (paint: plot-green,  thickness: 0.6pt, dash: "dashed")))
      plot.add(max-ph, style: (stroke: (paint: plot-purple, thickness: 0.6pt, dash: "dashed")))
      // Min (dotted, thin)
      plot.add(min-pd, style: (stroke: (paint: plot-blue,   thickness: 0.6pt, dash: "dotted")))
      plot.add(min-tn, style: (stroke: (paint: plot-green,  thickness: 0.6pt, dash: "dotted")))
      plot.add(min-ph, style: (stroke: (paint: plot-purple, thickness: 0.6pt, dash: "dotted")))
    }
  )
}

// ═════════════════════════════════════════════════════════════════════════════
= Appendix: Reach Task — Complete Training Metrics <appendix:reach_metrics>
// ═════════════════════════════════════════════════════════════════════════════

This appendix provides complete training curves for the reach task across all three tensegrity variants. All plots are cut at 48k timesteps (PD variant training duration).

// ── Data loading ────────────────────────────────────────────────────────────
#let rch(variant, metric) = parse-csv(csv(rl-base + "reach/" + variant + "/" + metric + ".csv"))

== Reward Overview <appendix:reach_reward_overview>

#faps-figure(
  stack(dir: ttb, spacing: 8pt,
    grid(
      columns: (auto, auto),
      column-gutter: 12pt,
      canvas(length: 1cm, {
        combined-reward-plot(
          rch("tensegrity_pd", "Reward_Totalrewardmean"),
          rch("tensegrity_tendon", "Reward_Totalrewardmean"),
          rch("tensegrity_physical", "Reward_Totalrewardmean"),
          rch("tensegrity_pd", "Reward_Totalrewardmax"),
          rch("tensegrity_tendon", "Reward_Totalrewardmax"),
          rch("tensegrity_physical", "Reward_Totalrewardmax"),
          rch("tensegrity_pd", "Reward_Totalrewardmin"),
          rch("tensegrity_tendon", "Reward_Totalrewardmin"),
          rch("tensegrity_physical", "Reward_Totalrewardmin"),
          x-max: reach-xmax,
        )
      }),
      reward-side-legend,
    ),
    canvas(length: 1cm, {
      stacked-plot-3v(
        title: [Policy Standard Deviation],
        rch("tensegrity_pd", "Policy_Standarddeviation"),
        rch("tensegrity_tendon", "Policy_Standarddeviation"),
        rch("tensegrity_physical", "Policy_Standarddeviation"),
        y-label: [$sigma$], x-max: reach-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Reach task reward overview. Top: total reward mean (solid), max (dashed), and min (dotted) for all three variants. Bottom: policy standard deviation ($sigma$).],
  short-caption: [Reach task reward overview],
) <fig:appendix_reach_rewards>

== Task Reward Components <appendix:reach_components>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(a) Position Tracking],
        rch("tensegrity_pd", "Info_Episode_Reward_end_effector_position_tracking"),
        rch("tensegrity_tendon", "Info_Episode_Reward_end_effector_position_tracking"),
        rch("tensegrity_physical", "Info_Episode_Reward_end_effector_position_tracking"),
        y-label: [Reward], x-max: reach-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(b) Orientation Tracking],
        rch("tensegrity_pd", "Info_Episode_Reward_end_effector_orientation_tracking"),
        rch("tensegrity_tendon", "Info_Episode_Reward_end_effector_orientation_tracking"),
        rch("tensegrity_physical", "Info_Episode_Reward_end_effector_orientation_tracking"),
        y-label: [Reward], x-max: reach-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(c) Fine-Grained Position],
        rch("tensegrity_pd", "Info_Episode_Reward_end_effector_position_tracking_fine_grained"),
        rch("tensegrity_tendon", "Info_Episode_Reward_end_effector_position_tracking_fine_grained"),
        rch("tensegrity_physical", "Info_Episode_Reward_end_effector_position_tracking_fine_grained"),
        y-label: [Reward], x-max: reach-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(d) Pose Reached],
        rch("tensegrity_pd", "Info_Episode_Reward_pose_reached"),
        rch("tensegrity_tendon", "Info_Episode_Reward_pose_reached"),
        rch("tensegrity_physical", "Info_Episode_Reward_pose_reached"),
        y-label: [Reward], x-max: reach-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Reach task reward components. *(a)*~End-effector position tracking ($1 - tanh(d_"pos")$). *(b)*~Orientation tracking ($1 - tanh(d_"orient")$). *(c)*~Fine-grained position tracking (tight tolerance). *(d)*~Pose reached bonus.],
  short-caption: [Reach task reward components],
) <fig:appendix_reach_components>

== Regularization and Training Diagnostics <appendix:reach_diagnostics>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(a) Action Rate Penalty],
        rch("tensegrity_pd", "Info_Episode_Reward_action_rate"),
        rch("tensegrity_tendon", "Info_Episode_Reward_action_rate"),
        rch("tensegrity_physical", "Info_Episode_Reward_action_rate"),
        y-label: [Penalty], x-max: reach-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(b) Joint Velocity Penalty],
        rch("tensegrity_pd", "Info_Episode_Reward_joint_vel"),
        rch("tensegrity_tendon", "Info_Episode_Reward_joint_vel"),
        rch("tensegrity_physical", "Info_Episode_Reward_joint_vel"),
        y-label: [Penalty], x-max: reach-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(c) Policy Loss],
        rch("tensegrity_pd", "Loss_Policyloss"),
        rch("tensegrity_tendon", "Loss_Policyloss"),
        rch("tensegrity_physical", "Loss_Policyloss"),
        y-label: [Loss], x-max: reach-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(d) Value Loss],
        rch("tensegrity_pd", "Loss_Valueloss"),
        rch("tensegrity_tendon", "Loss_Valueloss"),
        rch("tensegrity_physical", "Loss_Valueloss"),
        y-label: [Loss], x-max: reach-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Reach task regularization and training diagnostics. *(a)*~Action rate penalty (L2 action delta). *(b)*~Joint velocity penalty. *(c)*~Policy loss. *(d)*~Value function loss.],
  short-caption: [Reach task diagnostics],
) <fig:appendix_reach_diagnostics>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(a) Entropy Loss],
        rch("tensegrity_pd", "Loss_Entropyloss"),
        rch("tensegrity_tendon", "Loss_Entropyloss"),
        rch("tensegrity_physical", "Loss_Entropyloss"),
        y-label: [Loss], x-max: reach-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(b) Learning Rate],
        rch("tensegrity_pd", "Learning_Learningrate"),
        rch("tensegrity_tendon", "Learning_Learningrate"),
        rch("tensegrity_physical", "Learning_Learningrate"),
        y-label: [LR], x-max: reach-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Reach task additional diagnostics. *(a)*~Entropy loss (exploration incentive). *(b)*~KL-adaptive learning rate schedule.],
  short-caption: [Reach task additional diagnostics],
) <fig:appendix_reach_diagnostics2>

// ═════════════════════════════════════════════════════════════════════════════
= Appendix: Cube Place Task — Complete Training Metrics <appendix:place_metrics>
// ═════════════════════════════════════════════════════════════════════════════

This appendix provides complete training curves for the cube place task across all three tensegrity variants. All plots are cut at 300k timesteps (PD variant training duration).

// ── Data loading ────────────────────────────────────────────────────────────
#let plc(variant, metric) = parse-csv(csv(rl-base + "cube_place/" + variant + "/" + metric + ".csv"))

== Reward Overview <appendix:place_reward_overview>

#faps-figure(
  stack(dir: ttb, spacing: 8pt,
    grid(
      columns: (auto, auto),
      column-gutter: 12pt,
      canvas(length: 1cm, {
        combined-reward-plot(
          plc("tensegrity_pd", "Reward_Totalrewardmean"),
          plc("tensegrity_tendon", "Reward_Totalrewardmean"),
          plc("tensegrity_physical", "Reward_Totalrewardmean"),
          plc("tensegrity_pd", "Reward_Totalrewardmax"),
          plc("tensegrity_tendon", "Reward_Totalrewardmax"),
          plc("tensegrity_physical", "Reward_Totalrewardmax"),
          plc("tensegrity_pd", "Reward_Totalrewardmin"),
          plc("tensegrity_tendon", "Reward_Totalrewardmin"),
          plc("tensegrity_physical", "Reward_Totalrewardmin"),
          x-max: place-xmax, y-min: -50, y-max: 500,
        )
      }),
      reward-side-legend,
    ),
    canvas(length: 1cm, {
      stacked-plot-3v(
        title: [Policy Standard Deviation],
        plc("tensegrity_pd", "Policy_Standarddeviation"),
        plc("tensegrity_tendon", "Policy_Standarddeviation"),
        plc("tensegrity_physical", "Policy_Standarddeviation"),
        y-label: [$sigma$], x-max: place-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Cube place reward overview. Top: total reward mean (solid), max (dashed), and min (dotted) for all three variants. Bottom: policy standard deviation ($sigma$).],
  short-caption: [Cube place reward overview],
) <fig:appendix_place_rewards>

== Task Metrics <appendix:place_task_metrics>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(a) Grasp Rate],
        plc("tensegrity_pd", "Info_Metrics_grasp_rate"),
        plc("tensegrity_tendon", "Info_Metrics_grasp_rate"),
        plc("tensegrity_physical", "Info_Metrics_grasp_rate"),
        y-label: [Rate], y-min: 0, y-max: 1.0,
        x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(b) Place Success Rate],
        plc("tensegrity_pd", "Info_Metrics_place_success_rate"),
        plc("tensegrity_tendon", "Info_Metrics_place_success_rate"),
        plc("tensegrity_physical", "Info_Metrics_place_success_rate"),
        y-label: [Rate], y-min: 0, y-max: 1.0,
        x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(c) Red on Conveyor Rate],
        plc("tensegrity_pd", "Info_Metrics_red_on_conveyor_rate"),
        plc("tensegrity_tendon", "Info_Metrics_red_on_conveyor_rate"),
        plc("tensegrity_physical", "Info_Metrics_red_on_conveyor_rate"),
        y-label: [Rate], y-min: 0, y-max: 1.0,
        x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(d) Mean Episode Length],
        plc("tensegrity_pd", "Info_Metrics_mean_episode_length"),
        plc("tensegrity_tendon", "Info_Metrics_mean_episode_length"),
        plc("tensegrity_physical", "Info_Metrics_mean_episode_length"),
        y-label: [Steps], x-max: place-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Cube place task metrics. *(a)*~Grasp success rate. *(b)*~Place success rate. *(c)*~Red cube safety rate (remaining on conveyor). *(d)*~Mean episode length.],
  short-caption: [Cube place task metrics],
) <fig:appendix_place_metrics>

== Task Reward Components — Manipulation Pipeline <appendix:place_manipulation>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(a) Reaching Object],
        plc("tensegrity_pd", "Info_Episode_Reward_reaching_object"),
        plc("tensegrity_tendon", "Info_Episode_Reward_reaching_object"),
        plc("tensegrity_physical", "Info_Episode_Reward_reaching_object"),
        y-label: [Reward], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(b) Grasping],
        plc("tensegrity_pd", "Info_Episode_Reward_grasping"),
        plc("tensegrity_tendon", "Info_Episode_Reward_grasping"),
        plc("tensegrity_physical", "Info_Episode_Reward_grasping"),
        y-label: [Reward], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(c) Lifting Object],
        plc("tensegrity_pd", "Info_Episode_Reward_lifting_object"),
        plc("tensegrity_tendon", "Info_Episode_Reward_lifting_object"),
        plc("tensegrity_physical", "Info_Episode_Reward_lifting_object"),
        y-label: [Reward], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(d) Height Bonus],
        plc("tensegrity_pd", "Info_Episode_Reward_height_bonus"),
        plc("tensegrity_tendon", "Info_Episode_Reward_height_bonus"),
        plc("tensegrity_physical", "Info_Episode_Reward_height_bonus"),
        y-label: [Reward], x-max: place-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Cube place manipulation pipeline rewards. *(a)*~Distance shaping toward the cube. *(b)*~Grasp reward (closure $times$ proximity). *(c)*~Lifting binary reward. *(d)*~Height bonus (proportional to lift distance).],
  short-caption: [Cube place manipulation rewards],
) <fig:appendix_place_manipulation>

== Task Reward Components — Placement and Target <appendix:place_placement>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(a) Goal Tracking],
        plc("tensegrity_pd", "Info_Episode_Reward_goal_tracking"),
        plc("tensegrity_tendon", "Info_Episode_Reward_goal_tracking"),
        plc("tensegrity_physical", "Info_Episode_Reward_goal_tracking"),
        y-label: [Reward], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(b) Goal Tracking Fine],
        plc("tensegrity_pd", "Info_Episode_Reward_goal_tracking_fine"),
        plc("tensegrity_tendon", "Info_Episode_Reward_goal_tracking_fine"),
        plc("tensegrity_physical", "Info_Episode_Reward_goal_tracking_fine"),
        y-label: [Reward], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(c) Release],
        plc("tensegrity_pd", "Info_Episode_Reward_release"),
        plc("tensegrity_tendon", "Info_Episode_Reward_release"),
        plc("tensegrity_physical", "Info_Episode_Reward_release"),
        y-label: [Reward], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(d) Green in Target],
        plc("tensegrity_pd", "Info_Episode_Reward_green_in_target"),
        plc("tensegrity_tendon", "Info_Episode_Reward_green_in_target"),
        plc("tensegrity_physical", "Info_Episode_Reward_green_in_target"),
        y-label: [Reward], x-max: place-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Cube place placement rewards. *(a)*~Goal tracking shaping ($1 - tanh(d_"xy"/1.0)$, gated on grasp). *(b)*~Fine goal tracking ($1 - tanh(d_"xy"/0.20)$). *(c)*~Release reward. *(d)*~Green cube in target drum bonus.],
  short-caption: [Cube place placement rewards],
) <fig:appendix_place_placement>

== Regularization and Training Diagnostics <appendix:place_diagnostics>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(a) Action Rate Penalty],
        plc("tensegrity_pd", "Info_Episode_Reward_action_rate"),
        plc("tensegrity_tendon", "Info_Episode_Reward_action_rate"),
        plc("tensegrity_physical", "Info_Episode_Reward_action_rate"),
        y-label: [Penalty], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(b) Joint Velocity Penalty],
        plc("tensegrity_pd", "Info_Episode_Reward_joint_vel"),
        plc("tensegrity_tendon", "Info_Episode_Reward_joint_vel"),
        plc("tensegrity_physical", "Info_Episode_Reward_joint_vel"),
        y-label: [Penalty], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(c) Joint Torque Penalty],
        plc("tensegrity_pd", "Info_Episode_Reward_joint_torque"),
        plc("tensegrity_tendon", "Info_Episode_Reward_joint_torque"),
        plc("tensegrity_physical", "Info_Episode_Reward_joint_torque"),
        y-label: [Penalty], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(d) Belt Contact Penalty],
        plc("tensegrity_pd", "Info_Episode_Reward_belt_contact"),
        plc("tensegrity_tendon", "Info_Episode_Reward_belt_contact"),
        plc("tensegrity_physical", "Info_Episode_Reward_belt_contact"),
        y-label: [Penalty], x-max: place-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Cube place regularization terms. *(a)*~Action rate penalty (ramps at 200k). *(b)*~Joint velocity penalty (ramps at 200k). *(c)*~Joint torque penalty. *(d)*~Belt contact penalty.],
  short-caption: [Cube place regularization terms],
) <fig:appendix_place_regularization>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(a) Policy Loss],
        plc("tensegrity_pd", "Loss_Policyloss"),
        plc("tensegrity_tendon", "Loss_Policyloss"),
        plc("tensegrity_physical", "Loss_Policyloss"),
        y-label: [Loss], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(b) Value Loss],
        plc("tensegrity_pd", "Loss_Valueloss"),
        plc("tensegrity_tendon", "Loss_Valueloss"),
        plc("tensegrity_physical", "Loss_Valueloss"),
        y-label: [Loss], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(c) Entropy Loss],
        plc("tensegrity_pd", "Loss_Entropyloss"),
        plc("tensegrity_tendon", "Loss_Entropyloss"),
        plc("tensegrity_physical", "Loss_Entropyloss"),
        y-label: [Loss], x-max: place-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-3v(title: [(d) Learning Rate],
        plc("tensegrity_pd", "Learning_Learningrate"),
        plc("tensegrity_tendon", "Learning_Learningrate"),
        plc("tensegrity_physical", "Learning_Learningrate"),
        y-label: [LR], x-max: place-xmax,
      )
    }),
    variant-legend,
  ),
  caption: [Cube place training diagnostics. *(a)*~Policy loss. *(b)*~Value function loss. *(c)*~Entropy loss. *(d)*~KL-adaptive learning rate schedule.],
  short-caption: [Cube place training diagnostics],
) <fig:appendix_place_diagnostics>

// ═════════════════════════════════════════════════════════════════════════════
= Appendix: Cube Sort Task — Preliminary Training Metrics <appendix:sort_metrics>
// ═════════════════════════════════════════════════════════════════════════════

This appendix documents the preliminary cube sort training metrics using the PD tensegrity variant (20k timesteps, training ongoing). Only the PD variant has been trained on this task.

// ── Data loading ────────────────────────────────────────────────────────────
#let srt(metric) = parse-csv(csv(rl-base + "cube_sort/tensegrity_pd/" + metric + ".csv"))

== Reward Overview <appendix:sort_overview>

#faps-figure(
  stack(dir: ttb, spacing: 8pt,
    grid(
      columns: (auto, auto),
      column-gutter: 12pt,
      canvas(length: 1cm, {
        import draw: *
        plot.plot(
          size: (reward-pw, reward-h),
          x-label: [Timesteps], y-label: [Reward],
          x-min: 0, x-max: sort-xmax,
          x-grid: true, y-grid: true,
          legend: none,
          {
            plot.add(srt("Reward_Totalrewardmean"),
              style: (stroke: (paint: plot-blue, thickness: 1.2pt)))
            plot.add(srt("Reward_Totalrewardmax"),
              style: (stroke: (paint: plot-blue, thickness: 0.6pt, dash: "dashed")))
            plot.add(srt("Reward_Totalrewardmin"),
              style: (stroke: (paint: plot-blue, thickness: 0.6pt, dash: "dotted")))
          }
        )
      }),
      reward-side-legend-single,
    ),
    canvas(length: 1cm, {
      stacked-plot-1v(
        title: [Policy Standard Deviation],
        srt("Policy_Standarddeviation"),
        y-label: [$sigma$], x-max: sort-xmax,
      )
    }),
  ),
  caption: [Cube sort reward overview (PD variant). Top: total reward mean (solid), max (dashed), and min (dotted). Bottom: policy standard deviation ($sigma$).],
  short-caption: [Cube sort reward overview],
) <fig:appendix_sort_metrics>

== Task Metrics <appendix:sort_task_metrics>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(a) Green Grasp Rate],
        srt("Info_Metrics_green_grasp_rate"),
        y-label: [Rate], y-min: 0, y-max: 1.0,
        x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(b) Green Placement Rate],
        srt("Info_Metrics_green_placement_rate"),
        y-label: [Rate], y-min: 0, y-max: 1.0,
        x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(c) Green Miss Rate],
        srt("Info_Metrics_green_miss_rate"),
        y-label: [Rate], y-min: 0, y-max: 1.0,
        x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(d) Mean Episode Length],
        srt("Info_Metrics_mean_episode_length"),
        y-label: [Steps], x-max: sort-xmax,
      )
    }),
  ),
  caption: [Cube sort task metrics (PD variant). *(a)*~Green cube grasp rate. *(b)*~Green cube placement rate. *(c)*~Green cube miss rate. *(d)*~Mean episode length.],
  short-caption: [Cube sort task metrics],
) <fig:appendix_sort_task_metrics>

== Reward Components <appendix:sort_components>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(a) Reaching Object],
        srt("Info_Episode_Reward_reaching_object"),
        y-label: [Reward], x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(b) Grasping],
        srt("Info_Episode_Reward_grasping"),
        y-label: [Reward], x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(c) Lifting Object],
        srt("Info_Episode_Reward_lifting_object"),
        y-label: [Reward], x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(d) Goal Tracking],
        srt("Info_Episode_Reward_goal_tracking"),
        y-label: [Reward], x-max: sort-xmax,
      )
    }),
  ),
  caption: [Cube sort reward components (PD variant). *(a)*~Approach shaping. *(b)*~Grasp reward. *(c)*~Lifting reward. *(d)*~Goal tracking shaping.],
  short-caption: [Cube sort reward components],
) <fig:appendix_sort_components>

== Training Diagnostics <appendix:sort_diagnostics>

#faps-figure(
  stack(dir: ttb, spacing: 4pt,
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(a) Policy Loss],
        srt("Loss_Policyloss"),
        y-label: [Loss], x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(b) Value Loss],
        srt("Loss_Valueloss"),
        y-label: [Loss], x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(c) Entropy Loss],
        srt("Loss_Entropyloss"),
        y-label: [Loss], x-max: sort-xmax, show-x-label: false,
      )
    }),
    canvas(length: 1cm, {
      stacked-plot-1v(title: [(d) Learning Rate],
        srt("Learning_Learningrate"),
        y-label: [LR], x-max: sort-xmax,
      )
    }),
  ),
  caption: [Cube sort training diagnostics (PD variant). *(a)*~Policy loss. *(b)*~Value function loss. *(c)*~Entropy loss. *(d)*~KL-adaptive learning rate schedule.],
  short-caption: [Cube sort training diagnostics],
) <fig:appendix_sort_diagnostics>

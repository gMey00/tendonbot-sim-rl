// Chapter 5: Results
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *
#import "../../../shared/formatting/template.typ": faps-figure, faps-table
#import "../../../shared/formatting/plots.typ": *
#import "../../../shared/formatting/colors.typ": *

= Results <ch:results>

This chapter presents the quantitative and qualitative results of the model validation and #ac("RL") experiments. All results are produced by the automated training and plotting pipeline described in @ch:methodology.

== Model Validation Results <sec:validation_results>

=== Step-Response Analysis <subsec:step_results>

@tab:nrmse_comparison compares the NRMSE values obtained by the Isaac Sim PD and tendon-driven models against Klein's Gazebo-vs-real-robot reference values~@Klein2023. Both Isaac Sim models achieve comparable or better NRMSE across all joints and amplitudes. The PD model (using the tuned `ImplicitActuator` with $K = 400$, $D = 20$) exhibits the lowest NRMSE due to its direct joint-level control, while the tendon model introduces the physically realistic cable-mediated dynamics at a modest accuracy cost.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint*], [*Amplitude*], [*Klein (2023)*], [*PD (Isaac)*], [*Tendon (Isaac)*]),
    [`elbow_joint`], [$20 degree$], [39.1~%], [9.5~%], [12.9~%],
    [`elbow_joint`], [$30 degree$], [11.7~%], [9.5~%], [15.0~%],
    [`elbow_joint`], [$40 degree$], [12.1~%], [9.5~%], [17.0~%],
    [`wrist_y_joint`], [$10 degree$], [19.2~%], [8.3~%], [9.7~%],
    [`wrist_y_joint`], [$20 degree$], [11.6~%], [8.3~%], [12.4~%],
    [`wrist_y_joint`], [$30 degree$], [12.5~%], [8.3~%], [14.8~%],
    [`wrist_x_joint`], [$10 degree$], [25.5~%], [8.3~%], [9.7~%],
    [`wrist_x_joint`], [$20 degree$], [9.9~%], [8.3~%], [12.4~%],
    [`wrist_x_joint`], [$30 degree$], [11.2~%], [8.3~%], [14.8~%],
  ),
  caption: [NRMSE comparison across actuation modes. Klein~(2023) values are Gazebo-vs-real-robot from Table~4.2. PD and Tendon values are Isaac Sim disc-approximation model. Both Isaac Sim models match or outperform the Gazebo reference.],
  short-caption: [NRMSE comparison across actuation modes],
) <tab:nrmse_comparison>

@fig:step_responses shows representative step-response time traces for the elbow and wrist joints under PD and tendon control. The PD model settles rapidly with near-critical damping ($zeta approx 1.07$), while the tendon model exhibits a marginally slower rise due to the cable tension dynamics. Both models track the commanded step accurately and return to zero without residual offset.

// ── Data loading (step responses) ───────────────────────────────────────────
#let data-base = "../../../shared/data/"

// Load raw CSVs (resolved relative to this file)
#let raw-pd-e20  = csv(data-base + "step_pd_elbow_20.csv")
#let raw-pd-e30  = csv(data-base + "step_pd_elbow_30.csv")
#let raw-pd-e40  = csv(data-base + "step_pd_elbow_40.csv")
#let raw-tn-e20  = csv(data-base + "step_tendon_elbow_20.csv")
#let raw-tn-e30  = csv(data-base + "step_tendon_elbow_30.csv")
#let raw-tn-e40  = csv(data-base + "step_tendon_elbow_40.csv")
#let raw-pd-w10  = csv(data-base + "step_pd_wrist_y_10.csv")
#let raw-pd-w20  = csv(data-base + "step_pd_wrist_y_20.csv")
#let raw-pd-w30  = csv(data-base + "step_pd_wrist_y_30.csv")
#let raw-tn-w10  = csv(data-base + "step_tendon_wrist_y_10.csv")
#let raw-tn-w20  = csv(data-base + "step_tendon_wrist_y_20.csv")
#let raw-tn-w30  = csv(data-base + "step_tendon_wrist_y_30.csv")

// Parse into (time, actual) and (time, setpoint) arrays
#let pd-e20  = parse-csv(raw-pd-e20)
#let pd-e20s = parse-csv-setpoint(raw-pd-e20)
#let pd-e30  = parse-csv(raw-pd-e30)
#let pd-e30s = parse-csv-setpoint(raw-pd-e30)
#let pd-e40  = parse-csv(raw-pd-e40)
#let pd-e40s = parse-csv-setpoint(raw-pd-e40)

#let tn-e20  = parse-csv(raw-tn-e20)
#let tn-e20s = parse-csv-setpoint(raw-tn-e20)
#let tn-e30  = parse-csv(raw-tn-e30)
#let tn-e30s = parse-csv-setpoint(raw-tn-e30)
#let tn-e40  = parse-csv(raw-tn-e40)
#let tn-e40s = parse-csv-setpoint(raw-tn-e40)

#let pd-w10  = parse-csv(raw-pd-w10)
#let pd-w10s = parse-csv-setpoint(raw-pd-w10)
#let pd-w20  = parse-csv(raw-pd-w20)
#let pd-w20s = parse-csv-setpoint(raw-pd-w20)
#let pd-w30  = parse-csv(raw-pd-w30)
#let pd-w30s = parse-csv-setpoint(raw-pd-w30)

#let tn-w10  = parse-csv(raw-tn-w10)
#let tn-w10s = parse-csv-setpoint(raw-tn-w10)
#let tn-w20  = parse-csv(raw-tn-w20)
#let tn-w20s = parse-csv-setpoint(raw-tn-w20)
#let tn-w30  = parse-csv(raw-tn-w30)
#let tn-w30s = parse-csv-setpoint(raw-tn-w30)

// ── Helper: single step-response subplot ────────────────────────────────────
#let step-subplot(
  title: none,
  actual-data,
  setpoint-data,
  amplitudes,
  colors: amp-colors,
  x-max: 4.0,
  y-max: 45,
  y-step: 10,
) = {
  import draw: *

  plot.plot(
    size: (plot-half-width, plot-small-height),
    x-label: [Time (s)],
    y-label: [Angle (deg)],
    x-min: 0, x-max: x-max,
    y-min: -2, y-max: y-max,
    x-tick-step: 1.0,
    y-tick-step: y-step,
    x-grid: true,
    y-grid: true,
    legend: "inner-north-east",
    legend-style: (item: (spacing: 0.15), padding: 0.1),
    plot-style: (idx) => {
      // First 3 = setpoints (dashed), next 3 = actuals (solid)
      let base-idx = calc.rem(idx, 3)
      let c = colors.at(base-idx)
      if idx < 3 {
        (stroke: (paint: c, dash: "dashed", thickness: 0.6pt))
      } else {
        (stroke: (paint: c, thickness: 1.0pt))
      }
    },
    {
      // Setpoints (no legend label)
      for i in range(3) {
        plot.add(setpoint-data.at(i))
      }
      // Actuals (with legend)
      for i in range(3) {
        let amp = amplitudes.at(i)
        plot.add(actual-data.at(i), label: [#amp])
      }
    }
  )

  if title != none {
    content((plot-half-width / 2, plot-small-height + 0.35), text(size: 8pt, weight: "bold", title))
  }
}

#faps-figure(
  grid(
    columns: 2,
    column-gutter: 12pt,
    row-gutter: 16pt,
    // (a) PD elbow
    canvas(length: 1cm, {
      step-subplot(
        title: [(a) PD — Elbow Joint],
        (pd-e20, pd-e30, pd-e40),
        (pd-e20s, pd-e30s, pd-e40s),
        ([20°], [30°], [40°]),
      )
    }),
    // (b) Tendon elbow
    canvas(length: 1cm, {
      step-subplot(
        title: [(b) Tendon — Elbow Joint],
        (tn-e20, tn-e30, tn-e40),
        (tn-e20s, tn-e30s, tn-e40s),
        ([20°], [30°], [40°]),
      )
    }),
    // (c) PD wrist Y
    canvas(length: 1cm, {
      step-subplot(
        title: [(c) PD — Wrist Pitch],
        (pd-w10, pd-w20, pd-w30),
        (pd-w10s, pd-w20s, pd-w30s),
        ([10°], [20°], [30°]),
        y-max: 35,
      )
    }),
    // (d) Tendon wrist Y
    canvas(length: 1cm, {
      step-subplot(
        title: [(d) Tendon — Wrist Pitch],
        (tn-w10, tn-w20, tn-w30),
        (tn-w10s, tn-w20s, tn-w30s),
        ([10°], [20°], [30°]),
        y-max: 35,
      )
    }),
  ),
  caption: [Step-response time traces for the elbow and wrist pitch joints under PD and tendon actuation. Dashed lines show the commanded setpoint; solid lines show the measured joint angle. *(a)*~PD elbow ($K = 400$, $D = 20$); *(b)*~tendon-driven elbow ($K_p = 50$, $K_i = 4$, $K_d = 2$); *(c)*~PD wrist pitch; *(d)*~tendon-driven wrist pitch ($K_p = 10$, $K_i = 1.5$, $K_d = 0.6$). All configurations settle accurately with no residual offset.],
  short-caption: [Step-response time traces for elbow and wrist joints],
) <fig:step_responses>

The physical-linkage tendon variant uses body forces on the four-bar mechanism instead of the single-DOF elbow approximation. @tab:physical_metrics summarizes its performance, confirming that all joints settle within 300~ms across all amplitudes.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint*], [*Amplitude*], [*Rise (ms)*], [*Settle (ms)*], [*NRMSE*]),
    [`elbow_physical`], [$20 degree$], [92], [133], [12.7~%],
    [`elbow_physical`], [$30 degree$], [108], [167], [13.1~%],
    [`elbow_physical`], [$40 degree$], [117], [183], [14.0~%],
    [`wrist_y_joint`], [$10 degree$], [83], [150], [9.7~%],
    [`wrist_y_joint`], [$20 degree$], [100], [183], [12.4~%],
    [`wrist_y_joint`], [$30 degree$], [142], [217], [14.9~%],
    [`wrist_x_joint`], [$10 degree$], [83], [283], [10.1~%],
    [`wrist_x_joint`], [$20 degree$], [100], [192], [12.6~%],
    [`wrist_x_joint`], [$30 degree$], [133], [225], [14.9~%],
  ),
  caption: [Step-response metrics for the physical antiparallelogram tendon variant ($K_p = 75$, $K_i = 6$, $K_d = 3$ for elbow). All joints settle within 300~ms. NRMSE values are comparable to the disc-approximation variant, confirming correct four-bar linkage dynamics.],
  short-caption: [Physical-linkage tendon step-response metrics],
) <tab:physical_metrics>

@fig:nrmse_comparison and @fig:settling_comparison provide cross-variant visual summaries of the NRMSE and settling time distributions.

#import "@preview/cetz-plot:0.1.1": chart

#faps-figure(
  canvas(length: 1cm, {
    import draw: *

    chart.columnchart(
      size: (plot-full-width, plot-height),
      label-key: 0,
      value-key: (1, 2, 3),
      mode: "clustered",
      bar-style: (idx) => {
        let colors = (plot-amber, plot-blue, plot-green)
        (stroke: none, fill: colors.at(idx).transparentize(15%))
      },
      x-label: none,
      y-label: [NRMSE (\%)],
      labels: ([Klein (2023)], [PD (Isaac)], [Tendon (Isaac)]),
      (
        ([Elbow 20°],  39.1,   9.5,  12.9),
        ([Elbow 30°],  11.7,   9.5,  15.0),
        ([Elbow 40°],  12.1,   9.5,  17.0),
        ([Wrist Y 10°], 19.2,  8.3,   9.7),
        ([Wrist Y 20°], 11.6,  8.3,  12.4),
        ([Wrist Y 30°], 12.5,  8.3,  14.8),
        ([Wrist X 10°], 25.5,  8.3,   9.7),
        ([Wrist X 20°],  9.9,  8.3,  12.4),
        ([Wrist X 30°], 11.2,  8.3,  14.8),
      ),
    )
  }),
  caption: [NRMSE comparison across all joints and actuation modes. Both Isaac Sim models (PD and tendon) consistently outperform or match the Klein~(2023) Gazebo reference. The PD model achieves the lowest NRMSE due to direct joint-level actuation.],
  short-caption: [NRMSE comparison across actuation modes],
) <fig:nrmse_comparison>

#faps-figure(
  canvas(length: 1cm, {
    import draw: *

    chart.columnchart(
      size: (plot-full-width, plot-height),
      label-key: 0,
      value-key: (1, 2),
      mode: "clustered",
      bar-style: (idx) => {
        let colors = (plot-blue, plot-green)
        (stroke: none, fill: colors.at(idx).transparentize(15%))
      },
      x-label: none,
      y-label: [Settling Time (ms)],
      labels: ([PD ($D = 20$)], [Tendon]),
      (
        ([Elbow 20°],  108,  142),
        ([Elbow 30°],  117,  192),
        ([Elbow 40°],  117,  233),
        ([Wrist Y 10°], 100, 150),
        ([Wrist Y 20°], 100, 183),
        ([Wrist Y 30°], 100, 225),
        ([Wrist X 10°], 100, 158),
        ([Wrist X 20°], 100, 183),
        ([Wrist X 30°], 100, 225),
      ),
    )
  }),
  caption: [Settling time ($plus.minus 2%$ band) across all joints and actuation modes. All configurations settle within 300~ms, meeting the design target of 0.10--0.30~s.],
  short-caption: [Settling time comparison],
) <fig:settling_comparison>

*Damping sweep.* @fig:damping_sweep illustrates the effect of the PD damping parameter on the elbow step-response. The sweep from $D = 120$ (heavily overdamped) to $D = 15$ (slightly underdamped) confirms $D = 20$ as the optimal trade-off between settling speed and overshoot, consistent with the analytical prediction ($D_"crit" approx 18.7$).

// ── Damping sweep data ──────────────────────────────────────────────────────
#let raw-d120 = csv(data-base + "damping_D120.csv")
#let raw-d60  = csv(data-base + "damping_D60.csv")
#let raw-d30  = csv(data-base + "damping_D30.csv")
#let raw-d25  = csv(data-base + "damping_D25.csv")
#let raw-d20  = csv(data-base + "damping_D20.csv")
#let raw-d15  = csv(data-base + "damping_D15.csv")

#let d120 = parse-csv(raw-d120)
#let d60  = parse-csv(raw-d60)
#let d30  = parse-csv(raw-d30)
#let d25  = parse-csv(raw-d25)
#let d20  = parse-csv(raw-d20)
#let d15  = parse-csv(raw-d15)

#faps-figure(
  canvas(length: 1cm, {
    import draw: *

    plot.plot(
      size: (plot-full-width, plot-height + 0.5),
      x-label: [Time (s)],
      y-label: [Angle (deg)],
      x-min: 0, x-max: 4.0,
      y-min: -2, y-max: 35,
      x-tick-step: 0.5,
      y-tick-step: 5,
      x-grid: true,
      y-grid: true,
      legend: "inner-north-east",
      legend-style: (item: (spacing: 0.15), padding: 0.1),
      {
        // Setpoint reference (thin black dashed)
        plot.add(parse-csv-setpoint(raw-d20),
          label: [Setpoint],
          style: (stroke: (paint: luma(40), dash: "dashed", thickness: 0.6pt)))

        // D=120 (overdamped)
        plot.add(d120, label: [$D = 120$],
          style: (stroke: (paint: damping-colors.at(0), thickness: 0.8pt)))
        // D=60
        plot.add(d60, label: [$D = 60$],
          style: (stroke: (paint: damping-colors.at(1), thickness: 0.8pt)))
        // D=30
        plot.add(d30, label: [$D = 30$],
          style: (stroke: (paint: damping-colors.at(2), thickness: 0.8pt)))
        // D=25
        plot.add(d25, label: [$D = 25$],
          style: (stroke: (paint: damping-colors.at(3), thickness: 0.8pt)))
        // D=20 (optimal, thicker)
        plot.add(d20, label: [$D = 20$ (opt.)],
          style: (stroke: (paint: damping-colors.at(4), thickness: 1.4pt)))
        // D=15 (underdamped)
        plot.add(d15, label: [$D = 15$],
          style: (stroke: (paint: damping-colors.at(5), thickness: 0.8pt)))
      }
    )
  }),
  caption: [PD damping sweep for the elbow joint ($K = 400$ fixed, 30° step). Each curve corresponds to a different damping value. $D = 20$ (near-critically damped, $zeta approx 1.07$) achieves the fastest settling without significant overshoot. $D = 120$ (original) is heavily overdamped.],
  short-caption: [PD damping sweep for elbow joint],
) <fig:damping_sweep>

*Base joint validation.* The two prismatic base joints use PD control ($K = 8000$, $D = 800$) with gravitational feedforward for the $Z$ axis. @tab:base_metrics summarizes the results: settling times are below 200~ms and steady-state errors are negligible (${<} 0.003 "mm"$).

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint*], [*Amplitude*], [*Rise (ms)*], [*Settle (ms)*], [*NRMSE*], [*SS Error*]),
    [`base_y_joint`], [50~mm], [92], [183], [11.2~%], [0.002~mm],
    [`base_y_joint`], [100~mm], [92], [183], [11.2~%], [0.002~mm],
    [`base_y_joint`], [200~mm], [92], [183], [11.3~%], [0.001~mm],
    [`base_z_joint`], [30~mm], [100], [183], [11.2~%], [0.001~mm],
    [`base_z_joint`], [60~mm], [100], [183], [11.2~%], [0.001~mm],
    [`base_z_joint`], [100~mm], [100], [183], [11.3~%], [0.001~mm],
  ),
  caption: [Step-response metrics for the prismatic base joints ($K = 8000$, $D = 800$). Steady-state errors are negligible across all amplitudes, confirming accurate gravitational feedforward compensation on the $Z$ axis.],
  short-caption: [Base joint step-response metrics],
) <tab:base_metrics>

=== Workspace Analysis <subsec:workspace_results>

The Monte Carlo workspace analysis used $N = 2 000 000$ FK samples with the disc-approximation arm variant, ceiling-mounted at 2.30~m, across 4,096 parallel environments. @tab:workspace_stats summarizes the key statistics.

#faps-table(
  table(
    columns: (auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Metric*], [*Value*]),
    [Total FK samples], [$2 000 000$],
    [Samples inside desired workspace], [$701 472$ (35.1~%)],
    [Desired workspace coverage], [99.72~% ($33 904 "/"  34 000$ voxels)],
    [Yoshikawa manipulability (mean, in WS)], [$5.2 times 10^(-5)$],
    [Yoshikawa manipulability (median, in WS)], [$1 times 10^(-6)$],
    [Yoshikawa manipulability (p95, in WS)], [$1.91 times 10^(-4)$],
    [Voxel size], [$Delta = 0.02 "m"$],
  ),
  caption: [Monte Carlo workspace analysis statistics for the disc-approximation model. The desired workspace volume ($0.40 times 1.35 times 0.50 "m"^3$, derived from the Klein~(2023) task geometry) is covered to 99.7~%, confirming that the ceiling-mount height and base positioning provide adequate reach for all target tasks.],
  short-caption: [Workspace analysis statistics],
) <tab:workspace_stats>

@fig:workspace_metrics presents the three workspace quality metrics defined in @sec:workspace_analysis_sampling_metrics: reachability density, Yoshikawa manipulability, and inverse condition number. Each subplot shows a 3D voxelised scatter plot alongside three orthogonal cross-section heatmaps through the center of the desired workspace.

#faps-figure(
  grid(
    columns: 1,
    row-gutter: 1em,
    [*(a)* Reachability density],
    image("../../../shared/figures/tensegrity_workspace_density.png", width: 80%),
    [*(b)* Yoshikawa manipulability],
    image("../../../shared/figures/tensegrity_workspace_manipulability.png", width: 80%),
    [*(c)* Inverse condition number],
    image("../../../shared/figures/tensegrity_workspace_condition.png", width: 80%),
  ),
  caption: [Workspace quality metrics for the disc-approximation model (2M FK samples, 4096 parallel envs). *(a)*~Reachability density: high density near the workspace center indicates many joint configurations map to the task-relevant region. *(b)*~Yoshikawa manipulability: highest near the center, decreasing toward boundaries. The low absolute values ($cal(O)(10^(-5))$) reflect the rank-deficient $6 times 5$ Jacobian rather than indicating poor dexterity within the robot's 5-dimensional motion capability. *(c)*~Inverse condition number ($kappa^(-1) = sigma_min "/" sigma_max$): near-zero values ($cal(O)(10^(-8))$) throughout confirm the expected rank deficiency of the non-square Jacobian; the spatial variation nevertheless reveals regions of relatively better kinematic isotropy.],
  short-caption: [Workspace quality metrics (density, manipulability, condition)],
) <fig:workspace_metrics>

The disc-approximation model achieves near-complete coverage (99.7~%) of the desired workspace. As discussed in @subsec:disc_approx and @fig:elbow_comparison, this represents a conservative estimate: the physical antiparallelogram elbow can reach farther than the disc model at equivalent joint angles, so the true coverage of the physical robot is expected to be at least as high.

== Reach Task Results <sec:reach_results>

// ── Reach task data loading ─────────────────────────────────────────────────
#let rl-base = data-base + "rl_training/"

// Reach — Total reward (mean)
#let reach-pd-reward   = parse-csv(csv(rl-base + "reach/tensegrity_pd/Reward_Totalrewardmean.csv"))
#let reach-tn-reward   = parse-csv(csv(rl-base + "reach/tensegrity_tendon/Reward_Totalrewardmean.csv"))
#let reach-ph-reward   = parse-csv(csv(rl-base + "reach/tensegrity_physical/Reward_Totalrewardmean.csv"))

// Reach — Position tracking
#let reach-pd-pos   = parse-csv(csv(rl-base + "reach/tensegrity_pd/Info_Episode_Reward_end_effector_position_tracking.csv"))
#let reach-tn-pos   = parse-csv(csv(rl-base + "reach/tensegrity_tendon/Info_Episode_Reward_end_effector_position_tracking.csv"))
#let reach-ph-pos   = parse-csv(csv(rl-base + "reach/tensegrity_physical/Info_Episode_Reward_end_effector_position_tracking.csv"))

// Reach — Orientation tracking
#let reach-pd-orient = parse-csv(csv(rl-base + "reach/tensegrity_pd/Info_Episode_Reward_end_effector_orientation_tracking.csv"))
#let reach-tn-orient = parse-csv(csv(rl-base + "reach/tensegrity_tendon/Info_Episode_Reward_end_effector_orientation_tracking.csv"))
#let reach-ph-orient = parse-csv(csv(rl-base + "reach/tensegrity_physical/Info_Episode_Reward_end_effector_orientation_tracking.csv"))

// Reach — Fine-grained position
#let reach-pd-fine  = parse-csv(csv(rl-base + "reach/tensegrity_pd/Info_Episode_Reward_end_effector_position_tracking_fine_grained.csv"))
#let reach-tn-fine  = parse-csv(csv(rl-base + "reach/tensegrity_tendon/Info_Episode_Reward_end_effector_position_tracking_fine_grained.csv"))
#let reach-ph-fine  = parse-csv(csv(rl-base + "reach/tensegrity_physical/Info_Episode_Reward_end_effector_position_tracking_fine_grained.csv"))

// Reach — Policy std deviation
#let reach-pd-std   = parse-csv(csv(rl-base + "reach/tensegrity_pd/Policy_Standarddeviation.csv"))
#let reach-tn-std   = parse-csv(csv(rl-base + "reach/tensegrity_tendon/Policy_Standarddeviation.csv"))
#let reach-ph-std   = parse-csv(csv(rl-base + "reach/tensegrity_physical/Policy_Standarddeviation.csv"))

// ── Helper: multi-variant training curve subplot ────────────────────────────
#let variant-subplot(
  title: none,
  pd-data, tn-data, ph-data,
  y-label: none,
  y-min: auto, y-max: auto,
  y-tick-step: auto,
  x-max: auto,
  legend-pos: "inner-north-east",
  width: plot-half-width,
  height: plot-small-height,
) = {
  import draw: *

  plot.plot(
    size: (width, height),
    x-label: [Timesteps],
    y-label: y-label,
    x-min: 0, x-max: x-max,
    y-min: y-min, y-max: y-max,
    x-tick-step: auto,
    y-tick-step: y-tick-step,
    x-grid: true,
    y-grid: true,
    legend: legend-pos,
    legend-style: (item: (spacing: 0.15), padding: 0.1),
    {
      plot.add(pd-data, label: [PD],
        style: (stroke: (paint: plot-blue, thickness: 1.0pt)))
      plot.add(tn-data, label: [Tendon],
        style: (stroke: (paint: plot-green, thickness: 1.0pt)))
      plot.add(ph-data, label: [Physical],
        style: (stroke: (paint: plot-purple, thickness: 1.0pt)))
    }
  )

  if title != none {
    content((width / 2, height + 0.35), text(size: 8pt, weight: "bold", title))
  }
}

=== Training Convergence <subsec:reach_convergence>

@fig:reach_convergence compares the training convergence of the three tensegrity variants on the reach task. The PD and tendon variants, trained for 48k timesteps, converge rapidly to total rewards above $+0.75$, with the PD model reaching $+0.79$ and the tendon model $+0.76$. The physical tendon variant, trained for 150k timesteps with the dual-robot FK sampling architecture (@ch:methodology), converges more slowly to $-0.46$, reflecting the inherent difficulty of effort-based cable control through the four-bar linkage.

#faps-figure(
  grid(
    columns: 2,
    column-gutter: 12pt,
    row-gutter: 16pt,
    // (a) Total reward
    canvas(length: 1cm, {
      variant-subplot(
        title: [(a) Total Reward (Mean)],
        reach-pd-reward, reach-tn-reward, reach-ph-reward,
        y-label: [Reward],
        y-min: -2.5, y-max: 1.0,
      )
    }),
    // (b) Position tracking error
    canvas(length: 1cm, {
      variant-subplot(
        title: [(b) Position Tracking],
        reach-pd-pos, reach-tn-pos, reach-ph-pos,
        y-label: [Reward],
        legend-pos: "inner-south-east",
      )
    }),
    // (c) Orientation tracking error
    canvas(length: 1cm, {
      variant-subplot(
        title: [(c) Orientation Tracking],
        reach-pd-orient, reach-tn-orient, reach-ph-orient,
        y-label: [Reward],
        legend-pos: "inner-south-east",
      )
    }),
    // (d) Policy standard deviation
    canvas(length: 1cm, {
      variant-subplot(
        title: [(d) Policy Standard Deviation],
        reach-pd-std, reach-tn-std, reach-ph-std,
        y-label: [$sigma$],
        legend-pos: "inner-north-east",
      )
    }),
  ),
  caption: [Reach task training convergence for the three tensegrity variants. *(a)*~Mean total reward over training timesteps. *(b)*~Position tracking reward component. *(c)*~Orientation tracking reward component. *(d)*~Policy standard deviation ($sigma$), indicating exploration level. PD and tendon variants converge within 48k steps; the physical tendon variant requires 150k steps due to the effort-based control indirection.],
  short-caption: [Reach task training convergence],
) <fig:reach_convergence>

=== Final Performance <subsec:reach_performance>

@tab:reach_results summarizes the final performance metrics from the reach task training runs. The PD and tendon variants achieve comparable performance, with position tracking errors below 1~cm. The physical tendon variant achieves 2~cm position error, limited by the cable tension control indirection inherent to the four-bar linkage mechanism.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Variant*], [*Steps*], [*Total Reward*], [*Pos. Tracking*], [*Orient. Tracking*], [*Fine-Grained*], [*Policy $sigma$*]),
    [Tensegrity PD], [48k], [$+0.79$], [$-0.006$], [$-0.009$], [$0.087$], [$0.020$],
    [Tensegrity Tendon], [48k], [$+0.76$], [$-0.007$], [$-0.012$], [$0.085$], [$0.028$],
    [Tensegrity Physical], [150k], [$-0.46$], [$-0.020$], [$-0.044$], [$0.041$], [$0.113$],
  ),
  caption: [Reach task final performance metrics for the three tensegrity variants. Position and orientation tracking are per-step reward components (higher = better). Fine-grained measures sub-centimeter accuracy. The physical variant's larger policy $sigma$ indicates it has not yet fully converged.],
  short-caption: [Reach task final performance metrics],
) <tab:reach_results>

=== Cross-Variant Comparison <subsec:reach_comparison>

@fig:reach_comparison provides a direct comparison of the final reach task metrics across all three variants. The PD and tendon-driven models perform almost identically on task metrics, while the physical tendon variant shows a clear performance gap, particularly in orientation tracking. This gap is attributable to the effort-based control through the antiparallelogram mechanism, which introduces a nonlinear mapping between actuator commands and joint motion.

#faps-figure(
  canvas(length: 1cm, {
    import draw: *

    chart.columnchart(
      size: (plot-full-width, plot-height),
      label-key: 0,
      value-key: (1, 2, 3),
      mode: "clustered",
      bar-style: (idx) => {
        let colors = (plot-blue, plot-green, plot-purple)
        (stroke: none, fill: colors.at(idx).transparentize(15%))
      },
      x-label: none,
      y-label: [Value],
      labels: ([PD], [Tendon], [Physical]),
      (
        ([Total Reward],    0.79,   0.76,  -0.46),
        ([Pos. Tracking],  -0.006, -0.007, -0.020),
        ([Orient. Track.], -0.009, -0.012, -0.044),
        ([Fine-Grained],    0.087,  0.085,  0.041),
      ),
    )
  }),
  caption: [Cross-variant comparison of final reach task performance. The PD and tendon models achieve nearly identical scores, while the physical tendon variant shows reduced accuracy due to the effort-based cable actuation through the four-bar linkage.],
  short-caption: [Cross-variant reach task comparison],
) <fig:reach_comparison>

== Cube Place Results <sec:place_results>

// ── Cube Place data loading ─────────────────────────────────────────────────
// Total reward (mean)
#let place-pd-reward   = parse-csv(csv(rl-base + "cube_place/tensegrity_pd/Reward_Totalrewardmean.csv"))
#let place-tn-reward   = parse-csv(csv(rl-base + "cube_place/tensegrity_tendon/Reward_Totalrewardmean.csv"))
#let place-ph-reward   = parse-csv(csv(rl-base + "cube_place/tensegrity_physical/Reward_Totalrewardmean.csv"))

// Grasp rate
#let place-pd-grasp   = parse-csv(csv(rl-base + "cube_place/tensegrity_pd/Info_Metrics_grasp_rate.csv"))
#let place-tn-grasp   = parse-csv(csv(rl-base + "cube_place/tensegrity_tendon/Info_Metrics_grasp_rate.csv"))
#let place-ph-grasp   = parse-csv(csv(rl-base + "cube_place/tensegrity_physical/Info_Metrics_grasp_rate.csv"))

// Place success rate
#let place-pd-place   = parse-csv(csv(rl-base + "cube_place/tensegrity_pd/Info_Metrics_place_success_rate.csv"))
#let place-tn-place   = parse-csv(csv(rl-base + "cube_place/tensegrity_tendon/Info_Metrics_place_success_rate.csv"))
#let place-ph-place   = parse-csv(csv(rl-base + "cube_place/tensegrity_physical/Info_Metrics_place_success_rate.csv"))

// Red on conveyor rate (safety)
#let place-pd-red     = parse-csv(csv(rl-base + "cube_place/tensegrity_pd/Info_Metrics_red_on_conveyor_rate.csv"))
#let place-tn-red     = parse-csv(csv(rl-base + "cube_place/tensegrity_tendon/Info_Metrics_red_on_conveyor_rate.csv"))
#let place-ph-red     = parse-csv(csv(rl-base + "cube_place/tensegrity_physical/Info_Metrics_red_on_conveyor_rate.csv"))

// Policy std
#let place-pd-std     = parse-csv(csv(rl-base + "cube_place/tensegrity_pd/Policy_Standarddeviation.csv"))
#let place-tn-std     = parse-csv(csv(rl-base + "cube_place/tensegrity_tendon/Policy_Standarddeviation.csv"))
#let place-ph-std     = parse-csv(csv(rl-base + "cube_place/tensegrity_physical/Policy_Standarddeviation.csv"))

=== Training Convergence <subsec:place_convergence>

@fig:place_convergence presents the training convergence for the cube place task across all three tensegrity variants. The PD and tendon variants both reach total rewards above $330$ within 300k timesteps, demonstrating successful learning of the full grasp-lift-place pipeline. The physical tendon variant fails to learn the grasping sub-task, achieving a mean reward of only $58.9$, consistent with its lower policy convergence on the reach task.

#faps-figure(
  grid(
    columns: 2,
    column-gutter: 12pt,
    row-gutter: 16pt,
    // (a) Total reward
    canvas(length: 1cm, {
      variant-subplot(
        title: [(a) Total Reward (Mean)],
        place-pd-reward, place-tn-reward, place-ph-reward,
        y-label: [Reward],
        y-min: -50, y-max: 400,
      )
    }),
    // (b) Grasp rate
    canvas(length: 1cm, {
      variant-subplot(
        title: [(b) Grasp Rate],
        place-pd-grasp, place-tn-grasp, place-ph-grasp,
        y-label: [Rate],
        y-min: 0, y-max: 1.0,
        legend-pos: "inner-south-east",
      )
    }),
    // (c) Place success
    canvas(length: 1cm, {
      variant-subplot(
        title: [(c) Place Success Rate],
        place-pd-place, place-tn-place, place-ph-place,
        y-label: [Rate],
        y-min: 0, y-max: 1.0,
        legend-pos: "inner-south-east",
      )
    }),
    // (d) Policy std dev
    canvas(length: 1cm, {
      variant-subplot(
        title: [(d) Policy Standard Deviation],
        place-pd-std, place-tn-std, place-ph-std,
        y-label: [$sigma$],
      )
    }),
  ),
  caption: [Cube place training convergence for the three tensegrity variants. *(a)*~Mean total reward over training timesteps. *(b)*~Grasp success rate. *(c)*~Place success rate (cube placed in target drum). *(d)*~Policy standard deviation. Both the PD and tendon variants converge to high task success within 300k steps; the physical tendon variant does not acquire the grasping skill.],
  short-caption: [Cube place training convergence],
) <fig:place_convergence>

=== Final Performance <subsec:place_final>

@tab:place_results summarizes the cube place task evaluation metrics at the end of training. The PD and tendon variants achieve grasp rates above $94%$ and place success rates above $90%$, while maintaining red cube safety above $95%$. The physical tendon variant does not learn to grasp, consistent with its difficulty in precise position control.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Variant*], [*Steps*], [*Total Reward*], [*Grasp Rate*], [*Place Success*], [*Red Safe*]),
    [Tensegrity PD], [300k], [$333.3$], [$98.7%$], [$90.6%$], [$98.0%$],
    [Tensegrity Tendon], [300k], [$332.9$], [$94.7%$], [$93.4%$], [$95.2%$],
    [Tensegrity Physical], [300k], [$58.9$], [$0.5%$], [$0.0%$], [$97.8%$],
  ),
  caption: [Cube place final performance metrics. Grasp Rate: fraction of episodes achieving a stable grasp. Place Success: fraction of episodes where the green cube is placed in the target drum. Red Safe: fraction where the red distractor remains on the conveyor.],
  short-caption: [Cube place final performance metrics],
) <tab:place_results>

=== Curriculum Effect <subsec:curriculum_effect>

@fig:curriculum_effect illustrates the effect of the two curriculum transitions on training progression for the PD variant. At $100"k"$ timesteps, the red distractor cube is introduced, causing a brief dip in reward as the policy encounters a new obstacle. At $200"k"$ timesteps, action rate and joint velocity penalties ramp up, promoting smoother trajectories at a slight cost to peak reward. Both transitions demonstrate the curriculum design working as intended: the policy first acquires core skills, then learns robustness.

// Reward breakdown showing curriculum transitions
#let place-pd-action   = parse-csv(csv(rl-base + "cube_place/tensegrity_pd/Info_Episode_Reward_action_rate.csv"))
#let place-pd-joint-vel = parse-csv(csv(rl-base + "cube_place/tensegrity_pd/Info_Episode_Reward_joint_vel.csv"))
#let place-pd-red-rew  = parse-csv(csv(rl-base + "cube_place/tensegrity_pd/Info_Episode_Reward_cube_off_conveyor.csv"))

#faps-figure(
  grid(
    columns: 2,
    column-gutter: 12pt,
    row-gutter: 16pt,
    // (a) Total reward with curriculum markers
    canvas(length: 1cm, {
      import draw: *

      plot.plot(
        size: (plot-half-width, plot-small-height),
        x-label: [Timesteps],
        y-label: [Reward],
        x-min: 0, x-max: auto,
        y-min: -50, y-max: 400,
        x-grid: true, y-grid: true,
        legend: "inner-south-east",
        legend-style: (item: (spacing: 0.15), padding: 0.1),
        {
          plot.add(place-pd-reward, label: [Total Reward],
            style: (stroke: (paint: plot-blue, thickness: 1.2pt)))
          // Curriculum stage markers
          plot.add-vline(100000, label: [Red cube $in$],
            style: (stroke: (paint: plot-red, thickness: 0.8pt, dash: "dashed")))
          plot.add-vline(200000, label: [Reg. ramp],
            style: (stroke: (paint: plot-amber, thickness: 0.8pt, dash: "dashed")))
        }
      )

      content((plot-half-width / 2, plot-small-height + 0.35), text(size: 8pt, weight: "bold", [(a) Total Reward — PD Variant]))
    }),
    // (b) Grasp and place rates with markers
    canvas(length: 1cm, {
      import draw: *

      plot.plot(
        size: (plot-half-width, plot-small-height),
        x-label: [Timesteps],
        y-label: [Rate],
        x-min: 0, x-max: auto,
        y-min: 0, y-max: 1.0,
        x-grid: true, y-grid: true,
        legend: "inner-south-east",
        legend-style: (item: (spacing: 0.15), padding: 0.1),
        {
          plot.add(place-pd-grasp, label: [Grasp Rate],
            style: (stroke: (paint: plot-blue, thickness: 1.0pt)))
          plot.add(place-pd-place, label: [Place Success],
            style: (stroke: (paint: plot-green, thickness: 1.0pt)))
          plot.add(place-pd-red, label: [Red Safe Rate],
            style: (stroke: (paint: plot-red, thickness: 1.0pt)))
          plot.add-vline(100000, style: (stroke: (paint: plot-red, thickness: 0.8pt, dash: "dashed")))
          plot.add-vline(200000, style: (stroke: (paint: plot-amber, thickness: 0.8pt, dash: "dashed")))
        }
      )

      content((plot-half-width / 2, plot-small-height + 0.35), text(size: 8pt, weight: "bold", [(b) Task Metrics — PD Variant]))
    }),
    // (c) Action rate penalty evolution
    canvas(length: 1cm, {
      import draw: *

      plot.plot(
        size: (plot-half-width, plot-small-height),
        x-label: [Timesteps],
        y-label: [Penalty],
        x-min: 0, x-max: auto,
        x-grid: true, y-grid: true,
        legend: "inner-south-west",
        legend-style: (item: (spacing: 0.15), padding: 0.1),
        {
          plot.add(place-pd-action, label: [Action Rate],
            style: (stroke: (paint: plot-amber, thickness: 1.0pt)))
          plot.add(place-pd-joint-vel, label: [Joint Velocity],
            style: (stroke: (paint: plot-teal, thickness: 1.0pt)))
          plot.add-vline(200000, style: (stroke: (paint: plot-amber, thickness: 0.8pt, dash: "dashed")))
        }
      )

      content((plot-half-width / 2, plot-small-height + 0.35), text(size: 8pt, weight: "bold", [(c) Regularization Penalties]))
    }),
    // (d) Red cube penalty
    canvas(length: 1cm, {
      import draw: *

      plot.plot(
        size: (plot-half-width, plot-small-height),
        x-label: [Timesteps],
        y-label: [Penalty],
        x-min: 0, x-max: auto,
        x-grid: true, y-grid: true,
        legend: "inner-south-west",
        legend-style: (item: (spacing: 0.15), padding: 0.1),
        {
          plot.add(place-pd-red-rew, label: [Cube Off Conveyor],
            style: (stroke: (paint: plot-red, thickness: 1.0pt)))
          plot.add-vline(100000, label: [Red cube $in$],
            style: (stroke: (paint: plot-red, thickness: 0.8pt, dash: "dashed")))
        }
      )

      content((plot-half-width / 2, plot-small-height + 0.35), text(size: 8pt, weight: "bold", [(d) Red Cube Penalty]))
    }),
  ),
  caption: [Effect of curriculum stages on the PD variant's cube place training. *(a)*~Total reward with dashed lines marking the red cube introduction ($100"k"$) and regularization ramp ($200"k"$). *(b)*~Task success rates showing brief transient dips at curriculum transitions. *(c)*~Regularization penalty terms ramping up after $200"k"$ steps. *(d)*~Red cube off-conveyor penalty activating after $100"k"$ steps.],
  short-caption: [Curriculum stages effect on training],
) <fig:curriculum_effect>

== Cube Sort Preliminary Results <sec:sort_results>

// ── Cube Sort data loading ──────────────────────────────────────────────────
#let sort-pd-reward      = parse-csv(csv(rl-base + "cube_sort/tensegrity_pd/Reward_Totalrewardmean.csv"))
#let sort-pd-grasp-rate  = parse-csv(csv(rl-base + "cube_sort/tensegrity_pd/Info_Metrics_green_grasp_rate.csv"))
#let sort-pd-place-rate  = parse-csv(csv(rl-base + "cube_sort/tensegrity_pd/Info_Metrics_green_placement_rate.csv"))
#let sort-pd-miss-rate   = parse-csv(csv(rl-base + "cube_sort/tensegrity_pd/Info_Metrics_green_miss_rate.csv"))
#let sort-pd-red-grab    = parse-csv(csv(rl-base + "cube_sort/tensegrity_pd/Info_Metrics_red_grabbed_count.csv"))

The cube sort task represents an extension of the cube place pipeline to sequential multi-object sorting from a conveyor belt. @fig:sort_preliminary shows preliminary training results using the PD variant only, as training is ongoing. The agent begins to learn approach and grasp behaviors within 77k timesteps, though reliable sorting has not yet been achieved, as indicated by the still-increasing reward trajectory.

#faps-figure(
  grid(
    columns: 2,
    column-gutter: 12pt,
    row-gutter: 16pt,
    // (a) Total reward
    canvas(length: 1cm, {
      import draw: *

      plot.plot(
        size: (plot-half-width, plot-small-height),
        x-label: [Timesteps],
        y-label: [Reward],
        x-min: 0, x-max: auto,
        x-grid: true, y-grid: true,
        legend: "inner-north-west",
        legend-style: (item: (spacing: 0.15), padding: 0.1),
        {
          plot.add(sort-pd-reward, label: [Total Reward],
            style: (stroke: (paint: plot-blue, thickness: 1.2pt)))
        }
      )

      content((plot-half-width / 2, plot-small-height + 0.35), text(size: 8pt, weight: "bold", [(a) Total Reward (Mean)]))
    }),
    // (b) Grasp and placement rates
    canvas(length: 1cm, {
      import draw: *

      plot.plot(
        size: (plot-half-width, plot-small-height),
        x-label: [Timesteps],
        y-label: [Rate],
        x-min: 0, x-max: auto,
        y-min: 0, y-max: 1.0,
        x-grid: true, y-grid: true,
        legend: "inner-north-west",
        legend-style: (item: (spacing: 0.15), padding: 0.1),
        {
          plot.add(sort-pd-grasp-rate, label: [Green Grasp],
            style: (stroke: (paint: plot-green, thickness: 1.0pt)))
          plot.add(sort-pd-place-rate, label: [Green Place],
            style: (stroke: (paint: plot-blue, thickness: 1.0pt)))
          plot.add(sort-pd-miss-rate, label: [Green Miss],
            style: (stroke: (paint: plot-red, thickness: 1.0pt)))
        }
      )

      content((plot-half-width / 2, plot-small-height + 0.35), text(size: 8pt, weight: "bold", [(b) Green Cube Success Rates]))
    }),
  ),
  caption: [Preliminary cube sort training results using the PD tensegrity variant (77k timesteps, training ongoing). *(a)*~Total reward showing an upward trend. *(b)*~Green cube grasp, placement, and miss rates. The agent is beginning to acquire approach and grasp behaviors but has not yet achieved reliable sorting.],
  short-caption: [Preliminary cube sort training results],
) <fig:sort_preliminary>

== Summary of Key Results <sec:results_summary>

The experimental evaluation yields three main findings:

+ *PD and tendon variants achieve comparable RL performance.* On the reach task, both variants converge within 48k timesteps to total rewards above $+0.75$, with position tracking errors below $1"cm"$. On the cube place task, both achieve grasp rates above $94%$ and place success rates above $90%$. The cable-mediated tendon transmission does not introduce a measurable performance penalty in the RL setting, suggesting that the disc-approximation tension mapping is sufficiently transparent to the policy.

+ *The physical tendon variant presents a significant control challenge.* With effort-based actuation through the four-bar antiparallelogram linkage, the physical variant converges more slowly and to lower final performance on both tasks. On reach, it achieves a total reward of $-0.46$ (compared to $+0.79$ for PD) with $3 times$ the policy uncertainty. On cube place, it fails to acquire the grasping skill entirely. This result highlights the difficulty of RL with indirect force transmission and motivates future work on hierarchical control architectures.

+ *Curriculum learning enables robust multi-stage manipulation.* The staged introduction of the red distractor cube and regularization penalties in the cube place task produces smooth curriculum transitions with only brief transient performance dips, validating the curriculum design described in @ch:methodology.

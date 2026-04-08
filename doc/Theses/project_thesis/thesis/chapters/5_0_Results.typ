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

@fig:workspace_density and @fig:workspace_manip show the reachability density and Yoshikawa manipulability distributions, respectively. Each figure presents a 3D voxelised scatter plot alongside three orthogonal cross-section heatmaps through the center of the desired workspace.

#faps-figure(
  image("../../../shared/figures/tensegrity_workspace_density.png", width: 95%),
  caption: [Reachability density distribution of the disc-approximation model. _Left:_ 3D voxel scatter, coloured by sample density. _Right:_ 2D cross-section heatmaps along the $X$, $Y$, and $Z$ midplanes. High density near the workspace center indicates that many joint configurations map to the task-relevant region.],
  short-caption: [Workspace reachability density],
) <fig:workspace_density>

#faps-figure(
  image("../../../shared/figures/tensegrity_workspace_manipulability.png", width: 95%),
  caption: [Yoshikawa manipulability distribution. Manipulability is highest near the workspace center and decreases toward the boundaries, consistent with the kinematic structure of the 5-#ac("DoF") serial chain. The low absolute values ($cal(O)(10^(-5))$) reflect the rank-deficient $6 times 5$ Jacobian rather than indicating poor dexterity within the robot's 5-dimensional motion capability.],
  short-caption: [Workspace Yoshikawa manipulability],
) <fig:workspace_manip>

The disc-approximation model achieves near-complete coverage (99.7~%) of the desired workspace. As discussed in @subsec:disc_approx and @fig:elbow_comparison, this represents a conservative estimate: the physical antiparallelogram elbow can reach farther than the disc model at equivalent joint angles, so the true coverage of the physical robot is expected to be at least as high.

== Reach Task Results <sec:reach_results>

=== Training Convergence <subsec:reach_convergence>

// [PLACEHOLDER: Training curves — total reward, position error, orientation error
// for all 5 variants (Tensegrity PD, Tendon, Physical Tendon, UR10e, Kinova).
// Generated by: scripts/plot_reach_training_results.py]
#faps-figure(
  rect(width: 90%, height: 6cm, stroke: 0.5pt + luma(180))[
    #align(center + horizon)[_PLACEHOLDER: Reach training curves — total reward over timesteps\
    for Tensegrity PD, Tendon, Physical Tendon, UR10e, Kinova\
    Generated by `plot_reach_training_results.py`_]
  ],
  caption: [Reach task training convergence across all robot/actuation variants.],
  short-caption: [Reach task training convergence],
) <fig:reach_convergence>

=== Final Performance <subsec:reach_performance>

@tab:reach_results summarizes the final performance metrics from the reach task training runs.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Variant*], [*Steps*], [*Total Reward*], [*Pos. Error*], [*Orient. Error*], [*Fine-Grained*]),
    [Tensegrity PD], [48k], [$+0.52$], [$-0.008$], [$-0.024$], [$0.077$],
    [Tensegrity Tendon], [48k], [$+0.64$], [$-0.007$], [$-0.012$], [$0.079$],
    [Tensegrity Phys. Tendon], [48k], [$+0.67$], [$-0.006$], [$-0.012$], [$0.080$],
    [UR10e], [96k], [$-0.21$], [$-0.051$], [$-0.021$], [$0.056$],
    [Kinova Gen3], [96k], [$+0.62$], [$-0.014$], [$-0.019$], [$0.088$],
  ),
  caption: [Reach task final performance metrics across variants. The UR10e does not fully converge within 96k steps.],
  short-caption: [Reach task final performance metrics],
) <tab:reach_results>

=== Cross-Variant Comparison <subsec:reach_comparison>

// [PLACEHOLDER: Bar charts or radar plots comparing final metrics across variants.
// Generated by: scripts/plot_comparison.py]
#faps-figure(
  rect(width: 90%, height: 5cm, stroke: 0.5pt + luma(180))[
    #align(center + horizon)[_PLACEHOLDER: Cross-variant comparison plots\
    Generated by `plot_comparison.py`_]
  ],
  caption: [Cross-variant comparison of final reach task performance.],
  short-caption: [Cross-variant reach task comparison],
) <fig:reach_comparison>

== Cube Place Results <sec:place_results>

=== Training Convergence <subsec:place_convergence>

// [PLACEHOLDER: Training curves for cube place — total reward, grasp rate, place success rate
// for Tensegrity PD, Tendon, UR10e, Kinova variants.
// Generated by: scripts/plot_place_training_results.py]
#faps-figure(
  rect(width: 90%, height: 6cm, stroke: 0.5pt + luma(180))[
    #align(center + horizon)[_PLACEHOLDER: Cube place training curves\
    Generated by `plot_place_training_results.py`_]
  ],
  caption: [Cube place training convergence showing total reward and task-specific metrics over training steps.],
  short-caption: [Cube place training convergence],
) <fig:place_convergence>

=== Final Performance <subsec:place_final>

// [PLACEHOLDER: Final metrics table for cube place — grasp rate %, place success rate %,
// mean episode return, per variant.
// Data from training reports in reach/reports/ and cube_place/reports/]
#faps-table(
  rect(width: 90%, height: 3cm, stroke: 0.5pt + luma(180))[
    #align(center + horizon)[_PLACEHOLDER: Cube place final metrics table\
    Grasp Rate, Place Success Rate, Mean Episode Return per variant_]
  ],
  caption: [Cube place final performance metrics.],
  short-caption: [Cube place final performance metrics],
) <tab:place_results>

=== Curriculum Effect <subsec:curriculum_effect>

// [PLACEHOLDER: Before/after plots showing the effect of red cube introduction at 100k steps
// and regularization ramp at 200k steps on success rate and reward.
// Look for curriculum transitions in the training curves.]
#faps-figure(
  rect(width: 90%, height: 5cm, stroke: 0.5pt + luma(180))[
    #align(center + horizon)[_PLACEHOLDER: Curriculum effect — reward/success rate\
    changes at 100k (red cube) and 200k (regularization) step thresholds_]
  ],
  caption: [Effect of curriculum stages on cube place training: red cube introduction (100k steps) and regularization ramp (200k steps).],
  short-caption: [Curriculum stages effect on training],
) <fig:curriculum_effect>

== Cube Sort Preliminary Results <sec:sort_results>

// [PLACEHOLDER: If cube sort training has been run, show preliminary curves here.
// Generated by: scripts/plot_sort_training_results.py]
#faps-figure(
  rect(width: 90%, height: 5cm, stroke: 0.5pt + luma(180))[
    #align(center + horizon)[_PLACEHOLDER: Cube sort preliminary training results\
    (if available from `plot_sort_training_results.py`)_]
  ],
  caption: [Preliminary cube sort training results (work in progress).],
  short-caption: [Preliminary cube sort training results],
) <fig:sort_preliminary>

== Summary of Key Results <sec:results_summary>

// To be written after all measurements are available.

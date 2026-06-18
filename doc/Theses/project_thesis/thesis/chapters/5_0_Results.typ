// Chapter 5: Results
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *
#import "../../../shared/formatting/template.typ": faps-figure, faps-table, faps-subfigure
#import "../../../shared/formatting/plots.typ": *
#import "../../../shared/formatting/colors.typ": *
#import "@preview/physica:0.9.8": *

= Results <ch:results>

This chapter presents the quantitative and qualitative results of the model validation and #ac("RL") experiments. All results are produced by the training and plotting pipeline described in @ch:methodology.

== Model Validation Results <sec:validation_results>

The model validation results are structured into two main sections: step-response analysis and workspace analysis. The step-response analysis evaluates the dynamic performance of the three Isaac Sim arm variants against the Klein~@Klein2023 Gazebo reference across standard control metrics. The workspace analysis assesses the reachability and dexterity of the disc-approximation model within the task-relevant volume.

=== Step-Response Analysis <subsec:step_results>

@fig:step_metrics consolidates the four standard step-response performance indicators (rise time, settling time, overshoot and NRMSE) across the three Isaac Sim arm variants (PD, constant-$J^top$ tendon, physical tendon) and the Klein~@Klein2023 Gazebo-vs-real-robot reference. Bars are aggregated as the mean across the per-joint amplitude sweep (elbow: $20$/$30$/$40 degree$, wrist: $10$/$20$/$30 degree$). The corresponding per-trace time-domain plots are deferred to @appendix:step_responses to keep this section focused on the comparative summary.

The PD variant (`ImplicitActuator`, $K = 400$, $D = 20$) is the fastest and most accurate by every metric, as expected for direct joint-level control. The constant-$J^top$ tendon variant (elbow: $K_p = 50$, $K_i = 4$, $K_d = 2$ | wrist: $K_p = 10$, $K_i = 1.5$, $K_d = 0.6$) trades a moderate increase in rise/settling time and NRMSE for physically realistic cable-mediated dynamics. The physical tendon variant (elbow: $K_p = 75$, $K_i = 6$, $K_d = 3$) actuates the four-bar mechanism through body forces on the cables instead of the single-#acs("DoF") disc approximation. It matches the constant-$J^top$ tendon variant in NRMSE while exhibiting slightly longer settling times due to the linkage compliance. All three Isaac Sim configurations match or outperform the Klein~@Klein2023 Gazebo reference NRMSE and all joints settle within $300 "ms"$, satisfying the design target of $0.10$--$0.30 "s"$.

#faps-figure(
  image("../../assets/figures/results/step_metrics_overview.png", width: 100%),
  caption: [Cross-variant comparison of step-response performance metrics, aggregated as the mean across the per-joint amplitude sweep with error bars showing $plus.minus 1 sigma$ across the same sweep. Each panel compares the Klein~@Klein2023 Gazebo reference against the three Isaac Sim variants (PD, constant-$J^top$ tendon, physical tendon). Lower is better for all four metrics. For the simulation variants a non-overshooting trace contributes $0%$ to the overshoot panel. Hatched bars marked "n/a" denote Klein cells that were not reported in his Tabelle~4.2 (settling times in particular are not provided by Klein). Time-trace plots underlying these aggregates are provided in @appendix:step_responses.],
  short-caption: [Step-response performance metric comparison],
) <fig:step_metrics>

==== Base joint validation
The two prismatic base joints use PD control ($K = 8000$, $D = 800$). @tab:base_metrics summarizes the results. Settling times are below 200~ms and steady-state errors are negligible ($< 0.003 "mm"$).

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
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

The Monte Carlo workspace analysis used $N = 2 000 000$ FK samples with the disc-approximation arm variant, ceiling-mounted at 2.30~m, across 4,096 parallel environments. @tab:workspace_stats summarizes the key statistics and @fig:workspace_metrics presents the spatial distribution of the two reported quality metrics.

#faps-table(
  table(
    columns: (auto, auto),
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

==== Qualitative evaluation
The reachable volume forms the expected U-shaped envelope around the ceiling-mount centerline (@fig:workspace_metrics). The two prismatic base joints extend this envelope along $Y$, producing the rectangular plateau visible in the Top~$X Y$ slice and the wide saddle visible in the Front~$Y Z$ slice. The desired workspace (orange box) lies entirely inside the dense region of the envelope and is covered to $99.72~%$ ($33 904 "/" 34 000$ voxels, @tab:workspace_stats). The remaining $0.28~%$ are isolated corner voxels at the box boundary and lie within the discretisation error of the chosen $Delta = 0.02 "m"$ grid. The reachability density (panel~a) is highest in a horizontal band at $Z approx 0.9$--$1.3 "m"$, i.e.~directly below the mount and across the full prismatic stroke, which is precisely the region in which the conveyor surface and drum are placed. The narrow low-density notch visible above the mount in the Front~$Y Z$ slice corresponds to configurations where the arm reaches joint range limits and positioning is only provided by the prismatic base. This lack of configuration-space redundancy is not expected to impact the target tasks, which do not require dextrous manipulation at high elevation angles. Here the arm should only be able to hold a obljedt such that a second arm can approach it for a handover. The low density is not expected to cause any practical issues.

#faps-subfigure(
  columns: (1fr,),
  panels: (
    (image("../../assets/figures/validation/tensegrity_workspace_density.png", width: 100%), [Reachability density]),
    (image("../../assets/figures/validation/tensegrity_workspace_manipulability.png", width: 100%), [Yoshikawa manipulability]),
  ),
  caption: [Workspace quality metrics for the disc-approximation model (2M FK samples, 4096 parallel envs). *(a)*~Reachability density: high density near the workspace center indicates many joint configurations map to the task-relevant region. *(b)*~Yoshikawa manipulability: highest near the center, decreasing toward boundaries. The low absolute values ($cal(O)(10^(-5))$) reflect the rank-deficient $6 times 5$ Jacobian rather than indicating poor dexterity within the robot's 5-dimensional motion capability.],
  short-caption: [Workspace quality metrics (density, manipulability)],
  label: <fig:workspace_metrics>,
)

==== Magnitude of the Yoshikawa index
The absolute values of the Yoshikawa manipulability are small ($cal(O)(10^(-5))$, see @tab:workspace_stats). This is a direct consequence of the index definition $w(vb(q)) = sqrt(det(J(vb(q)) J(vb(q))^T))$ on a rank-deficient $6 times 5$ geometric Jacobian: with only five actuated #acp("DoF") the smallest singular value of $J J^T$ is structurally close to zero, and the determinant under the square root scales with the product of the remaining singular values, which all carry units of metres. Concretely, the linear singular values are bounded by the link lengths ($cal(O)(10^(-1)) "m"$) and the angular ones are dimensionless of order unity, so $w$ takes values in the $10^(-5)$--$10^(-4)$ range even far from any singularity. The relevant information therefore lies in the _spatial distribution_ of $w$, not in its absolute magnitude: panel~(b) shows the expected pattern of higher values near the workspace center and a smooth decay toward the boundaries, without any internal singularity surface inside the desired workspace.

==== Omission of the inverse condition number
The inverse condition number $kappa^(-1) = sigma_"min" "/" sigma_"max"$ was computed alongside the Yoshikawa index but is not reported as a figure. Because the geometric Jacobian is $6 times 5$, it has at most rank~5 and $sigma_"min"$ of $J J^T$ evaluates to zero (within numerical precision) at _every_ sample. The resulting field is uniformly zero and therefore carries no spatial information. Including it would only add a featureless heatmap. This is a property of the kinematic structure (5~#acp("DoF") cannot independently control all 6 Cartesian directions), not of the specific design choices made in this work.

==== Suitability for the target application
The combined evidence supports that the arm is suitable for the sorting tasks targeted in this work and in the subsequent master's thesis. First, the desired workspace is covered to $99.7~%$, so every pick, place, and reach pose used by the #ac("RL") tasks in @sec:reach_results and @sec:place_results is kinematically reachable. Second, the Yoshikawa field is smooth and strictly positive across the entire desired volume, meaning no internal singularities will be encountered during policy rollouts. Third, the disc approximation is conservative with respect to the physical antiparallelogram elbow (@subsec:disc_approx, @fig:elbow_comparison). The migrating instantaneous center of rotation of the four-bar linkage shifts the forearm tip outward at large elbow angles, so the physical robot reaches at least as far as the disc model at every joint configuration. Coverage of the desired workspace on the physical hardware is therefore expected to be $>= 99.7~%$, and any base-positioning, mount-height and reward-region decisions taken from this analysis remain valid with additional margin.

==== Motivation for the workspace-analysis tool
Beyond validating the current design, the Monte Carlo sampler, voxeliser and visualiser (`workspace_sample.py` / `workspace_visualize.py`) are deliberately implemented in a robot-agnostic fashion. This enables further concrete uses in the master's thesis and other future work on the robot or in Isaac~Lab in general. Re-running the analysis on alternative tensegrity configurations (e.g.~modified link lengths, different mount heights or an additional active wrist #ac("DoF")) without re-implementing any of the sampling or post-processing infrastructure and supporting the planned wrist-transfer experiment in which a standard $6$-#ac("DoF") industrial arm is evaluated _with_ and _without_ the tendon-driven wrist module. The same pipeline can produce a directly comparable coverage and Yoshikawa map for both, quantifying analytically how much of the dexterous workspace the tendon wrist recovers or sacrifices relative to the bare arm.



== Reach Task Results <sec:reach_results>

==== Training Convergence <subsec:reach_convergence>

@fig:reach_convergence compares the training convergence of the three tensegrity variants on the reach task. The #ac("PD") and constant-$J^top$ tendon variants are trained for $48"k"$ timesteps and reach plateau rewards of $+0.77$ and $+0.75$ respectively within the first $20"k"$ steps. The physical tendon variant is trained for $150"k"$ timesteps. Its mean total reward rises monotonically from approximately $-1.8$ and plateaus near $-0.71$ over the last $10%$ of training. The policy standard deviation $sigma$ in panel~(d) drops from the prior of $1.0$ to $0.021$ (#ac("PD")), $0.029$ (tendon) and $0.092$ (physical tendon), a four-fold residual exploration margin in the physical tendon variant that confirms the policy has not fully consolidated within the training horizon.

The per-component reward curves in @fig:appendix_reach_rewards and @fig:appendix_reach_components show that the position-tracking term contributes the largest learning gradient for the #ac("PD") and tendon variants, while the fine-grained proximity bonus (tanh shaped, $sigma_"std" = 0.05 "m"$) saturates after roughly $30"k"$ steps and stabilises the end-effector near the target. The physical tendon variant approaches but never reaches this saturation level, indicating that the effort-based actuation interface through the antiparallelogram linkage limits the precision attainable within the $150"k"$-step budget.

#faps-figure(
  image("../../assets/figures/results/reach_convergence.png", width: 100%),
  caption: [Reach task training convergence for the three tensegrity variants. *(a)*~Mean total reward over training timesteps. *(b)*~Position-reached success rate ($d_"pos" < 5 "cm"$). *(c)*~Full pose-reached success rate (position and orientation, $d_"ori" < 15 degree$). *(d)*~Policy standard deviation ($sigma$), indicating exploration level. PD and tendon variants converge within 48k steps. The physical tendon variant requires 150k steps due to the effort-based control indirection.],
  short-caption: [Reach task training convergence],
) <fig:reach_convergence>

==== Final Performance (Seed-Aggregated Evaluation) <subsec:reach_performance>

The training-time success metrics in @subsec:reach_convergence are computed inside the curriculum-shaped reward loop and over a single seed per variant. They therefore conflate exploration noise, curriculum effects, and the actual task competence of the converged policy. Following the recommendations of Henderson _et al._~@Henderson2018DeepRLMatters and Colas _et al._~@Colas2018HowManySeeds, the final policies are re-evaluated in a clean play-mode environment with the evaluation harness described in @subsec:reach_eval_harness. $N = 5$ independent training seeds per variant, $500$ episodes per (variant, seed) cell, and the same $5 "cm"$ success threshold that the reward term uses.
// Henderson2018DeepRLMatters — p.2 §"Experimental Analysis" (five trials per evaluation) and p.5 §"Random Seeds and Trials" (seed-to-seed variance makes a single curve unreliable). Uploaded arXiv PDF.
// Colas2018HowManySeeds — p.2 §1 (deep-RL papers commonly use "5 seeds or less", which "can lead to claim that two algorithms performances are different when they are not", motivating averaging over more seeds). Uploaded PDF.

@tab:reach_results reports the resulting seed-aggregated metrics for all four agents (the trained #ac("PPO") policy plus the three reference agents from @subsec:reach_eval_harness). For each (variant, agent) cell, success rate, final-step position error, final-step orientation error, and time-to-reach are first averaged over the $500$ evaluation episodes within a seed and then aggregated across seeds as mean~$plus.minus$~standard deviation. The DLS-IK heuristic is defined only for the #ac("PD") variant.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    table.header(
      [*Variant*], [*Agent*],
      [*Success*], [*Pos.\ err.~[m]*], [*Ori.\ err.~[rad]*], [*Reach~[s]*],
    ),
    [Tensegrity PD], [Zero],      [$0.019 plus.minus 0.006$], [$0.479$], [$0.952$], [$0.119$],
    [Tensegrity PD], [Random],    [$0.150 plus.minus 0.014$], [$0.522$], [$1.043$], [$2.119$],
    [Tensegrity PD], [DLS-IK],    [$0.793 plus.minus 0.009$], [$0.024$], [$0.717$], [$0.489$],
    [Tensegrity PD], [PPO],       [$bold(0.997 plus.minus 0.003)$], [$0.010$], [$0.050$], [$0.325$],
    [Tensegrity Tendon], [Zero],     [$0.019 plus.minus 0.006$], [$0.479$], [$0.952$], [$0.118$],
    [Tensegrity Tendon], [Random],   [$0.112 plus.minus 0.008$], [$0.519$], [$1.021$], [$2.181$],
    [Tensegrity Tendon], [PPO],      [$bold(0.979 plus.minus 0.039)$], [$0.010$], [$0.064$], [$0.376$],
    [Tensegrity Physical], [Zero],   [$0.034 plus.minus 0.005$], [$0.648$], [$1.440$], [$0.276$],
    [Tensegrity Physical], [Random], [$0.123 plus.minus 0.015$], [$0.776$], [$1.732$], [$2.281$],
    [Tensegrity Physical], [PPO],    [$bold(0.314 plus.minus 0.144)$], [$0.217$], [$0.709$], [$1.428$],
  ),
  caption: [Reach task seed-aggregated evaluation metrics. Each cell reports the mean and standard deviation across $N = 5$ training seeds of the per-seed mean over $500$ play-mode episodes ($50$ parallel environments $times$ $10$ episodes). Success: indicator of $||bold(p) - bold(p)^*|| lt.eq 0.05 "m"$ at any step. Position and orientation errors are reported at episode termination. Reach time is conditioned on successful episodes. PPO entries are bold for readability. The DLS-IK heuristic is defined only for the #ac("PD") variant (cf.~@subsec:reach_eval_harness).],
  short-caption: [Reach task seed-aggregated evaluation metrics],
) <tab:reach_results>

The trained #ac("PPO") policy clears the $5 "cm"$ success threshold in $99.7 plus.minus 0.3 %$ of episodes on the #ac("PD") variant and in $97.9 plus.minus 3.9 %$ on the constant-$J^top$ tendon variant. The per-seed standard deviation on the tendon variant is dominated by a single seed at $91.0 %$ while the remaining four seeds cluster between $98.0 %$ and $100.0 %$ (@fig:reach_per_seed_scatter). For both variants the final-step position error collapses to $1.0 "cm"$ (#ac("PD")) and $0.5$ to $2.5 "cm"$ (tendon), well below the threshold. The physical tendon variant only succeeds in $31.4 plus-minus 14.4 %$ of episodes, and the across-seed range is the entire interval from $8.8 %$ (seed 4) to $46.2 %$ (seed 3). Its mean final-step position error of $21.7 "cm"$ exceeds the success threshold by more than a factor of four, confirming that the policies for this variant struggle to reach the target.

==== Baseline Comparison <subsec:reach_baselines>

@fig:reach_baselines shows the seed-aggregated success rate of each agent on each variant as grouped bars. The zero agent succeeds in $1.9$ to $3.4 %$ of episodes for all three variant. The small non-zero value comes from targets sampled inside the $5 "cm"$ ball around the initial end-effector pose. The uniform-random agent reaches $11.2$ to $15.0 %$, which quantifies the inherent reachability of the desired workspace under unstructured exploration over a $6 "s"$ episode. The DLS-IK heuristic on the #ac("PD") variant achieves $79.3 plus.minus 0.9 %$ and serves as the strongest non-learned reference. It is not run on the two tendon variants for the reasons stated in @subsec:reach_eval_harness, which is marked _n/a_ in the figure.

#faps-figure(
  image("../../assets/figures/results/reach_baselines.png", width: 90%),
  caption: [Reach-task baseline comparison. Grouped bars show the seed-aggregated success rate (mean~$plus.minus$~standard deviation across $N = 5$ seeds) of each agent on each actuation variant. _n/a_ marks the DLS-IK heuristic for the two tendon variants, which is undefined because the action space is cable tension rather than joint position. The trained #ac("PPO") policy outperforms every baseline on all variants.],
  short-caption: [Reach-task baseline comparison],
) <fig:reach_baselines>

The trained #ac("PPO") policy improves on the strongest available baseline by $20.4$ percentage points on the #ac("PD") variant ($99.7 %$ vs.~$79.3 %$ DLS-IK), by $86.7$ percentage points on the constant-$J^top$ tendon variant ($97.9 %$ vs.~$11.2 %$ random), and by $19.0$ percentage points on the physical tendon variant ($31.4 %$ vs.~$12.3 %$ random). On the latter, the trained policy lies on the same order of magnitude as random exploration, in stark contrast to the more than $8 times$ gap on the other two variants. The four-panel summary in @fig:reach_results_aggregate also reports the corresponding mean position error (cleared by both #ac("PD") and tendon, missed by physical), orientation error and reach time.

#faps-figure(
  image("../../assets/figures/results/reach_results_aggregate.png", width: 90%),
  caption: [Reach-task seed-aggregated #ac("PPO") evaluation metrics. *(a)*~Success rate. *(b)*~Final-step position error with the $5 "cm"$ threshold marked as a dashed line. *(c)*~Final-step orientation error. *(d)*~Time-to-first-reach over successful episodes. Bars show the mean~$plus.minus$~standard deviation over $N = 5$ training seeds. Individual seed means are overlaid as dots.],
  short-caption: [Reach-task seed-aggregated PPO metrics],
) <fig:reach_results_aggregate>

==== Per-Seed Distribution <subsec:reach_per_seed>

@fig:reach_per_seed_scatter visualises the per-seed dispersion of the two most informative metrics (success rate and final-step position error) on a single axis. The #ac("PD") variant occupies a tight band at the top of the success-rate panel (range $0.991$ to $1.000$) and the bottom of the position-error panel ($0.94$ to $1.18 "cm"$), with no overlap to the other two variants on either metric. The tendon variant is shifted slightly down on success rate by the single underperforming seed mentioned above, but its four healthy seeds reach the same $0.4$ to $1.0 "cm"$ position-error band as #ac("PD") and confirm that the constant-$J^top$ tendon mapping does not impose a precision penalty when training succeeds. The physical tendon variant fans out across an order of magnitude on the log-scaled position-error panel ($14.3$ to $31.5 "cm"$) and across nearly all of the success-rate axis ($0.088$ to $0.462$). No seed for this variant brings the median episode below the $5 "cm"$ threshold (dashed line).

#faps-figure(
  image("../../assets/figures/results/reach_per_seed_scatter.png", width: 100%),
  caption: [Per-seed #ac("PPO") evaluation distribution. *(a)*~Per-seed success rate by variant. Horizontal lines mark the seed-aggregated mean. *(b)*~Per-seed final-step position error on logarithmic scale with the $5 "cm"$ success threshold dashed. All #ac("PD") and four of five tendon seeds clear the threshold by an order of magnitude. No physical tendon seed reaches it.],
  short-caption: [Per-seed PPO evaluation distribution],
) <fig:reach_per_seed_scatter>

The cross-seed dispersion confirms that #ac("PD") and constant-$J^top$ tendon are reliable training targets at the current hyperparameter setting, while the physical tendon variant is bimodal in outcome quality. Half the seeds reach a stable approach near $15 "cm"$ and half stall near $30 "cm"$. The implication for the planned re-run on a larger seed budget is discussed in @ch:discussion.

==== Statistical Significance <subsec:reach_significance>

To check that the head-to-head differences in @tab:reach_results are not artifacts of the small $N = 5$ seed budget, the per-seed success-rate vectors of #ac("PPO") and the strongest available baseline on each variant are compared with a Welch two-sample $t$-test (`scipy.stats.ttest_ind(equal_var=False)`). On the #ac("PD") variant, #ac("PPO") versus DLS-IK gives $t = 46.4$ at $nu approx 4.6$ degrees of freedom and a two-sided $p = 2.3 times 10^(-7)$. On the constant-$J^top$ tendon variant, #ac("PPO") versus uniform-random gives $t = 49.2$ at $nu approx 4.3$ and $p = 4.3 times 10^(-7)$. On the physical tendon variant, #ac("PPO") versus uniform-random gives $t = 3.0$ at $nu approx 4.1$ and $p = 0.041$.

The first two comparisons are highly significant and survive any reasonable correction for the three comparisons made here. The physical tendon improvement is significant at the conventional $alpha = 0.05$ but not at the Bonferroni-corrected level $alpha / 3 approx 0.017$. This borderline result must be read with caution: Colas _et al._ show that at $N = 5$ a two-sample test is severely underpowered (the type-II error in their worked example is $beta = 0.51$) and that the realised small-sample false-positive rate is inflated above the nominal $alpha$, so a difference that is significant only at the uncorrected level should be treated as exploratory rather than confirmatory~@Colas2018HowManySeeds.
// Colas2018HowManySeeds — p.13 §4.2 + Fig.4 ("β = 0.51 for N = 5") and p.16 §5.1 + Fig.6 (small-N type-I inflation; bootstrap CI test needs N≈20). Uploaded PDF.
The third result is therefore reported as an exploratory finding and motivates the recommendation in @sec:discussion_recommendations to refine and repeat the physical tendon training with $N gt.eq 10$ seeds in future work.

== Cube Place Results <sec:place_results>

==== Training Convergence <subsec:place_convergence>

@fig:place_convergence reports the training convergence for the cube place task. The #ac("PD") and constant-$J^top$ tendon variants both converge within the $300"k"$ timestep budget. The #ac("PD") variant reaches a plateau total reward of $+332$ after roughly $175"k"$ steps, the tendon variant a plateau of $+342$ after $100"k"$ steps. Both curves remain stable for the remainder of training, including the curriculum transitions at $125"k"$ and $150"k"$. The physical tendon variant rises rapidly during the first $25"k"$ steps as the reaching reward saturates and then plateaus near $+58$ without acquiring grasping (see @fig:appendix_place_manipulation_a, @fig:appendix_place_manipulation_b). The policy standard deviation in panel~(d) settles at $0.196$ (#ac("PD")), $0.338$ (tendon) and $0.393$ (physical tendon), consistent with the residual exploration observed in the reach task and pointing to a less consolidated policy in the tendon variants.

#faps-figure(
  image("../../assets/figures/results/place_convergence.png", width: 100%),
  caption: [Cube place training convergence for the three tensegrity variants. *(a)*~Mean total reward over training timesteps. *(b)*~Grasp success rate. *(c)*~Place success rate (cube placed in target drum). *(d)*~Policy standard deviation. PD and tendon variants converge to high task success within 300k steps. The physical tendon variant does not acquire the grasping skill.],
  short-caption: [Cube place training convergence],
) <fig:place_convergence>

The per-phase reward curves in @fig:appendix_place_manipulation_a, @fig:appendix_place_manipulation_b and @fig:appendix_place_placement document the sequential acquisition of the task skills. The reaching reward rises first and saturates within $30"k"$ steps for the #ac("PD") and tendon variants, followed by grasping and lifting (within $50"k"$ steps), the coarse goal-tracking reward (after the `was_grasped` latch becomes active, roughly $60"k"$ steps) and finally the fine goal-tracking and release rewards. The physical tendon variant follows the same ordering, but only the reaching reward saturates within the training budget. Grasping and lifting remain near zero throughout, which prevents the downstream gated rewards from contributing.

==== Final Performance <subsec:place_final>

@tab:place_results summarises the cube place evaluation metrics at the end of training (mean over the last $10 %$). The #ac("PD") variant achieves a grasp rate of $97.3 %$ and a place success rate of $90.1 %$. The tendon variant matches it with a grasp rate of $95.1 %$ and a higher place success rate of $92.1 %$. Both variants keep the red distractor on the conveyor in over $97 %$ of episodes, confirming that the active push-aside reward terms (`red_clearance`, `red_green_separation`) successfully shape selective manipulation. The physical tendon variant reaches the cube but fails to close on it, recording a grasp rate of $0.2 %$ and a place success rate of $0.0 %$.

Unlike the reach evaluation in @subsec:reach_performance, these cube-place metrics are reported for a single training seed per variant and are computed from the last $10 %$ of training rather than from a separate seed-aggregated play-mode evaluation. The multi-seed protocol was omitted here for time and scope reasons (cf.~the place evaluation scope in @sec:rl_tasks and @sec:discussion_limitations); the values should therefore be read as single-seed point estimates without a cross-seed dispersion measure.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    table.header([*Variant*], [*Steps*], [*Total Reward*], [*Grasp Rate*], [*Place Success*], [*Red Safe*]),
    [Tensegrity PD], [300k], [$+332$], [$97.3%$], [$90.1%$], [$98.7%$],
    [Tensegrity Tendon], [300k], [$+342$], [$95.1%$], [$92.1%$], [$97.6%$],
    [Tensegrity Physical], [300k], [$+58$], [$0.2%$], [$0.0%$], [$99.3%$],
  ),
  caption: [Cube place final performance metrics (mean over the last $10 %$ of training). Grasp Rate: fraction of episodes achieving a stable grasp. Place Success: fraction of episodes where the green cube is placed in the target drum. Red Safe: fraction where the red distractor remains on the conveyor.],
  short-caption: [Cube place final performance metrics],
) <tab:place_results>

The #ac("PD") and tendon variants trade roughly $2 %$ in opposite directions on the grasp and place metrics. The #ac("PD") variant grasps more reliably but releases slightly less precisely over the drum, while the tendon variant places more cubes successfully at the cost of two failed grasps per hundred episodes. The fact that the tendon variant matches or marginally exceeds the #ac("PD") baseline on the harder placement metric replicates the equivalence between the two actuation interfaces, already observed for reach (@subsec:reach_performance). The physical tendon variant departs from this pattern. It learns the reaching sub-task ( visible as a non-zero reaching reward in @fig:appendix_place_manipulation_a ) but does not bridge the transition into grasping. The release and goal-tracking rewards in @fig:appendix_place_placement remain at zero throughout the run, gated by the `was_grasped` latch that never activates.

The diagnostic curves in @fig:appendix_place_diagnostics confirm that this failure is not driven by an unstable optimisation. The policy loss, value loss and #ac("PPO") clip fraction follow the same patterns observed in the converging #ac("PD") and tendon variants. The bottleneck is therefore the body-force actuation interface itself, which couples five cable efforts through the closed-chain antiparallelogram into joint torques and makes the discrete transition from reaching to grasping substantially harder for direct policy learning than the simplyfied interfaces of the other two variants.

==== Effect of the sequential reward structure
The shaping scheme described in @subsec:place_reward_shaping is validated by the phase-wise reward curves of the converging variants. The paired coarse/fine reaching rewards (@fig:appendix_place_manipulation_a, panels~(a) and~(b)) rise in the predicted order. The coarse term saturates first as the policy learns to approach the cube across the initial offset, and the fine term saturates next once the gripper enters the pre-grasp envelope. The product grasping term (@fig:appendix_place_manipulation_a, panel~(c)) only activates after both reaching terms have plateaued, which confirms that the closure gradient is not exploited as a shortcut before contact. The coarse and fine goal-tracking terms (@fig:appendix_place_placement, panels~(a) and~(b)) become non-zero after the `was_grasped` latch flips and rise to the same ordered saturation pattern as the reaching terms. The release reward (panel~(c)) reaches a stable positive value of roughly $12$ in both converging variants, which is consistent with the design intent that the elevated weight of $25.0$ overcomes the per-step holding incentive that the earlier reward iterations had created. In the absence of this re-weighting the agent would prefer to carry the cube indefinitely. The success bonus in panel~(d) converges close to its theoretical ceiling for the longest in-drum dwell times the $5 "s"$ episode allows. The red-cube safety terms in @fig:appendix_place_safety (panels~(b) and~(c)) follow the curriculum schedule. Both terms remain at zero until $125"k"$ steps and then rise to positive values within $40"k"$ steps, mirroring the increase in the red-on-conveyor metric. The shaping scheme therefore meets its three design objectives (phase-wise gradient coverage, suppression of the holding local optimum, and active distractor separation) for the two variants that solve the task.

==== Curriculum Effect <subsec:curriculum_effect>

@fig:curriculum_effect illustrates the effect of the two scheduled curriculum transitions on training progression, using the #ac("PD") variant as a reference (the curriculum schedule is identical across variants). At $125"k"$ timesteps, the red distractor cube's reward terms are activated. The cube itself has been spawned on the conveyor and included in the observations from step~$0$, so this transition introduces only a reward-side perturbation rather than an observation-space discontinuity. The total reward in panel~(a) shows a brief dip of approximately $5%$ followed by recovery within $25"k"$ steps, with the grasp and place success rates in panel~(b) remaining within $2$ percentage points of the pre-transition values. At $150"k"$ timesteps, the action-rate and joint-velocity penalty weights ramp from $-10^(-4)$ to $-2 times 10^(-3)$, increasing the magnitude of both regularisation terms in panel~(c) by an order of magnitude. The cube-off-conveyor penalty in panel~(d) becomes non-zero at $125"k"$ when the red cube starts contributing to the safety reward terms.

Both curriculum transitions therefore meet their design intent. The Manipulation skills acquired before $125"k"$ steps persist through the addition of the distractor and the tightening of the regularisation budget, while the new reward signals shape the desired behaviour (red-cube separation in the first transition, smoother joint trajectories in the second). The same schedule is applied unchanged to the tendon variants and produces the same qualitative pattern (see @fig:appendix_place_metrics, @fig:appendix_place_safety, @fig:appendix_place_regularization_a). For the physical tendon variant the curriculum has no observable effect, because the prerequisite grasping skill is not acquired.

#faps-figure(
  image("../../assets/figures/results/place_curriculum.png", width: 100%),
  caption: [Effect of curriculum stages on the cube-place training (PD variant). *(a)*~Total reward with dashed lines marking the red cube introduction ($125"k"$) and regularization ramp ($150"k"$). *(b)*~Task success rates showing brief transient dips at curriculum transitions. *(c)*~Regularization penalty terms (action rate, joint velocity) ramping up after $150"k"$ steps. *(d)*~Cube-off-conveyor penalty activating after $125"k"$ steps.],
  short-caption: [Curriculum stages effect on training],
) <fig:curriculum_effect>

// The experimental evaluation yields three main findings:

// + *PD and tendon variants achieve comparable RL performance.* On the reach task, both variants converge within 48k timesteps to total rewards above $+0.75$, with position tracking errors below $1"cm"$. On the cube place task, both achieve grasp rates above $94%$ and place success rates above $90%$. The cable-mediated tendon transmission does not introduce a measurable performance penalty in the RL setting, suggesting that the disc-approximation tension mapping is sufficiently transparent to the policy.

// + *The physical tendon variant presents a significant control challenge.* With effort-based actuation through the four-bar antiparallelogram linkage, the physical tendon variant converges more slowly and to lower final performance on both tasks. On reach, it achieves a total reward of $-0.46$ (compared to $+0.79$ for PD) with $3 times$ the policy uncertainty. On cube place, it fails to acquire the grasping skill entirely. This result highlights the difficulty of RL with indirect force transmission and motivates future work on hierarchical control architectures.

// + *Curriculum learning enables robust multi-stage manipulation.* The staged introduction of the red distractor cube and regularization penalties in the cube place task produces smooth curriculum transitions with only brief transient performance dips, validating the curriculum design described in @ch:methodology.

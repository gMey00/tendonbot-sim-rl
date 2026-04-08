// §4.5 Model Validation
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table

== Model Validation <sec:model_validation>

The simulated robot model is validated through step-response testing and workspace analysis, following the methodology established by Klein~@Klein2023 for the same manipulator in the Gazebo simulator. This section describes the testing protocols and evaluation metrics. Quantitative results are presented in @ch:results.

=== Step-Response Testing Protocol <subsec:step_response>

Step-response tests characterize the dynamic behavior of each arm joint under controlled conditions. The protocol is designed to mirror Klein's experimental procedure~@Klein2023 to enable direct comparison between Isaac Sim, Gazebo, and hardware measurements.

==== Test procedure
Each arm joint receives a step position command while all other joints are held at their zero position. The joint position response is recorded at 120~Hz (matching the physics simulation rate of $1/120 "s"$ substeps at decimation~2). Each test consists of four phases: a 0.5~s warm-up at zero (not recorded), a 0.5~s baseline recording at zero, a 2.5~s hold phase during which the joint tracks the step command, and a 1.0~s return-to-zero phase. The test is performed for three step amplitudes per joint.

==== Controller configuration
A #ac("PID") position controller computes the torque command for each arm joint during tendon-driven testing. The control law is:
$ tau = K_p (q^* - q) + K_i integral_0^t (q^* - q) d t' - K_d dot(q) + tau_g $ <eq:pid_control>
where $q^*$ is the target position, $q$ is the measured position, and $tau_g$ is a feedforward gravity compensation term. The derivative term acts on the measured velocity ($-K_d dot(q)$) rather than on the error derivative, which naturally filters PhysX velocity noise and avoids chattering on the wrist joints. The gravity compensation for the elbow joint is computed as:
$ tau_g = m_e dot g dot l_c dot sin(theta_"elbow") $ <eq:gravity_comp>
with $m_e = 2.49 "kg"$ (disc, forearm, and end-effector links combined), $g = 9.81 "m/s"^2$, $l_c = 0.26 "m"$ (combined center of mass from elbow axis), and $theta_"elbow"$ the current elbow angle. For the wrist joints, gravity compensation uses $m_e = 0.40 "kg"$ and $l_c = 0.068 "m"$.

The PID gains used in Isaac Sim are scaled approximately $133 times$ from Klein's original Gazebo gains ($K_p = 0.3$, $K_i = 0.03$, $K_d = 0.02$). This scaling compensates for the different effective rotational inertia representation between the two simulators (Gazebo uses ODE/DART with generalized-coordinate dynamics; Isaac Sim uses PhysX with reduced-coordinate articulations that exhibit different numerical conditioning, yielding an effective rotational inertia of approximately $0.103 "kg" dot "m"^2$ for the elbow). @tab:step_test_matrix summarizes the test parameters.

#faps-table(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint*], [*Step amplitudes*], [*PID gains*], [*Saturation*], [*Min. tension*]),
    [`elbow_joint`], [$20 degree$, $30 degree$, $40 degree$], [$K_p = 50, K_i = 4, K_d = 2$], [400~N], [6~N],
    [`wrist_y_joint`], [$10 degree$, $20 degree$, $30 degree$], [$K_p = 10, K_i = 1.5, K_d = 0.6$], [600~N], [5~N],
    [`wrist_x_joint`], [$10 degree$, $20 degree$, $30 degree$], [$K_p = 10, K_i = 1.5, K_d = 0.6$], [600~N], [5~N],
  ),
  caption: [Step-response test matrix for the tendon-driven arm. PID gains are scaled from Klein's originals~@Klein2023 to compensate for simulator-specific inertia representation. The minimum tension maintains cable pre-tension on inactive tendons to prevent slack.],
  short-caption: [Step-response test matrix],
) <tab:step_test_matrix>

The *physical-linkage variant* uses moderately increased elbow gains ($K_p = 75$, $K_i = 6$, $K_d = 3$, saturation 500~N) to account for the higher effective inertia of the four-bar linkage mechanism. Wrist gains remain identical to the disc-approximation variant.

==== Performance metrics
Four standard metrics quantify the step-response behavior:

- *Rise time $t_r$* (10%--90%): The time for the response to transition from 10% to 90% of the step amplitude. It characterizes the actuator speed under the given controller gains.
- *Settling time $t_s$* ($plus.minus 2%$ band): The time after which the response remains within $plus.minus 2%$ of the steady-state value. It captures the combined effect of damping, inertia, and integrator windup.
- *Overshoot $M_p$*: The maximum percentage deviation above the target, indicating the degree of underdamping.
- *#ac("NRMSE")*: Defined as $"NRMSE" = 100 times ||bold(q) - bold(q)^*||_2 / (q_"max" - q_"min")$ over the hold phase, providing a single scalar comparison metric. The #acs("NRMSE") values are compared against Klein's reported Gazebo-vs-hardware #acs("NRMSE")~@Klein2023 to assess whether the Isaac Sim model achieves comparable or better agreement.

=== PD Gain Tuning <subsec:pd_tuning>

In addition to the explicit PID tendon controller, the arm joints are also tested with Isaac Sim's built-in `ImplicitActuator` PD drives. The PD gains were determined through a systematic 5-phase tuning procedure:

+ *Step-response baseline* — all base (2 joints $times$ 3 amplitudes) and arm (3 joints $times$ 3 amplitudes) steps at the current gains.
+ *Arm damping sweep* — stiffness $K = 400$ fixed; damping $D$ swept through $\{120, 60, 30, 25, 20, 15\}$. The optimal damping is selected by minimizing the average settling time across all arm joints.
+ *Cross-coupling check* — each arm joint steps $plus.minus 0.30 "rad"$; base drift is monitored with a threshold of $0.01 "rad/m"$ to ensure arm motion does not destabilize the base.
+ *Gripper grasp test* — close fingers on a test cube, lift, hold, and verify grip is maintained under the tuned arm dynamics.
+ *Summary comparison* — current vs. recommended gains are presented side-by-side.

The analytical critical damping for the elbow ($J_"eff" approx 0.22 "kg" dot "m"^2$) is $D_"crit" approx 18.7 "N" dot "m" dot "s/rad"$. The initial configuration used $D = 120$ ($zeta approx 6.4$, massively overdamped); the damping sweep confirmed $D = 20$ as the optimal value ($zeta approx 1.07$, near-critically damped), providing the fastest settling without significant overshoot. The base prismatic joints use $K = 8000$, $D = 800$, tuned analogously via a damping sweep over $\{1600, 1200, 800, 600, 400, 200\}$.

=== Monte Carlo Workspace Analysis <subsec:workspace_analysis>

The reachable workspace and its quality are characterized using Monte Carlo forward-kinematics sampling, following the methodology of Rastegar and Fardanesh~@Rastegar1990WorkspaceMonteCarlo.

==== Sampling procedure
$N = 2 000 000$ joint configurations are sampled uniformly within the joint limits from @tab:joint_ranges across 4,096 parallel simulation environments. For each sample, the Isaac Sim physics engine computes the forward kinematics and the geometric Jacobian at the end-effector body. A gripper-tip offset of $(0, 0, -0.225) "m"$ along the end-effector body's local $Z$ axis accounts for the Robotiq 2F-140 fingertip position. The resulting point cloud is partitioned into a regular 3D voxel grid (cell size $Delta = 0.02 "m"$).

==== Quality metrics
Two per-voxel metrics are computed:

- *Reachability density*: The number of sample configurations mapping to a given voxel, normalized by the total sample count. Dense regions indicate portions of the workspace that are kinematically accessible from many configurations~@Dong2013WorkspaceDensity.
- *Yoshikawa manipulability*~@Yoshikawa1985Manipulability: $w = sqrt(det(bold(J)(bold(q)) bold(J)^top (bold(q))))$, where $bold(J)$ is the $6 times 5$ geometric Jacobian. This scalar quantifies the directional ease of end-effector motion; $w = 0$ indicates a kinematic singularity.

The inverse condition number ($kappa^(-1) = sigma_min / sigma_max$)~@Salisbury1982ArticulatedHands is not used because the 5-#ac("DoF") manipulator's $6 times 5$ Jacobian is rank-deficient by construction ($sigma_min = 0$), making $kappa^(-1) = 0$ everywhere.

==== Model selection
Only the disc-approximation arm variant is used for workspace analysis. As demonstrated in @fig:elbow_comparison, the disc model underestimates the physical elbow's reach because its fixed pivot cannot reproduce the outward shift of the antiparallelogram's migrating instantaneous center of rotation. The disc-based workspace therefore represents a conservative lower bound on the physical robot's reachable volume. Reliable FK sampling of the physical four-bar variant proved infeasible: the PhysX loop-closure constraint (`excludeFromArticulation`) degrades stochastically under large Monte Carlo batches, causing the body-chain integrity filter to reject an unpredictable fraction of samples.

The per-voxel metrics are averaged over all sample configurations mapping to that voxel, producing a volumetric quality map of the workspace.

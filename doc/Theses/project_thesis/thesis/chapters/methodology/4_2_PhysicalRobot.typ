// §4.2 Robot Design
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table
// Schematic is generated externally by
// `doc/Tensegrity_robot/kinematic_schematic.ipynb` and exported as SVG.
// The previous native-CeTZ implementation is preserved (commented out)
// in `shared/formatting/diagrams.typ` under `tensegrity-kinematic-schematic`.

== Robot Design <sec:physical_robot>

This section describes the composite manipulator system used throughout this work. Its design brings together three distinct subsystems of different origin: a tendon-driven tensegrity arm built as physical hardware at FAPS, an industrial parallel-jaw gripper added for its versatility, and a prismatic positioning base created purely in simulation. Understanding each component's design rationale is essential for interpreting the simulation model and the choices made during task development.

=== Kinematic Chain and Degrees of Freedom <subsec:kinematic_chain>

The tensegrity manipulator is a ceiling-mounted, 5-#ac("DoF") serial kinematic chain. It consists of three functional subsystems arranged in series: a 2-#ac("DoF") prismatic positioning base, a 3-#ac("DoF") tendon-driven arm, and a Robotiq 2F-140 parallel-jaw gripper. @fig:kinematic_chain_diagram shows the side-view kinematic schematic of the assembled 5-#ac("DoF") manipulator, and @tab:link_params and @tab:joint_ranges summarize the corresponding link and joint parameters used in simulation.

#faps-figure(
  image("../../../assets/figures/robot/tensegrity_kinematic_schematic.svg", width: 100%),
  caption: [Kinematic schematic of the 5-#ac("DoF") tensegrity manipulator (side view, Y–Z plane). The ceiling-mounted prismatic base ($Y$, $Z$) is followed by the tendon-driven arm with three revolute #ac("DoF") (elbow, wrist~$x$, wrist~$y$) and the Robotiq 2F-140 gripper at the end-effector flange. The dashed inset on the right depicts the alternative *physical antiparallelogram* elbow variant, in which the single revolute `elbow_joint` is replaced by a four-bar linkage with crossed side links.],
  short-caption: [Kinematic schematic of the tensegrity manipulator],
) <fig:kinematic_chain_diagram>

#faps-table(
  table(
    columns: (auto, auto, auto, auto),
    table.header([*Link*], [*Segment length (mm)*], [*Sim.\ mass (kg)*], [*Description*]),
    [Root link], [430], [2.000], [Aluminium upper-arm bracket, fixed to the base],
    [Forearm link], [406], [2.092], [Aluminium forearm tube and elbow disc (disc-approx.\ variant)],
    [Wrist link], [136], [0.400], [Lower wrist platform with #ac("IMU") housing],
    [Tool link], [—], [0.001], [Massless #ac("TCP") frame (kinematic only)],
  ),
  caption: [Segment lengths between consecutive joint frames (root~$arrow$~elbow, elbow~$arrow$~wrist, wrist~$arrow$~#ac("TCP")) and per-link simulation masses for the disc-approximation variant. The total root-to-#ac("TCP") length is 972~mm. The 406~mm forearm segment is the centre-to-centre distance from the elbow disc axis to the wrist base ring used in the simulation. In the physical antiparallelogram variant the forearm link is split into a forearm tube (1.900~kg) and two crossing rod links (0.096~kg each), conserving the total mass.],
  short-caption: [Link segment lengths and simulation masses],
) <tab:link_params>

#faps-table(
  table(
    columns: (auto, auto, auto, auto),
    table.header([*Joint*], [*Type & Axis*], [*Range of Motion*], [*Effort limit (Nm)*]),
    [`base_y_joint`], [Prismatic ($Y$)], [$plus.minus 0.5$~m], [300],
    [`base_z_joint`], [Prismatic ($Z$)], [$-0.5$ to $0.0$~m], [200],
    [`elbow_joint`], [Revolute (pitch)], [$plus.minus 70 degree$], [35.0],
    [`wrist_y_joint`], [Revolute (pitch)], [$plus.minus 50 degree$], [3.5],
    [`wrist_x_joint`], [Revolute (roll)], [$plus.minus 50 degree$], [3.5],
    [`finger_joint`], [Revolute (gripper)], [$0$ to $0.7854$ rad], [200],
  ),
  caption: [Actuated joints of the integrated 5-#ac("DoF") tensegrity manipulator. Joint ranges correspond to the practical workspace measured by Klein~@Klein2023. Effort limits are derived from the motor and tendon specifications (see @subsec:drive_system). The physical antiparallelogram variant additionally exposes the four passive linkage joints `rod_left_joint`, `rod_right_joint`, `coupler_left_joint` and `coupler_right_joint`, the last of which acts as the PhysX loop-closure constraint and is excluded from the articulation tree.],
  short-caption: [Actuated joints],
) <tab:joint_ranges>

==== Tendon-driven arm
The tendon-driven arm is the central subsystem of the manipulator and the only part that exists as physical hardware at FAPS. It was designed and built by #highlight[CITATION NEEDED] and is the subject of the ongoing scientific development that this thesis supports. The remaining subsystems (prismatic base and gripper) are deliberately treated as interchangeable additions around this fixed core, as discussed below.

The arm comprises a upper arm fixed to the base, a forearm, and a two-axis wrist, connected by three rotational #ac("DoF") (one elbow, two wrist axes). All three #ac("DoF") are actuated exclusively through proximal cable transmissions. The five Maxon EC60 motors are mounted on the ceiling frame and drive the joints through Bowden-routed Dyneema tendons, with no direct motor-to-joint coupling at any joint (see @sec:bg_tendon_actuation for the underlying actuation principle and @subsec:drive_system for the routing details). Structural elements are aluminium profiles and machined aluminium parts (grey/silver in @fig:robot_variants). Cable redirection pulleys, brackets, and several mounting parts are 3D-printed from #ac("PLA") (green/white)~@Klein2023.

In the kinematic schematic of @fig:kinematic_chain_diagram and in the simulation model derived from it, the arm is represented as three independent revolute joints. This is a deliberate simplification of the physical hardware:

- The elbow is in reality not a single revolute joint but a planar antiparallelogram four-bar linkage with crossed side links of length $l_e = 150 "mm"$ and frame/coupler bars of length $k_e = 60 "mm"$~@Klein2023. The introduction to crossed four-bar linkages is given in @sec:bg_four_bar, and the mechanism's kinematics, centrode behavior, and tendon attachment are detailed in @subsec:antiparallelogram. The single-revolute representation is the disc approximation used for fast simulation and workspace analysis and was introduced by Klein~@Klein2023. The loop-closed physical-linkage variant coexists in the model and is selected when the configuration-dependent transmission ratio is required.
- The two-axis wrist is in reality the cable-driven parallel mechanism of Nemoto et al.~@Nemoto2022CableWrist, in which three active and three passive cables span a base ring and a platform ring connected by a passive universal joint. Its topology and structure matrix are described in @subsec:wrist_mechanism. The two orthogonal revolute joints (`wrist_x_joint`, `wrist_y_joint`) used in the kinematic chain are the equivalent serial decomposition of the universal joint that the parallel mechanism actuates.

The practical joint ranges of the assembled hardware are $plus.minus 70 degree$ at the elbow and $plus.minus 50 degree$ about each wrist axis, as measured by Klein~@Klein2023 and used in @tab:joint_ranges.

==== Prismatic base
The two prismatic joints translate the arm mounting plate along the horizontal $Y$-axis and the vertical $Z$-axis. The base is a purely simulative addition that does not exist in hardware. It was created using the Isaac Sim Robot Wizard (BETA) and is intentionally kept minimal to act as a placeholder for future base configurations. The $Y$-axis extends the arm's reach along the cross-section of the conveyor belt, the $Z$-axis lowers the arm toward the belt, and an $X$-axis is omitted because the conveyor itself provides transport in that direction. The wrist's two rotational #ac("DoF") absorb residual positioning errors introduced by conveyor tolerances. The base provides a planar positioning range of $plus.minus 0.5$~m in $Y$ and $-0.5$~m to $0.0$~m in $Z$ (downward from the ceiling). Future iterations of the platform may replace this prismatic base with an articulated arm mount, a linear gantry, or a mobile base without affecting the tendon-driven arm itself.

==== Gripper
The Robotiq 2F-140 is an adaptive parallel-jaw gripper with a 140~mm stroke, adjustable grip force from 10~N to 125~N, and a payload capacity of approximately 2.5~kg~@Robotiq2F140Datasheet. It was selected for its versatility across the rigid objects and deformable cloth envisioned in the trash-sorting application, and is likewise treated as a prototypical addition rather than a fixed part of the robot. It is loaded as a self-contained articulated asset from the Isaac Sim built-in asset library, with a single actuated joint driving both fingers symmetrically via an internal linkage. As with the base, the gripper is a swappable end-effector that can be replaced (for example by a suction or soft gripper) in later iterations of the system.

==== Modular assembly
The tendon-driven arm is therefore the only fixed component of the manipulator design. The prismatic base and the parallel-jaw gripper are prototypical additions that bracket it into the conveyor scenario used in this work. To preserve this distinction at the implementation level, the robot's #ac("USD") description, Isaac Lab configuration, and task structure expose the base, arm, and gripper as independently exchangeable subsystems. The same modularity is reflected in the validation and #ac("RL") pipelines (@sec:model_validation, @sec:rl_tasks, @sec:training_infrastructure), which are written to accept alternative robot configurations without modifying the arm model or the core  training code.

@tab:link_params summarizes the principal link segment lengths and the per-link masses used in simulation. Segment lengths are taken directly from the #ac("CAD") assembly. Masses are the values configured on the simulation rigid bodies and approximate the assembled aluminium and 3D-printed components reported by Klein~@Klein2023.

=== Antiparallelogram Elbow Mechanism <subsec:antiparallelogram>

The elbow joint uses a tensegrity-inspired antiparallelogram four-bar linkage as its primary actuation mechanism. This crossed linkage topology provides inherent compliance and impact isolation, a key design goal for safe human--robot interaction described by Walter et al.~@Walter2023Tensegrity.

==== Linkage geometry
The antiparallelogram is a planar four-bar mechanism in which the two side links (length $l_e = 150 "mm"$) cross each other, while the frame and coupler bars share a common length $k_e = 60 "mm"$~@Klein2023. @fig:antiparallelogram_schematic illustrates the topology. The joints labeled $A$ and $B$ are fixed to the frame (root link). Joints $C$ and $D$ are attached to the coupler (forearm) and trace the output motion. The four revolute joints of this mechanism are the minimum representation required to capture the kinematic constraint.

#faps-figure(
  image("../../../assets/figures/robot/antiparallelogram_static.svg", width: 100%),
  caption: [Five-pose overview of the antiparallelogram four-bar linkage forming the elbow mechanism, generated from the kinematic model. The linkage sweeps from $-75 degree$ to $+75 degree$ elbow angle. Joints $A$ and $B$ are grounded on the root link (frame, length $k_e$). Joints $C$ and $D$ are connected by the coupler (length $k_e$). The crossed side links each have length $l_e$. Centrode ellipses, instantaneous center of rotation (ICR), and tendon force directions are shown.],
  short-caption: [Antiparallelogram elbow linkage],
) <fig:antiparallelogram_schematic>

==== Kinematic closure condition
The configuration of a four-bar linkage is fully determined by a single input angle. Following the analytical framework of McCarthy and Soh~@McCarthy2011, the loop-closure equation for the antiparallelogram relates the input crank angle $theta$ (measured at joint $A$) to the output angle $phi$ (measured at joint $B$) through a set of algebraic constraints derived from the polygon-closure condition:
$ bold(Z)_1 e^(i theta) + bold(Z)_2 e^(i beta) = bold(Z)_3 e^(i phi) + bold(Z)_4 $ <eq:loop_closure>
where $bold(Z)_i$ denote the complex link vectors and $beta$ is the coupler angle. For the specific case of the antiparallelogram ($|bold(Z)_1| = |bold(Z)_3| = l_e$, $|bold(Z)_2| = |bold(Z)_4| = k_e$), this reduces to a tangent-half-angle substitution yielding a closed-form expression for $phi(theta)$~@McCarthy2011.

==== Centrode geometry of the FAU elbow
The general antiparallelogram result of Dijksman~@Dijksman1976MotionGeometry ( the fixed centrode is an ellipse with foci at the two grounded pivots and a focal-string length equal to the side-link length ) is summarised in @sec:tendon_linkage_mechanisms. Substituting the manufactured dimensions $l_e = 150 "mm"$ and $k_e = 60 "mm"$~@Klein2023 yields the specific values
$ a = l_e / 2 = 75.00 "mm", quad b = (sqrt(l_e^2 - k_e^2)) / 2 approx 68.74 "mm", quad c = k_e / 2 = 30.00 "mm" $ <eq:centrode_axes>
for the semi-major axis $a$, semi-minor axis $b$ and linear eccentricity $c$ of the fixed centrode ellipse, with the two foci coinciding with the grounded joints $A = (-30, 0) "mm"$ and $B = (+30, 0) "mm"$. The numerical eccentricity is $e = c/a = k_e/l_e = 0.4$, and the aspect ratio $b/a approx 0.917$ is close to unity. The moving centrode is a congruent ellipse with foci at the coupler joints $C$ and $D$, and the two ellipses roll on each other without slipping~@Dijksman1976MotionGeometry @BryantSangwin2008HowRound.

The instant centre of rotation #ac("ICR") is the rolling-contact point. At the symmetric equilibrium ($theta = theta_0 = arcsin(k_e / l_e) approx 23.58 degree$, the equilibrium rod inclination from the vertical) the #ac("ICR") sits on the symmetry axis at $(0, -b) = (0, -68.74) "mm"$. That is, $68.74 "mm"$ below the line through $A$ and $B$. As the elbow flexes from $-75 degree$ to $+75 degree$, the #ac("ICR") migrates along the upper half of the centrode ellipse between the two extreme rolling-contact points, displacing horizontally by up to $plus.minus 48 "mm"$ and vertically by $approx 16 "mm"$ relative to the equilibrium location. The output (coupler) angle $phi$ is therefore *not* a rigid rotation about a fixed pivot, and a single revolute joint at the geometric centre of the linkage is only an approximation to the true motion.

==== Effective transmission and singularity margin
Because the #ac("ICR") migrates, the effective moment arm $r_"eff" (phi)$ between an applied tendon force and the elbow output angle is configuration-dependent. The disc-approximation variant (@subsec:disc_approx) collapses this geometry onto a single revolute joint with the constant lever arm $r_e = 72.5 "mm"$ that Klein~@Klein2023 fitted to the cable routing of the prototype, while the physical-linkage variant (@subsec:physical_linkage) retains the four-bar geometry and so reproduces the configuration-dependent torque. Wang et al.~@Wang2023humanoid quantify the same effect for an analogous #ac("APM") elbow by comparing the elliptical centrode trajectory against an ideal pure-circular-rolling joint and reporting a deviation
$ d(theta_A) = x_p / sin theta_A - (h_p / 2 + d_0) $ <eq:wang_deviation>
which grows monotonically toward the workspace boundary. The FAPS elbow inherits the same qualitative error budget from the disc approximation.

Two further geometric consequences bound the usable workspace. First, the antiparallelogram has a parallelogram-bifurcation singularity when the two crossed side links become collinear with the frame (nominally at $phi = plus.minus 90 degree$) so the empirical $plus.minus 75 degree$ limit reported by Klein~@Klein2023 keeps a margin from this kinematic singularity. Second, the input-to-output relation $phi(theta)$ derived from the closure condition (@eq:loop_closure) is non-linear. A uniform sweep of the elbow output through $plus.minus 75 degree$ corresponds to an asymmetric rod-deviation interval of approximately $[-21 degree, +35 degree]$ around $theta_0$, which directly motivates the set joint limits within the physical elbow model variant.

==== Tendon actuation of the elbow
Two antagonistic tendons are attached at the midpoints of the side links and route via Bowden sheaths to the ceiling-mounted Maxon EC60 spool drums (cf.~@subsec:drive_system). When one tendon is tensioned, the crossed side links rotate, the coupler (forearm) swings about the migrating #ac("ICR") of @eq:centrode_axes, and the second tendon lengthens under residual pre-tension. The elbow torque is therefore a function of both the applied cable tension and the instantaneous linkage geometry. For the disc-approximation variant this relationship is linearised through the constant lever arm $r_e = 72.5 "mm"$, yielding the antagonistic torque envelope reported in @eq:effort_elbow. For the physical variant the torque is recomputed at every simulation step from the actual force-application geometry of the current configuration.

=== Cable-Driven Wrist Mechanism <subsec:wrist_mechanism>

The wrist provides two rotational degrees of freedom (pitch and roll) and is based on the cable-driven parallel mechanism described by Nemoto et al.~@Nemoto2022CableWrist. It connects the forearm tube to the end-effector plate through a passive universal joint, and motion is controlled entirely by three active cables. @fig:wrist_3d shows a three-dimensional overview of the wrist mechanism, including the six cable attachment points and the workspace envelope.

==== Topology
The wrist consists of a base ring (attached to the forearm), a platform ring (attached to the end-effector), and six cables arranged symmetrically. Three cables are _passive_ (structural, constraining the platform) and three are _active_ (driven by motors via tendons routed through the forearm)~@Nemoto2022CableWrist. The active cables are spaced at $120 degree$ intervals around the circumference and pass through guide holes in the base ring at a radius $r_w$ from the wrist center.

==== Structure matrix
The mapping from cable tensions to platform wrenches is described by a structure matrix $bold(A) in bb(R)^(2 times 3)$ that captures the geometric arrangement of the three active cables. For small angular deflections, the cable direction vectors project onto the two rotational axes of the wrist, yielding the approximate relationship:
$ bold(tau)_w = bold(A) bold(T)_w $ <eq:wrist_structure_matrix>
where $bold(tau)_w = [tau_"pitch", tau_"roll"]^top$ and $bold(T)_w = [T_1, T_2, T_3]^top$ denotes the three active cable tensions. The entries of $bold(A)$ are determined by the angular positions of the cable attachment points and the wrist radius. At the zero configuration with cables at $0 degree$, $120 degree$, and $240 degree$, the structure matrix takes the form:
$ bold(A) = r_w mat(
  cos(0 degree), cos(120 degree), cos(240 degree);
  sin(0 degree), sin(120 degree), sin(240 degree)
) = r_w mat(
  1, -1/2, -1/2;
  0, sqrt(3)/2, -sqrt(3)/2
) $ <eq:structure_matrix_explicit>

This arrangement is redundantly actuated (three cables, two #ac("DoF")), ensuring that the tension distribution can always satisfy the unilateral cable constraint $bold(T)_w >= 0$ within the wrist's operating range~@Pott2009ClosedFormForce @Verhoeven2004TendonPlatforms.

#faps-figure(
  image("../../../assets/figures/robot/wrist_static_3d.svg", width: 100%),
  caption: [Three-dimensional overview of the cable-driven 2-#ac("DoF") wrist mechanism, generated from the kinematic model. The base ring (radius $R_B = 80 "mm"$) is shown with six cable attachment points: three active cables (red) and three passive cables (blue). The platform ring (radius $r_w = 20 "mm"$) tilts about two perpendicular axes. Ghost poses illustrate the workspace envelope. The semi-transparent green cone visualizes the lower wrist component acting as a compression element in the sense of tensegrity.],
  short-caption: [Cable-driven wrist mechanism],
) <fig:wrist_3d>

=== Drive System and Tendon Routing <subsec:drive_system>

The five tendons (two for elbow, three for wrist) are driven by five identical Maxon EC60 flat brushless #ac("DC") motors (150~W, 24~V nominal), each controlled by a Maxon EPOS4 70/15 positioning controller in current-regulation mode~@MaxonEC60Datasheet @MaxonEPOS4Datasheet @Klein2023. The motors are mounted on the ceiling frame and the cables are routed through Bowden sheaths (Dyneema cable, $diameter = 1 "mm"$) to their respective attachment points on the arm~@Klein2023.

==== Spool and pulley system
Each motor winds a Dyneema cable onto a spool with radius $r_s = 5 "mm"$. For the elbow, the cable additionally passes through a block-and-tackle pulley arrangement (3D-printed #acs("PLA") pulleys, $diameter = 37 "mm"$) that produces a 3:1 mechanical advantage between the motor spool and the cable attachment on the antiparallelogram side links~@Klein2023. The wrist tendons are direct-drive (1:1, no additional reduction). The effective lever arm at the joint is independent of the spool radius. For the elbow it is $r_e = 72.5 "mm"$ (perpendicular distance from cable attachment to the effective elbow rotation axis at zero configuration), and for the wrist it equals the platform radius $r_w = 20 "mm"$~@Klein2023.

==== Maximum tensions
The maximum cable tension at the motor spool is $T_"motor" = tau_"motor,max" / r_s$. With the nominal continuous motor torque $tau_"motor" = 0.401 "Nm"$ this yields a per-motor force of $80 "N"$, and at the power-supply peak torque ($0.79 "Nm"$) it reaches $158 "N"$~@Klein2023. The 3:1 elbow mechanical advantage triples the tension at the cable attachment point, so the elbow attachment-side tension is up to $T_"max,elbow" approx 474 "N"$ (peak) or $240 "N"$ (continuous). The wrist cables, being direct-drive, never exceed $T_"max,wrist" approx 158 "N"$ (peak) or $80 "N"$ (continuous):
$ T_"max,elbow" = 3 dot tau_"motor,max" / r_s, quad T_"max,wrist" = tau_"motor,max" / r_s. $ <eq:max_tension_elbow>
Reflecting Klein's controller saturation limits (160~N motor-side elbow, 80~N wrist), the simulation models the cable saturations per tendon: $T_"max,elbow" = 480 "N"$ at the elbow attachment side (160~N $times$ 3:1 pulley) and $T_"max,wrist" = 160 "N"$ at the wrist attachment side.

==== Effort limits
The joint effort limits used by the PhysX solver cap the maximum torque producible through the PD driven joints in the substitution PD base model. Because the two elbow tendons act antagonistically, the worst-case elbow torque is generated when one cable pulls at $T_"max,elbow"$ while the other is slack:
$ tau_"max,elbow" approx T_"max,elbow" dot r_e = 480 dot 0.0725 approx 34.8 "Nm". $ <eq:effort_elbow>
The simulation rounds this up to a configured elbow effort limit of $35 "Nm"$~@Klein2023. For the wrist, the analogous antagonistic worst-case torque from two contributing active cables at the continuous tension limit is
$ tau_"max,wrist" approx T_"cable,max" dot r_w = 160 dot 0.020 = 3.2 "Nm", $ <eq:effort_wrist>
rounded up to the configured wrist effort limit of $3.5 "Nm"$ per axis. The resulting joint-level effort limits are tabulated together with the kinematic ranges in @tab:joint_ranges (§ @subsec:kinematic_chain).

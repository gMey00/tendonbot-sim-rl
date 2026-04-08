// §4.2 Robot Design
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table
#import "../../../../shared/formatting/diagrams.typ": kinematic-chain-diagram

== Robot Design <sec:physical_robot>

This section describes the composite manipulator system used throughout this work. Its design brings together three distinct subsystems of different origin: a tendon-driven tensegrity arm built as physical hardware at FAPS, an industrial parallel-jaw gripper added for its versatility, and a prismatic positioning base created purely in simulation. Understanding each component's provenance and design rationale is essential for interpreting the simulation model and the choices made during task development.

The tensegrity arm is the only part of the robot that exists as real hardware; it was designed and built at FAPS, and its mechanical design and initial characterization are documented by Klein~@Klein2023 and Walter et al.~@Walter2023Tensegrity. The gripper and base are simulated additions that complete the manipulator for the target application of autonomous waste sorting, where objects of widely varying size and shape — including cloth items in the planned master thesis extension — must be reliably grasped and placed.

=== Kinematic Chain and Degrees of Freedom <subsec:kinematic_chain>

The tensegrity manipulator is a ceiling-mounted, 5-#ac("DoF") serial kinematic chain. It consists of three functional subsystems arranged in series: a 2-#ac("DoF") prismatic positioning base, a 3-#ac("DoF") tendon-driven arm, and a Robotiq 2F-140 parallel-jaw gripper. @fig:kinematic_chain_diagram shows the kinematic topology, and @fig:robot_variants shows the fully assembled manipulator variants rendered in simulation.

#faps-figure(
  kinematic-chain-diagram(),
  caption: [Kinematic topology of the 5-#ac("DoF") tensegrity manipulator. The ceiling-mounted base provides two prismatic #ac("DoF"). The tendon-driven arm contributes three revolute #ac("DoF") (elbow, wrist pitch, wrist roll). The Robotiq 2F-140 gripper attaches at the wrist output flange.],
  short-caption: [Kinematic topology of the tensegrity manipulator],
) <fig:kinematic_chain_diagram>

#faps-figure(
  image("../../../../shared/figures/tensegrity_variants.png", width: 90%),
  caption: [Robot model variants rendered in Isaac Sim. From left to right: 2-#ac("DoF") prismatic base, 3-#ac("DoF") tendon-driven arm (disc approximation), assembled 5-#ac("DoF") manipulator, and 5-#ac("DoF") manipulator with Robotiq 2F-140 gripper. Each variant is available with full-resolution and low-resolution collision meshes, yielding eight #ac("USD") assemblies in total.],
  short-caption: [Robot model variants rendered in Isaac Sim],
) <fig:robot_variants>

*Prismatic base.* The two prismatic joints translate the arm mounting plate along the horizontal $Y$-axis and the vertical $Z$-axis. The base is a purely simulative addition; it was created using the Isaac Sim Robot Wizard (BETA) and is intentionally oversimplified to allow rapid prototyping of different base configurations. Its $Z$-axis provides linear motion in the same direction as the elbow swing, extending the arm's vertical reach, while the $Y$-axis approximates the lateral positioning that a ceiling-mounted conveyor rail would provide in the physical setup. The conveyor itself handles the remaining transport direction; the wrist's two rotational degrees of freedom compensate for residual positioning errors introduced by conveyor tolerances. The base provides a planar positioning range of $plus.minus 0.5 "m"$ in $Y$ and $-0.5 "m"$ to $0.0 "m"$ in $Z$ (downward from the ceiling).

*Tendon-driven arm.* The arm is the sole hardware component and comprises an upper arm (root link), a forearm, and a two-axis wrist, connected by three revolute joints. All three joints are actuated exclusively through cable-driven transmissions; no direct motor-to-joint coupling exists. The elbow joint provides $plus.minus 70 degree$ of rotation in the sagittal plane. The wrist mechanism provides $plus.minus 50 degree$ about each of two perpendicular axes (pitch and roll), enabling the end-effector to orient relative to the forearm axis~@Klein2023.

*Gripper.* The Robotiq 2F-140 is an adaptive parallel-jaw gripper with a 140~mm stroke, adjustable grip force from 10~N to 125~N, and a payload capacity of approximately 2.5~kg~@Robotiq2F140Datasheet. It was selected for its versatility and adaptability to a wide range of object sizes and shapes, which is critical for the envisioned trash-sorting application where the robot must handle everything from rigid cubes to deformable cloth. The gripper is modeled as a self-contained articulated asset loaded from the Isaac Sim built-in asset library. A single actuated joint drives both fingers symmetrically via an internal linkage.

*Modular assembly.* The robot's #ac("USD") description, Isaac Lab configuration, and task structure are designed so that the base, arm, and gripper are independently exchangeable. This modularity makes it straightforward to explore alternative base or gripper combinations — for example, replacing the prismatic base with an articulated arm mount or substituting a suction gripper — without modifying the training pipeline.

@tab:link_params summarizes the principal link dimensions and masses of the tendon-driven arm, extracted from the #ac("CAD") assembly and verified against Klein~@Klein2023.

#faps-table(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Link*], [*Length (mm)*], [*Mass (kg)*], [*Description*]),
    [Root link], [150], [2.6737], [Upper arm, elbow motor housing],
    [Forearm], [300], [0.8891], [Carbon-fiber tube with cable guides],
    [Wrist plate], [55], [0.0653], [2-#ac("DoF") cable-driven joint platform],
    [EE link], [10], [0.001], [End-effector mounting flange],
  ),
  caption: [Link dimensions and masses of the tendon-driven arm as specified in the #ac("CAD") model~@Klein2023. The root link contains the primary elbow actuation mechanism. The forearm serves as the structural conduit for the wrist tendons.],
  short-caption: [Link dimensions and masses of the tendon-driven arm],
) <tab:link_params>

=== Antiparallelogram Elbow Mechanism <subsec:antiparallelogram>

The elbow joint uses a tensegrity-inspired antiparallelogram four-bar linkage as its primary actuation mechanism. This crossed linkage topology provides inherent compliance and impact isolation, a key design goal for safe human--robot interaction described by Walter et al.~@Walter2023Tensegrity.

*Linkage geometry.* The antiparallelogram is a planar four-bar mechanism in which the two side links (length $l_e = 150 "mm"$) cross each other, while the frame and coupler bars share a common length $k_e = 60 "mm"$~@Klein2023. @fig:antiparallelogram_schematic illustrates the topology. The joints labeled $A$ and $B$ are fixed to the frame (root link); joints $C$ and $D$ are attached to the coupler (forearm) and trace the output motion. The four revolute joints of this mechanism are the minimum representation required to capture the kinematic constraint.

#faps-figure(
  image("../../../../shared/figures/antiparallelogram_static.svg", width: 100%),
  caption: [Five-pose overview of the antiparallelogram four-bar linkage forming the elbow mechanism, generated from the kinematic model. The linkage sweeps from $-75 degree$ to $+75 degree$ elbow angle. Joints $A$ and $B$ are grounded on the root link (frame, length $k_e$). Joints $C$ and $D$ are connected by the coupler (length $k_e$). The crossed side links each have length $l_e$. Centrode ellipses, instantaneous center of rotation (ICR), and tendon force directions are shown.],
  short-caption: [Five-pose overview of the antiparallelogram elbow linkage],
) <fig:antiparallelogram_schematic>

*Kinematic closure condition.* The configuration of a four-bar linkage is fully determined by a single input angle. Following the analytical framework of McCarthy and Soh~@McCarthy2010Linkages, the loop-closure equation for the antiparallelogram relates the input crank angle $theta$ (measured at joint $A$) to the output angle $phi$ (measured at joint $B$) through a set of algebraic constraints derived from the polygon-closure condition:
$ bold(Z)_1 e^(i theta) + bold(Z)_2 e^(i beta) = bold(Z)_3 e^(i phi) + bold(Z)_4 $ <eq:loop_closure>
where $bold(Z)_i$ denote the complex link vectors and $beta$ is the coupler angle. For the specific case of the antiparallelogram ($|bold(Z)_1| = |bold(Z)_3| = l_e$, $|bold(Z)_2| = |bold(Z)_4| = k_e$), this reduces to a tangent-half-angle substitution yielding a closed-form expression for $phi(theta)$~@McCarthy2010Linkages.

*Centrode analysis.* The instantaneous center of rotation of the coupler relative to the frame traces a curve known as the fixed centrode. For the antiparallelogram, Dijksman~@Dijksman1977Antiparallelogram showed that the centrodes are ellipses, reflecting the fact that the mechanism's instant center migrates continuously during motion rather than remaining fixed. This migration produces a nonlinear transmission ratio: the effective moment arm of the tendons with respect to the elbow output angle changes as a function of the elbow configuration. The disc approximation model (described in @subsec:disc_approx) replaces this configuration-dependent behavior with a constant lever arm, while the physical linkage model (@subsec:physical_linkage) preserves it.

*Tendon actuation of the elbow.* Two antagonistic tendons are attached at the midpoints of the side links and route to spool drums on Maxon EC60 motors. When one tendon is tensioned, the crossed links rotate and the coupler (forearm) swings about the effective instant center. The elbow torque is therefore a function of both the applied cable tension and the instantaneous geometry of the linkage. For the disc approximation variant, this relationship is linearized using a constant lever arm $r_e = 72.5 "mm"$ (the perpendicular distance from tendon attachment to the revolute axis at zero configuration). For the physical variant, the torque depends on the actual force application geometry at the current linkage configuration.

=== Cable-Driven Wrist Mechanism <subsec:wrist_mechanism>

The wrist provides two rotational degrees of freedom (pitch and roll) and is based on the cable-driven parallel mechanism described by Nemoto et al.~@Nemoto2022CableWrist. It connects the forearm tube to the end-effector plate through a passive universal joint, and motion is controlled entirely by three active cables.

*Topology.* The wrist consists of a base ring (attached to the forearm), a platform ring (attached to the end-effector), and six cables arranged symmetrically. Three cables are _passive_ (structural, constraining the platform) and three are _active_ (driven by motors via tendons routed through the forearm)~@Nemoto2022CableWrist. The active cables are spaced at $120 degree$ intervals around the circumference and pass through guide holes in the base ring at a radius $r_w$ from the wrist center.

*Structure matrix.* The mapping from cable tensions to platform wrenches is described by a structure matrix $bold(A) in bb(R)^(2 times 3)$ that captures the geometric arrangement of the three active cables. For small angular deflections, the cable direction vectors project onto the two rotational axes of the wrist, yielding the approximate relationship:
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
  image("../../../../shared/figures/wrist_static_3d.svg", width: 100%),
  caption: [Three-dimensional overview of the cable-driven 2-#ac("DoF") wrist mechanism, generated from the kinematic model. The base ring (radius $R = 72.5 "mm"$) is shown with six cable attachment points: three active cables (red) and three passive cables (blue). The platform ring (radius $r = 25 "mm"$) tilts about two perpendicular axes. Ghost poses illustrate the workspace envelope.],
  short-caption: [3D overview of the cable-driven wrist mechanism],
) <fig:wrist_3d>

#faps-figure(
  image("../../../../shared/figures/tensegrity_wrist_joints.png", width: 55%),
  caption: [Wrist joint axes of the cable-driven 2-#ac("DoF") wrist as modeled in Isaac Sim. The two revolute joints are arranged perpendicular to each other, providing pitch and roll motion of the end-effector plate relative to the forearm. The three active cable attachment points are visible on the wrist platform ring.],
  short-caption: [Wrist joint axes of the cable-driven wrist],
) <fig:wrist_joints>

=== Drive System and Tendon Routing <subsec:drive_system>

The five tendons (two for elbow, three for wrist) are driven by five identical Maxon EC60 flat brushless DC motors (150~W, 24~V nominal), each controlled by a Maxon EPOS4 70/15 positioning controller~@MaxonEC60Datasheet @MaxonEPOS4Datasheet. The motors are mounted in the root link housing, and cables are routed through the structural tubes to their respective attachment points.

==== Spool and pulley system
Each motor winds a cable onto a spool with radius $r_s = 5 "mm"$. For the elbow, a 3:1 pulley reduction between the motor spool and the tendon attachment amplifies the mechanical advantage, yielding an effective elbow lever arm of:
$ r_"eff,elbow" = 3 times r_s = 15 "mm" $ <eq:elbow_lever>
The wrist tendons are direct-drive (no additional reduction), so the effective wrist lever arm equals the wrist platform radius~@Klein2023.

==== Maximum tensions
The maximum continuous motor torque ($tau_"motor,max"$) divided by the spool radius determines the maximum cable tension per motor. With the 3:1 elbow pulley, the maximum elbow tendon tension is three times the single-motor maximum:
$ T_"max,elbow" = 3 dot tau_"motor,max" / r_s $ <eq:max_tension_elbow>
The wrist tendons each carry at most $T_"max,wrist" = tau_"motor,max" / r_s$ per cable. In simulation, a unified saturation limit of $T_"max" = 500 "N"$ per tendon is used for all five tendons, derived from the motor specifications and verified to remain within the 10~A continuous current limit of the EPOS4 controller~@Klein2023.

==== Effort limits
The corresponding joint effort limits used by the physics solver are computed from the maximum torque producible through the tendon arrangement:
$ tau_"max,elbow" = 2 dot T_"max" dot r_"eff,elbow" = 2 times 500 times 0.0725 = 72.5 "Nm" $ <eq:effort_elbow>
For the wrist, maximum single-joint effort depends on the tendon geometry and the structure matrix. The configured effort limits are 72.5~Nm for the elbow and 80~Nm for each wrist joint (accounting for the redundant cable arrangement that allows all three cables to contribute cooperatively).

@tab:joint_ranges summarizes the joint parameters of the assembled 5-#ac("DoF") manipulator.

#faps-table(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Joint*], [*Type & Axis*], [*Range of Motion*], [*Effort limit (Nm)*]),
    [`base_y_joint`], [Prismatic ($Y$)], [$plus.minus 0.5$~m], [200],
    [`base_z_joint`], [Prismatic ($Z$)], [$-0.5$ to $0.0$~m], [200],
    [`elbow_joint`], [Revolute (pitch)], [$plus.minus 70 degree$], [72.5],
    [`wrist_y_joint`], [Revolute (pitch)], [$plus.minus 50 degree$], [80],
    [`wrist_x_joint`], [Revolute (roll)], [$plus.minus 50 degree$], [80],
    [`finger_joint`], [Revolute (gripper)], [$0$ to $0.7854$ rad], [200],
  ),
  caption: [Actuated joints of the integrated 5-#ac("DoF") tensegrity manipulator. Joint ranges correspond to the practical workspace reported by Klein~@Klein2023. Effort limits are derived from the motor and tendon specifications.],
  short-caption: [Actuated joints of the tensegrity manipulator],
) <tab:joint_ranges>

// §2.2 Tendon-Driven Mechanisms and Linkage Kinematics
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table
#import "../../../../shared/formatting/diagrams.typ": tendon-antagonistic, tensegrity-concept
#reg-sym("sym:theta", "sym:T", "sym:tau")

== Tendon-Driven Mechanisms and Linkage Kinematics <sec:tendon_linkage_mechanisms>

This section introduces the mechanical and kinematic foundations of the robot used in this thesis. The manipulator combines a tendon-driven actuation paradigm with a linkage-based elbow joint and a cable-driven wrist, all motivated by the structural principles of tensegrity. Understanding these concepts is necessary for interpreting the robot models (@sec:physical_robot), the tendon actuation modes (@sec:tendon_actuation), and the simulation choices made in the methodology.

=== Tendon-Driven Actuation Principles <sec:bg_tendon_actuation>

In tendon-driven robots, joint motion is produced by tensile elements (tendons or cables) that transmit forces from remotely placed actuators to the joints through pulleys, guides, or routing channels. The actuators can therefore be positioned proximally, keeping the distal structure lightweight and reducing its inertia. This actuation paradigm is widely used in cable-driven mechanisms, robotic hands, and bio-inspired designs. Particularly where low distal inertia, compact joint packaging, or intrinsic compliance is desired.~@Rao2021TDCRModeling @Nemoto2022CableWrist

==== Antagonistic arrangements
 Because cables can only pull, a single tendon cannot produce bidirectional joint torque. Joints are therefore actuated by _antagonistic_ tendon pairs (or groups), as shown in @fig:tendon_antagonistic. Differential tension produces a net joint torque, while simultaneous co-contraction modulates apparent joint stiffness without changing the joint angle~@Kobayashi1998NullSpaceStiffness@MuralidharanWenger2021. The resulting feasible torque–stiffness space is illustrated in @fig:torque_stiffness. For an $n$-#ac("DoF") system, Tsai and Lee~@TsaiLee1989 established that $n + 1$ cables are the minimum required for full force closure (bidirectional torque on all joints with positive cable tensions), while additional cables beyond $n + 1$ provide redundancy that can be exploited for stiffness modulation and tension optimization~@TsaiLee1989@Kobayashi1998NullSpaceStiffness.

==== Transmission non-idealities
The mechanical transmission through cables introduces modeling complexity beyond direct-drive systems. Tendon stretch, cable--pulley compliance, and frictional effects cause configuration-dependent motion errors, phase lags, and hysteresis loops under load. These effects break the assumption of a unique, memoryless mapping between actuator displacement and end-effector pose. As a result, tendon-driven systems are often controlled using local joint sensing to close the loop in joint space, or model-based compensation to improve the actuator-to-joint mapping.~@Miyasaka2020HysteresisFriction@Kato2016Neuroendoscopy

#faps-figure(
  image("../../../assets/figures/background/concept_antagonistic_joint.svg", width: 100%),
  caption: [Antagonistic tendon actuation at a single revolute joint shown in three configurations. Two cables $T_1$ (red) and $T_2$ (orange) are routed from proximal motors $M_1$ and $M_2$ on the fixed frame through a pulley of radius $r$ at the joint centre. Both cables attach to opposite sides of the pulley rim. 
  //In the neutral configuration (centre), equal tensions produce zero net torque. Differential tension produces a net torque $tau = r(T_1 - T_2)$ that rotates the output link in opposite directions for the left and right panels. Simultaneous co-contraction ($T_1 + T_2$) modulates apparent joint stiffness $kappa$ independently of position, as shown quantitatively in @fig:torque_stiffness.
  ],
  short-caption: [Antagonistic tendon actuation at a revolute joint],
) <fig:tendon_antagonistic>

#faps-figure(
  image("../../../assets/figures/background/concept_torque_stiffness.svg", width: 100%),
  caption: [Feasible torque--stiffness space for an antagonistic tendon pair with tensions bounded by $0 <= T_i <= T_"max"$. The diamond-shaped region shows all achievable combinations of normalised net torque $tau slash tau_"max"$ and normalised apparent stiffness $kappa slash kappa_"max"$. The vertices correspond to the extremal tension combinations. 
  //maximum co-contraction at the top ($T_1 = T_2 = T_"max"$, maximum stiffness), zero tension at the bottom, and maximum unilateral torque on the sides. 
  The dashed purple line indicates the null-space direction (co-contraction at zero torque), along which stiffness can be modulated without affecting the joint torque. The green lines illustrate the available stiffness range at two fixed torque values.
  //demonstrating that stiffness modulation decreases as the demanded torque increases.
  ],
  short-caption: [Feasible torque--stiffness region of an antagonistic tendon pair],
) <fig:torque_stiffness>

=== Planar Four-Bar Linkages and the Antiparallelogram <sec:bg_four_bar>

The elbow joint of the #highlight(fill:red)[tensegrity manipulator] uses an antiparallelogram four-bar linkage. This subsection introduces the relevant linkage theory. The specific kinematic closure equations for the elbow are derived in @subsec:antiparallelogram of the methodology.

==== Four-bar linkage fundamentals
 A planar four-bar linkage consists of four rigid bars connected by four revolute joints, forming a single closed loop with one #ac("DoF")~@McCarthy2011 @Uicker2017. One bar is conventionally designated as the _frame_ (grounded link). The remaining links are the _input crank_, _coupler_, and _output rocker_. The Grashof condition classifies four-bar mechanisms by the range of motion of each link: if the sum of the shortest and longest link lengths does not exceed the sum of the other two, at least one link can rotate fully, yielding crank--rocker, double-crank, or double-rocker configurations~@Uicker2017. The configuration is fully determined by a single input angle through the _loop-closure equation_, which can be expressed in complex-number form as

$ bold(Z)_1 e^(i theta) + bold(Z)_2 e^(i beta) = bold(Z)_3 e^(i phi) + bold(Z)_4 $ <eq:bg_loop_closure>

where $bold(Z)_i$ denote the complex link vectors, $theta$ is the input angle, $phi$ the output angle, and $beta$ the coupler angle~@McCarthy2011 @Uicker2017. The _transmission angle_ (the angle between the coupler and the output link) characterizes force transmission quality. Values near $90 degree$ maximize force transfer, while values approaching $0 degree$ or $180 degree$ indicate near-singular configurations~@Uicker2017 @Hartenberg1955.

 #faps-figure(
  image("../../../assets/figures/background/concept_four_bar_types.svg", width: 100%),
  caption: [Comparison of the two four-bar linkage topologies relevant to tendon-driven manipulators. Left: the standard parallelogram, where both side links (blue) remain parallel throughout motion and the instantaneous centre of rotation (#ac("ICR"), purple cross) lies at infinity, producing pure translational coupler motion. Right: the antiparallelogram, where the side links cross each other. The #ac("ICR") migrates along a centrode ellipse (purple dashed), creating a configuration-dependent lever arm. 
  // This nonlinear transmission ratio distinguishes the antiparallelogram from a simple revolute joint and requires either the physical linkage model or a disc approximation for torque computation. Multiple ghost poses at different input angles illustrate the swept workspace of each topology.
  ],
  short-caption: [Linkage topologies],
) <fig:four_bar_types>

==== Antiparallelogram topology
 The antiparallelogram is a special case of the four-bar linkage in which the two side links (cranks) are of equal length and _cross_ each other, while the frame and coupler share a common length~@Dijksman1976MotionGeometry @McCarthy2011. This crossed topology produces qualitatively different kinematics from the more common parallelogram arrangement. @fig:four_bar_types contrasts both topologies and visualises the swept workspace and #ac("ICR") behaviour.

==== Centrode geometry and #ac("ICR") migration
 The #ac("ICR") of the coupler relative to the frame traces a curve called the _fixed centrode_. For a standard revolute joint, the #ac("ICR") is fixed at the joint axis. For the antiparallelogram, Dijksman~@Dijksman1976MotionGeometry showed that the fixed centrode is an _ellipse_, meaning the #ac("ICR") migrates continuously during motion. Bryant and Sangwin~@BryantSangwin2008HowRound give the ellipse semi-axes as $beta$ and $sqrt(beta^2 - alpha^2)$, where $alpha$ and $beta$ are determined by the link lengths.
This migrating #ac("ICR") produces a _nonlinear, configuration-dependent transmission ratio_. The effective lever arm between the tendon attachment point and the output rotation axis changes continuously with the elbow angle, complicating the force-torque relationship.

==== Implications for tendon actuation
 The variable transmission ratio has direct consequences for tendon-driven actuation. As the elbow moves away from the zero configuration, the effective moment arm of the tendon attachment changes, meaning the same cable tension produces different joint torques at different configurations.~@FuretWenger2019

==== Loop-closure enforcement
 The antiparallelogram can bifurcate into a parallelogram configuration at an instability point where the two side links are collinear. In physical mechanisms, cables or mechanical stops prevent this mode-switching by constraining the linkage to the crossed configuration.~@FuretWenger2019

=== #acp("CDPM") <sec:bg_cdpm>

The wrist mechanism of the #highlight(fill:red)[tensegrity manipulator] is a #ac("CDPM") where three active cables, arranged symmetrically around a universal joint, control two rotational degrees of freedom. This subsection introduces the general framework. The specific structure matrix for the thesis robot's wrist is derived in @subsec:wrist_mechanism of the methodology.

==== Concept
A #ac("CDPM") consists of a platform (end-effector) connected to a fixed base frame by $m$ cables, where platform motion is controlled by adjusting cable lengths~@Pott2018CDPR. Unlike rigid-link parallel robots, cables can only transmit tensile forces, which fundamentally constrains the feasible wrench (forces and/or torques) space~@Pott2018CDPR @Verhoeven2004TendonPlatforms. @fig:cdpm_concept illustrates the general concept of a #ac("CDPM"), showing a possible cable arrangement and the resulting platform wrench.

==== Structure matrix formalism
 The static relationship between cable tensions and the platform wrench is described by the _structure matrix_ $bold(A)^top$. Following the notation established by Verhoeven~@Verhoeven2004TendonPlatforms and formalized by Bruckmann _et al._~@Bruckmann2008StructureMatrix, the equilibrium condition is

$ bold(w) = bold(A)^top bold(t), $ <eq:bg_structure_matrix>

where $bold(w) in bb(R)^n$ is the desired platform wrench, $bold(t) in bb(R)^m$ is the cable tension vector, and $bold(A)^top in bb(R)^(n times m)$ is the structure matrix whose columns are determined by the cable direction vectors and attachment geometry~@Bruckmann2008StructureMatrix @Pott2009ClosedFormForce. The term "structure matrix" is due to Verhoeven~@Verhoeven2004TendonPlatforms, while Gosselin uses the equivalent term "wrench matrix"~@GouttefardeGosselin2006. The structure matrix is the transpose of the Jacobian that maps platform velocities to cable length rates, establishing a kineto-static duality~@Craig2005Robotics @MurrayLiSastry1994.

==== Redundant actuation and positive tensions 
Since cables can only pull ($bold(t) >= 0$), a minimum of $n + 1$ cables is required for an $n$-#ac("DoF") platform to achieve _wrench closure_ (the ability to generate arbitrary wrenches at a given pose)~@GouttefardeGosselin2006. The _wrench-feasible workspace_ is the more practical criterion, defined as the set of poses where all required wrenches can be sustained with tensions within prescribed bounds $T_min <= T_i <= T_max$, accounting for both minimum tension (preventing cable slack) and actuator saturation~@Bosscher2006WrenchFeasible. For a system with $m > n + 1$ cables, the null space of $bold(A)^top$ provides freedom to adjust the tension distribution without affecting the net wrench. Pott _et al._~@Pott2009ClosedFormForce developed a closed-form algorithm for computing feasible tension distributions in this redundant case, decomposing the solution as $bold(t) = bold(t)_"particular" + bold(N) lambda$, where $bold(N)$ spans the null space and $lambda$ is chosen to satisfy the tension bounds.

#faps-figure(
  image("../../../assets/figures/background/concept_cdpm.svg", width: 100%),
  caption: [Three-dimensional schematic of a #ac("CDPM") with $m = 3$ cables and $n = 2$ rotational degrees of freedom, illustrating the general #ac("CDPM") concept underlying the wrist mechanism. Three cables (red, blue, orange) connect base frame anchors $b_i$ at $120 degree$ intervals on the fixed ring to platform anchors $p_i$ on the tilted movable ring. Each cable's unit direction vector $bold(u)_i$ and the resultant platform wrench $bold(w)$ are indicated. 
  // The static equilibrium is governed by the structure matrix: $bold(w) = bold(A)^top bold(t)$, where the one degree of redundancy ($m - n = 1$) enables tension optimisation under the positive-tension constraint $bold(t) >= 0$.
  ],
  short-caption: [#ac("CDPM") concept with three cables],
) <fig:cdpm_concept>

=== Jacobian-Transpose Force Mapping <sec:bg_jacobian_transpose>

The fundamental relationship between cable tensions and joint torques in tendon-driven rigid-link robots is the Jacobian-transpose mapping. This formalism underlies all three actuation modes implemented in the methodology (@sec:tendon_actuation).

==== Kineto-static duality 
For a tendon-driven mechanism with joint coordinates $bold(q) in bb(R)^n$ and cable lengths $bold(l) in bb(R)^m$, the differential kinematics of the transmission are described by

$ dot(bold(l)) = bold(R)(bold(q)) dot(bold(q)), $

where $bold(R)(bold(q)) in bb(R)^(m times n)$ is the transmission Jacobian (also called the routing matrix)~@MurrayLiSastry1994. By virtual work, the joint torques $bold(tau)$ produced by cable tensions $bold(T)$ are~@Craig2005Robotics @MurrayLiSastry1994 @Tsai1999RobotAnalysis

$ bold(tau) = bold(R)^top (bold(q)) bold(T). $ <eq:bg_jacobian_transpose>

This is the _Jacobian-transpose mapping_, establishing that the static force map is the transpose of the kinematic velocity map~@Craig2005Robotics. The formulation was introduced for tendon-driven hands by Salisbury and Craig~@Salisbury1982ArticulatedHands and generalized by Tsai and Lee~@TsaiLee1989.

==== Constant versus configuration-dependent Jacobian
A key distinction arises from the tendon routing geometry. Tsai and Lee~@TsaiLee1989 showed that when tendons wrap around circular pulleys of constant radius, the transmission Jacobian $bold(R)$ is _constant_ (i.e., independent of $bold(q)$). The entries are simply the pulley radii with appropriate signs. This is valid because circular pulleys maintain a fixed moment arm regardless of joint angle. When tendon routing involves non-circular geometry,such as the antiparallelogram linkage's migrating #ac("ICR"), the effective moment arm is configuration-dependent, and $bold(R)(bold(q))$ must be recomputed at each configuration~@FuretWenger2019 @MurrayLiSastry1994. A common engineering approximation linearizes $bold(R)$ at the zero configuration ($bold(q) = 0$) and treats the resulting constant matrix as valid over the operating range. This approximation is exact for circular-pulley routing and increasingly inaccurate as the actual routing geometry departs from the linearization point.

==== Positive-tension constraint
Since cables can only produce tensile forces, the feasible joint torque space is limited to the image of the positive orthant $bold(T) >= 0$ under $bold(R)^top$. For a system with $m$ cables and $n$ joints, the $n + 1$ rule states that at least $n + 1$ cables are necessary to generate torques in all directions~@Salisbury1982ArticulatedHands@TsaiLee1989. Additional cables provide a null space that can be used for co-contraction (stiffness modulation) without changing the net torque~@Kobayashi1998NullSpaceStiffness.

=== Tensegrity Principles in Robot Design <sec:bg_tensegrity>

The tensegrity concept provides the structural design rationale for the joint designs of the robot used in this thesis.

==== Definition and origins 
The term Tensegrity is a combination of _tensile integrity_ introduced by Buckminster Fuller. Tensegrity describes structures in which isolated compression members (struts, bars) are held in stable equilibrium by a continuous network of tension members (cables, springs). Skelton and de Oliveira provide the engineering definition and introduce the _class-$k$_ taxonomy, illustrated in @fig:tensegrity_concept. In a class-$k$ tensegrity, up to $k$ rigid bodies may be in contact through joints (class-1 means no struts touch). The concept originated with Kenneth Snelson's _X-Piece_ sculpture (1948) and Fuller's subsequent patent work, and has since been developed into a mature engineering discipline spanning architecture, structural engineering, and robotics.~@ShahTensegrityRobotics2022

==== Key properties 
Tensegrity structures offer several properties attractive for robotic applications:
- *Inherent compliance and impact absorption:* The tensile network distributes and absorbs impact forces passively, providing safety benefits faster than any control loop can react~@Bicchi2004FastSoftArm@Walter2023Tensegrity.
- *High strength-to-weight ratio:* Compression members carry primarily axial loads, with bending stresses minimised, enabling lightweight design~@ShahTensegrityRobotics2022.
- *Controllable stiffness:* By adjusting the pretension in cables, stiffness can be modulated without dedicated variable-stiffness hardware~@MuralidharanWenger2021@Kobayashi1998NullSpaceStiffness.
- *Remote actuation:* Motors can be placed at the base, with force transmitted through cables to distal joints, minimizing moving mass~@FuretWenger2019@Fasquelle2020BioInspired3DOF.

#faps-figure(
  image("../../../assets/figures/background/concept_tensegrity_classes.svg", width: 100%),
  caption: [Tensegrity structure classification according to Skelton and de Oliveira. Left: a class-1 tensegrity prism (3 struts, 9 cables) where no compression elements are in direct contact
  // struts float within the continuous tension network of edge cables (solid red) and cross-face cables (dashed red). 
  Right: a class-2 tensegrity module, where up to two struts may share a node (red dots at the four corners). This topology corresponds to the antiparallelogram X-joint used in the elbow mechanism of the thesis robot.
  // , where the crossed struts form a planar linkage connected by cables and a coupler platform (green). 
  // The legend distinguishes compression elements (struts, grey), edge and cross-face tension elements (cables, red), antagonistic cables (orange), and the coupler link (green).
  ],
  short-caption: [Tensegrity class-1 and class-2 structure classification],
) <fig:tensegrity_concept>

==== Tensegrity in robotics 
Shah _et al._ position tensegrity as bridging rigid and soft robotics. The rigid compression elements provide structural load capacity while the elastic tension network provides compliance and deformability. Existing tensegrity robots span locomotion platforms (rolling and hopping), continuum manipulators (stacked-module assemblies), and serial manipulators with tensegrity-inspired joints.~@ShahTensegrityRobotics2022

For manipulation specifically, Lessard _et al._~@Lessard2016TensegrityManipulator demonstrated a bio-inspired tensegrity arm with structurally compliant joints, while Fasquelle _et al._~@Fasquelle2020BioInspired3DOF proposed a 3-#ac("DoF") manipulator using antiparallelogram X-joints, where each joint is a class-2 tensegrity mechanism with bars loaded only axially.

==== Tensegrity-inspired design in the thesis robot 

The FAPS #highlight(fill:red)[tensegrity manipulator]~@Klein2023 embodies the tensegrity principle through its cable-driven actuation. The antiparallelogram elbow is a class-2 tensegrity mechanism whose crossed links carry compressive loads while cables maintain the kinematic constraint. The cable-driven wrist similarly distributes actuation through a tension network, with three cables suspending the wrist platform in a configuration directly analogous to a small-scale cable-driven parallel robot. Walter _et al._ demonstrated that this wrist architecture achieves up to 83% impact force isolation in modal-hammer hardware experiments, confirming the compliance benefits predicted by tensegrity theory for the wrist module.~@Walter2023Tensegrity

==== Compliance and safety context 
Bicchi and Tonietti~@Bicchi2004FastSoftArm established that mechanical compliance is essential for safe human--robot interaction, as passive compliance reacts faster than any control loop during fast impacts. The series elastic actuator concept introduces deliberate compliance in series with the actuator, trading bandwidth for shock tolerance and force control accuracy~@Pratt1995SEA.

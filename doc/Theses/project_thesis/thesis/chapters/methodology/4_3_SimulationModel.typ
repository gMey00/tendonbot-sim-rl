// §4.3 Simulation Model Construction
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table, faps-subfigure
#import "../../../../shared/formatting/diagrams.typ": mesh-processing-pipeline

== Simulation Model Construction <sec:sim_model_construction>

Translating the #ac("CAD") assembly of the tensegrity manipulator into a physics-ready Isaac Sim asset requires a multi-stage pipeline that processes raw geometry, defines articulation topology, assigns physics properties, and assembles the final robot. This section describes each stage in detail.

#faps-figure(
  image("../../../assets/figures/simulation/tensegrity_variants.png", width: 95%),
  caption: [Robot model variants rendered in Isaac Sim. Next to the elbow-disc model established by Klein~@Klein2023, a four-bar antiparallelogram model was added. Each available with full- and low-resolution visual meshes.],
  short-caption: [Robot model variants rendered in Isaac Sim],
) <fig:robot_variants>

=== Mesh Processing Pipeline <subsec:mesh_pipeline>

The original #ac("CAD") assembly is authored in Autodesk Fusion~360 and exported in FBX format. All subsequent processing is performed in Blender to prepare the meshes for physics simulation. @fig:mesh_pipeline illustrates the complete pipeline from #ac("CAD") to Isaac Lab asset.

#faps-figure(
  mesh-processing-pipeline(),
  caption: [Mesh processing pipeline from #ac("CAD") to Isaac Lab asset. Fusion~360 exports the assembly as FBX. Blender performs geometry preparation (joining, origin correction, decimation, and collision mesh extraction). The #ac("USD") Builder Script assigns articulation, rigid-body, and collision #acp("API"). The Robot Assembler combines base, arm, and gripper into a complete manipulator.],
  short-caption: [Mesh processing pipeline from #ac("CAD") to Isaac Lab asset],
) <fig:mesh_pipeline>

The Blender processing stage performs three principal operations:

- *Part joining and origin correction:* Individual part meshes from the #ac("CAD") export are joined per rigid link (e.g., all screws, tubes, and brackets belonging to the forearm are merged into a single mesh object). Each link's origin is set so that it coincides with the corresponding kinematic frame defined in the #ac("CAD") assembly, ensuring that the simulated joint axes align with the physical mechanism.

- *Visual mesh decimation:* For each visual mesh, a low-polygon variant is generated using Blender's _Decimate_ modifier (ratio $approx 0.1$--$0.3$ depending on mesh complexity). Both the full-resolution and the decimated variants are kept for rendering only. Collision is handled by separate, hand-built meshes.

- *Hand-built collision meshes:* For every link a strongly simplified, hand-built convex mesh was modelled in Blender. Automatic convex-hull or convex-decomposition tools were avoided because, on the original #ac("CAD") geometry, they produce collision shells with an unnecessarily high triangle count that significantly slow down the GPU-accelerated PhysX broad- and narrow-phase. The hand-built shells are sufficient for the contact events that occur in the manipulation tasks of this work and result in markedly lower collision overhead. The same collision meshes are shared between the full-resolution and low-resolution variants. @fig:collider_comparison contrasts the resulting collision shells of both arm variants.

All mesh outputs are exported in USDC format and organised in a per-link structure expected by the custom #ac("USD") builder script.

#faps-subfigure(
  columns: (1fr, 1fr),
  panels: (
    (image("../../../assets/figures/simulation/tensegrity_colliders_approx.png", width: 70%), [Disc-approximation variant]),
    (image("../../../assets/figures/simulation/tensegrity_colliders_physical.png", width: 70%), [Physical antiparallelogram variant]),
  ),
  caption: [Hand-built collision geometry of the two arm variants. All shells are modelled manually in Blender as strongly simplified convex hulls. Automatically generated convex decompositions were avoided to keep the #ac("GPU") collision cost low.],
  short-caption: [Hand-built collision geometry of the two arm variants],
  label: <fig:collider_comparison>,
)

=== Articulation and Physics Configuration <subsec:usd_builder>

A Python-based #ac("USD") builder script, executed within the NVIDIA Omniverse Kit runtime, constructs the complete articulated asset from the prepared meshes. The script performs the following per link:

+ *Prim hierarchy creation:* Each link is represented as a #ac("USD") Xform prim under the articulation root. Visual and collision meshes are imported as child prims of their respective link.

+ *RigidBodyAPI application:* The `UsdPhysics.RigidBodyAPI` is applied to each link prim, enabling rigid-body simulation. Mass properties (mass, center of mass, and inertia tensor) are assigned from the #ac("CAD") specifications listed in @tab:link_params.

+ *CollisionAPI application:* Collision shapes are assigned by applying `UsdPhysics.MeshCollisionAPI` to the hand-built collision mesh prims. The approximation type is set uniformly to `convexHull`, since the meshes are already authored as low-vertex convex shells in Blender.

+ *Joint definition:* Joints are created using `UsdPhysics.RevoluteJoint` or `UsdPhysics.PrismaticJoint` prims between parent and child links. Each joint prim specifies the axis of rotation or translation, position limits, effort limits (from @tab:joint_ranges), and velocity limits. @fig:joint_comparison visualizes the joint axes of the two arm variants in Isaac Sim.

+ *ArticulationRootAPI:* The top-level prim receives `UsdPhysics.ArticulationRootAPI`, marking the tree root for the PhysX reduced-coordinate solver. Solver position and velocity iteration counts are tunable per asset.

#faps-subfigure(
  columns: (1fr, 1fr),
  panels: (
    (image("../../../assets/figures/simulation/tensegrity_joints_approx.png", width: 60%), [Disc approximation variant]),
    (image("../../../assets/figures/simulation/tensegrity_joints_physical.png", width: 60%), [Physical antiparallelogram variant]),
  ),
  caption: [Joint visualization of the two arm variants in Isaac Sim. Both share identical wrist joints.],
  short-caption: [Joint visualization of the two arm variants],
  label: <fig:joint_comparison>,
)

=== Disc Approximation Variant <subsec:disc_approx>

The disc approximation simplifies the antiparallelogram elbow mechanism into a single revolute joint. The four-bar linkage is replaced by a direct rotation of the forearm about a fixed pivot coinciding with the center of the linkage's frame bar. This is geometrically equivalent to assuming a circular (disc-shaped) centrode where the instant center remains fixed~@Uicker2017.

The resulting kinematic chain contains exactly five joints (two prismatic, three revolute) and five rigid links. The collision geometry consists of the hand-built simplified convex shells described in @subsec:mesh_pipeline (see @fig:collider_comparison, left). This variant is computationally efficient, with low collision complexity and a minimal joint count, making it suitable for large-scale parallel #ac("RL") training (see @subsec:sim_params).

The key simplification is that the effective tendon lever arm for the elbow is treated as constant ($r_e = 72.5 "mm"$) regardless of joint angle. As discussed in @subsec:antiparallelogram, the physical mechanism exhibits a configuration-dependent lever arm due to centrode migration. The disc approximation can therefore not reproduce the elbow's nonlinear torque-angle characteristic at extreme flexion/extension angles.

@fig:elbow_comparison quantifies the kinematic deviation. Because the antiparallelogram's migrating #ac("ICR") shifts the coupler outward at large deflections, the disc arc always falls _inside_ the physical trajectory. The maximum deviation remains below 30~mm across the operational range ($plus.minus 70 degree$), confirming that the approximation is acceptable for the simulation tasks. Importantly, the disc model _underestimates_ the elbow's effective reach, which means workspace analyses and task designs based on this variant are conservative. The physical robot can reach at least as far.

#faps-figure(
  image("../../../assets/figures/robot/elbow_comparison_static.svg", width: 80%),
  caption: [Kinematic comparison of the antiparallelogram mechanism and its disc approximation. _Upper:_ Forearm-tip trajectories for both models across $plus.minus 75 degree$. _Lower:_ Positional deviation as a function of elbow angle. The disc approximation consistently underestimates the physical reach, making workspace designs based on it conservative.],
  short-caption: [Antiparallelogram vs. disc approximation kinematic comparison],
) <fig:elbow_comparison>

=== Physical Antiparallelogram Linkage Variant <subsec:physical_linkage>

The physical variant replaces the single revolute elbow joint with the full four-bar antiparallelogram mechanism. The implementation adds four revolute joints and two additional rigid links (the two side bars of the crossed linkage) to the kinematic tree. The resulting articulation therefore contains six revolute joints (four for the linkage, two for the wrist) plus the two prismatic base joints and the gripper joint, for a total of nine actuated joints.

==== Loop-closure constraint
The antiparallelogram is a closed kinematic chain, but the PhysX articulation solver operates on tree-structured (open-chain) topologies. To enforce the loop closure, the right rod is connected to the coupler (forearm) through an additional `PhysxSchema.PhysxJointAPI` (`coupler_right_joint`) with the `excludeFromArticulation` flag set to `true`. This instructs the solver to treat this joint as a _constraint_ joint rather than as part of the reduced-coordinate articulation, effectively closing the kinematic loop at the constraint-solver level~@physx_541_gpu_rigid_bodies.

==== Link masses
The two crossing rod assemblies of the linkage are lightweight. Each `rod_left_link` / `rod_right_link` is assigned a mass of 0.096~kg, derived from the corresponding aluminium rod geometry in the #ac("CAD") model. The remaining forearm mass is reduced accordingly to 1.900~kg so that the total assembled arm mass is conserved with respect to the disc-approximation variant~@Klein2023.

=== Five-#ac("DoF") Manipulator Assembly <subsec:robot_assembly>

The final robot is assembled from three independently constructed #ac("USD") assets using Isaac Sim's Robot Assembler tool:

+ *2-#ac("DoF") articulated base:* two prismatic joints on rigid link geometry.
+ *3-#ac("DoF") tendon-driven arm:* the elbow and wrist articulation in disc or physical variant.
+ *Robotiq 2F-140 gripper:* from the Isaac Sim built-in asset library.

The Robot Assembler creates fixed joints at the attachment interfaces, producing a single articulation tree. Four base variants emerge from the $2 times 2$ combination matrix of elbow model and mesh resolution (@tab:usd_variants).

#faps-table(
  table(
    columns: (auto, auto, auto, auto),
    table.header([*Variant*], [*Elbow model*], [*Mesh resolution*], [*Total joints*]),
    [Approx.\ full-res], [Disc (1 revolute)], [Full], [5 + gripper],
    [Approx.\ low-res], [Disc (1 revolute)], [Low-poly], [5 + gripper],
    [Physical full-res], [Antiparallelogram (4 revolute)], [Full], [8 + gripper],
    [Physical low-res], [Antiparallelogram (4 revolute)], [Low-poly], [8 + gripper],
  ),
  caption: [Robot #ac("USD") variant matrix. Each is assembled with the prismatic base and Robotiq gripper to form a complete manipulator.],
  short-caption: [Robot #ac("USD") variant matrix],
) <tab:usd_variants>

@fig:base_and_gripper shows the base joint axes and the Robotiq 2F-140 gripper as rendered in Isaac Sim.

#faps-subfigure(
  columns: (1fr, 1fr),
  panels: (
    (image("../../../assets/figures/simulation/base_joints.png", width: 86.5%), [Prismatic base joint axes ]),
    (image("../../../assets/figures/simulation/robotiq_2f140.png", width: 100%), [Robotiq 2F-140 gripper]),
  ),
  caption: [Base joints and gripper as rendered in Isaac Sim.],
  short-caption: [Prismatic base joints and Robotiq 2F-140 gripper],
  label: <fig:base_and_gripper>,
)

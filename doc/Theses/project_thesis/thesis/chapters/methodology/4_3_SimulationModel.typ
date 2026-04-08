// §4.3 Simulation Model Construction
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table
#import "../../../../shared/formatting/diagrams.typ": mesh-processing-pipeline

== Simulation Model Construction <sec:sim_model_construction>

Translating the #ac("CAD") assembly of the tensegrity manipulator into a physics-ready Isaac Sim asset requires a multi-stage pipeline that processes raw geometry, defines articulation topology, assigns physics properties, and assembles the final robot. This section describes each stage in detail.

=== Mesh Processing Pipeline <subsec:mesh_pipeline>

The original #ac("CAD") assembly is authored in Autodesk Fusion~360 and exported in FBX format. All subsequent processing is performed in Blender to prepare the meshes for physics simulation. @fig:mesh_pipeline illustrates the complete pipeline from #ac("CAD") to Isaac Lab asset.

#faps-figure(
  mesh-processing-pipeline(),
  caption: [Mesh processing pipeline from #ac("CAD") to Isaac Lab asset. Fusion~360 exports the assembly as FBX. Blender performs geometry preparation (joining, origin correction, decimation, and collision mesh extraction). The #ac("USD") Builder Script assigns articulation, rigid-body, and collision APIs. The Robot Assembler combines base, arm, and gripper into a complete manipulator.],
  short-caption: [Mesh processing pipeline from CAD to Isaac Lab asset],
) <fig:mesh_pipeline>

The Blender processing stage performs three principal operations:

- *Part joining and origin correction:* Individual part meshes from the #ac("CAD") export are joined per rigid link (e.g., all screws, tubes, and brackets belonging to the forearm are merged into a single mesh object). Each link's origin is set to its center of mass as specified in the #ac("CAD") data, ensuring that inertial properties align with the mesh geometry.

- *Mesh decimation:* For each visual mesh, a low-polygon variant is generated using Blender's _Decimate_ modifier (ratio $approx 0.1$--$0.3$ depending on mesh complexity). The full-resolution meshes are retained for rendering, while the decimated variants are used for collision detection in GPU-accelerated simulation where mesh complexity directly affects solver throughput.

- *Collision mesh extraction:* Separate collision meshes are produced per link. For the disc approximation variant, convex hull decompositions of the full link geometry are used. For the physical antiparallelogram variant, the individual linkage bars require tighter collision representations to avoid self-intersection during four-bar motion; here, oriented bounding boxes or tight convex hulls per bar segment are generated.

All mesh outputs are exported in OBJ or glTF format and organized in a per-link directory structure expected by the #ac("USD") builder script.

=== Articulation and Physics Configuration <subsec:usd_builder>

A Python-based #ac("USD") builder script, executed within the NVIDIA Omniverse Kit runtime, constructs the complete articulated asset from the prepared meshes. The script performs the following per link:

+ *Prim hierarchy creation:* Each link is represented as a #ac("USD") Xform prim under the articulation root. Visual and collision meshes are imported as child prims of their respective link.

+ *RigidBodyAPI application:* The `UsdPhysics.RigidBodyAPI` is applied to each link prim, enabling rigid-body simulation. Mass properties (mass, center of mass, and inertia tensor) are assigned from the #ac("CAD") specifications listed in @tab:link_params.

+ *CollisionAPI application:* Collision shapes are assigned by applying `UsdPhysics.MeshCollisionAPI` to the collision mesh prims. The approximation type is set to `convexHull` for standard links and to `convexDecomposition` for geometrically complex links (e.g., the wrist plate with cable guide holes).

+ *Joint definition:* Joints are created using `UsdPhysics.RevoluteJoint` or `UsdPhysics.PrismaticJoint` prims between parent and child links. Each joint prim specifies the axis of rotation or translation, position limits, effort limits (from @tab:joint_ranges), and velocity limits.

+ *ArticulationRootAPI:* The top-level prim receives `UsdPhysics.ArticulationRootAPI`, marking the tree root for the PhysX reduced-coordinate solver. Solver position and velocity iteration counts are tunable per asset.

#faps-figure(
  grid(
    columns: 2,
    gutter: 8pt,
    image("../../../../shared/figures/tensegrity_joints_approx.png", width: 100%),
    image("../../../../shared/figures/tensegrity_joints_physical.png", width: 100%),
  ),
  caption: [Joint visualization of the two arm variants in Isaac Sim. _Left:_ Disc approximation variant with a single revolute elbow joint. _Right:_ Physical antiparallelogram variant with four revolute joints forming the crossed linkage. Both share identical wrist joints.],
  short-caption: [Joint visualization of the two arm variants],
) <fig:joint_comparison>

=== Disc Approximation Variant <subsec:disc_approx>

The disc approximation simplifies the antiparallelogram elbow mechanism into a single revolute joint. The four-bar linkage is replaced by a direct rotation of the forearm about a fixed pivot coinciding with the center of the linkage's frame bar. This is geometrically equivalent to assuming a circular (disc-shaped) centrode — hence the name — where the instant center remains fixed~@Uicker2011Mechanisms.

The resulting kinematic chain contains exactly five joints (two prismatic, three revolute) and five rigid links. The collision geometry consists of convex hull approximations of the full link meshes (@fig:collider_comparison, left). This variant is computationally efficient, with low collision complexity and a minimal joint count, making it suitable for large-scale parallel #ac("RL") training (4,096+ environments).

The key simplification is that the effective tendon lever arm for the elbow is treated as constant ($r_e = 72.5 "mm"$) regardless of joint angle. As discussed in @subsec:antiparallelogram, the physical mechanism exhibits a configuration-dependent lever arm due to centrode migration. The disc approximation can therefore not reproduce the elbow's nonlinear torque-angle characteristic at extreme flexion/extension angles.

@fig:elbow_comparison quantifies the kinematic deviation. The left panel overlays the forearm-tip trajectories of the physical antiparallelogram and the disc approximation across the full $plus.minus 75 degree$ range; the right panel shows the positional error as a function of elbow angle. Because the antiparallelogram's migrating instantaneous center of rotation shifts the coupler outward at large deflections, the disc arc always falls _inside_ the physical trajectory. The maximum deviation remains below 10~mm across the operational range ($plus.minus 70 degree$), confirming that the approximation is acceptable for the simulation tasks. Importantly, the disc model _underestimates_ the elbow's effective reach, which means workspace analyses and task designs based on this variant are conservative: the physical robot can reach at least as far.

#faps-figure(
  image("../../../../shared/figures/elbow_comparison_static.svg", width: 100%),
  caption: [Kinematic comparison of the antiparallelogram mechanism and its disc approximation. _Left:_ Forearm-tip trajectories for both models across $plus.minus 75 degree$. _Right:_ Positional deviation as a function of elbow angle. The disc approximation consistently underestimates the physical reach, making workspace designs based on it conservative.],
  short-caption: [Antiparallelogram vs. disc approximation kinematic comparison],
) <fig:elbow_comparison>

=== Physical Antiparallelogram Linkage Variant <subsec:physical_linkage>

The physical variant replaces the single revolute elbow joint with the full four-bar antiparallelogram mechanism. The implementation adds four revolute joints and two additional rigid links (the two side bars of the crossed linkage) to the kinematic tree. The resulting articulation contains seven revolute joints (four for the linkage, two for the wrist, one for the gripper) plus the two prismatic base joints.

==== Loop-closure constraint 
The antiparallelogram is a closed kinematic chain, but the PhysX articulation solver operates on tree-structured (open-chain) topologies. To enforce the loop closure, the fourth bar of the linkage is connected to the frame through a `PhysxSchema.PhysxJointAPI` with the `excludeFromArticulation` flag set to `true`. This instructs the solver to treat this joint as a _constraint_ joint rather than as part of the reduced-coordinate articulation, effectively closing the kinematic loop at the constraint-solver level~@physx_541_gpu_rigid_bodies.

==== Link masses
The two side bars of the linkage are lightweight; each has a mass of 0.05~kg, derived from the aluminum bar geometry in the #ac("CAD") model. The upper and lower linkage pivot blocks add 0.03~kg each. These masses are small compared to the root link (2.67~kg) and forearm (0.89~kg) but are included for accurate dynamic simulation.

#faps-figure(
  grid(
    columns: 2,
    gutter: 8pt,
    image("../../../../shared/figures/tensegrity_colliders_approx.png", width: 100%),
    image("../../../../shared/figures/tensegrity_colliders_physical.png", width: 100%),
  ),
  caption: [Collision geometry comparison. _Left:_ Disc approximation variant with convex-hull colliders per link. _Right:_ Physical antiparallelogram variant with additional capsule colliders for the individual linkage bars and separate collision shapes for the upper and lower pivot blocks.],
  short-caption: [Collision geometry comparison of arm variants],
) <fig:collider_comparison>

=== Five-DoF Manipulator Assembly <subsec:robot_assembly>

The final robot is assembled from three independently constructed #ac("USD") assets using Isaac Sim's Robot Assembler tool:

+ *2-#ac("DoF") articulated base* — two prismatic joints on rigid link geometry, providing $Y$/$Z$ planar positioning.
+ *3-#ac("DoF") tendon-driven arm* — the elbow and wrist articulation in either disc or physical variant.
+ *Robotiq 2F-140 gripper* — loaded from the Isaac Sim built-in asset library and attached at the wrist output flange.

The Robot Assembler creates fixed joints at the attachment interfaces, producing a single articulation tree. Four base variants emerge from the two $times$ two combination matrix of elbow model and mesh resolution (@tab:usd_variants).

#faps-table(
  table(
    columns: (auto, auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Variant*], [*Elbow model*], [*Mesh resolution*], [*Total joints*]),
    [Approx.\ full-res], [Disc (1 revolute)], [Full], [5 + gripper],
    [Approx.\ low-res], [Disc (1 revolute)], [Low-poly], [5 + gripper],
    [Physical full-res], [Antiparallelogram (4 revolute)], [Full], [8 + gripper],
    [Physical low-res], [Antiparallelogram (4 revolute)], [Low-poly], [8 + gripper],
  ),
  caption: [Robot #ac("USD") variant matrix. Two elbow models and two mesh resolutions produce four arm assets. Each is assembled with the prismatic base and Robotiq gripper to form a complete manipulator.],
  short-caption: [Robot USD variant matrix],
) <tab:usd_variants>

@fig:base_and_gripper shows the base joint axes and the Robotiq 2F-140 gripper as rendered in Isaac Sim.

#faps-figure(
  grid(
    columns: 2,
    gutter: 8pt,
    image("../../../../shared/figures/base_joints.png", width: 100%),
    image("../../../../shared/figures/robotiq_2f140.png", width: 100%),
  ),
  caption: [_Left:_ Prismatic base joint axes ($Y$ horizontal, $Z$ vertical) as defined in the simulation. _Right:_ Robotiq 2F-140 gripper with visual mesh and finger linkage.],
  short-caption: [Prismatic base joints and Robotiq 2F-140 gripper],
) <fig:base_and_gripper>

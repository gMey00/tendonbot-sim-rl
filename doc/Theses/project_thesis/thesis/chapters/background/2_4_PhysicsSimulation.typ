// §2.4 Physics Simulation
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/diagrams.typ": physx-pipeline
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table

== Physics Simulation <sec:physics_sim>

Physics-based simulation in Isaac~Sim~5.1 is implemented as a layered stack that separates scene and asset description, binding of these descriptions to a concrete backend, and numerical solvers that evolve the physical state over time. In the Omniverse ecosystem, physics properties are authored in #ac("USD") through the `UsdPhysics` schema (engine-agnostic) and `PhysXSchema` (backend-specific), allowing assets and environments to be defined declaratively in the scene graph.~@openusd_usdphysics_schema@omni_physx_overview

During simulation, the `omni.physx` runtime parses the #ac("USD") stage, constructs the corresponding physics scene, advances the simulation, and writes back poses and state to #ac("USD")~@omni_physx_overview. Isaac~Sim~5.1 uses NVIDIA PhysX~5 as its primary physics backend, providing rigid-body dynamics, articulations, contacts, and particle-based phenomena, with optional #ac("GPU")-accelerated pipelines for throughput-critical workloads such as parallel #ac("RL") training~@isaac_sim_simulation_fundamentals_510@physx_welcome_513@physx_gpu_rigid_bodies_510.

Sim-to-real transfer is fundamentally limited by discrepancies between simulated and real dynamics (the _reality gap_), where unmodelled effects, incorrect parameters, and solver artefacts can lead to policies that exploit simulator-specific regularities~@Jakobi1995RealityGap @Salvato2021RealityGapSurvey. In #ac("RL"), the simulator acts as a data-generating process. Thus, modelling and solver choices directly shape the transition distribution used for learning, influencing both policy quality and generalization~@Salvato2021RealityGapSurvey @Peng2018DynamicsRandomization. This implies that solver selection, time-step choice, contact modelling, and collision geometry approximations must be treated as part of the learning system design~@Peng2018DynamicsRandomization @physx_best_practices_541.

#faps-figure(
  physx-pipeline(),
  caption: [Conceptual physics simulation stack in Isaac~Sim~5.1: physics is authored in #ac("USD") via `UsdPhysics` and `PhysXSchema`, interpreted by `omni.physx`, and simulated using PhysX~5 with optional #ac("GPU")-accelerated pipelines.],
  short-caption: [Isaac Sim physics simulation stack],
) <fig:isaac_sim_physics_stack>

=== Rigid-Body Dynamics and Articulations

Rigid-body simulation assumes that each object moves as a rigid transform (no deformation), with state represented by position/orientation and linear/angular velocity. PhysX represents the simulated world as actors (static, kinematic, dynamic) with attached collision shapes and material properties. Dynamics are advanced under external loads, contact constraints, and joint constraints.~@physx_welcome_513@physx_rigid_body_dynamics_511

==== Articulated bodies 
Robotic manipulators are modelled as articulated mechanisms. Rigid links connected by joints that constrain relative motion and optionally apply actuation. PhysX provides an _articulation_ abstraction simulated in _reduced coordinates_, where the configuration is parameterized by the root pose and joint coordinates rather than the world pose of each link. This formulation improves numerical fidelity for mechanisms with large mass ratios and long kinematic chains compared to independently simulating rigid bodies connected by generic constraints.~@physx_articulations_513

==== Articulation tendons
PhysX exposes _articulation tendons_ as additional constraint structures within an articulation. _Fixed tendons_ couple joint coordinates through constant gearing ratios, while _spatial tendons_ define spring-damper and limit constraints between attachment points across links, producing configuration-dependent forces. These constructs enable cable-like actuation or joint coupling to be represented at the physics level without external force computation.~@physx_articulations_513 @omni_physics_tendon_authoring

See @sec:tendon_actuation for their application in the methodology.

=== Contact and Friction Models

Contact response is formulated as a constrained dynamics problem. Contacts prevent interpenetration and transfer or dissipate energy through restitution and friction~@physx_rigid_body_collision_510. Friction and restitution are specified via materials attached to shapes, with friction parameterized by static and dynamic coefficients~@physx_rigid_body_dynamics_511@physx_material_api_510. Because contact-rich manipulation is common in robotic grasping, parameter choices for friction and restitution are directly coupled to learning outcomes and sim-to-real robustness~@Salvato2021RealityGapSurvey@Peng2018DynamicsRandomization.

==== Constraint solving
PhysX supports a classic #acs("PGS")-style solver and the newer #acs("TGS") solver, with #acs("TGS") recommended for improved convergence and joint drive accuracy~@physx_rigid_body_dynamics_511. Both operate iteratively over constraint manifolds (contacts, joints, friction), trading accuracy for performance by limiting iteration counts. For stiff constraint systems, replacing multiple solver iterations at a long time step by multiple smaller substeps can yield superior stability at comparable computational cost~@Macklin2019SmallSteps@physx_best_practices_541.

=== Simulation Parameters  <sec:bg_sim_params>

#reg-sym("sym:h")

#faps-table(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    table.header(
      [*Parameter*], [*Location*], [*Primary effect*],
    ),
    [Time step $h$], [Physics scene], [Accuracy vs.~throughput; contact/joint stability~@physx_simulation_541],
    [Solver iterations], [Actor / island], [Constraint convergence; joint error; contact quality~@physx_rigid_body_dynamics_511],
    [`contactOffset`/`restOffset`], [Shape], [Contact prediction and resting separation~@physx_advanced_collision_detection_512],
    [Friction coefficients], [Material], [Sliding/sticking behavior; grasp stability~@physx_material_api_510],
  ),
  caption: [Selected rigid-body simulation parameters relevant to robot learning in Isaac~Sim / PhysX.],
  short-caption: [Key rigid-body simulation parameters],
) <tab:physx_rigid_body_parameters>

~@tab:physx_rigid_body_parameters summarizes key simulation parameters that influence the fidelity and stability of rigid-body dynamics in PhysX.
The time step $h$ determines how frequently dynamics are integrated and how often the policy interacts with the environment. Overly large $h$ can lead to missed collisions and unstable joint behavior, whereas smaller $h$ improves accuracy at increased compute cost~@physx_simulation_541 @physx_best_practices_541. 

Collision detection cost depends on the collider representation. Primitive shapes (spheres, boxes, capsules) provide the best performance and robustness, while convex meshes provide a practical middle ground for complex objects~@physx_rigid_body_collision_510 @physx_geometry_541. Collider selection trades geometric fidelity against computational cost and solver stability. This tradeoff must be balanced against the throughput requirements of #ac("RL") training.

=== GPU-Accelerated Simulation for Parallel RL Environments 
#highlight(fill: red)[TODO: Recheck this section for accuracy and correct citations]

PhysX provides a #ac("GPU") rigid-body pipeline that accelerates broad-phase, narrow-phase, contact generation, and constraint solving on CUDA-capable hardware. Large performance gains arise when many rigid bodies are active, which aligns with #ac("RL") workloads where hundreds to thousands of environment replicas are simulated in parallel. The key advantage is that physics stepping and policy learning can share device memory, avoiding #ac("CPU")--#ac("GPU") transfer bottlenecks. This design has been demonstrated to yield order-of-magnitude throughput improvements in GPU-based robot learning platforms.~@physx_gpu_rigid_bodies_510@Makoviychuk2021IsaacGym

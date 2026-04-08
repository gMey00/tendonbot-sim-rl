// §2.6 Robot and Scene Description Formats
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/diagrams.typ": format-conversion-pipeline
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table

== Robot and Scene Description Formats <sec:description_formats>

Robotic learning workflows in simulation require at least two layers of model description: a _robot model_ capturing kinematic structure and physical properties, and a _scene model_ capturing environment layout, lighting, and simulator-specific physics configuration. In the Omniverse ecosystem, the scene is represented as a composed #ac("USD") stage, while robots are typically authored in robotics-oriented formats such as #ac("URDF") and subsequently imported into #ac("USD") for simulation.

#faps-figure(
  format-conversion-pipeline(),
  caption: [Schematic pipeline illustrating how robot descriptions (e.g., #ac("URDF")) are converted into #ac("USD") assets and integrated with environment elements on a composed #ac("USD") stage.],
  short-caption: [URDF to USD conversion pipeline],
) <fig:format_pipeline_usd_urdf_mjcf>

=== USD as Scene Graph

#ac("USD") is a system for encoding scalable, hierarchically organized data with the primary purpose of interchange and augmentation across digital content creation applications~@openusd_usd_faq. Beyond being a file encoding, #ac("USD") composes a single stage from scene description distributed across many layers and assets, supporting sparse overrides via composition operators (references, variants)~@openusd_usd_faq. This design makes #ac("USD") the foundation of Omniverse-based simulation: worlds are assembled from reusable assets (robots, fixtures, task objects), while experiment-specific changes (randomization parameters, material settings) are authored as non-destructive overrides.

Physics semantics are provided by _UsdPhysics_, which defines schemas for rigid bodies, joints, constraints, and joint drives on the scene graph~@openusd_usdphysics_schema. Simulator-specific extensions (e.g., PhysX solver parameters, tendon constraints) are exposed through additional _PhysXSchema_ additions~@omni_physics_physx_joint_schema.

=== URDF as Robot Model Format

#ac("URDF") is an #ac("XML")-based format in the ROS ecosystem for specifying robot geometry and organization~@ros2_urdf_main. A #ac("URDF") model describes the robot as rigid links connected through joints, augmented with collision geometry, inertial parameters, and joint limits~@ros2_urdf_physical_properties. This focus on kinematic structure makes #ac("URDF") the standard interchange format for robotic manipulators.

==== Structural limitations for tendon mechanisms
 Standard #ac("URDF") supports only serial kinematic chains; closed-loop dependencies are handled through simulator-specific mechanisms~@ros2_control_joint_kinematics. The _mimic joint_ concept allows linear coupling between joints (e.g., for belt-and-pulley systems), but tendon routing with wrapping, varying moment arms, elastic stretch, and multi-joint coupling is not representable with standard #ac("URDF") alone~@ros2_control_joint_kinematics. More complex tendon actuation must be introduced through simulator-specific extensions or higher-level actuation models.

=== Format Pipeline and Tendon Workarounds

#faps-table(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    table.header(
      [*Format*], [*Primary scope*], [*Role in simulation*],
    ),
    [#ac("USD")], [Scene graph with composition], [World assembly, physics/rendering schemas~@openusd_usd_faq @openusd_usdphysics_schema],
    [#ac("URDF")], [Robot kinematics + rigid-body data], [Robot model interchange; converted into #ac("USD")~@ros2_urdf_main @isaacsim_import_urdf_510],
  ),
  caption: [Roles of #ac("USD") and #ac("URDF") in the simulation pipeline used in this thesis.],
  short-caption: [USD and URDF roles in simulation],
) <tab:usd_urdf_roles>

In this thesis, the tendon-driven arm is authored as a modified #ac("URDF") and imported into Isaac~Sim using the provided URDF-to-USD importer~@isaacsim_import_urdf_510. Isaac~Lab recommends converting external formats into #ac("USD") and performing subsequent editing in the #ac("USD") representation~@isaaclab_import_new_asset. Once converted, the robot is integrated with environment assets on a composed #ac("USD") stage, with physics semantics attached through _UsdPhysics_ and _PhysXSchema_~@openusd_usdphysics_schema @omni_physics_physx_joint_schema.

==== Tendon modeling workarounds
 Since standard #ac("URDF") cannot express tendon routing and PhysX articulation tendons are listed among known limitations in Omniverse Physics~@omni_physics_current_limitations, tendon actuation is approximated through joint drives and coupling simplifications rather than explicit tendon constraints (@sec:tendon_actuation details the three actuation modes implemented in the methodology). Mimic-joint coupling is used where tendon effects can be approximated as linear joint relationships, and external force computation is used where configuration-dependent tendon geometry must be accounted for~@omni_physics_current_limitations @ros2_control_joint_kinematics.

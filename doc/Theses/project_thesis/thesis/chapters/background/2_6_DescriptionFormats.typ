// §2.6 Robot and Scene Description Formats
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table

== Robot and Scene Description Formats <sec:description_formats>


Robotic learning workflows in simulation require at least two layers of model description: a _robot model_ capturing kinematic structure and physical properties, and a _scene model_ capturing environment layout, lighting, and simulator-specific physics configuration. In the Omniverse ecosystem, the scene is represented as a composed #ac("USD") stage, while robots are typically authored in robotics-oriented formats such as #ac("URDF") and subsequently imported into #ac("USD") for simulation.

=== #ac("USD") as Scene Graph

#ac("USD") is a system for composing scalable, hierarchically organised scene description from many layers and assets, and additionally provides interchange services across digital content creation applications. Composition operators (references, variants) support sparse, non-destructive overrides on the assembled scenegraph. This design makes #ac("USD") the foundation of Omniverse-based simulation. Worlds are assembled from reusable assets (robots, fixtures, task objects), while experiment-specific changes (randomization parameters, material settings) are authored as overrides without modifying the source assets.~@openusd_usd_faq

Physics semantics are provided by _UsdPhysics_, which defines schemas for rigid bodies, joints, constraints, and joint drives on the scene graph~@openusd_usdphysics_schema. Simulator-specific extensions (e.g., PhysX solver parameters, tendon constraints) are exposed through additional _PhysXSchema_ additions~@omni_physics_physx_joint_schema.

=== #ac("URDF") as Robot Model Format

#ac("URDF") is an #ac("XML")-based format in the #ac("ROS") ecosystem for specifying robot geometry and organization~@ros2_urdf_main. A #ac("URDF") model describes the robot as rigid links connected through joints, augmented with collision geometry, inertial parameters, and joint limits~@ros2_urdf_physical_properties. This focus on kinematic structure makes #ac("URDF") the standard interchange format for robotic manipulators.

==== Structural limitations for tendon mechanisms
Standard #ac("URDF") supports only serial kinematic chains. Closed-loop dependencies are handled through simulator-specific mechanisms. The _mimic joint_ concept allows linear coupling between joints (e.g., for belt-and-pulley systems), but tendon routing with wrapping, varying moment arms, elastic stretch, and multi-joint coupling is not representable with standard #ac("URDF") alone. More complex tendon actuation must be introduced through simulator-specific extensions or higher-level actuation models.~@ros2_control_joint_kinematics

=== Format Pipeline and Tendon Workarounds

#faps-table(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    table.header(
      [*Format*], [*Primary scope*], [*Role in simulation*],
    ),
    [#ac("USD")], [Scene graph with composition], [World assembly, physics/rendering schemas~@openusd_usd_faq @openusd_usdphysics_schema],
    [#ac("URDF")], [Robot kinematics + rigid-body data], [Robot model interchange, converted into #ac("USD")~@ros2_urdf_main @isaacsim_import_urdf_510],
  ),
  caption: [Roles of #ac("USD") and #ac("URDF") in typical robotics simulation pipelines.],
  short-caption: [#ac("USD") and #ac("URDF") roles in simulation],
) <tab:usd_urdf_roles>

In this thesis, rather than relying on an intermediate #ac("URDF") conversion for the arm mechanics, the tendon-driven arm is constructed directly using a Python-based #ac("USD") builder script acting on prepared #ac("CAD") meshes (see @sec:sim_model_construction). However, Isaac~Lab recommends converting external formats into #ac("USD") and performing subsequent editing in the #ac("USD") representation~@isaaclab_import_new_asset, a pattern followed for supporting assets like the gripper. Ultimately, the robot is integrated with environment assets on a composed #ac("USD") stage, with physics semantics attached through _UsdPhysics_ and _PhysXSchema_~@openusd_usdphysics_schema @omni_physics_physx_joint_schema. To ensure the created robot models remain available and interchangeable with other simulation ecosystems, the Isaac~Sim #ac("URDF") exporter is used to generate #ac("URDF") representations of the final #ac("USD") assets.

==== Tendon modeling workarounds
Since standard #ac("URDF") cannot express tendon routing and native PhysX spatial tendons present numerical instability and lack tension reporting, tendon actuation is implemented through custom mapping models rather than native constraint objects (@sec:tendon_actuation details the three actuation modes implemented in the methodology). Constant Jacobian mappings to joint efforts are used where tendon effects can be approximated linearly, and explicit body force computation is used where configuration-dependent tendon geometry must be accounted for~@omni_physics_current_limitations @ros2_control_joint_kinematics.

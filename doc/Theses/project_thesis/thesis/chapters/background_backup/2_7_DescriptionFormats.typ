// §2.5 (file numbered 2_7) Robot and Scene Description Formats
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/diagrams.typ": format-conversion-pipeline
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table

== Robot and Scene Description Formats <sec:description_formats>

Robotic learning workflows in simulation typically require at least two layers of model description: (i) a _robot model_ capturing kinematic structure and physical properties, and (ii) a _scene/world model_ capturing the placement of robots and objects, lighting and sensors, and simulator-specific physics configuration. In the #ac("USD")-based _Omniverse_ ecosystem, the scene is represented as a composed #ac("USD") stage, while robots are typically authored in robotics-oriented formats such as #ac("URDF") and subsequently imported into #ac("USD") for simulation. This section explains the role of #ac("USD") and #ac("URDF") in this separation of concerns and clarifies how both formats are used in this thesis.

#faps-figure(
  format-conversion-pipeline(),
  caption: [Schematic pipeline illustrating how robot descriptions (e.g., #ac("URDF")) are converted into #ac("USD") assets and integrated with environment elements (rigid objects, deformable bodies, lights, sensors) on a composed #ac("USD") stage.],
  short-caption: [URDF to USD conversion pipeline],
) <fig:format_pipeline_usd_urdf_mjcf>

=== #acs("USD") as a scene graph for simulation worlds

*Core scene description concept.*
#ac("USD") is a system for encoding scalable, hierarchically organized data (static as well as time-sampled) with the primary purpose of interchange and augmentation across digital content creation applications~@openusd_usd_faq. Beyond being a file encoding, #ac("USD") is designed to _compose_ a single stage from scene description distributed across many layers and assets, supporting sparse overrides and asset aggregation via composition operators such as references and variants~@openusd_usd_faq. This composition-centric design is the main reason #ac("USD") is used as the foundation of _NVIDIA Omniverse_-based applications, including robotics simulation: a simulation world can be assembled from reusable assets (robots, fixtures, conveyors, task objects), while experiment-specific changes (randomization parameters, object placements, material settings) can be authored as non-destructive overrides.

*Extensibility via schemas.*
A central mechanism in #ac("USD") is _schema-based extensibility_: standardized or custom schemas define what properties are available on a prim and how data are interpreted by downstream tools~@openusd_usd_faq @openusd_usdphysics_schema. For robotics simulation, this is crucial because a scene must not only store geometry and transforms, but also physics and sensor semantics. In practice, simulation-ready #ac("USD") stages rely on multiple cooperating schema families: (i) rendering-related schemas (e.g., light prims), and (ii) physics-related schemas that define rigid bodies, collisions, articulation joints, and constraints, enabling a simulator backend to interpret the stage as a dynamical system.

*Lighting and sensor-relevant elements.*
Lighting in #ac("USD") is represented by the #ac("USD") lighting schema _UsdLux_, which provides a portable representation for common light types and attributes (e.g., intensity/exposure and color) intended for interchange across rendering environments~@openusd_usdlux_schema. This is directly relevant for robotics simulation, because perception pipelines and synthetic data generation depend on reproducible lighting setups. The use of common light schemas also reduces simulator lock-in at the scene-description level, since the same lighting structure can be understood by multiple Omniverse tools and render backends~@openusd_usdlux_schema.

*Physics properties encoded in #acs("USD").*
Physics semantics in #ac("USD") are provided by _UsdPhysics_, which defines prim and property schemas for representing physics simulation state and configuration on a stage~@openusd_usdphysics_schema. Within this schema family, rigid bodies, joints, and constraints are described as #ac("USD") prims and applied #ac("API") schemas, allowing physics engines to identify which parts of the stage are simulated and how they are constrained~@openusd_usdphysics_schema. Joint actuation is commonly represented via drive attributes that behave as implicit damped springs, i.e., forces/accelerations are defined via stiffness and damping terms toward a target position/velocity~@openusd_usdphysics_schema. In Omniverse, additional simulator-specific tuning parameters are exposed through _PhysX schema additions_; for example, PhysX joint schemas provide per-axis parameters and actuator performance characteristics extending the generic _UsdPhysics_ joint representation~@omni_physics_physx_joint_schema.

=== #acs("URDF") as a robotics-oriented robot model format

*Scope and typical usage.*
#ac("URDF") is an #ac("XML")-based format used in the _Robot Operating System_ ecosystem for specifying robot geometry and organization~@ros2_urdf_main. A #ac("URDF") model typically focuses on the robot as a kinematic structure, described by rigid links connected through joints, plus the geometric information used for visualization and collision checking, and the physical properties required for rigid-body simulation (e.g., link inertias)~@ros2_urdf_physical_properties. This focus makes #ac("URDF") a common interchange format for robotic manipulators and mobile robots, especially when robots are also integrated into _ROS_ pipelines for control, state publishing, and visualization~@ros2_urdf_main.

*Kinematic structure, joint limits, and inertial properties.*
Within the #ac("URDF") workflow, links are augmented with collision geometry and inertial parameters, and joints are augmented with joint dynamics and limits. The ROS 2 #ac("URDF") tutorials explicitly emphasize that collision and inertial properties are needed for simulation, and that joint dynamics parameters (e.g., friction) can be specified as part of the joint description~@ros2_urdf_physical_properties. Joint limits (position bounds, effort/velocity bounds) are also represented at the joint level in #ac("URDF")-based tooling, and are typically used both for controller safety and for ensuring physically plausible simulation states~@ros2_urdf_physical_properties.

*Structural limitations relevant to tendon mechanisms.*
A key limitation of #ac("URDF") is that, in standard robotics tooling, only serial kinematic chains are supported in a direct and portable way; more complex kinematic dependencies are generally handled through additional simulator- or controller-specific mechanisms~@ros2_control_joint_kinematics. A commonly supported exception is the _mimic joint_ concept, which allows joint coordinates to be coupled through a linear dependency. Mimic joints are explicitly described as a pragmatic abstraction used to represent simplified closed-loop dependencies or mechanically coupled degrees of freedom, such as belt-and-pulley systems~@ros2_control_joint_kinematics. This capability is relevant for tendon-driven or cable-driven mechanisms when coupling can be approximated as a simple linear relationship. However, tendon routing with wrapping, varying moment arms, elastic cable stretch, frictional loss, and multi-joint coupling is generally not fully representable with the standard #ac("URDF") kinematic abstraction alone~@ros2_control_joint_kinematics. In such cases, either simulator-specific extensions or higher-level actuation models must be introduced.

=== Format use in this thesis and tendon-driven modeling workarounds

*Responsibilities of each format in the pipeline.*
In the context of Isaac Sim and Isaac Lab, #ac("USD") is treated as the _runtime scene format_: robots and environment assets are ultimately represented as #ac("USD") so that they can be loaded, referenced, and simulated on a #ac("USD") stage~@isaaclab_import_new_asset. #ac("URDF") is used as the _authoring and interchange format_ for the robot model, capturing: (i) robot topology (links and joints), (ii) joint types and joint limits, and (iii) rigid-body physical properties (e.g., inertial parameters and collision geometry) required for simulation~@ros2_urdf_physical_properties. By contrast, #ac("USD") is used to integrate the robot with the rest of the simulated world and to attach simulation-specific semantics (physics schema applications, light prims, and renderer-facing scene elements)~@openusd_usd_faq @openusd_usdphysics_schema @openusd_usdlux_schema.

#faps-table(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    table.header(
      [*Format*], [*Primary scope*], [*Typical role in simulation*],
    ),
    [#ac("USD")], [Scene graph with composition], [World assembly, physics schemas, lights/sensors~@openusd_usd_faq @openusd_usdphysics_schema @openusd_usdlux_schema],
    [#ac("URDF")], [Robot kinematics + basic rigid-body data], [Robot model interchange; converted into #ac("USD") for Omniverse~@ros2_urdf_main @isaacsim_import_urdf_510],
  ),
  caption: [Roles of #ac("USD") and #ac("URDF") in the simulation pipeline used in this thesis.],
  short-caption: [USD and URDF roles in simulation],
) <tab:usd_urdf_roles>

*Which format is used for the tendon-driven robot and why.*
In this thesis, the custom tendon-driven arm component is provided as a _modified_ #ac("URDF") description and is imported into Isaac Sim, where it becomes a #ac("USD") robot asset for simulation. This choice is motivated by the practical availability of #ac("URDF") as a widely used robotics interchange format and by the fact that Isaac Sim provides an explicit import workflow that converts #ac("URDF") robot descriptions into #ac("USD") assets for use on a #ac("USD") stage~@isaacsim_import_urdf_510. In addition, Isaac Lab recommends converting external robot model formats into #ac("USD") using the provided importers and then performing subsequent editing, tuning, and large-scale simulation using the #ac("USD") representation~@isaaclab_import_new_asset. The final integrated robot is therefore simulated as #ac("USD") within Isaac Sim, consistent with Omniverse's #ac("USD")-native scene representation~@isaaclab_import_new_asset.

*Integration of robot and scene as USD assets.*
Once the robot components exist as #ac("USD") assets, integration into a complete simulation scene is performed by placing them on the #ac("USD") stage together with the environment geometry and task objects (including manipulation targets and supporting structures) and then attaching physics and rendering semantics through the relevant schemas~@openusd_usdphysics_schema @openusd_usdlux_schema. In Omniverse-based robotics workflows, modular robot assemblies are commonly created by combining separate #ac("USD") robot assets (e.g., a base, an arm, and a gripper) via fixed joints. Isaac Sim's _Robot Assembler_ tool is explicitly designed for this use case: it combines two #ac("USD") assets through a physically simulated fixed joint and produces a resulting #ac("USD") asset that can be saved and loaded as a single composite robot~@isaacsim_robot_assembler_510.

*URDF modifications and tendon modeling workarounds.*
Standard #ac("URDF") does not provide a portable, first-class representation of tendon routing with wrapping, elastic elements, and multi-joint cable coupling~@ros2_control_joint_kinematics. Tendon-driven mechanisms are therefore approximated in #ac("URDF")-to-simulator pipelines by mapping tendon actuation to joint-level drive targets (position/velocity/effort) and by encoding only those couplings that can be captured with simple kinematic relationships (e.g., mimic joints)~@ros2_control_joint_kinematics. In Omniverse workflows, practical #ac("URDF")-to-#ac("USD") conversion constraints further motivate simplification: Isaac Lab explicitly documents that some #ac("URDF") tags used in other simulators (e.g., `<transmission>` and `<gazebo>` tags) are not supported by the #ac("URDF") importer, and that pre-processing may therefore remove such tags prior to conversion~@isaaclab_import_new_asset.

In the present thesis, the tendon routing/transmission constructs in the imported arm description are simplified such that each tendon-driven joint is represented as an actuated revolute joint driven directly by joint targets rather than an explicit cable model. Moreover, joint limits are enforced in the description to improve stability, while complex tendon dynamics (stretch, multi-joint coupling, backlash) are not explicitly simulated at this stage.

*Why PhysX tendon features are not used in the current implementation.*
Although PhysX articulations support tendon constraints at the articulation level, including fixed tendons (constraints on joint positions) and spatial tendons (distance constraints along attachment points)~@physx_articulations_550, limitations in simulation fidelity and behavior have been documented for articulation tendons in Omniverse Physics. Specifically, articulation tendons are listed among known limitations, and workarounds are recommended: mimic-joint style coupling is suggested for fixed tendon behavior, and external forces are suggested to mimic spatial tendon effects~@omni_physics_current_limitations. In this thesis, these limitations motivate a simplified approach in which tendon effects are approximated via direct joint drives (spring-damper-like actuation toward joint targets) rather than via explicit PhysX tendon constraints, thereby prioritizing stability and controllability of the learning environment over detailed tendon mechanics.

- *URDF role:* The tendon-driven arm geometry, kinematic chain, joint limits, collision shapes, and rigid-body inertial properties are authored in #ac("URDF") and then converted for Omniverse simulation~@ros2_urdf_physical_properties @isaacsim_import_urdf_510.
- *USD role:* The robot (after conversion) and the environment assets are integrated on a composed #ac("USD") stage; physics and lighting semantics are expressed via _UsdPhysics_ and _UsdLux_, and PhysX-specific parameters are expressed via PhysX schema additions~@openusd_usdphysics_schema @openusd_usdlux_schema @omni_physics_physx_joint_schema.
- *Tendon workarounds:* Due to representational limitations of #ac("URDF") and documented limitations of articulation tendon fidelity in Omniverse Physics, tendon actuation is approximated through joint drives and coupling simplifications rather than explicit tendon constraints~@ros2_control_joint_kinematics @omni_physics_current_limitations.

// Chapter 7: Conclusion and Outlook
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *

= Conclusion and Outlook <ch:conclusion>

This chapter reconnects the results to the central project-thesis question, summarises the key findings without restating the detailed numbers already discussed in @ch:discussion, and separates the project-thesis scope from the subsequent master-thesis programme. The first section presents the compact conclusion. The second section formulates the outlook toward the master thesis.

==== Conclusion <sec:conclusion>

The project thesis set out to answer whether a custom tendon-driven manipulator can be modeled, validated, and controlled through #ac("RL") in NVIDIA Isaac~Sim and Isaac~Lab for conveyor-based rigid-object pick-and-place tasks. The interpretation in @ch:discussion answers this question positively for the validated simulation model, the #ac("PD") baseline, and the constant-Jacobian tendon variant, and negatively for direct end-to-end control of the physical antiparallelogram tendon variant.

The thesis delivers four contributions to this question. First, it provides a reusable Isaac~Sim and Isaac~Lab model of the ceiling-mounted 5-#ac("DoF") tendon-driven manipulator, with the prismatic base, the antiparallelogram elbow, the cable-driven wrist, the Robotiq gripper, the conveyor scene, and the drum task built modularly described in @ch:methodology. Second, it offers a quantitative validation of the model against the predecessor work~@Klein2023 by means of step-response tests and Monte-Carlo workspace analysis (@subsec:step_results, @subsec:workspace_results), which together satisfy the validation success criterion of @ch:research_gap. Third, it establishes an #ac("RL") baseline on the reach and cube-place tasks for which the #ac("PD") and constant-Jacobian tendon variants reach the comparative performance levels reported in @subsec:reach_performance and @subsec:place_final. Fourth, it identifies the remaining technical bottleneck. Direct policy control through closed-chain body-force tendon actuation does not yet acquire reliable grasping and placing, which restricts the failure analysis from a general simulation problem to a specific actuation-interface problem.

The thesis does not demonstrate textile sorting, deformable-cloth manipulation, multi-agent coordination, perception-based classification, or sim-to-real transfer. These aspects are deliberately excluded by the scope in @ch:introduction and are deferred to the master thesis. The validated rigid-body simulation, the modular task infrastructure, and the diagnostic three-variant actuation comparison form the foundation on which the master-thesis goals are subsequently built.

==== Outlook <sec:outlook>

The master thesis starts from the foundation established here and extends it along five directions, in line with the partial goals stated for the master thesis programme. The directions are listed in increasing distance from the project-thesis scope, so the dependency between the rigid-body baseline and the deformable-object goals remains visible.

The first direction is the transition from rigid cube manipulation to conveyor-based textile sorting. The robot must selectively retrieve garments from a mixed waste stream rather than known rigid objects, which transforms validated pick-and-place control into selectivity-aware manipulation with new task-success criteria such as sorting accuracy, false-pick rate, and missed-pick rate.

The second direction is the integration of deformable cloth simulation and perception. The project thesis uses simulation state observations and rigid objects. The master thesis requires garment models with realistic material parameters, simulated camera views, and a damage-condition classifier. The Isaac~Lab scene, the logging pipeline, and the curriculum structure of the project thesis remain reusable, while the object model and the observation space require explicit extension rather than reuse from the cube-place task.

The third direction is multi-agent system design. A second collaborative arm stretches and presents the garment for the camera, while the tendon-driven picker continues to retrieve textiles from the conveyor. The extension introduces a second robot model, compatible task interfaces, an inter-agent coordination protocol, and metrics for handover quality, classification accuracy, and cycle time.

The fourth direction is the tendon-wrist transfer experiment. The 2-#ac("DoF") tendon-driven wrist developed within the manipulator is isolated and applied to a conventional industrial arm under matched training conditions. The experiment turns the compliant wrist design from an assumed advantage into a testable research question with a controlled comparison against the native rigid wrist.

The fifth direction is preparation for physical deployment. The Isaac~Lab configuration is extended with domain randomization, sensor and lighting variation, an #ac("ROS")~2 interface, safety constraints, and transfer-readiness evaluation procedures. This step is meaningful only after the rigid-body baseline of the project thesis remains reproducible and after the physical-tendon-control bottleneck identified in @sec:discussion_outcomes is resolved.

In summary, the project thesis establishes the simulation, validation, and learning foundation. The master thesis extends that foundation toward multi-agent textile sorting.

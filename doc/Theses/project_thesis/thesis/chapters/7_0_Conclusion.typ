// Chapter 7: Conclusion and Outlook
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *

= Conclusion and Outlook <ch:conclusion>

This chapter reconnects the results to the central project-thesis question, summarises the key findings without restating the detailed numbers already discussed in @ch:discussion, and separates the project-thesis scope from the subsequent master-thesis programme. The first section presents the compact conclusion. The second section formulates the outlook toward the master thesis.

==== Conclusion <sec:conclusion>

The project thesis set out to answer whether a custom tendon-driven manipulator can be modeled, validated, and controlled through #ac("RL") in NVIDIA Isaac~Sim and Isaac~Lab for conveyor-based rigid-object pick-and-place tasks. The interpretation in @ch:discussion answers this question positively for the validated simulation model, the #ac("PD") baseline, and the constant-$J^top$ tendon variant, and negatively for direct end-to-end control of the physical tendon variant.

The thesis makes six contributions to this question, which together correspond to the objectives stated in @ch:introduction and the success criteria of @ch:research_gap.

+ *Modular and reusable simulation model:* It provides a reproducible Isaac~Sim and Isaac~Lab model of the ceiling-mounted 5-#ac("DoF") tendon-driven manipulator, assembled modularly from the 2-#ac("DoF") prismatic base, the antiparallelogram elbow, the 2-#ac("DoF") cable-driven wrist, and the Robotiq~2F-140 gripper, together with the shared conveyor scene and drum task (@ch:methodology). The model is realised as three actuation variants that share a common kinematic interface, namely a #ac("PD")-driven reference, a constant-$J^top$ tendon variant, and a physical tendon variant. To the author's knowledge this is the first tendon-driven, antiparallelogram-jointed tensegrity manipulator modelled for #ac("RL") in the Isaac~Sim/Isaac~Lab ecosystem (@sec:research_gap_id), extending the predecessor work of Klein~@Klein2023, which provided a workspace analysis and Gazebo model but no learning-based control.

+ *Custom tendon actuation layer:* It implements tendon-driven actuation directly in Isaac~Lab through an analytic Jacobian-transpose torque mapping and, for the physical tendon variant, body-force application on the closed-chain linkage (@sec:tendon_actuation). The mapping is compatible with the PhysX reduced-coordinate articulation solver, differentiable for gradient-based training, and parallelizable across thousands of environments, which closes the absence of a native, #ac("RL")-compatible tendon primitive in Omniverse and PhysX noted in @sec:prior_tendon_sim.

+ *Quantitative model validation:* It validates the three actuation variants against the predecessor work~@Klein2023 by means of a #ac("PID") step-response benchmark and a Monte-Carlo workspace and manipulability characterization (@subsec:step_results, @subsec:workspace_results), which together satisfy the validation success criterion of @ch:research_gap. The accompanying robot-agnostic sampling, voxelisation, and visualisation tooling is reusable for future robot configurations.

+ *Reinforcement-learning tasks and comparative baseline:* It formulates two manager-based #ac("RL") tasks of increasing difficulty, the reach task and the selective cube-place task with phase-gated composite shaping rewards and a difficulty curriculum, and trains `skrl` #ac("PPO") policies for each actuation variant (@sec:rl_tasks). The reach evaluation is reported as a seed-aggregated comparison against zero, uniform-random, and DLS-IK baselines, and shows that the constant-$J^top$ tendon variant matches the #ac("PD") reference with no measurable task-performance penalty on either task (@subsec:reach_performance, @subsec:place_final).

+ *Diagnostic three-variant comparison and bottleneck localisation:* The deliberate separation into #ac("PD"), constant-$J^top$, and physical tendon variants isolates actuation-model effects from task-design effects, and thereby localises the remaining technical bottleneck. Direct end-to-end policy control through closed-chain body-force tendon actuation does not yet acquire reliable grasping and placing. This narrows the open problem from a general simulation question to a specific actuation-interface question (@sec:discussion_outcomes).

+ *Reproducible infrastructure for the master thesis:* It delivers the simulation, validation, and training infrastructure as a documented external Isaac~Lab extension (`tensegrity_pick`) with registered Gymnasium environments per task and variant, an evaluation harness, an automated plotting pipeline, and a custom #acs("TUI")/web training monitor (@sec:training_infrastructure). This provides a foundation that future work can extend without revisiting the tendon model itself.

The thesis does not demonstrate textile sorting, deformable-cloth manipulation, multi-agent coordination, perception-based classification, or sim-to-real transfer. These aspects are deliberately excluded by the scope in @ch:introduction and are deferred to the master thesis. The validated rigid-body simulation, the modular task infrastructure, and the diagnostic three-variant actuation comparison form the foundation on which the master-thesis goals are subsequently built.

==== Outlook <sec:outlook>

The master thesis starts from the foundation established here and extends it along five directions, in line with the partial goals stated for the master thesis programme. The directions are listed in increasing distance from the project-thesis scope, so the dependency between the rigid-body baseline and the deformable-object goals remains visible.

The first direction is the transition from rigid cube manipulation to conveyor-based textile sorting. The robot must selectively retrieve garments from a mixed waste stream rather than known rigid objects, which transforms validated pick-and-place control into selectivity-aware manipulation with new task-success criteria such as sorting accuracy, false-pick rate, and missed-pick rate.

The second direction is the integration of deformable cloth simulation and perception. The project thesis uses simulation state observations and rigid objects. The master thesis requires garment models with realistic material parameters, simulated camera views, and a damage-condition classifier. The Isaac~Lab scene, the logging pipeline, and the curriculum structure of the project thesis remain reusable, while the object model and the observation space require explicit extension rather than reuse from the cube-place task.

The third direction is multi-agent system design. A second collaborative arm stretches and presents the garment for the camera, while the tendon-driven picker continues to retrieve textiles from the conveyor. The extension introduces a second robot model, compatible task interfaces, an inter-agent coordination protocol, and metrics for handover quality, classification accuracy, and cycle time.

The fourth direction is the tendon-wrist transfer experiment. The 2-#ac("DoF") tendon-driven wrist developed within the manipulator is isolated and applied to a conventional industrial arm under matched training conditions. The experiment turns the compliant wrist design from an assumed advantage into a testable research question with a controlled comparison against the native rigid wrist.

The fifth direction is preparation for physical deployment. The Isaac~Lab configuration is extended with domain randomization, sensor and lighting variation, an #ac("ROS")~2 interface, safety constraints, and transfer-readiness evaluation procedures. This step is meaningful only after the rigid-body baseline of the project thesis remains reproducible and after the physical tendon control bottleneck identified in @sec:discussion_outcomes is resolved.

In summary, the project thesis establishes the simulation, validation, and learning foundation. The master thesis extends that foundation toward multi-agent textile sorting.

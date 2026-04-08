// §4.1 Overview of Approach
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table
#import "../../../../shared/formatting/diagrams.typ": methodology-overview

== Overview of Approach <sec:approach_overview>

The experimental methodology follows three stages, illustrated in @fig:methodology_overview:

+ *Model construction and validation.* A physically representative simulation model of the tendon-driven tensegrity manipulator is constructed from #ac("CAD") data and hardware specifications. Two distinct elbow model variants are implemented: a simplified disc approximation and the full antiparallelogram four-bar linkage. Each variant is validated through step-response testing and workspace analysis against reference data from Klein~@Klein2023.

+ *Progressive task formulation.* Two #ac("RL") tasks of increasing complexity are defined on the validated model. A _reach_ task trains the agent to track end-effector poses, verifying that each actuation mode supports effective policy learning. A _cube place_ task extends reach to a sequential pick-transport-place problem with the Robotiq 2F-140 gripper, requiring both arm coordination and gripper control.

+ *Cross-variant evaluation.* Each task is trained under three actuation modes (#ac("PD")-driven, tendon-driven, and physical tendon-driven), producing a systematic comparison of learning performance, motion quality, and task success across different levels of physical fidelity.

#faps-figure(
  methodology-overview(),
  caption: [Overview of the experimental pipeline. Stage~1 constructs and validates the robot model with three actuation modes. Stage~2 formulates progressive #ac("RL") tasks. Stage~3 evaluates task performance across all robot-actuation variants.],
  short-caption: [Overview of the experimental pipeline],
) <fig:methodology_overview>

The following sections describe each component in the order listed above. @sec:physical_robot introduces the robot design — the tensegrity arm hardware and the simulated base and gripper that complete the manipulator. @sec:sim_model_construction details the translation from #ac("CAD") geometry to a physics-ready #ac("USD") asset. @sec:tendon_actuation derives the tendon-to-joint mapping used in simulation. @sec:model_validation presents the validation protocol. @sec:rl_tasks and @sec:training_infrastructure describe the #ac("RL") task formulations and training setup, respectively.

// §4.1 Overview of Approach
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure
#import "../../../../shared/formatting/diagrams.typ": methodology-overview

== Overview of Approach <sec:approach_overview>

The experimental methodology follows three stages, illustrated in @fig:methodology_overview:

+ *Model construction and validation:* A physically representative simulation model of the tendon-driven tensegrity manipulator is constructed from #ac("CAD") data and hardware specifications. Two distinct elbow model variants are implemented: a simplified disc approximation and the full antiparallelogram four-bar linkage. Each variant is validated through step-response testing and workspace analysis against reference data from Klein~@Klein2023.

+ *Progressive task formulation:* Two #ac("RL") tasks of increasing complexity are defined on the validated model. A _reach_ task trains the agent to track end-effector poses, verifying that each actuation mode supports effective policy learning. A _cube place_ task extends reach to a sequential pick-transport-place problem with the Robotiq 2F-140 gripper, requiring both arm coordination and gripper control.

+ *Cross-variant evaluation: * Each task is trained under three actuation modes (PD-driven, tendon-driven (constant $J^top$), and physical tendon (body-force)), producing a systematic comparison of learning performance, motion quality, and task success across different levels of physical fidelity.

#faps-figure(
  methodology-overview(),
  caption: [Overview of the experimental pipeline. Stage~1 constructs and validates the robot model. Stage~2 formulates progressive #ac("RL") tasks on the validated model. Stage~3 evaluates each task across the models.],
  short-caption: [Overview of the experimental pipeline],
) <fig:methodology_overview>

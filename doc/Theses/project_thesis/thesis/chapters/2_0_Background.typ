// Chapter 2: Background
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *

= Background <ch:background>
//
// #highlight(fill:red)[TODO: exchange "tensegrity manipulator" with actual name of Robot.]
//
This chapter provides an overview of the theoretical and technical background relevant to this thesis. It covers fundamental concepts in #ac("RL"), tendon-driven robotic mechanisms, workspace analysis, physics simulation, and the software architecture of Isaac Sim and Isaac Lab. The goal is to establish a common foundation for understanding the design choices and methodologies employed in the subsequent chapters.

// §2.1 — #ac("RL") fundamentals: MDP/#ac("POMDP") formalism, policy gradients, #ac("PPO"), #ac("GAE")
#include "background/2_1_ReinforcementLearning.typ"

#pagebreak()

// §2.2 — Tendon-driven mechanisms: four-bar linkages, cable-driven wrists, J^T mapping, tensegrity
#include "background/2_2_TendonLinkageMechanisms.typ"

// §2.3 — Workspace analysis and dexterity metrics
#include "background/2_3_WorkspaceAnalysis.typ"

#pagebreak()

// §2.4 — Rigid-body physics simulation in PhysX 5
#include "background/2_4_PhysicsSimulation.typ"

// §2.5 — Isaac Sim 5.1 and Isaac~Lab architecture
#include "background/2_5_IsaacSimLab.typ"

#pagebreak()

// §2.6 — Robot and scene description formats (#ac("USD"), #ac("URDF"))
#include "background/2_6_DescriptionFormats.typ"

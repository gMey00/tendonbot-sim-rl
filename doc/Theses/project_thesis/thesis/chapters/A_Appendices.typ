// Appendices — wrapper that includes per-section sub-files.
// Shared imports/helpers live inside the individual `appendices/*.typ` files
// because `#include` does not propagate definitions from this scope.

// Reset heading numbering for appendix style
#set heading(numbering: "A.1")
#counter(heading).update(0)

// §A.1 — Simulation and Control Parameters
#include "appendices/A_1_SimulationParams.typ"

// §A.2 — Kinematic Reference Data (Link Origins & Reference Points)
#include "appendices/A_1b_KinematicReference.typ"

// §A.3 — PPO Hyperparameters
#include "appendices/A_2_PPOHyperparams.typ"

// §A.3 — Cube Place Reward Terms
#include "appendices/A_3_PlaceRewards.typ"

// §A.4 — Reach Task: Complete Training Metrics
#include "appendices/A_4_ReachMetrics.typ"

// §A.5 — Cube Place Task: Complete Training Metrics
#include "appendices/A_5_PlaceMetrics.typ"

// §A.6 — Training Monitor
#include "appendices/A_6_TrainingMonitor.typ"

// §A.7 — Step-Response Time Traces
#include "appendices/A_7_StepResponses.typ"

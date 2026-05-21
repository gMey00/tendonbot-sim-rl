// Appendices — wrapper that includes per-section sub-files.
// Shared imports/helpers live inside the individual `appendices/*.typ` files
// because `#include` does not propagate definitions from this scope.

// Reset heading numbering for appendix style
#set heading(numbering: "A.1")
#counter(heading).update(0)

// §A.1 — Kinematic Reference Data (Link Origins & Reference Points)
#include "appendices/A_1_KinematicReference.typ"
  
// §A.2 — Simulation and Control Parameters
#include "appendices/A_2_SimulationParams.typ"

// §A.3 — Step-Response Time Traces
#include "appendices/A_3_StepResponses.typ"

// §A.4 — PPO Hyperparameters
#include "appendices/A_4_PPOHyperparams.typ"

// §A.5 — Reach Task: Complete Training Metrics
#include "appendices/A_5_ReachMetrics.typ"

// §A.6 — Cube Place Reward Terms
#include "appendices/A_6_PlaceRewards.typ"

// §A.7 — Cube Place Task: Complete Training Metrics
#include "appendices/A_7_PlaceMetrics.typ"

// §A.8 — Training Monitor
#include "appendices/A_8_TrainingMonitor.typ"

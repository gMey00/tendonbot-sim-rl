// Chapter 4: Methodology
// Completely rewritten — detailed treatment of robot model construction,
// kinematics, tendon actuation, validation, and #ac("RL") task formulation.
// All content derived from:  doc/Tensegrity_robot/ documentation,
// src/tensegrity_pick/ code, and the referenced bibliography.
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *
#import "../../../shared/formatting/diagrams.typ": methodology-overview

= Methodology <ch:methodology>

This chapter describes the complete workflow for constructing, validating, and using a tendon-driven #highlight(fill: red)[tensegrity manipulator] in simulation for #ac("RL")-based manipulation tasks. First, the robot design is introduced (@sec:physical_robot). Next, the process for translating the #ac("CAD") geometry into a physics-ready Isaac Sim simulation model is detailed (@sec:sim_model_construction). The tendon actuation layer that maps cable tensions to joint torques in simulation is then derived (@sec:tendon_actuation). Validation of the resulting model is presented (@sec:model_validation). Finally, the formulation of #ac("RL") tasks built on top of this model is described (@sec:rl_tasks), together with the training infrastructure (@sec:training_infrastructure).

All implementation is carried out in NVIDIA Isaac Sim~5.1.0 with the IsaacLab framework, using the `skrl` library for #ac("PPO")-based policy training~@SerranoMunoz2023skrl @Schulman2017PPO. The source code is organized as an external IsaacLab extension (`tensegrity_pick`) that provides registered Gymnasium environments for each task and robot variant.

// ── §4.1  Overview of Approach ───────────────────────────────
#include "methodology/4_1_Overview.typ"

// ── §4.2  Robot Design ──────────────────────────────
#include "methodology/4_2_PhysicalRobot.typ"

// ── §4.3  Simulation Model Construction ──────────────────────
#include "methodology/4_3_SimulationModel.typ"

// ── §4.4  Tendon Actuation in Simulation ─────────────────────
#include "methodology/4_4_TendonActuation.typ"

// ── §4.5  Model Validation ───────────────────────────────────
#include "methodology/4_5_Validation.typ"

// ── §4.6  #ac("RL") Tasks ───────────────────────
#include "methodology/4_6_RLTasks.typ"

// ── §4.7  Training Infrastructure ────────────────────────────
#include "methodology/4_7_Infrastructure.typ"

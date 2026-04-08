// §4.7 Training Infrastructure
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table

== Training Infrastructure <sec:training_infrastructure>

This section describes the #ac("PPO") configuration, simulation parameters, and hardware setup used for all experiments.

=== PPO Configuration <subsec:ppo_config>

All tasks use #ac("PPO")~@Schulman2017PPO from the `skrl` library~@SerranoMunoz2023skrl with #ac("GAE")~@Schulman2015GAE for advantage estimation. @tab:ppo_summary lists the principal hyperparameters. The complete configuration is provided in @appendix:ppo_hyperparams.

#faps-table(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Parameter*], [*Value*], [*Notes*]),
    [Policy / value network], [$[64, 64]$ / $[256, 128, 64]$ ELU], [Reach / Cube Place],
    [Rollout horizon], [24 / 48 steps], [Reach / Cube Place],
    [Learning epochs], [5], [Per rollout batch],
    [Mini-batches], [4 / 8], [Reach / Cube Place],
    [Learning rate], [$10^(-3)$ / $10^(-4)$], [KL-adaptive scheduler],
    [Discount $gamma$], [0.99], [—],
    [GAE $lambda$], [0.95], [—],
    [Training environments], [4,096], [Parallel GPU instances],
    [Evaluation environments], [50], [Separate evaluation pass],
  ),
  caption: [#ac("PPO") hyperparameter summary. All robot/actuation variants within a given task share identical hyperparameters to ensure a fair comparison.],
  short-caption: [PPO hyperparameter summary],
) <tab:ppo_summary>

=== Simulation Parameters <subsec:sim_params>

#faps-table(
  table(
    columns: (auto, auto, auto),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Parameter*], [*Reach task*], [*Cube place task*]),
    [Physics timestep], [$1/60$~s], [$1/50$~s],
    [Decimation factor], [2], [2],
    [Effective control frequency], [30~Hz], [25~Hz],
    [Episode length], [12.0~s (360 steps)], [5.0~s (250 steps)],
    [Parallel environments], [4,096 (train) / 50 (eval)], [4,096 (train) / 50 (eval)],
  ),
  caption: [Simulation parameters per task. The physics timestep and episode length are adapted to the complexity of each task. The decimation factor holds the number of physics substeps per control step.],
  short-caption: [Simulation parameters per task],
) <tab:sim_params>

=== Software and Hardware <subsec:sw_hw>

All experiments are conducted on a single workstation equipped with an NVIDIA RTX~A6000 GPU (48~GB VRAM), running Isaac Sim~5.1.0, IsaacLab (main branch, March 2026 snapshot), and `skrl`~1.3. The robot model is stored as #ac("USD") assets in the repository, and tasks are registered as Gymnasium environments within the `tensegrity_pick` IsaacLab extension. Training is managed through `skrl`'s sequential trainer, with TensorBoard logging for reward curves, episode metrics, and custom diagnostic signals. A custom monitoring application allows the live tracking of all relevant data in a #ac("TUI") or as a web application.

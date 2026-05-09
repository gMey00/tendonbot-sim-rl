// §4.7 Training Infrastructure
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table, faps-figure
#import "../../../../shared/formatting/diagrams.typ": training-infrastructure-stack

== Training Infrastructure <sec:training_infrastructure>

This section describes the #ac("PPO") configuration, simulation parameters, and hardware setup used for all experiments.

=== #ac("PPO") Configuration <subsec:ppo_config>

All tasks use #ac("PPO")~@Schulman2017PPO from the `skrl` library~@SerranoMunoz2023skrl with #ac("GAE")~@Schulman2015GAE for advantage estimation. The complete configuration is provided in @appendix:ppo_hyperparams.

=== Simulation Parameters <subsec:sim_params>

#faps-table(
  table(
    columns: (auto, auto, auto),
    table.header([*Parameter*], [*Reach task*], [*Cube place task*]),
    [Physics timestep], [$1/60$~s], [$1/100$~s],
    [Decimation factor], [2], [2],
    [Effective control frequency], [30~Hz], [50~Hz],
    [Episode length], [12.0~s (360 steps)], [5.0~s (250 steps)],
    [Parallel environments], [2,048 (train) / 50 (eval)], [2,048 (train) / 50 (eval)],
  ),
  caption: [Simulation parameters per task. The physics timestep and episode length are adapted to the complexity of each task. The decimation factor holds the number of physics substeps per control step.],
  short-caption: [Simulation parameters per task],
) <tab:sim_params>

=== Software and Hardware <subsec:sw_hw>

@fig:training_infrastructure summarises the four-layer software/hardware stack used for all training runs. From top to bottom: the *algorithm layer* hosts the #ac("PPO") agent in `skrl`~$gt.eq$~1.4.3 together with the sequential trainer and the logging back-ends. The *environment layer* exposes each task as a Gymnasium environment defined inside the `tensegrity_pick` IsaacLab extension and decomposed by the IsaacLab #ac("MDP") managers (observations, actions, rewards, terminations, curricula). The *simulation layer* is provided by IsaacSim~5.1.0 (#ac("USD") scene graph, `Cloner`-based scene replication) on top of the PhysX 5 backend (rigid bodies, articulations, fixed and spatial tendon constraints). The *hardware layer* is a single workstation with an NVIDIA RTX~A6000 #ac("GPU") (48~GB VRAM) running a dedicated conda environment.

#faps-figure(
  training-infrastructure-stack(),
  caption: [Training infrastructure stack used in this work. The #ac("PPO") learner (`skrl`~$gt.eq$~1.4.3) drives a vectorized Gymnasium environment that is realised through IsaacLab manager classes. Environment cloning, physics, and rendering execute on a single RTX~A6000 #ac("GPU") through the IsaacSim/PhysX backend. Custom diagnostic signals are streamed to TensorBoard and to a #acs("TUI")/web monitoring application.],
  short-caption: [Training infrastructure stack],
) <fig:training_infrastructure>

*Logging and monitoring.* `skrl`'s sequential trainer streams scalar reward components, episode lengths, and KL/clipping diagnostics to TensorBoard. In addition, each task overrides the IsaacLab `RewardManager` to emit individually-weighted reward terms, which are essential for the per-term ablation tables in @ch:results. A custom monitoring application under `tools/monitor/` exposes the live training scalars both as a #ac("TUI") (`monitor.sh`) and as a web dashboard (`monitor.sh --web`). Both back-ends share the same parser and can be attached to a remote training run via SSH port forwarding. More details on the monitoring tool are provided in @appendix:training_monitor.

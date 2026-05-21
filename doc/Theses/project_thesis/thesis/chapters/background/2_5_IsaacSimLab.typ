// §2.5 NVIDIA Isaac Sim and IsaacLab
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/diagrams.typ": isaac-stack
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table

== NVIDIA Isaac Sim and IsaacLab <sec:isaac_sim_isaaclab_architecture>

NVIDIA Isaac~Sim (Version 5.1.0) is used as the primary simulation environment in this thesis because it integrates robotics-oriented scene authoring, high-fidelity multi-physics simulation, and sensor-realistic rendering within a single Omniverse-based application~@nvidia_isaac_sim_510_what_is. The associated IsaacLab framework structures the learning workflow, providing a modular interface for designing robotics tasks as #acp("MDP") and training policies at scale using GPU-accelerated parallel simulation~@Mittal2025IsaacLab.

=== Isaac Sim in the Omniverse Stack

Isaac~Sim is a reference application on top of NVIDIA Omniverse, designed for developing, simulating, and testing #ac("AI")-driven robots in physically based virtual environment. A central architectural decision is the use of #ac("USD") as the scene representation. Robotics assets and environments are expressed as a hierarchical scene graph, serving as the common data layer across physics, rendering, and robotics tooling. Isaac~Sim is built on Omniverse Kit, a modular framework based on lightweight plugins and an extension mechanism. Capabilities such as physics interfaces, sensor simulation, and robotics middleware bridges are exposed as extensions that can be enabled, configured, and scripted from Python, supporting both interactive development and headless execution in automated pipelines.~@nvidia_isaac_sim_510_what_is The resulting software stack is shown in @fig:isaacsim_isaaclab_stack.

#faps-figure(
  isaac-stack(),
  caption: [Schematic view of the software stack used in this thesis. Isaac~Sim is built on Omniverse Kit with #ac("USD") as the scene data layer and PhysX~5 as the physics backend. IsaacLab provides task abstractions (manager-based #ac("MDP") environments) and training utilities on top.],
  short-caption: [Isaac Sim and IsaacLab software stack],
) <fig:isaacsim_isaaclab_stack>

The most relevant capabilities for this thesis are:
- *PhysX~5-based multi-physics:* GPU-accelerated rigid-body and articulated dynamics for contact-rich robotics simulation (@sec:physics_sim)~@nvidia_isaac_sim_510_what_is @physx_gpu_rigid_bodies_510.
- *Environment cloning:* The `isaacsim.core.cloner` extension efficiently replicates prims and environments with optional inter-clone collision filtering, enabling the throughput scaling required for #ac("RL")~@nvidia_isaac_sim_510_extensions_api.
- *#ac("USD")-centric asset pipeline:* Structured scene graphs with physics and rendering metadata support scalable instancing and non-destructive composition of complex environments~@nvidia_isaac_sim_510_what_is.

=== IsaacLab as a Robot Learning Framework

IsaacLab is the successor to Isaac Gym and extends GPU-native simulation into large-scale robot learning while remaining integrated with Isaac~Sim for high-fidelity physics~@Mittal2025IsaacLab. It consolidates the GPU-native paradigm of Isaac Gym~@Makoviychuk2021IsaacGym and the modular environment design of the Orbit framework~@Mittal2023Orbit into a unified training runtime.

==== Manager-based #ac("MDP") workflow
 IsaacLab structures simulation tasks as #acp("MDP") through a manager-based architecture. A base environment is composed from reusable managers (action, observation, event), then extended to an #ac("RL") task environment by adding reward, termination, command, and curriculum component. This decomposition separates environment implementation from task specification. Task specific behavior is encoded in configuration objects rather than by modifying core environment classes, enabling consistent reuse across tasks and reducing duplication across experimental variants.~@isaaclab_manager_based_rl_env

The `ManagerBasedRLEnv` interface returns #ac("MDP")-relevant signals while remaining compatible with standard environment creation patterns (e.g., via `gymnasium.make`), supporting integration with multiple #ac("RL") libraries without rewriting the simulator interface layer~@isaaclab_manager_based_rl_env.

==== GPU-native training
  The key performance advantage is that physics stepping and policy learning share device memory, avoiding #ac("CPU")--#ac("GPU") transfer bottlenecks. Isaac Gym demonstrated that this GPU-resident design can yield order-of-magnitude improvements in end-to-end training throughput compared to #ac("CPU")-based simulation~@Makoviychuk2021IsaacGym. IsaacLab extends this paradigm with documented multi-#ac("GPU") and multi-node training support, enabling scale-out execution for #ac("RL") workloads when training is bottlenecked by rollout collection~@Mittal2025IsaacLab @isaaclab_multi_gpu_multinode.

=== Comparison to Alternative Simulators

#faps-table(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header(
      [*Platform*], [*Primary strength*], [*Thesis fit*],
    ),
    [Isaac Sim + IsaacLab], [Integrated Omniverse stack: PhysX, RTX, #ac("RL") tooling~@nvidia_isaac_sim_510_what_is @Mittal2025IsaacLab], [High],
    [MuJoCo], [Efficient dynamics for model-based control~@Todorov2012MuJoCo], [Medium],
    [PyBullet/Bullet], [Accessible real-time physics~@pybullet_homepage], [Medium],
  ),
  caption: [Qualitative comparison of simulation platforms with respect to thesis requirements. Isaac~Sim is selected for its integrated physics, rendering, and scalable #ac("RL") training in a single runtime.],
  short-caption: [Qualitative simulator comparison],
) <tab:simulator_comparison>

MuJoCo~@Todorov2012MuJoCo is widely adopted in continuous-control research for efficient per-step dynamics but does not natively provide sensor-realistic rendering or a scalable parallel training framework within the same architecture. PyBullet~@pybullet_homepage offers rapid prototyping through Python-accessible simulation but requires assembly of separate components for rendering, training infrastructure, and middleware integration. Isaac~Sim~5.1 with IsaacLab is selected because it provides an integrated #ac("USD")-based scene representation, PhysX-backed multi-physics, and a principled #ac("MDP")-based task design workflow with GPU-native training support in a single runtime~@nvidia_isaac_sim_510_what_is @Mittal2025IsaacLab.

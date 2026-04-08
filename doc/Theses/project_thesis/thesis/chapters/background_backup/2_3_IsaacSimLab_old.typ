// §2.3 NVIDIA Isaac Sim and IsaacLab Architecture
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/diagrams.typ": isaac-stack
#import "../../../../shared/formatting/template.typ": faps-figure, faps-table

== NVIDIA Isaac Sim and IsaacLab Architecture <sec:isaac_sim_isaaclab_architecture>

NVIDIA Isaac Sim (version 5.1) is used as the primary simulation environment in this thesis because it integrates robotics-oriented scene authoring, high-fidelity multi-physics simulation, and sensor-realistic rendering within a single Omniverse-based application stack~@nvidia_isaac_sim_510_what_is. In addition, the associated IsaacLab framework is used to structure the learning and evaluation workflow, because it provides a unified, modular interface for designing robotics tasks as #acp("MDP") and training policies at scale using GPU-accelerated parallel simulation~@Mittal2025IsaacLab. The resulting toolchain aims to minimize integration overhead between simulation fidelity (e.g., contact-rich multi-body dynamics), robotic middleware interfaces, and the training pipeline used for robot learning.

=== Isaac Sim in the Omniverse application stack

Isaac Sim is implemented as a reference application on top of NVIDIA Omniverse and is designed to support development, simulation, and testing of AI-driven robots in physically based virtual environments~@nvidia_isaac_sim_510_what_is. A central architectural decision is the use of #ac("USD") as the scene representation and interchange format: robotics assets and environments are expressed as a hierarchical scene graph with reusable primitives, enabling non-destructive composition and scalable instancing workflows~@nvidia_isaac_sim_510_what_is. This representation is used as the common "data layer" across physics, rendering, and robotics-specific tooling, which reduces interface mismatches between asset creation and simulation-time execution.

Isaac Sim is built on Omniverse Kit, which is a modular application framework based on lightweight plugins and an extension mechanism for composing functionality~@nvidia_isaac_sim_510_what_is. Consequently, Isaac Sim capabilities (e.g., physics interfaces, sensor simulation, and robotics middleware bridges) are exposed as extensions that can be enabled, configured, and scripted from Python, which supports both interactive development in the graphical application and headless execution in automated pipelines~@nvidia_isaac_sim_510_what_is. This extension-centric architecture is relevant to robot learning workloads because it enables the simulator to be configured as a reproducible experiment runtime, rather than as a monolithic interactive tool.

#faps-figure(
  isaac-stack(),
  caption: [Schematic view of the software stack used in this thesis. Isaac Sim is built on Omniverse Kit with #ac("USD") as the scene data layer and PhysX 5 as the physics backend; RTX-based rendering is used for sensor simulation. IsaacLab is implemented on top of Isaac Sim and provides task abstractions (e.g., manager-based #ac("MDP") environments) and training utilities (e.g., multi-#acp("GPU") execution).],
  short-caption: [Isaac Sim and IsaacLab software stack],
) <fig:isaacsim_isaaclab_stack>

=== Core simulation features in Isaac Sim 5.1

Isaac Sim combines high-fidelity physics, photorealistic rendering, and robotics integration within a single runtime~@nvidia_isaac_sim_510_what_is. These features are not independent: physics fidelity affects contact-rich manipulation, rendering realism affects perception and sim-to-real transfer, and middleware integration affects how simulation is connected to external robot software stacks. The most relevant capabilities for this thesis are summarized below.

- *Omniverse and #ac("USD")-centric scene representation:* Scenes are represented in #ac("USD"), which serves as the unifying data interchange format for robot assets, environments, and simulation metadata~@nvidia_isaac_sim_510_what_is. This enables structured scene graphs and supports scalable instancing patterns for large simulation workloads.
- *High-fidelity physics based on PhysX 5:* Isaac Sim uses a GPU-based PhysX engine and provides access to rigid-body and articulated dynamics required for robotics simulation~@nvidia_isaac_sim_510_what_is. GPU rigid-body simulation in PhysX provides GPU-accelerated broad phase, contact generation, and constraint solving when enabled, while other components remain on the CPU~@physx_gpu_rigid_bodies_510.
- *Photorealistic RTX-based sensor simulation:* Isaac Sim supports multi-sensor RTX rendering and uses GPU access to simulate sensors such as cameras and lidar in an industrially scalable manner~@nvidia_isaac_sim_510_what_is. RTX-based sensor simulation is exposed via dedicated extensions (e.g., RTX sensors) within the Kit extension framework~@nvidia_isaac_sim_510_extensions_api.
- *Robotics middleware support via #ac("ROS")~2 bridging:* A dedicated #ac("ROS")~2 Bridge extension is provided to publish and subscribe to #ac("ROS")~2 topics and services from within Isaac Sim, enabling bidirectional communication between simulation and external #ac("ROS")~2 systems~@nvidia_isaac_sim_510_extensions_api.

==== GPU acceleration and parallel simulation

GPU acceleration in the Isaac ecosystem is achieved at multiple levels, ranging from physics backend execution to batched simulation of many environments. At the physics-backend level, PhysX GPU rigid bodies are configured at the scene level and enable GPU-accelerated rigid-body pipeline components including broad phase, contact generation, and the constraint solver~@physx_gpu_rigid_bodies_510. Since not all features are executed on the GPU, the performance benefit depends strongly on scene composition and on whether large numbers of active bodies are simulated concurrently~@physx_gpu_rigid_bodies_510. This architecture is aligned with robot learning workloads in which many similar environments are executed in parallel to increase sample throughput.

At the application level, Isaac Sim exposes explicit support for cloning and replicating environments. In particular, the `isaacsim.core.cloner` extension is provided to clone prims and environments efficiently and to optionally filter collisions across clones, which is a practical requirement when many environments share a single process runtime~@nvidia_isaac_sim_510_extensions_api. This cloning functionality forms a key bridge between a high-fidelity simulator and the throughput requirements of #ac("RL"), since scalable policy training is typically bottlenecked by the rate of environment rollouts rather than by single-scene fidelity.

=== IsaacLab as a unified robot learning framework

IsaacLab is used in this thesis as the primary framework for structuring robot learning experiments on top of Isaac Sim. IsaacLab is described as the successor to Isaac Gym and is intended to extend GPU-native simulation into large-scale, multi-modal robot learning, while remaining integrated with Isaac Sim for high-fidelity physics and RTX-based sensing~@Mittal2025IsaacLab. The framework is designed to unify workflows spanning #ac("RL"), imitation learning, and motion planning, while exposing reusable abstractions for robot assets, environments, and training pipelines~@Mittal2025IsaacLab.

A key motivation for IsaacLab is the unification of two prior lines of development in the Isaac ecosystem. First, Isaac Gym demonstrated that physics simulation and policy learning can be executed entirely on the #ac("GPU") with direct tensor access, yielding large speedups over CPU-based simulators when training complex robot policies~@Makoviychuk2021IsaacGym. Second, the Orbit framework established a modular environment library on top of Isaac Sim, providing benchmark tasks and supporting policy training and demonstration collection using GPU-based parallelization~@Mittal2023Orbit. IsaacLab is positioned as a consolidated framework that carries forward the GPU-native paradigm of Isaac Gym and the modular task and environment design principles of Orbit, but within an Isaac Sim-integrated runtime that supports richer sensing and broader multi-physics capabilities~@Mittal2025IsaacLab.

==== Manager-based MDP workflow and code reuse

IsaacLab structures simulation tasks explicitly as #acp("MDP"), i.e., as environments that provide state (or observations), actions, and reward/termination signals suitable for learning. Within the manager-based workflow, a base environment is composed from reusable managers (e.g., action, observation, and event managers) and is then extended to an #ac("RL") task environment by adding reward, termination, command, and curriculum components~@isaaclab_manager_based_rl_env. This decomposition is relevant for thesis-scale experimentation because it separates environment implementation from task specification: instead of modifying core environment classes directly, task-specific behavior is encoded in configuration objects, which enables consistent reuse across tasks and reduces duplication across experimental variants~@isaaclab_manager_based_rl_env.

The manager-based formulation also encourages a stable interface between simulation and training. In the documented workflow, the `ManagerBasedRLEnv` interface returns #ac("MDP")-relevant signals (including reward and termination status) while remaining compatible with standard environment creation patterns (e.g., via `gymnasium.make`), which supports integration with multiple #ac("RL") libraries without rewriting the simulator interface layer~@isaaclab_manager_based_rl_env. The resulting abstraction is intended to enable systematic experimentation across robot morphologies and task configurations while preserving a consistent training loop.

==== Distributed training support in IsaacLab

Robot learning in high-dimensional, contact-rich tasks often requires a large number of environment interactions, which motivates distributed execution. IsaacLab provides explicit support for multi-#ac("GPU") and multi-node #ac("RL") training, with documented workflows using PyTorch distributed tooling (e.g., `torchrun`) and a process-per-#ac("GPU") execution model in which each process instantiates its own IsaacLab environment and synchronizes gradients across processes~@isaaclab_multi_gpu_multinode. This design is aligned with on-policy #ac("RL") settings, in which large batch sizes and parallel rollouts can substantially increase training throughput, provided that simulator stepping and observation extraction can be executed efficiently.

The documented multi-#ac("GPU") pipeline is also operationally relevant: each process is responsible for its own environment instance and rollout collection buffers, while synchronization occurs only during gradient updates~@isaaclab_multi_gpu_multinode. This reduces coupling between simulation instances and avoids centralized simulation bottlenecks, at the expense of higher per-process resource usage. For thesis experiments that require extensive policy iteration or hyperparameter exploration, this design provides a practical path toward scaling without abandoning the integrated Isaac Sim runtime.

==== Imitation learning and motion planning integration

Beyond #ac("RL"), IsaacLab supports imitation learning workflows that combine human demonstration segments with automated planning. The SkillGen pipeline is documented as integrating motion planning into the IsaacLab Mimic demonstration workflow, thereby generating collision-free trajectories between skill segments using an external planner~@isaaclab_skillgen. In this workflow, cuRobo is used as the underlying motion planner for generating smooth, collision-free motions, and the planner is explicitly described as GPU-accelerated~@isaaclab_skillgen.

cuRobo itself is formulated as a GPU-parallel motion generation approach for manipulators, with a design that exploits massive parallelism (e.g., many optimization seeds and parallel collision checks) to achieve low-latency planning and inverse kinematics computations~@Sundaralingam2023cuRobo. The availability of such a planner within the IsaacLab ecosystem is relevant for tasks where expert trajectories must be generated efficiently, either to bootstrap learning (e.g., as demonstrations) or to define hybrid learning-and-planning pipelines.

=== Rationale for selecting Isaac Sim 5.1 with IsaacLab

The combined selection of Isaac Sim 5.1 and IsaacLab is driven by the need to simultaneously address (i) contact-rich manipulation in articulated robots and (ii) scalable learning infrastructure. These requirements are tightly coupled in the targeted thesis use case: reliable contact dynamics are essential for manipulation tasks, while learning-based policies require large numbers of rollouts to converge.

Scalability is addressed through GPU-native simulation and distributed training utilities. Isaac Gym established that GPU-resident simulation and learning with direct tensor access can yield large improvements in end-to-end training throughput compared to CPU-based simulation paired with GPU-based neural network training~@Makoviychuk2021IsaacGym. IsaacLab extends this paradigm within an Isaac Sim-integrated stack and provides documented support for multi-#ac("GPU") and multi-node training, enabling scale-out execution for #ac("RL") workloads when training is bottlenecked by rollout collection~@Mittal2025IsaacLab @isaaclab_multi_gpu_multinode. When combined with Isaac Sim cloning utilities for replicated environments~@nvidia_isaac_sim_510_extensions_api, the simulator can be configured to support large-scale parallel rollouts within a consistent runtime, which is particularly advantageous in on-policy scenarios where sample throughput dominates wall-clock training time.

Finally, robotics integration requirements are addressed at the middleware level. Isaac Sim exposes a #ac("ROS")~2 Bridge extension for publishing and subscribing to #ac("ROS")~2 topics and services through OmniGraph nodes and action graphs~@nvidia_isaac_sim_510_extensions_api. This bridge supports a common development pattern in which simulation is used for software-in-the-loop or hardware-in-the-loop testing, and it provides a practical interface to connect simulation-time controllers and perception to external robot software stacks without bespoke transport layers.

=== Comparison to alternative simulation platforms

The choice of simulator is constrained by trade-offs between fidelity, performance, and integration effort. For this thesis, the critical differentiator is the ability to combine contact-rich articulated manipulation with scalable learning infrastructure within a unified and scriptable runtime. Alternative platforms are briefly contrasted below to justify the selection of Isaac Sim 5.1 with IsaacLab for this specific use case.

MuJoCo is a widely used physics engine for model-based control that emphasizes efficient dynamics and contact simulation, and it is commonly adopted as a baseline simulator in continuous-control research~@Todorov2012MuJoCo. Its design makes it well-suited for rigid-body control benchmarks and for iterative controller development when the primary bottleneck is per-step simulation performance. However, the thesis use case requires integrated sensor-realistic rendering and a scalable parallel training framework within the same architecture, and these requirements typically necessitate additional tooling beyond a core rigid-body simulator.

PyBullet is widely used as a Python-accessible simulation interface built around the Bullet physics engine, and it is often adopted in robotics prototyping due to its ease of scripting and broad ecosystem integration~@pybullet_homepage. The underlying Bullet engine is documented in the context of real-time physics simulation and is frequently used in interactive applications~@Coumans2015Bullet. In practice, PyBullet is frequently selected when rapid iteration and lightweight deployment are prioritized, but the simulator stack used in this thesis additionally requires an integrated high-fidelity rendering and sensor simulation pipeline and a unified large-scale training framework, which are provided natively by Isaac Sim and IsaacLab rather than assembled from separate components.

SOFA is an open-source simulation framework primarily targeted at medical simulation and is designed around an advanced software architecture for building complex physical simulations by composing algorithms, solvers, and collision models~@Allard2007SOFA. This makes SOFA a strong candidate for high-fidelity soft-body modeling, including scenarios where accurate mechanics are required. Nevertheless, the overall thesis workflow requires scalable robot learning infrastructure with high-throughput parallel environment execution and integrated robotics middleware interfaces, which are not the primary design objectives of SOFA and typically require additional integration layers.

#faps-table(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header(
      [*Platform*], [*Primary strength (per source)*], [*Fit to this thesis use case*],
    ),
    [Isaac Sim + IsaacLab], [Integrated Omniverse stack with PhysX, RTX, and #ac("RL") tooling~@nvidia_isaac_sim_510_what_is @Mittal2025IsaacLab], [High (scalable RL training)],
    [MuJoCo], [Efficient physics engine for model-based control~@Todorov2012MuJoCo], [Medium (fidelity/tooling trade-offs)],
    [PyBullet/Bullet], [Accessible real-time physics simulation ecosystem~@pybullet_homepage @Coumans2015Bullet], [Medium (integration required)],
    [SOFA], [Modular framework for complex deformable simulation~@Allard2007SOFA], [Medium (learning-scale integration required)],
  ),
  caption: [Qualitative comparison of simulators with respect to thesis requirements.],
  short-caption: [Qualitative simulator comparison],
) <tab:simulator_comparison>

Overall, Isaac Sim 5.1 with IsaacLab is selected because it provides an integrated software architecture that aligns with the requirements of contact-rich manipulation and scalable robot learning. The Isaac stack combines a #ac("USD")-based scene representation, PhysX-backed multi-physics simulation, RTX-based sensing, and robotics middleware integration in a single Omniverse runtime~@nvidia_isaac_sim_510_what_is @nvidia_isaac_sim_510_extensions_api. IsaacLab further provides a principled #ac("MDP")-based task design workflow and documented multi-#ac("GPU")/multi-node training support, which reduces the engineering effort required to perform large-scale learning experiments while retaining simulation fidelity~@isaaclab_manager_based_rl_env @isaaclab_multi_gpu_multinode @Mittal2025IsaacLab.

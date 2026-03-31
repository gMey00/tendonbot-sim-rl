# Simulation & Physics Engines

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## NVIDIA Isaac Sim / Isaac Lab

### Journal Articles & Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Mittal2023Orbit | Mittal, M. et al. | "Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments" | 2023 | 🟢 | 🟢 | 🟢 | Introduces the Orbit/Isaac Lab simulation stack on NVIDIA Omniverse. Foundation for the entire RL training pipeline in this project. [DOI](https://doi.org/10.1109/LRA.2023.3270034) |
| Mittal2025IsaacLab | NVIDIA & Mittal, M. et al. | "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning" | 2025 | 🟢 | 🟢 | 🟢 | Describes Isaac Lab's architecture, manager-based environments, and GPU-accelerated training. Core framework reference. [arXiv](https://arxiv.org/abs/2511.04831) |
| Makoviychuk2021 | Makoviychuk, V. et al. | "Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning" | 2021 | 🟡 | 🟢 | 🟢 | Predecessor to Isaac Lab; demonstrates GPU-based parallel simulation for RL. Context for the evolution of the simulation platform. [OpenReview](https://openreview.net/forum?id=fgFBtYgJQX_) |

### Documentation (Online)

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| isaac_sim_510_overview | NVIDIA | What Is Isaac Sim? | 🟢 | 🟡 | 🟡 | Top-level overview of Isaac Sim 5.1.0 capabilities. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) |
| isaac_sim_510_release_notes | NVIDIA | Release Notes (5.1.0) | 🟡 | 🔴 | 🟡 | Version-specific feature list and known issues. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/release_notes.html) |
| isaac_sim_510_physics_fundamentals | NVIDIA | Physics Simulation Fundamentals | 🟢 | 🟢 | 🟡 | Explains PhysX integration, time stepping, and solver configuration in Isaac Sim. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html) |
| isaac_sim_ros2_reference_architecture | NVIDIA | ROS 2 Reference Architecture | 🟡 | 🟡 | 🟡 | Integration architecture for ROS 2 with Isaac Sim. Relevant for future sim-to-real pathway. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/reference_architecture.html) |
| isaaclab_overview_doc | Isaac Lab Developers | Overview | 🟢 | 🟡 | 🟡 | Isaac Lab high-level architecture and API overview. [Link](https://isaac-sim.github.io/IsaacLab/v1.0.0/index.html) |
| isaaclab_mdp_api | Isaac Lab Developers | isaaclab.envs.mdp API | 🟢 | 🟡 | 🟡 | MDP manager API for observations, actions, rewards, terminations. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html) |
| isaaclab_multi_gpu_doc | Isaac Lab Developers | Multi-GPU and Multi-Node Training | 🟡 | 🟡 | 🟡 | Documentation for distributed training setup. [Link](https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html) |
| isaaclab_manager_based_rl_env | Isaac Lab Developers | Creating a Manager-Based RL Environment | 🟢 | 🟡 | 🟡 | Tutorial for the environment design pattern used in this project. [Link](https://isaac-sim.github.io/IsaacLab/v1.0.0/source/tutorials/03_envs/create_manager_rl_env.html) |
| isaaclab_task_design_workflows | Isaac Lab Developers | Task Design Workflows | 🟢 | 🟡 | 🟡 | Manager-based vs direct workflow comparison. [Link](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html) |
| isaaclab_wrap_env | Isaac Lab Developers | Wrapping Environments | 🟢 | 🟡 | 🟡 | How to wrap Isaac Lab envs for external RL libraries. [Link](https://isaac-sim.github.io/IsaacLab/main/source/how-to/wrap_rl_env.html) |
| isaaclab_training_rl_agent | Isaac Lab Developers | Training with an RL Agent | 🟢 | 🟡 | 🟡 | End-to-end RL training tutorial. [Link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html) |
| isaaclab_rl_api | Isaac Lab Developers | isaaclab_rl API | 🟢 | 🔴 | 🟡 | RL wrapper API reference. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html) |
| isaaclab_skillgen | Isaac Lab Developers | SkillGen for Automated Demonstration Generation | 🔴 | 🟡 | 🟡 | Automated demo generation. Potential future use for imitation learning. [Link](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html) |
| isaaclab_import_new_asset | Isaac Lab Developers | Importing a New Asset | 🟢 | 🟡 | 🟡 | URDF/MJCF to USD import workflow. [Link](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html) |
| isaacsim_urdf_importer | NVIDIA | URDF Importer Extension | 🟢 | 🟡 | 🟡 | URDF to USD conversion tool. [Link](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html) |
| isaacsim_import_urdf_510 | NVIDIA | Tutorial: Import URDF (5.1.0) | 🟢 | 🟡 | 🟡 | Step-by-step URDF import for Isaac Sim 5.1.0. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html) |
| isaacsim_robot_assembler_510 | NVIDIA | Robot Assembler (5.1.0) | 🟢 | 🟡 | 🟡 | Modular robot assembly from USD components. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/assemble_robots.html) |
| isaacsim_cloner_tutorial | NVIDIA | Getting Started with Cloner | 🟢 | 🟡 | 🟡 | Parallel environment cloning for GPU-accelerated RL. [Link](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/isaac_lab_tutorials/tutorial_cloner.html) |
| nvidia_isaaclab_mdp_managers | NVIDIA | Manager Configuration — Getting Started | 🟢 | 🟡 | 🟡 | MDP manager setup tutorial. [Link](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-second-robot-with-isaac-lab/03-implementing-mdp-with-managers.html) |
| isaaclab_sim_doc | NVIDIA | isaaclab.sim — Simulation Config & GPU Physics | 🟢 | 🟡 | 🟡 | Simulation parameter configuration API. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html) |
| isaaclab_managers_api | Isaac Lab Developers | isaaclab.managers API | 🟢 | 🔴 | 🟡 | Manager classes API reference. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html) |
| isaaclab_envs_api | Isaac Lab Developers | isaaclab.envs API | 🟢 | 🔴 | 🟡 | Environment base classes API. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html) |
| isaacsim_mjcf_tendon_api | NVIDIA | MJCFTendon Python API | 🟢 | 🟡 | 🟡 | MJCF tendon import API for Isaac Sim. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/classisaacsim_1_1asset_1_1importer_1_1mjcf_1_1_m_j_c_f_tendon.html) |

---

## PhysX & Omniverse Physics

### Documentation (Online)

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| physx_541_gpu_rigid_bodies | NVIDIA | GPU Simulation (PhysX 5.4.1) | 🟢 | 🟢 | 🟡 | GPU-accelerated rigid body simulation details. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html) |
| physx_articulations_550 | NVIDIA | Articulations (PhysX 5.5.0) — incl. Articulation Tendons | 🟢 | 🟢 | 🟡 | PhysX articulation tendons (fixed/spatial). Core reference for tendon simulation in PhysX. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html) |
| physx_articulations_513 | NVIDIA | Articulations (PhysX 5.1.3) | 🟢 | 🟡 | 🟡 | Earlier articulation docs for version compatibility reference. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/Articulations.html) |
| physx_rigid_body_dynamics_511 | NVIDIA | Rigid Body Dynamics (PhysX 5.1.1) | 🟡 | 🟡 | 🟡 | Rigid body dynamics solver details. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/docs/RigidBodyDynamics.html) |
| physx_rigid_body_collision_510 | NVIDIA | Rigid Body Collision (PhysX 5.1.0) | 🟡 | 🟡 | 🟡 | Collision detection and response. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/RigidBodyCollision.html) |
| physx_simulation_541 | NVIDIA | Simulation (PhysX 5.4.1) | 🟡 | 🟡 | 🟡 | General simulation pipeline documentation. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Simulation.html) |
| physx_best_practices_541 | NVIDIA | Best Practices Guide (PhysX 5.4.1) | 🟡 | 🟡 | 🟡 | Performance and stability tuning. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/BestPractices.html) |
| physx_geometry_541 | NVIDIA | Geometry (PhysX 5.4.1) | 🟡 | 🟡 | 🟡 | Collision geometry types and usage. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html) |
| physx_material_api_510 | NVIDIA | PxMaterial API Reference (PhysX 5.1.0) | 🟡 | 🔴 | 🟡 | Material property API (friction, restitution). [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/class_px_material.html) |
| physx_advanced_collision_detection_512 | NVIDIA | Advanced Collision Detection (PhysX 5.1.2) | 🔴 | 🟡 | 🟡 | CCD and advanced contact generation. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html) |
| physx_pbd_particlesystem_510 | NVIDIA | Particle System (PhysX 5.1.0) | 🟢 | 🟢 | 🟡 | PBD-based particle and cloth simulation in PhysX. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/ParticleSystem.html) |
| physx_particlesystem_513 | NVIDIA | Particle System (PhysX 5.1.3) | 🟢 | 🟡 | 🟡 | Updated particle system docs. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/ParticleSystem.html) |
| physx_pbd_particlesystem_api | NVIDIA | PxPBDParticleSystem API Reference | 🟢 | 🔴 | 🟡 | API for PBD particle cloth. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/class_px_p_b_d_particle_system.html) |
| omni_physics_articulations_tendons | NVIDIA | Articulations — Tendons and Mimic Joint Compliance | 🟢 | 🟡 | 🟡 | Omniverse-level tendon authoring on articulations. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html) |
| omni_physics_tendon_authoring | NVIDIA | Tendon Visual Authoring | 🟢 | 🟡 | 🟡 | GUI-based tendon authoring in Omniverse. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.usdphysics.ui/docs/dev_guide/tendon_authoring.html) |
| omni_physics_particles | NVIDIA | Particles (Omni Physics) | 🟢 | 🟡 | 🟡 | Particle-based cloth and deformable bodies in Omniverse. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html) |
| omni_physx_overview | NVIDIA | Omni PhysX — Overview | 🟡 | 🟡 | 🟡 | High-level Omni PhysX extension overview. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/index.html) |
| omni_physics_physx_joint_schema | NVIDIA | PhysX Joint Schema | 🟡 | 🟡 | 🟡 | Joint types and configuration in Omniverse. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/joints/physx_joint_schema.html) |
| omni_physics_current_limitations | NVIDIA | Known Physics Limitations | 🟡 | 🟢 | 🟡 | Critical for documenting simulation constraints and workarounds. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/current_limitations.html) |
| omni_deformable_migration_guide | NVIDIA | Deformable Body Migration Guide | 🟢 | 🟡 | 🟡 | Migration path for deformable body simulation. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformable_bodies/deformable_migration.html) |
| omniverse_rtx_renderer_doc | NVIDIA | Omniverse RTX Renderer | 🔴 | 🟡 | 🟡 | RTX rendering pipeline in Omniverse. [Link](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html) |
| isaac_sim_physics_resources_510 | NVIDIA | Physics and PhysX SDK Limitations (5.1.0) | 🟡 | 🟢 | 🟡 | Known physics limitations for version 5.1.0. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/physics_resources.html) |
| physx_welcome_513 | NVIDIA | Welcome to PhysX (5.1.3) | 🔴 | 🔴 | 🟡 | General PhysX introduction. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/index.html) |

---

## Cloth Simulation Methods (Physics)

### Seminal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| BaraffWitkin1998 | Baraff, D. & Witkin, A. | "Large Steps in Cloth Simulation" | 1998 | 🔴 | 🟢 | 🟢 | Implicit integration for cloth simulation enabling large time steps. Foundational reference for understanding modern cloth solvers. [DOI](https://doi.org/10.1145/280814.280821) |
| muller2007pbd | Müller, M. et al. | "Position Based Dynamics" | 2007 | 🟢 | 🟢 | 🟢 | Introduces PBD, the method underlying PhysX particle cloth simulation used in this project. [DOI](https://doi.org/10.1016/j.jvcir.2007.01.005) |
| macklin2016xpbd | Macklin, M. et al. | "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" | 2016 | 🟢 | 🟢 | 🟢 | Extended PBD with physically meaningful compliance. Directly used in PhysX cloth. [DOI](https://doi.org/10.1145/2994258.2994272) |
| Bridson2002 | Bridson, R. et al. | "Robust Treatment of Collisions, Contact and Friction for Cloth Animation" | 2002 | 🔴 | 🟡 | 🟢 | Robust collision handling for cloth. Background for understanding contact challenges. [DOI](https://doi.org/10.1145/566570.566623) |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Nealen2006 | Nealen, A. et al. | "Physically Based Deformable Models in Computer Graphics" | 2006 | 🔴 | 🟢 | 🟢 | Comprehensive survey of deformable model methods (MSS, FEM, PBD). Provides taxonomy for the thesis background chapter. [DOI](https://doi.org/10.1111/j.1467-8659.2006.01000.x) |
| Jiang2016MPM | Jiang, C. et al. | "The Material Point Method for Simulating Continuum Materials" | 2016 | 🔴 | 🟡 | 🟢 | MPM tutorial for continuum simulation. Background alternative to PBD. [DOI](https://doi.org/10.1145/2897826.2927348) |
| chen2024vbd | Chen, A. H. et al. | "Vertex Block Descent" | 2024 | 🟡 | 🟢 | 🟢 | Novel GPU solver for deformable simulation. Relevant as the method behind Newton physics engine. [DOI](https://doi.org/10.1145/3658179) |
| FratarcangeliPellacini2013 | Fratarcangeli, M. & Pellacini, F. | "A GPU-Based Implementation of Position Based Dynamics for Interactive Deformable Bodies" | 2013 | 🟡 | 🟡 | 🟢 | GPU-parallel PBD implementation. Context for understanding PhysX GPU cloth. [DOI](https://doi.org/10.1080/2165347X.2015.1030525) |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| SifakisBarbic2012 | Sifakis, E. & Barbič, J. | "FEM Simulation of 3D Deformable Solids: A Practitioner's Guide" | 2012 | 🔴 | 🟡 | 🟢 | Practical FEM guide for deformable solids. Context for alternative simulation approaches. [DOI](https://doi.org/10.1145/2343483.2343501) |
| Macklin2019SmallSteps | Macklin, M. et al. | "Small Steps in Physics Simulation" | 2019 | 🟡 | 🟡 | 🟢 | Sub-stepping strategy for PBD stability. Relevant to PhysX solver configuration. [Link](https://mmacklin.com/smallsteps.pdf) |

---

## Newton GPU Solvers

### Documentation & Blog Posts

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| newton_solvers_doc | NVIDIA | Newton GPU Solvers Documentation | 🟢 | 🟡 | 🟡 | Next-generation GPU physics engine for robotics. [Link](https://docs.omniverse.nvidia.com/kit/docs/newton/latest/index.html) |
| newton_physics_doc | Newton Contributors | Newton Physics Documentation | 🟢 | 🟡 | 🟡 | Open-source Newton physics documentation. [Link](https://newton-physics.github.io/newton/) |
| nvidia_newton_announcement_2025 | NVIDIA | Announcing Newton Physics Engine | 🔴 | 🟡 | 🟡 | Newton announcement with VBD cloth simulation. [Link](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/) |
| isaaclab_newton_blog_2025 | NVIDIA | Newton GPU Physics for Robotics — Isaac Lab Blog | 🟡 | 🟡 | 🟡 | Newton integration with Isaac Lab. [Link](https://developer.nvidia.com/blog/advancing-robot-learning-with-newton-the-next-generation-gpu-accelerated-physics-engine/) |
| nvidia_isaaclab_newton_blog_2025 | NVIDIA | Train Quadruped & Simulate Cloth with Isaac Lab and Newton | 🟢 | 🟡 | 🟡 | Practical cloth simulation with Newton in Isaac Lab. [Link](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/) |

---

## MuJoCo

### Documentation

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| MuJoCoOverviewDoc | DeepMind | MuJoCo Documentation: Overview | 🟡 | 🟢 | 🟡 | MuJoCo overview including tendon geometry and coupling. [Link](https://mujoco.readthedocs.io/en/2.3.7/overview.html) |
| MuJoCoModelingDoc | DeepMind | MuJoCo Documentation: Modeling | 🟡 | 🟢 | 🟡 | MJCF design and URDF vs. MJCF scope. Important for format conversion discussion. [Link](https://mujoco.readthedocs.io/en/stable/modeling.html) |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Todorov2012 | Todorov, E. et al. | "MuJoCo: A Physics Engine for Model-Based Control" | 2012 | 🔴 | 🟢 | 🟢 | Introduces MuJoCo with native tendon support. Key comparator for simulator selection discussion. [DOI](https://doi.org/10.1109/IROS.2012.6386109) |

---

## PyBullet / Bullet

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Coumans2015 | Coumans, E. | "Bullet Physics Simulation" (SIGGRAPH Course) | 2015 | 🔴 | 🟡 | 🟢 | Bullet physics engine overview. Background for simulator comparison. [DOI](https://doi.org/10.1145/2776880.2792704) |
| pybullet_homepage | PyBullet contributors | Bullet Real-Time Physics Simulation: PyBullet | — | 🔴 | 🟡 | 🟡 | PyBullet project website. [Link](https://pybullet.org/) |

---

## SOFA

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Allard2007 | Allard, J. et al. | "SOFA — an open source framework for medical simulation" | 2007 | 🔴 | 🟡 | 🟢 | Open-source medical simulation framework. Alternative simulator for deformable objects. [PMID: 17377224](https://pubmed.ncbi.nlm.nih.gov/17377224/) |

---

## OpenUSD & Scene Description

### Documentation

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| openusd_usd_faq | Pixar | USD Frequently Asked Questions | 🟡 | 🟡 | 🟡 | USD format overview and design rationale. [Link](https://openusd.org/release/usdfaq.html) |
| openusd_usdphysics_schema | Pixar | UsdPhysics: USD Physics Schema | 🟢 | 🟡 | 🟡 | Physics schema for USD scenes. [Link](https://openusd.org/docs/api/usd_physics_page_front.html) |
| openusd_usdlux_schema | Pixar | UsdLux: USD Lighting Schema | 🔴 | 🔴 | 🟡 | Lighting schema for USD. [Link](https://openusd.org/docs/api/usd_lux_page_front.html) |

---

## Robot Description Formats (URDF / MJCF / ROS 2)

### Documentation

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| ros2_urdf_main | Open Robotics | URDF Main (ROS 2 Rolling) | 🟢 | 🟡 | 🟡 | URDF format specification. [Link](https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/URDF-Main.html) |
| ros2_urdf_physical_properties | Open Robotics | Adding Physical and Collision Properties to URDF | 🟢 | 🟡 | 🟡 | URDF physical property specification. [Link](https://docs.ros.org/en/foxy/Tutorials/Intermediate/URDF/Adding-Physical-and-Collision-Properties-to-a-URDF-Model.html) |
| ros2_control_joint_kinematics | Open Robotics | Joint Kinematics for ros2_control | 🟡 | 🟡 | 🟡 | Mimic joints and serial-chain URDF kinematics limitations. [Link](https://docs.ros.org/en/rolling/p/hardware_interface/doc/joints_userdoc.html) |

---

## Motion Planning

### Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Sundaralingam2023 | Sundaralingam, B. et al. | "cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation" | 2023 | 🟡 | 🟡 | 🟢 | GPU-accelerated motion planning. Potential integration for path planning. [arXiv](https://arxiv.org/abs/2310.17274) |

### Documentation

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| Robotiq2F140Datasheet | Robotiq Inc. | 2F-85 and 2F-140 Adaptive Gripper — Specifications | 🟢 | 🟡 | 🟡 | Gripper specifications for the Robotiq 2F-140 used in the simulation. [Link](https://robotiq.com/products/adaptive-grippers) |

# Simulation & Physics Engines

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## NVIDIA Isaac Sim / Isaac Lab

### Journal Articles & Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Mittal2023Orbit](sources/simulation/Papers/Mittal2023Orbit.pdf) | Mittal, M. et al. | "Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments" | 2023 | 🟢 | 🟢 | 🟢 | Introduces the Orbit/Isaac Lab simulation stack on NVIDIA Omniverse. Foundation for the entire RL training pipeline in this project. [DOI](https://doi.org/10.1109/LRA.2023.3270034) |
| [Mittal2025IsaacLab](sources/simulation/Papers/Mittal2025IsaacLab.pdf) | NVIDIA & Mittal, M. et al. | "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning" | 2025 | 🟢 | 🟢 | 🟢 | Describes Isaac Lab's architecture, manager-based environments, and GPU-accelerated training. Core framework reference. [arXiv](https://arxiv.org/abs/2511.04831) |
| [Makoviychuk2021](sources/simulation/Papers/Makoviychuk2021.pdf) | Makoviychuk, V. et al. | "Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning" | 2021 | 🟡 | 🟢 | 🟢 | Predecessor to Isaac Lab; demonstrates GPU-based parallel simulation for RL. Context for the evolution of the simulation platform. [OpenReview](https://openreview.net/forum?id=fgFBtYgJQX_) |

### Documentation (Online)

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| [isaac_sim_510_overview](sources/simulation/Documentation/isaac_sim_510_overview.html) | NVIDIA | What Is Isaac Sim? | 🟢 | 🟡 | 🟡 | Top-level overview of Isaac Sim 5.1.0 capabilities. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) |
| [isaac_sim_510_release_notes](sources/simulation/Documentation/isaac_sim_510_release_notes.html) | NVIDIA | Release Notes (5.1.0) | 🟡 | 🔴 | 🟡 | Version-specific feature list and known issues. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/release_notes.html) |
| [isaac_sim_510_physics_fundamentals](sources/simulation/Documentation/isaac_sim_510_physics_fundamentals.html) | NVIDIA | Physics Simulation Fundamentals | 🟢 | 🟢 | 🟡 | Explains PhysX integration, time stepping, and solver configuration in Isaac Sim. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html) |
| [isaac_sim_ros2_reference_architecture](sources/simulation/Documentation/isaac_sim_ros2_reference_architecture.html) | NVIDIA | ROS 2 Reference Architecture | 🟡 | 🟡 | 🟡 | Integration architecture for ROS 2 with Isaac Sim. Relevant for future sim-to-real pathway. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/reference_architecture.html) |
| [isaaclab_overview_doc](sources/simulation/Documentation/isaaclab_overview_doc.html) | Isaac Lab Developers | Overview | 🟢 | 🟡 | 🟡 | Isaac Lab high-level architecture and API overview. [Link](https://isaac-sim.github.io/IsaacLab/v1.0.0/index.html) |
| [isaaclab_mdp_api](sources/simulation/Documentation/isaaclab_mdp_api.html) | Isaac Lab Developers | isaaclab.envs.mdp API | 🟢 | 🟡 | 🟡 | MDP manager API for observations, actions, rewards, terminations. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html) |
| [isaaclab_multi_gpu_doc](sources/simulation/Documentation/isaaclab_multi_gpu_doc.html) | Isaac Lab Developers | Multi-GPU and Multi-Node Training | 🟡 | 🟡 | 🟡 | Documentation for distributed training setup. [Link](https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html) |
| [isaaclab_manager_based_rl_env](sources/simulation/Documentation/isaaclab_manager_based_rl_env.html) | Isaac Lab Developers | Creating a Manager-Based RL Environment | 🟢 | 🟡 | 🟡 | Tutorial for the environment design pattern used in this project. [Link](https://isaac-sim.github.io/IsaacLab/v1.0.0/source/tutorials/03_envs/create_manager_rl_env.html) |
| [isaaclab_task_design_workflows](sources/simulation/Documentation/isaaclab_task_design_workflows.html) | Isaac Lab Developers | Task Design Workflows | 🟢 | 🟡 | 🟡 | Manager-based vs direct workflow comparison. [Link](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html) |
| [isaaclab_wrap_env](sources/simulation/Documentation/isaaclab_wrap_env.html) | Isaac Lab Developers | Wrapping Environments | 🟢 | 🟡 | 🟡 | How to wrap Isaac Lab envs for external RL libraries. [Link](https://isaac-sim.github.io/IsaacLab/main/source/how-to/wrap_rl_env.html) |
| [isaaclab_training_rl_agent](sources/simulation/Documentation/isaaclab_training_rl_agent.html) | Isaac Lab Developers | Training with an RL Agent | 🟢 | 🟡 | 🟡 | End-to-end RL training tutorial. [Link](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html) |
| [isaaclab_rl_api](sources/simulation/Documentation/isaaclab_rl_api.html) | Isaac Lab Developers | isaaclab_rl API | 🟢 | 🔴 | 🟡 | RL wrapper API reference. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html) |
| [isaaclab_import_new_asset](sources/simulation/Documentation/isaaclab_import_new_asset.html) | Isaac Lab Developers | Importing a New Asset | 🟢 | 🟡 | 🟡 | URDF/MJCF to USD import workflow. [Link](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html) |
| [isaacsim_urdf_importer](sources/simulation/Documentation/isaacsim_urdf_importer.html) | NVIDIA | URDF Importer Extension | 🟢 | 🟡 | 🟡 | URDF to USD conversion tool. [Link](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/ext_isaacsim_asset_importer_urdf.html) |
| [isaacsim_import_urdf_510](sources/simulation/Documentation/isaacsim_import_urdf_510.html) | NVIDIA | Tutorial: Import URDF (5.1.0) | 🟢 | 🟡 | 🟡 | Step-by-step URDF import for Isaac Sim 5.1.0. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html) |
| [isaacsim_robot_assembler_510](sources/simulation/Documentation/isaacsim_robot_assembler_510.html) | NVIDIA | Robot Assembler (5.1.0) | 🟢 | 🟡 | 🟡 | Modular robot assembly from USD components. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/assemble_robots.html) |
| [isaacsim_cloner_tutorial](sources/simulation/Documentation/isaacsim_cloner_tutorial.html) | NVIDIA | Getting Started with Cloner | 🟢 | 🟡 | 🟡 | Parallel environment cloning for GPU-accelerated RL. [Link](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/isaac_lab_tutorials/tutorial_cloner.html) |
| [nvidia_isaaclab_mdp_managers](sources/simulation/Documentation/nvidia_isaaclab_mdp_managers.html) | NVIDIA | Manager Configuration — Getting Started | 🟢 | 🟡 | 🟡 | MDP manager setup tutorial. [Link](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-second-robot-with-isaac-lab/03-implementing-mdp-with-managers.html) |
| [isaaclab_sim_doc](sources/simulation/Documentation/isaaclab_sim_doc.html) | NVIDIA | isaaclab.sim — Simulation Config & GPU Physics | 🟢 | 🟡 | 🟡 | Simulation parameter configuration API. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html) |
| [isaaclab_managers_api](sources/simulation/Documentation/isaaclab_managers_api.html) | Isaac Lab Developers | isaaclab.managers API | 🟢 | 🔴 | 🟡 | Manager classes API reference. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html) |
| [isaaclab_envs_api](sources/simulation/Documentation/isaaclab_envs_api.html) | Isaac Lab Developers | isaaclab.envs API | 🟢 | 🔴 | 🟡 | Environment base classes API. [Link](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html) |
| [isaacsim_mjcf_tendon_api](sources/simulation/Documentation/isaacsim_mjcf_tendon_api.html) | NVIDIA | MJCFTendon Python API | 🟢 | 🟡 | 🟡 | MJCF tendon import API for Isaac Sim. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/classisaacsim_1_1asset_1_1importer_1_1mjcf_1_1_m_j_c_f_tendon.html) |

---

## PhysX & Omniverse Physics

### Documentation (Online)

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| [physx_541_gpu_rigid_bodies](sources/simulation/Documentation/physx_541_gpu_rigid_bodies.html) | NVIDIA | GPU Simulation (PhysX 5.4.1) | 🟢 | 🟢 | 🟡 | GPU-accelerated rigid body simulation details. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html) |
| [physx_articulations_550](sources/simulation/Documentation/physx_articulations_550.html) | NVIDIA | Articulations (PhysX 5.5.0) — incl. Articulation Tendons | 🟢 | 🟢 | 🟡 | PhysX articulation tendons (fixed/spatial). Core reference for tendon simulation in PhysX. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/Articulations.html) |
| [physx_articulations_513](sources/simulation/Documentation/physx_articulations_513.html) | NVIDIA | Articulations (PhysX 5.1.3) | 🟢 | 🟡 | 🟡 | Earlier articulation docs for version compatibility reference. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/Articulations.html) |
| [physx_rigid_body_dynamics_511](sources/simulation/Documentation/physx_rigid_body_dynamics_511.html) | NVIDIA | Rigid Body Dynamics (PhysX 5.1.1) | 🟡 | 🟡 | 🟡 | Rigid body dynamics solver details. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.1/docs/RigidBodyDynamics.html) |
| [physx_rigid_body_collision_510](sources/simulation/Documentation/physx_rigid_body_collision_510.html) | NVIDIA | Rigid Body Collision (PhysX 5.1.0) | 🟡 | 🟡 | 🟡 | Collision detection and response. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/RigidBodyCollision.html) |
| [physx_simulation_541](sources/simulation/Documentation/physx_simulation_541.html) | NVIDIA | Simulation (PhysX 5.4.1) | 🟡 | 🟡 | 🟡 | General simulation pipeline documentation. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Simulation.html) |
| [physx_best_practices_541](sources/simulation/Documentation/physx_best_practices_541.html) | NVIDIA | Best Practices Guide (PhysX 5.4.1) | 🟡 | 🟡 | 🟡 | Performance and stability tuning. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/BestPractices.html) |
| [physx_geometry_541](sources/simulation/Documentation/physx_geometry_541.html) | NVIDIA | Geometry (PhysX 5.4.1) | 🟡 | 🟡 | 🟡 | Collision geometry types and usage. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html) |
| [physx_material_api_510](sources/simulation/Documentation/physx_material_api_510.html) | NVIDIA | PxMaterial API Reference (PhysX 5.1.0) | 🟡 | 🔴 | 🟡 | Material property API (friction, restitution). [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/class_px_material.html) |
| [physx_advanced_collision_detection_512](sources/simulation/Documentation/physx_advanced_collision_detection_512.html) | NVIDIA | Advanced Collision Detection (PhysX 5.1.2) | 🔴 | 🟡 | 🟡 | CCD and advanced contact generation. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html) |
| [physx_pbd_particlesystem_api](sources/simulation/Documentation/physx_pbd_particlesystem_api.html) | NVIDIA | PxPBDParticleSystem API Reference | 🟢 | 🔴 | 🟡 | API for PBD particle cloth. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/_api_build/class_px_p_b_d_particle_system.html) |
| [omni_physics_articulations_tendons](sources/simulation/Documentation/omni_physics_articulations_tendons.html) | NVIDIA | Articulations — Tendons and Mimic Joint Compliance | 🟢 | 🟡 | 🟡 | Omniverse-level tendon authoring on articulations. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html) |
| [omni_physics_tendon_authoring](sources/simulation/Documentation/omni_physics_tendon_authoring.html) | NVIDIA | Tendon Visual Authoring | 🟢 | 🟡 | 🟡 | GUI-based tendon authoring in Omniverse. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.usdphysics.ui/docs/dev_guide/tendon_authoring.html) |
| [omni_physics_particles](sources/simulation/Documentation/omni_physics_particles.html) | NVIDIA | Particles (Omni Physics) | 🟢 | 🟡 | 🟡 | Particle-based cloth and deformable bodies in Omniverse. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/particles/particles.html) |
| [omni_physx_overview](sources/simulation/Documentation/omni_physx_overview.html) | NVIDIA | Omni PhysX — Overview | 🟡 | 🟡 | 🟡 | High-level Omni PhysX extension overview. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/index.html) |
| [omni_physics_physx_joint_schema](sources/simulation/Documentation/omni_physics_physx_joint_schema.html) | NVIDIA | PhysX Joint Schema | 🟡 | 🟡 | 🟡 | Joint types and configuration in Omniverse. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/joints/physx_joint_schema.html) |
| [omni_physics_current_limitations](sources/simulation/Documentation/omni_physics_current_limitations.html) | NVIDIA | Known Physics Limitations | 🟡 | 🟢 | 🟡 | Critical for documenting simulation constraints and workarounds. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/current_limitations.html) |
| [omni_deformable_migration_guide](sources/simulation/Documentation/omni_deformable_migration_guide.html) | NVIDIA | Deformable Body Migration Guide | 🟢 | 🟡 | 🟡 | Migration path for deformable body simulation. [Link](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformable_bodies/deformable_migration.html) |
| [isaac_sim_physics_resources_510](sources/simulation/Documentation/isaac_sim_physics_resources_510.html) | NVIDIA | Physics and PhysX SDK Limitations (5.1.0) | 🟡 | 🟢 | 🟡 | Known physics limitations for version 5.1.0. [Link](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/physics_resources.html) |
| [physx_welcome_513](sources/simulation/Documentation/physx_welcome_513.html) | NVIDIA | Welcome to PhysX (5.1.3) | 🔴 | 🔴 | 🟡 | General PhysX introduction. [Link](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/index.html) |

---

## Cloth Simulation Methods (Physics)

### Seminal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [muller2007pbd](sources/simulation/Papers/Muller2007.pdf) | Müller, M. et al. | "Position Based Dynamics" | 2007 | 🟢 | 🟢 | 🟢 | Introduces PBD, the method underlying PhysX particle cloth simulation used in this project. [DOI](https://doi.org/10.1016/j.jvcir.2007.01.005) |
| [macklin2016xpbd](sources/simulation/Papers/Macklin2016XPBD.pdf) | Macklin, M. et al. | "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" | 2016 | 🟢 | 🟢 | 🟢 | Extended PBD with physically meaningful compliance. Directly used in PhysX cloth. [DOI](https://doi.org/10.1145/2994258.2994272) |
| [Bridson2002](sources/simulation/Papers/Bridson2002.pdf) | Bridson, R. et al. | "Robust Treatment of Collisions, Contact and Friction for Cloth Animation" | 2002 | 🔴 | 🟡 | 🟢 | Robust collision handling for cloth. Background for understanding contact challenges. [DOI](https://doi.org/10.1145/566570.566623) |

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Nealen2006](sources/simulation/Papers/Nealen2006.pdf) | Nealen, A. et al. | "Physically Based Deformable Models in Computer Graphics" | 2006 | 🔴 | 🟢 | 🟢 | Comprehensive survey of deformable model methods (MSS, FEM, PBD). Provides taxonomy for the thesis background chapter. [DOI](https://doi.org/10.1111/j.1467-8659.2006.01000.x) |
| [Jiang2016MPM](sources/simulation/Papers/Jiang2016MPM.pdf) | Jiang, C. et al. | "The Material Point Method for Simulating Continuum Materials" | 2016 | 🔴 | 🟡 | 🟢 | MPM tutorial for continuum simulation. Background alternative to PBD. [DOI](https://doi.org/10.1145/2897826.2927348) |
| [chen2024vbd](sources/simulation/Papers/Chen2024VBD.pdf) | Chen, A. H. et al. | "Vertex Block Descent" | 2024 | 🟡 | 🟢 | 🟢 | Novel GPU solver for deformable simulation. Relevant as the method behind Newton physics engine. [DOI](https://doi.org/10.1145/3658179) |
| [FratarcangeliPellacini2013](sources/simulation/Papers/FratarcangeliPellacini2013.pdf) | Fratarcangeli, M. & Pellacini, F. | "A GPU-Based Implementation of Position Based Dynamics for Interactive Deformable Bodies" | 2013 | 🟡 | 🟡 | 🟢 | GPU-parallel PBD implementation. Context for understanding PhysX GPU cloth. [DOI](https://doi.org/10.1080/2165347X.2015.1030525) |

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [SifakisBarbic2012](sources/simulation/Papers/SifakisBarbic2012.pdf) | Sifakis, E. & Barbič, J. | "FEM Simulation of 3D Deformable Solids: A Practitioner's Guide" | 2012 | 🔴 | 🟡 | 🟢 | Practical FEM guide for deformable solids. Context for alternative simulation approaches. [DOI](https://doi.org/10.1145/2343483.2343501) |
| [Macklin2019SmallSteps](sources/simulation/Papers/Macklin2019SmallSteps.pdf) | Macklin, M. et al. | "Small Steps in Physics Simulation" | 2019 | 🟡 | 🟡 | 🟢 | Sub-stepping strategy for PBD stability. Relevant to PhysX solver configuration. [Link](https://mmacklin.com/smallsteps.pdf) |

---

## Newton GPU Solvers

### Documentation & Blog Posts

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| [newton_solvers_doc](sources/simulation/Documentation/newton_solvers_doc.html) | NVIDIA | Newton GPU Solvers Documentation | 🟢 | 🟡 | 🟡 | Next-generation GPU physics engine for robotics. [Link](https://docs.omniverse.nvidia.com/kit/docs/newton/latest/index.html) |
| [newton_physics_doc](sources/simulation/Documentation/newton_physics_doc.html) | Newton Contributors | Newton Physics Documentation | 🟢 | 🟡 | 🟡 | Open-source Newton physics documentation. [Link](https://newton-physics.github.io/newton/) |
| [nvidia_newton_announcement_2025](sources/simulation/Documentation/nvidia_newton_announcement_2025.html) | NVIDIA | Announcing Newton Physics Engine | 🔴 | 🟡 | 🟡 | Newton announcement with VBD cloth simulation. [Link](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/) |
| [nvidia_isaaclab_newton_blog_2025](sources/simulation/Documentation/nvidia_isaaclab_newton_blog_2025.html) | NVIDIA | Train Quadruped & Simulate Cloth with Isaac Lab and Newton | 🟢 | 🟡 | 🟡 | Practical cloth simulation with Newton in Isaac Lab. [Link](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/) |

---

## MuJoCo

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Todorov2012](sources/simulation/Papers/Todorov2012.pdf) | Todorov, E. et al. | "MuJoCo: A Physics Engine for Model-Based Control" | 2012 | 🔴 | 🟢 | 🟢 | Introduces MuJoCo with native tendon support. Key comparator for simulator selection discussion. [DOI](https://doi.org/10.1109/IROS.2012.6386109) |

---

## PyBullet / Bullet

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [pybullet_homepage](sources/simulation/Documentation/pybullet_org.html) | Bullet Physics Project | Bullet Real-Time Physics Simulation: Home of Bullet and PyBullet | 2026 | 🔴 | 🟡 | 🟡 | PyBullet/Bullet project homepage. Background for simulator-comparison discussion. [Link](https://pybullet.org/wordpress/) |

---

## SOFA

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|

---

## OpenUSD & Scene Description

### Documentation

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| [openusd_usd_faq](sources/simulation/Documentation/openusd_usd_faq.html) | Pixar | USD Frequently Asked Questions | 🟡 | 🟡 | 🟡 | USD format overview and design rationale. [Link](https://openusd.org/release/usdfaq.html) |
| [openusd_usdphysics_schema](sources/simulation/Documentation/openusd_usdphysics_schema.html) | Pixar | UsdPhysics: USD Physics Schema | 🟢 | 🟡 | 🟡 | Physics schema for USD scenes. [Link](https://openusd.org/docs/api/usd_physics_page_front.html) |

---

## Robot Description Formats (URDF / MJCF / ROS 2)

### Documentation

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| [ros2_urdf_main](sources/simulation/Documentation/ros2_urdf_main.html) | Open Robotics | URDF Main (ROS 2 Rolling) | 🟢 | 🟡 | 🟡 | URDF format specification. [Link](https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/URDF-Main.html) |
| [ros2_urdf_physical_properties](sources/simulation/Documentation/ros2_urdf_physical_properties.html) | Open Robotics | Adding Physical and Collision Properties to URDF | 🟢 | 🟡 | 🟡 | URDF physical property specification. [Link](https://docs.ros.org/en/foxy/Tutorials/Intermediate/URDF/Adding-Physical-and-Collision-Properties-to-a-URDF-Model.html) |
| [ros2_control_joint_kinematics](sources/simulation/Documentation/ros2_control_joint_kinematics.html) | Open Robotics | Joint Kinematics for ros2_control | 🟡 | 🟡 | 🟡 | Mimic joints and serial-chain URDF kinematics limitations. [Link](https://docs.ros.org/en/rolling/p/hardware_interface/doc/joints_userdoc.html) |

---

## Motion Planning

### Documentation

| Ref | Source | Title | Impl. | Thesis | Cred. | Summary |
|-----|--------|-------|-------|--------|-------|---------|
| [Robotiq2F140Datasheet](sources/simulation/Documentation/Robotiq2F140Datasheet.pdf) | Robotiq Inc. | 2F-85 and 2F-140 Adaptive Gripper — Specifications | 🟢 | 🟡 | 🟡 | Gripper specifications for the Robotiq 2F-140 used in the simulation. [Link](https://robotiq.com/products/adaptive-grippers) |

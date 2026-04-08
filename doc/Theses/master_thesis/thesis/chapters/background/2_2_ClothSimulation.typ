// §2.2 Cloth and Deformable Object Simulation
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *

== Cloth and Deformable Object Simulation <sec:cloth_sim>

Deformable object simulation relaxes the rigid-body assumption by allowing the object's shape to change under load. This introduces two coupled complexities: the state dimension increases dramatically (many vertices/particles instead of a single rigid transform), and contacts become configuration-dependent because deformation continuously alters the set of potential contact points and normals~@Nealen2006DeformableModels @Longhini2025ReviewClothManipulation. For textiles, this coupling is particularly pronounced: applied forces not only translate the object but also create folds, wrinkles, and contact-rich self-interactions that alter future dynamics. These properties are repeatedly identified as key contributors to the reality gap in robotic cloth manipulation~@Longhini2025ReviewClothManipulation.

*Common modelling approaches.*
Three modelling families are most relevant in robotics simulation. First, mass--spring(-damper) models discretize cloth as a mesh whose vertices are connected by springs encoding stretch and (optionally) bending penalties; such models can be made stable with implicit integration techniques but require careful parameterization and can be numerically stiff~@BaraffWitkin1998LargeStepsCloth. Second, particle-based constraint methods represent cloth by particles and enforce desired behavior via geometric (holonomic) constraints, which can be solved iteratively with high robustness under collisions; #ac("PBD") and its variants are canonical examples~@muller2007pbd @macklin2016xpbd. Third, continuum formulations discretize elastic energy and constitutive laws (e.g., in #ac("FEM")) or use hybrid particle--grid methods (e.g., #ac("MPM")), providing a pathway to parameters with clearer physical interpretation at increased computational cost~@SifakisBarbic2012FEM @Jiang2016MPM.

*Challenges specific to cloth dynamics.*
Cloth exhibits high effective #ac("DoF"), frequent self-collision, and strong sensitivity to material parameters such as bending stiffness, friction, and damping. Robust self-contact and friction handling is widely recognized as a hard requirement for stable cloth simulation, particularly under large deformations and repeated collisions~@Bridson2002ClothCollisions @Longhini2025ReviewClothManipulation. From a learning perspective, this sensitivity implies that small modelling errors (e.g., in friction or damping) can change grasp success rates, sliding behavior, and fold formation, which in turn can qualitatively alter the learned policy's strategy and its transferability to hardware~@Salvato2021RealityGapSurvey @Longhini2025ReviewClothManipulation.

=== #acs("PBD")---PhysX Particle Cloth

PhysX provides particle-based simulation through `PxPBDParticleSystem`, whose solver is explicitly described as using #ac("PBD") for particle--particle dynamics and as supporting phenomena including cloth, inflatables, and other deformable behaviors~@physx_particle_system_513 @physx_pbd_particlesystem_api. In Omniverse, particle simulation is #ac("GPU")-accelerated and is documented as supporting fluids, granular media, and cloth, including interaction with rigid bodies and articulations~@omni_physics_particles. At the feature level, particle cloth is treated as a particle-based #ac("PBD") mass--spring-style system and is marked as deprecated in favor of newer deformable-body features, which highlights that "cloth" in Isaac~Sim can refer to multiple, evolving solver families rather than a single canonical model~@omni_physics_particles @isaac_sim_physics_resources_510.

*Time-step structure and iterative projection.*
In standard #ac("PBD"), the deformable is represented by particle positions $bold(x)_i in bb(R)^3$ with masses $m_i$; desired behavior is enforced by a set of constraints $C_j (bold(x))$ (equalities or inequalities), typically solved by iterative projection in a Gauss--Seidel fashion~@muller2007pbd. A schematic time step consists of: (i)~updating velocities under external forces, (ii)~predicting positions by an explicit step, (iii)~iteratively projecting predicted positions to reduce constraint violations, and (iv)~updating velocities from the corrected positions (and applying damping if configured)~@muller2007pbd. For a constraint $C_j (bold(x)) = 0$, linearization yields a correction direction based on the constraint gradient. Using the inverse mass matrix $bold(M)^(-1)$ (diagonal in the particle basis), one common update can be written as~@muller2007pbd @macklin2016xpbd:
$ Delta bold(x) = k_j dot s_j dot bold(M)^(-1) nabla C_j (bold(x)), quad s_j = frac(-C_j (bold(x)), nabla C_j (bold(x))^top bold(M)^(-1) nabla C_j (bold(x))), $ <eq:pbd_correction>
where $k_j in [0, 1]$ is a user-defined stiffness-like scaling applied per constraint.

*Effective stiffness and dependence on $h$ and iteration count.*
A central practical implication of the iterative projection view is that the _effective_ stiffness of constraints depends on numerical settings. Increasing the number of projection iterations typically yields stiffer constraint satisfaction at fixed $h$, while increasing $h$ tends to reduce stability and can require either more iterations or smaller substeps to maintain comparable behavior~@muller2007pbd @macklin2016xpbd @Macklin2019SmallSteps. This is beneficial for robustness and controllability, but it complicates mapping simulator parameters to physically measured material properties (e.g., bending stiffness in SI units), which is a recurring issue when calibrating cloth simulation for sim-to-real transfer~@macklin2016xpbd @Salvato2021RealityGapSurvey.

*#acs("XPBD") compliance formulation.*
#ac("XPBD") extends #ac("PBD") by introducing a compliance parameter and an explicit Lagrange multiplier state to reduce time-step dependence and recover meaningful constraint force estimates~@macklin2016xpbd. For compliance $alpha >= 0$ and time step $h$, the incremental multiplier update for a constraint can be expressed as~@macklin2016xpbd:
$ Delta lambda = frac(-C(bold(x)) - frac(alpha, h^2) lambda, sum_i w_i ||nabla_(bold(x)_i) C(bold(x))||^2 + frac(alpha, h^2)), quad Delta bold(x)_i = w_i nabla_(bold(x)_i) C(bold(x)) Delta lambda, $ <eq:xpbd_update>
with inverse masses $w_i = 1 / m_i$. This formulation decouples the user parameter $alpha$ from the iteration count in a way that improves interpretability of stiffness-like behavior across different $h$ and solver settings~@macklin2016xpbd.

*Why #ac("PBD") is used in PhysX for particle cloth.*
#ac("PBD") is well-aligned with #ac("GPU") execution because constraints can be partitioned into independent sets and processed in parallel; graph coloring is a common strategy to schedule constraint batches without write conflicts on shared particles~@FratarcangeliPellacini2013GPUPBD. Omniverse documentation explicitly emphasizes #ac("GPU")-accelerated #ac("PBD") particles and their interaction with rigid bodies and articulations, which is essential for cloth manipulation scenes in robotics~@omni_physics_particles.

=== #acs("VBD")---Newton Thin Deformables

While PhysX particle cloth is based on explicit prediction and iterative position projection, implicit methods aim to improve stability under stiffness and large time steps by solving a time-discrete optimization problem. #ac("VBD") is introduced as a block coordinate descent solver for the variational form of implicit Euler time integration, targeting convergence to the implicit Euler solution while maintaining unconditional stability under limited iteration budgets~@chen2024vbd.

*Variational implicit Euler and global energy minimization.*
Let $bold(x)_t$ denote positions, $bold(v)_t$ velocities, $h$ the time step, and $bold(M)$ the mass matrix. In the variational formulation, the next positions are obtained by minimizing a global energy that combines inertia and potential energy~@chen2024vbd:
$ bold(x)_(t+1) = arg min_(bold(x)) G(bold(x)), quad G(bold(x)) = frac(1, 2 h^2) ||bold(x) - bold(y)||_(bold(M))^2 + E(bold(x)), $ <eq:vbd_global_energy>
where $bold(y) = bold(x)_t + h bold(v)_t + h^2 bold(a)_"ext"$ and $E(bold(x))$ is the total potential energy. Velocities follow from the implicit Euler relationship $bold(v)_(t+1) = (bold(x)_(t+1) - bold(x)_t) / h$~@chen2024vbd.

*Vertex-level block coordinate descent and local Newton step.*
#ac("VBD") solves the global problem by iterating over vertices and updating one vertex at a time while temporarily holding the others fixed, yielding a vertex-level Gauss--Seidel pattern with favorable parallelization via vertex coloring~@chen2024vbd. For vertex $i$, a local energy is minimized:
$ bold(x)_i <- arg min_(bold(x)_i) G_i (bold(x)), quad G_i (bold(x)) = frac(m_i, 2 h^2) ||bold(x)_i - bold(y)_i||^2 + sum_(j in F_i) E_j (bold(x)), $ <eq:vbd_local_energy>
where $F_i$ is the set of force/energy elements incident to vertex $i$. Because $G_i$ has only three degrees of freedom, a local Newton step is formed by solving a $3 times 3$ linear system per vertex~@chen2024vbd:
$ bold(H)_i Delta bold(x)_i = bold(f)_i, $ <eq:vbd_newton_step>
with $bold(H)_i$ the local Hessian and $bold(f)_i$ the local gradient/force term.

*Advantages for stiff, contact-rich thin deformables.*
The primary advantages of #ac("VBD") are: (i)~unconditional stability inherited from the energy descent property, (ii)~convergence toward the implicit Euler solution as iterations increase, and (iii)~#ac("GPU")-oriented parallelization via vertex coloring with small, localized linear solves~@chen2024vbd. These properties are attractive for cloth-like thin deformables, which often exhibit near-inextensible stretch (high stiffness) combined with softer bending and frequent contact events~@chen2024vbd.

*Newton in the robotics simulation ecosystem.*
Newton is presented as an open-source, #ac("GPU")-accelerated physics engine designed for robotics simulation and robot learning, built on NVIDIA Warp and intended to integrate with robot learning frameworks such as IsaacLab~@newton_physics_doc @nvidia_newton_announcement_2025. In NVIDIA's robotics communication, Newton is explicitly associated with thin-deformable simulation via a #ac("VBD") solver and with multiphysics capabilities that include implicit #ac("MPM") and other solver components~@nvidia_isaaclab_newton_blog_2025 @nvidia_newton_announcement_2025.

=== Comparison: #acs("PBD") vs. #acs("VBD") for Cloth in #ac("RL") Workflows

*Mathematical targets and implications.*
The core distinction is the optimization target. #ac("PBD") advances dynamics by predicting positions and then _projecting_ them to reduce constraint violations, where the resulting behavior depends on solver scheduling and iteration count, and where constraint "stiffness" is typically not time-step invariant without extensions~@muller2007pbd @macklin2016xpbd. #ac("VBD"), in contrast, directly minimizes the variational implicit Euler energy and therefore targets a principled implicit integration objective; stability is maintained by construction via energy descent~@chen2024vbd.

*Practical tradeoffs for learning-centric simulation.*
For #ac("RL"), #ac("PBD")-based particle cloth is attractive because it is simple, robust under frequent contacts, and aligned with #ac("GPU")-parallel execution~@omni_physics_particles @FratarcangeliPellacini2013GPUPBD @Makoviychuk2021IsaacGym. However, because effective stiffness depends on $h$ and iteration count, parameter tuning can be difficult to interpret physically and can amplify sim-to-real sensitivity~@macklin2016xpbd @Salvato2021RealityGapSurvey. #ac("VBD") is attractive when stability under stiffness, large time steps, and mixed constraint types is prioritized~@chen2024vbd @nvidia_isaaclab_newton_blog_2025.

*Relation to the Omniverse ecosystem and thesis choice.*
In Omniverse/Isaac~Sim, particle cloth is documented as deprecated in favor of newer deformable-body features~@omni_physics_particles @isaac_sim_physics_resources_510. For the experiments in this thesis, the PhysX/#ac("PBD") particle cloth pathway is motivated by #ac("GPU") throughput and robustness for large-scale training in Isaac~Sim~5.1, whereas Newton/#ac("VBD") is conceptually aligned with scenarios where greater stability under stiff cloth behavior and larger time steps is required~@omni_physics_particles @chen2024vbd @nvidia_isaaclab_newton_blog_2025.

*Rationale for an integrated cloth--rigid simulation stack.*
Integrated cloth and deformable dynamics are supported within the Isaac Sim / PhysX stack through GPU-accelerated particle-based simulation that can represent cloth-like behavior and interact with articulations and other physics objects within a shared simulation scene~@omni_physics_particles. In addition, IsaacLab explicitly highlights support for deformable objects (including cloth) within a framework built on Isaac Sim and designed for multi-modal learning~@Mittal2025IsaacLab. This integration is beneficial because it avoids ad-hoc coupling between an external deformable simulator and a separate rigid-body engine, which would otherwise require custom synchronization, contact handling, and state transfer between solvers.

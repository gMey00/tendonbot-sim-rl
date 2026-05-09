// §2.6 Prior Work
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *

== Prior Work <sec:prior_work>

This section reviews prior work along four dimensions that converge in the present thesis: (i)~GPU-accelerated #ac("RL") for manipulation, (ii)~simulation and control of tendon- and cable-driven robots, (iii)~tensegrity-inspired serial arm manipulators, and (iv)~computational workspace analysis. The review focuses on contributions that directly inform the methodological choices made in this thesis and concludes by identifying the specific research gap that the present work addresses.

=== GPU-Accelerated #ac("RL") for Manipulation <subsec:prior_gpu_rl>

*Isaac Gym and large-scale parallel training.*
The transition from #ac("CPU") to GPU-accelerated simulation was catalyzed by Makoviychuk _et~al._~@Makoviychuk2021IsaacGym, who demonstrated that running both the PhysX step and neural-network forward passes entirely on the #ac("GPU") eliminates data transfer bottlenecks. In their benchmarking suite, 16#sym.space.thin{}384 parallel agents trained with #ac("PPO") (#ac("PPO"), see @sec:reinforcement_learning) converged in under 25~minutes on a single #ac("GPU"). This directly enabled the massive parallelization scale utilized in the current methodology (@sec:infrastructure), overcoming prior computational bottlenecks in #ac("RL"). 

*Contact-rich manipulation and assembly.*
Building upon parallelized simulation, Narang _et~al._ introduced _Factory_, an environment suite focused on contact-rich assembly tasks. By improving collision representations and constraint solvers, Factory reduced simulation artifacts in tight-tolerance tasks~@Narang2022Factory. Extending this, Tang _et~al._ (_IndustReal_) achieved successful sim-to-real transfer of peg-insertion policies with clearances of $lt.eq 0.5$--$0.6 "mm"$~@Tang2023IndustReal. _IndustReal_ specifically introduced a simulation-aware policy update mechanism and sampling-based curricula for progressive task difficulty. These curriculum strategies heavily inform the multi-stage reward and initialization design employed for the pick-and-place tasks formulated in this thesis (@sec:rl_tasks).

*Sim-to-real transfer for dexterous manipulation.*
For high-degree-of-freedom end-effectors, Allshire _et~al._ demonstrated transfer of a 6-#ac("DoF") manipulation policy to a physical TriFinger robot using #ac("DR") and a keypoint-based pose representation~@Allshire2022TriFinger. Handa _et~al._ (_DeXtreme_) further improved on this using automatic #ac("DR") to adapt manipulation policies dynamically during training without manual tuning~@Handa2023DeXtreme. Both of these approaches demonstrate the efficacy of asymmetric actor-critic architectures---where the critic receives privileged simulator state while the actor relies only on observable states. While the present methodology operates exclusively in simulation, these established sim-to-real techniques validate the choice of Isaac Sim as a foundation for future physical deployment.

*Frameworks: Orbit, Isaac Lab, and skrl.*
The environment management surrounding the #ac("GPU") pipeline has matured into standardized frameworks. Mittal _et~al._ presented _Orbit_ to wrap the Isaac Sim backend into modular environments conforming to common #ac("RL") #ac("API") standards~@Mittal2023Orbit. _Orbit_ evolved into _Isaac Lab_~@Mittal2025IsaacLab, achieving state-of-the-art throughput and introducing closed-loop kinematic chain support, which is critical for modeling the tensegrity arm's physical linkage variant (@sec:sim_model_construction). For the training backend, the `skrl` library provides native support for Isaac Lab environments, offering competitive performance with greater modularity than monolithic libraries~@SerranoMunoz2023skrl. The integration of Isaac Lab and `skrl` forms the core framework stack of the present work.

=== Tendon- and Cable-Driven Robot Simulation <subsec:prior_tendon_sim>

*Rigid-link tendon-driven manipulation.*
The simulation of tendon routing has been extensively explored, though predominantly outside the Isaac ecosystem. Or _et~al._ presented _Robostrich_, an 18-#ac("DoF") tendon-driven robot modeled and trained within MuJoCo. They demonstrated that curriculum-based #ac("RL") is necessary for underactuated tendon systems to overcome challenging exploration landscapes~@Or2023Robostrich. Guist _et~al._ (_PAMY2_) successfully applied model-based #ac("RL") to a pneumatic-tendon arm, also leveraging MuJoCo's built-in tendon elements to map cable contraction to joint actuation~@Guist2024PAMY2. Morimoto _et~al._ compared tendon-driven continuum arms with rigid manipulators, highlighting the utility of tendon platforms in uncertain environments~@Saito2022ContinuumRL. 

*Modeling benchmarking and implementation constraints.*
Rao _et~al._ systematically benchmarked computational models of tendon-driven continuum robots, exposing the trade-offs between physical fidelity (e.g., Cosserat rod models) and simulation speed~@Rao2021TDCRModeling. In contrast to MuJoCo, which provides native, #ac("RL")-compatible tendon elements, Omniverse physics handles spatial tendons as constraint objects that often induce numerical instability and obscure tension variables (@sec:physics_sim). Consequently, to simulate tendon interactions reliably under #ac("RL") at scale, custom Jacobian and body-force mappings must be formulated, as is implemented in @sec:tendon_actuation.

=== Tensegrity-Inspired Serial Arm Manipulators <subsec:prior_tensegrity>

*Antiparallelogram joints in tensegrity arms.*
The integration of tensegrity principles into robot manipulation relies heavily on specific joint geometries. As outlined in @sec:tendon_mechanisms, Furet and Wenger~@FuretWenger2019 proposed a planar tensegrity 2-X manipulator using antiparallelogram (crossed four-bar) joints and conducted formal kinetostatic analysis. Muralidharan and Wenger~@MuralidharanWenger2021 contrasted these X-joints against traditional R-joints, verifying that the antiparallelogram joint provides a superior orientation range. Fasquelle _et~al._~@Fasquelle2020BioInspired3DOF extended this concept to a physical 3-#ac("DoF") prototype with closed-loop position control.

*FAPS tensegrity manipulator.*
The arm architecture examined in this thesis was physically introduced by Walter _et~al._, who demonstrated its compliance through hardware impact isolation tests~@Walter2023Tensegrity. Klein~@Klein2023 developed the initial forward-kinematic workspace analysis and a baseline Isaac Sim model for this manipulator but stopped short of applying learning-based unified control.

*Tensegrity #ac("RL").*
Notably, the application of #ac("RL") to tensegrity systems has been strictly confined to locomotion robots. Zhang _et~al._ applied model-based #ac("RL") to a NASA tensegrity rover~@Zhang2017tensRL, with subsequent research focusing on altering morphologies for rolling or crawling. At present, no literature exists concerning #ac("RL") applied to serial tensegrity manipulators or antiparallelogram-based tensegrity arms for pick-and-place tasks.

=== Computational Workspace Analysis <subsec:prior_workspace>

Analyzing the performance envelope of non-standard kinematic structures requires numerical simulation approaches (@sec:workspace_analysis_bg). Rastegar and Fardanesh formalized the Monte Carlo forward-kinematics sampling method, allowing complex coupled constraints to be mapped probabilistically into task-space volumes~@Rastegar1990WorkspaceMonteCarlo. Building upon this, Peidró _et~al._ introduced Gaussian Growth techniques to densify sampling near theoretical boundaries, resolving the sparsity problem inherent to uniform joint-space sampling~@Peidro2017GaussianGrowth.

Regarding manipulability, while Yoshikawa first quantified kinematic conditioning~@Yoshikawa1985Manipulability, Vahrenkamp and Asfour's extension to constrained mechanisms is particularly relevant for cable-driven structures. They utilized joint-angle distance-to-limit constraints to modulate manipulability near bounds~@Vahrenkamp2015ConstrainedManipulability. This thesis draws inspiration from their methodology by substituting traditional joint limits with analytical cable structural bounds (@sec:workspace_analysis), effectively assessing safe, tension-feasible manipulation volumes.

=== Identified Research Gap <subsec:prior_gap>

The literature review exposes three distinct domains that have not yet intersected. First, GPU-accelerated #ac("RL") has proven highly effective for contact-rich manipulation within the Isaac ecosystem~@Tang2023IndustReal @Handa2023DeXtreme, yet these tools have exclusively been applied to conventional rigid-body robots. Second, learning-based control of tendon-driven systems relies explicitly on external simulators like MuJoCo~@Or2023Robostrich @Guist2024PAMY2, circumventing the native modeling limitations of PhysX. Third, while the kinematics of antiparallelogram-jointed tensegrity arms are well-understood theoretically~@FuretWenger2019 @Walter2023Tensegrity, closed-loop neural control has never been generalized to this manipulator class.

As of May~2026, no published work integrates an (i)~antiparallelogram-jointed tensegrity manipulator with (ii)~scalable tendon-actuation models into (iii)~the Isaac Lab GPU-accelerated ecosystem for the purpose of (iv)~#ac("RL")-based manipulation. The present thesis directly bridges this gap. By developing robust physical models and custom force-mappings (@sec:sim_model_construction and @sec:tendon_actuation) and by employing curriculum-based #ac("PPO") (@sec:rl_tasks), this work validates the feasibility of controlling complex compliant structures using massive parallel simulation.

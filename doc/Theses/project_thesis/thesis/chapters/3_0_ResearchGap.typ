// Chapter 3: State of the Art and Research Gap
// Combines prior work review with research gap identification
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *

= State of the Art and Research Gap <ch:research_gap>

This chapter reviews prior work along three dimensions that converge in the present thesis: #ac("GPU")-accelerated #ac("RL") for manipulation, simulation and control of tendon- and cable-driven robots, and tensegrity-inspired serial arm manipulators. The review focuses on contributions that directly inform the methodological choices made in this thesis and concludes by identifying the specific research gap that the present work addresses.

== GPU-Accelerated #ac("RL") for Manipulation <sec:prior_gpu_rl>

==== Isaac Gym and large-scale parallel training
The transition from #ac("CPU") to GPU-accelerated simulation was catalyzed by Makoviychuk _et~al._~@Makoviychuk2021IsaacGym, who demonstrated that running both the PhysX step and neural-network forward passes entirely on the #ac("GPU") eliminates data transfer bottlenecks. In their benchmarking suite, 16~384 parallel agents trained with #ac("PPO") (see @sec:rl_manipulation) converged in under 25~minutes on a single #ac("GPU"). This directly enabled the massive parallelization scale utilized in the current methodology (@sec:training_infrastructure), overcoming prior computational bottlenecks in #ac("RL").

==== Contact-rich manipulation and assembly
Building upon parallelized simulation, Narang _et~al._ introduced _Factory_, an environment suite focused on contact-rich assembly tasks. By improving collision representations and constraint solvers, Factory achieves order-of-magnitude speedups over prior contact simulation in tight-tolerance tasks~@Narang2022Factory. Extending this, Tang _et~al._ (_IndustReal_) achieved successful sim-to-real transfer of peg-insertion policies with clearances of $lt.eq 0.5$--$0.6 "mm"$~@Tang2023IndustReal. _IndustReal_ specifically introduced a simulation-aware policy update mechanism and sampling-based curricula for progressive task difficulty. These curriculum strategies heavily inform the multi-stage reward and initialization design employed for the pick-and-place tasks formulated in this thesis (@sec:rl_tasks).

==== Sim-to-real transfer for dexterous manipulation
For high-degree-of-freedom end-effectors, Allshire _et~al._ demonstrated transfer of a 6-#ac("DoF") manipulation policy to a physical TriFinger robot using #ac("DR") and a keypoint-based pose representation~@Allshire2022TriFinger. Handa _et~al._ (_DeXtreme_) further improved on this using automatic #ac("DR") to adapt manipulation policies dynamically during training without manual tuning~@Handa2023DeXtreme. Both approaches demonstrate the efficancy of asymmetric actor-critic architectures, where the critic receives privileged simulator state while the actor relies only on observable states. While the present methodology operates exclusively in simulation, these established sim-to-real techniques validate the choice of Isaac Sim as a foundation for future physical deployment.

==== Frameworks: Orbit, Isaac Lab, and skrl
The environment management surrounding the #ac("GPU") pipeline has matured into standardized frameworks. Mittal _et~al._ presented _Orbit_ to wrap the Isaac Sim backend into modular environments conforming to common #ac("RL") #ac("API") standards~@Mittal2023Orbit. _Orbit_ evolved into _Isaac Lab_, reaching up to 1.6~million frames per second for state-based manipulation tasks and adding closed-loop kinematic chain support, which is critical for modeling the tensegrity arm's physical linkage variant (@sec:sim_model_construction)~@Mittal2025IsaacLab. For the training backend, the `skrl` library provides native support for Isaac Lab environments, offering competitive performance with greater modularity than monolithic libraries~@SerranoMunoz2023skrl. The integration of Isaac Lab and `skrl` forms the core framework stack of the present work.

== Tendon- and Cable-Driven Robot Simulation <sec:prior_tendon_sim>

==== Rigid-link tendon-driven manipulation
The simulation of tendon routing has been extensively explored, though predominantly outside the Isaac ecosystem. Or _et~al._ presented _Robostrich_, a 9-tendon, 18-#ac("DoF") rigid-link tendon-driven manipulator modeled and trained within MuJoCo. They demonstrated that curriculum-based #ac("RL") outperforms flat training for underactuated tendon arms~@Or2023Robostrich. Guist _et~al._ (_PAMY2_) successfully applied model-based #ac("RL") to a pneumatic-tendon arm, also leveraging MuJoCo's built-in tendon elements to map cable contraction to joint actuation~@Guist2024PAMY2. Morimoto _et~al._ compared tendon-driven continuum arms with rigid manipulators, highlighting the utility of tendon platforms in uncertain environments~@Saito2022ContinuumRL.

==== Cable-driven parallel robot simulation with #ac("RL")
Dhakate _et~al._ (_CaRoSaC_) used Unity3D with Obi Rope for flexible cable simulation via #ac("XPBD"), validated against a real industrial 4-cable-suspended parallel robot, with a TD3-based controller outperforming classical kinematic solvers near workspace boundaries~@Dhakate2025CaRoSaC. Garrido Campos _et~al._ validated a cable-driven parallel robot simulation against a hardware prototype, achieving a peak position divergence of only 0.27% across various speed scenarios~@Garrido2024CDPR.

==== Modeling benchmarks and implementation constraints
Rao _et~al._ systematically benchmarked computational models of tendon-driven continuum robots, exposing the trade-offs between physical fidelity (e.g., Cosserat rod models) and simulation speed~@Rao2021TDCRModeling. In contrast to MuJoCo, which provides native, #ac("RL")-compatible tendon elements, Omniverse physics handles spatial tendons as constraint objects that often induce numerical instability and obscure tension variables (@sec:physics_sim). Consequently, to simulate tendon interactions reliably under #ac("RL") at scale, custom Jacobian and body-force mappings must be formulated, as is implemented in @sec:tendon_actuation.

== Tensegrity-Inspired Serial Arm Manipulators <sec:prior_tensegrity>

==== Antiparallelogram joints in tensegrity arms
The integration of tensegrity principles into robot manipulation relies heavily on specific joint geometries. As outlined in @sec:tendon_linkage_mechanisms, Furet and Wenger~@FuretWenger2019 proposed a planar tensegrity 2-X manipulator using antiparallelogram (crossed four-bar) joints and conducted formal kinetostatic analysis. Hamon and Aoustin~@HamonAoustin2010 employed crossed four-bar linkages for the knees of a planar bipedal robot, exploiting the nonlinear torque transmission. Fasquelle _et~al._~@Fasquelle2020BioInspired3DOF extended the tensegrity 2-X concept to a physical 3-#ac("DoF") prototype with closed-loop position control. Muralidharan and Wenger~@MuralidharanWenger2021 contrasted these X-joints against traditional R-joints, verifying that the antiparallelogram joint provides a substantially larger orientation range for the same link lengths.

==== FAPS tensegrity manipulator
The arm architecture examined in this thesis was physically introduced by Walter _et~al._, who demonstrated the wrists compliance through hardware experiments isolating impact forces by up to 83%~@Walter2023Tensegrity. Klein~@Klein2023 developed the initial workspace analysis and a simulation model for this manipulator but did not apply learning-based control.

==== Tensegrity #ac("RL")
The application of #ac("RL") to tensegrity systems has so far been confined to the locomotion domain. Zhang _et~al._ applied model-based #ac("RL") to the SUPERball tensegrity locomotive~@Zhang2017tensRL, with subsequent research focusing on altering morphologies for rolling or crawling. No published work applies #ac("RL") to serial tensegrity manipulators or antiparallelogram-based tensegrity arms for pick-and-place tasks.

// == Computational Workspace Analysis <sec:prior_workspace>

// Analyzing the performance envelope of non-standard kinematic structures requires numerical sampling approaches (see @sec:workspace_analysis_sampling_metrics). Rastegar and Fardanesh formalized the Monte Carlo forward-kinematics sampling method, allowing complex coupled constraints to be mapped probabilistically into task-space volumes~@Rastegar1990WorkspaceMonteCarlo. Building upon this, Peidró _et~al._ introduced a Gaussian Growth method that iteratively densifies sampling near theoretical boundaries, resolving the sparsity problem inherent to uniform joint-space sampling~@Peidro2017GaussianGrowth.

// Regarding manipulability, Yoshikawa first quantified kinematic conditioning through the manipulability ellipsoid~@Yoshikawa1985Manipulability. Vahrenkamp and Asfour subsequently extended this measure by penalizing the Jacobian based on proximity to joint limits and self-collision distances~@Vahrenkamp2015ConstrainedManipulability. By analogy with their joint-limit penalty, the present thesis adapts this constrained-manipulability approach to the tendon-driven case by treating cable tension bounds in place of joint-position bounds, thereby assessing tension-feasible manipulation volumes. This analogy is the present author's extension and is not stated in @Vahrenkamp2015ConstrainedManipulability.

== Identified Research Gap <sec:research_gap_id>

The literature reviewed above reveals three distinct bodies of work that have not been combined. First, GPU-accelerated #ac("RL") for manipulation has advanced rapidly within the Isaac Gym / Isaac Sim ecosystem, with demonstrated sim-to-real transfer for contact-rich tasks involving conventional rigid-body robots~@Makoviychuk2021IsaacGym @Narang2022Factory @Tang2023IndustReal @Allshire2022TriFinger @Handa2023DeXtreme. Second, tendon- and cable-driven robot simulation for #ac("RL") has been explored predominantly in MuJoCo~@Or2023Robostrich @Guist2024PAMY2 @Saito2022ContinuumRL or in specialized engines~@Dhakate2025CaRoSaC, but not in Isaac Sim. Third, tensegrity-inspired manipulators with antiparallelogram joints have been studied from a mechanism-design perspective~@FuretWenger2019 @Fasquelle2020BioInspired3DOF @MuralidharanWenger2021, while #ac("RL") for tensegrity robots has been limited to locomotion~@Zhang2017tensRL.

As of May~2026, no published work combines an antiparallelogram-jointed tensegrity manipulator with scalable tendon-actuation models in the Isaac~Sim / Isaac~Lab ecosystem for #ac("RL")-based policy training. Furthermore, the FAPS-internal predecessor work by Klein~@Klein2023 established workspace analysis and simulation for this cable-driven manipulator but did not employ learning-based control. The present thesis addresses this gap by constructing a validated tendon-driven tensegrity robot model in Isaac~Sim (@sec:sim_model_construction), formulating custom Jacobian and body-force tendon mappings (@sec:tendon_actuation), and training manipulation policies using `skrl` #ac("PPO")~@SerranoMunoz2023skrl @Schulman2017PPO under progressive #ac("RL") task curricula (@sec:rl_tasks).

== Success Criteria

The following criteria define successful completion of the thesis objectives:

+ The simulated robot model covers the desired kinematic workspace defined by Klein~@Klein2023, validated through Monte Carlo workspace analysis and step-response testing.
+ #ac("RL") policies trained with #ac("PPO") converge to functional reach and cube-place behavior for at least one actuation variant (#ac("PD") or tendon-driven).
+ Comparative evaluation of #ac("PD")-driven and tendon-driven actuation variants provides quantitative metrics (success rate, convergence speed, final reward) enabling informed selection.
+ The simulation, validation and training infrastructure is documented and reproducible, providing a good foundation for the subsequent master thesis.

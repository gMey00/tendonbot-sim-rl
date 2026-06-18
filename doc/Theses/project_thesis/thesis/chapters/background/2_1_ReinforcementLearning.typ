// §2.1 #ac("RL") for Robotic Manipulation
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/diagrams.typ": rl-agent-env-loop
#import "../../../../shared/formatting/template.typ": faps-figure

== #ac("RL") for Robotic Manipulation <sec:rl_manipulation>

Robotic manipulation requires generating control commands that achieve a goal while dealing with uncertainty, partial observability, contacts, and high-dimensional configuration spaces. Classical approaches rely on analytical models, trajectory optimisation, and carefully engineered feedback control. However, accurate models of frictional contact, deformable objects, or cluttered interactions are often difficult to obtain, and manually designing robust policies can be time-consuming. #ac("RL") provides an alternative. It enables learning control policies from interaction data by optimising a task objective, thereby reducing reliance on explicit modelling and hand-crafted strategies. General #ac("RL") background and formalism are treated in standard texts such as Sutton and Barto~@SuttonBarto2018, and robotics-specific perspectives (including practical constraints and common failure modes) are discussed in Kober _et al._~@Kober2013Survey.

#reg-sym("sym:S", "sym:A", "sym:r", "sym:gamma", "sym:theta", "sym:pi", "sym:J", "sym:Ahat", "sym:epsilon", "sym:lambda")

#faps-figure(
  rl-agent-env-loop(),
  caption: [The #ac("RL") interaction loop: the agent selects action~$a_t$ from policy~$pi$. The environment transitions and returns observation~$o_(t+1)$ and reward~$r_(t+1)$, fed back to the agent via a unit delay~$z^(-1)$.],
  short-caption: [#ac("RL") agent-environment interaction loop],
) <fig:rl_agent_environment_loop>

==== #ac("MDP") viewpoint and partial observability
A manipulation task can be modelled as a #ac("MDP") $cal(M) = (cal(S), cal(A), P, r, gamma)$, where $cal(S)$ is the state space, $cal(A)$ the action space, $P(s_(t+1) | s_t, a_t)$ the transition dynamics, $r(s_t, a_t)$ the reward, and $gamma in [0,1)$ the discount factor. A policy $pi_theta (a | s)$ (parameterized by $theta$) induces trajectories $(s_0, a_0, r_0, dots, s_T)$ and is trained to maximize the expected discounted return $J(theta) = EE_(pi_theta)[sum_(t=0)^T gamma^t r_t].$~@SuttonBarto2018

#pagebreak()

In real manipulation systems, the full state $s_t$ is typically not directly accessible. Instead, control is based on observations $o_t$ (e.g., proprioception, gripper state, pose estimates, perception features). This partial observability is formally captured by the #ac("POMDP") framework, which generalises the #ac("MDP") with an observation model $O(o_t | s_t, a_(t-1))$ and renders the optimal policy history-dependent in general~@Spaan2012POMDP @SuttonBarto2018. In practice, this motivates careful observation design and, where appropriate, augmenting policies with memory (e.g., recurrent networks) or stacking observation histories to approximate the Markov property~@Kober2013Survey. @fig:rl_agent_environment_loop illustrates this interaction loop with the manipulation-specific action, observation, and reward structure used in this thesis.

==== Key challenges in manipulation
Three aspects commonly make #ac("RL") for manipulation demanding:
- *continuous, high-dimensional control*: Actions may correspond to joint torques, velocities, or Cartesian increments for many degrees of freedom.
- *contact-rich and nonlinear dynamics*: Small modelling errors in friction or contact impulses can lead to qualitatively different outcomes.
- *sparse or delayed feedback*: Success may only be measurable after long sequences of intermediate steps. 
Further practical constraints include safety, limited real-world interaction budgets, and the need for robustness across object variations and perception noise. These characteristics influence algorithm choice, reward design, termination conditions, and the role of simulation during training.~@Kober2013Survey

==== Reward design and task specification
In #ac("RL"), the reward function encodes the task objective. For manipulation, rewards are often composed of shaping terms (e.g., distance-to-target, alignment, grasp quality proxies) and sparse success signals (e.g., "object lifted above height"). Well-shaped rewards can accelerate learning but risk biasing the learned behavior toward unintended shortcuts. Conversely, sparse rewards better reflect the true objective but can make exploration difficult. Robotics #ac("RL") workflows therefore often iterate on reward definitions and termination criteria to balance learnability and task fidelity.~@Kober2013Survey @SuttonBarto2018

==== Sequential reward gating for multi-phase tasks
Multi-step manipulation tasks such as pick-and-place naturally decompose into sequential phases (reach, grasp, transport, place), each requiring distinct behaviors. Ng _et al._~@Ng1999RewardShaping proved that adding a _potential-based_ shaping reward $F(s, a, s') = gamma Phi(s') - Phi(s)$ preserves all optimal policies of the original #ac("MDP"), which provides theoretical justification for adding phase-dependent intermediate rewards.

Devlin and Kudenko extended this result to _dynamic_ potentials that depend on time-varying state augmentations (e.g., persistent latch flags), showing that optimality is preserved even when the potential changes during an episode~@Devlin2012DynamicPBRS.

In practice, Popov _et al._ demonstrated the effectiveness of _composite shaping rewards_ for Lego brick stacking, where each reward component is gated by a task-phase predicate (e.g., transport reward activates only after a "grasped" condition is met)~@Popov2017DataEfficientManipulation.

This gated activation pattern is equivalent to a two-state _reward machine_ in the formalism of Toro Icarte _et al._~@Icarte2022RewardMachines, where finite-state automata define per-phase reward functions driven by propositional events.

Benchmark environments such as robosuite~@Zhu2020robosuite implement this principle through explicit `staged_rewards()` methods with sequentially gated reach, grasp, lift, and hover components~@robosuite_pickplace_repo.

=== Learning objectives and algorithm families <sec:rl_algorithm_families>

For continuous control, modern deep #ac("RL") methods commonly use _policy gradient_ and _actor--critic_ approaches. Policy gradient methods directly optimise the policy parameters by ascending an estimate of $nabla_theta J(theta)$, which naturally supports continuous actions through differentiable policy parameterizations. Actor--critic methods introduce a learned critic (value function) that provides lower-variance learning signals for the actor, often improving stability and learning speed in practice. A central quantity is the advantage function $A^pi (s_t, a_t)$, which measures how much better an action is compared to the policy's baseline behaviour at a state. In practice, advantage estimates $hat(A)_t$ are computed from value predictions and collected rollouts, and then used to weight policy updates.~@SuttonBarto2018

==== Bias--variance trade-off in advantage estimation
A practical challenge is balancing low-variance learning signals with accurate credit assignment over time. #ac("GAE") provides a widely used mechanism to interpolate between Monte Carlo returns and bootstrapped temporal-difference estimates by exponentially weighting temporal-difference residuals. This improves learning stability in many continuous-control applications and is commonly paired with actor--critic training.~@Schulman2015GAE

=== Exploration and data collection in manipulation <sec:rl_exploration>

A defining challenge in #ac("RL") is the _exploration--exploitation trade-off_: the agent must explore sufficiently to discover high-reward behaviours while exploiting learned knowledge to improve performance~@SuttonBarto2018. In manipulation, exploration is complicated by long horizons, sparse success signals, and irreversible or unsafe states (in real systems)~@Kober2013Survey. Common practical measures include:

- *Stochastic policies and action sampling*, which induce exploratory behavior while still allowing gradient-based optimization~@SuttonBarto2018. In manipulation, _hybrid action spaces_ combining continuous arm control with simplified (binary or low-dimensional) gripper commands are common, as they reduce the exploration burden on the gripper dimension while retaining continuous dexterity for arm motions~@Kalashnikov2018QTOpt @Zhu2020robosuite.
- *Reward shaping and curricula*, where training starts from simplified conditions (e.g., easier initial object poses) and progressively increases difficulty to maintain a learnable signal~@Kober2013Survey.
- *Episode resets and termination design*, which affect state visitation and can substantially change the data distribution seen by the learner~@Kober2013Survey.

These aspects tie the learning algorithm closely to environment design. Even a robust optimiser can fail if exploration is insufficient or if the reward/termination specification induces pathological behavior~@Kober2013Survey @SuttonBarto2018.

=== Simulation-to-reality considerations <sec:rl_sim2real>

#ac("RL") policies are frequently trained in simulation to avoid wear, risk, and slow data collection on real robots~@Kober2013Survey. However, transferring policies to the real world is challenging due to the _reality gap_. Discrepancies in geometry, contact parameters, actuator dynamics, sensor noise, and unmodeled effects can degrade performance. Common strategies include *#ac("DR")*, where visual and physical parameters are randomized during training so that the policy learns behaviors that are robust across a range of plausible conditions~@Tobin2017DomainRandomization, and *dynamics randomization*, which targets uncertainties in physical parameters (e.g., mass, friction, motor gains, delays) to improve transfer for control policies~@Peng2018DynamicsRandomization.

=== #ac("PPO") <sec:ppo>

Among actor--critic methods, #ac("PPO") is a popular choice due to its robustness and simplicity. #ac("PPO") performs iterative policy improvement by collecting trajectories with the current policy and updating the policy with a surrogate objective that discourages overly large steps in policy space. Define the importance ratio $r_t (theta) = pi_theta (a_t | s_t) / pi_(theta_"old") (a_t | s_t)$. #ac("PPO") maximizes the clipped objective

$ cal(L)^"CLIP" (theta) = EE [min(r_t (theta) hat(A)_t, "clip"(r_t (theta), 1 - epsilon, 1 + epsilon) hat(A)_t)] $ <eq:ppo_clip>

where $hat(A)_t$ is an advantage estimate (commonly computed with #ac("GAE")). Intuitively, clipping limits the incentive to change the policy too aggressively, which helps prevent performance collapse from unstable updates. This is a practical consideration in contact-rich manipulation where learning can be sensitive to abrupt policy changes.~@Schulman2015GAE~@Schulman2017PPO@Kober2013Survey

//==== Why #ac("PPO") is often used for robotic manipulation <sec:ppo_for_manipulation>

From a robotics perspective, #ac("PPO") has several properties that make it attractive in manipulation pipelines:

- *Stable learning dynamics:* The clipped objective yields conservative and typically reliable updates~@Schulman2017PPO.
- *Compatibility with continuous actions:* #ac("PPO") naturally supports Gaussian policies over continuous controls~@SuttonBarto2018.
- *Practical training loop:* The rollout-and-update structure integrates well with simulator-based training and supports systematic iteration on reward and termination design~@Kober2013Survey.

These considerations motivate #ac("PPO") as a strong baseline for manipulation, especially when training stability and predictable iteration cycles are prioritized.

=== Statistical Evaluation of #ac("RL") Experiments <sec:rl_statistical_eval>

Deep #ac("RL") algorithms exhibit substantial run-to-run variance even when hyperparameters, environment, and implementation are held fixed. Henderson _et al._ show that training the same algorithm on the same task while varying only the random seed can produce learning curves that do not fall within the same distribution, and that splitting a single ten-seed experiment into two groups of five seeds yields averaged curves that a two-sample $t$-test deems significantly different ($t = -9.09$, $p = 0.0016$).
// Henderson2018DeepRLMatters — p.2 §"Experimental Analysis" ("To ensure fairness we run five experiment trials for each evaluation, each with a different preset random seed") and p.5 §"Random Seeds and Trials" + Fig.5 and its caption ("averaged over two sets of 5 different random seeds each. The average 2-sample t-test ... resulted in t = -9.0916, p = 0.0016"). Page numbers refer to the uploaded arXiv PDF (= AAAI'18, pp. 3207--3214).
This sensitivity to seed choice means a single learning curve is not a reliable summary. Henderson _et al._ recommend averaging over multiple independent runs and quantifying the outcome with significance tests, bootstrap confidence bounds, and bootstrap power analysis, while explicitly noting that no single, universally valid trial count can be prescribed.~@Henderson2018DeepRLMatters
// Henderson2018DeepRLMatters — p.5 §"Random Seeds and Trials" ("While there can be no specific number of trials specified as a recommendation ... power analysis methods can be used"), p.4 Table 3 (bootstrap mean and 95% confidence bounds), and p.6 §"Confidence Bounds"/"Power Analysis"/"Significance" (averaging over seeds gives insight into the population distribution; bootstrap and significance testing). Uploaded arXiv PDF.

The number of seeds is itself a statistical question. Colas _et al._ frame the comparison of two algorithms as a power-analysis problem and recommend Welch's $t$-test over Student's $t$-test, because it remains valid under the unequal variances that are the rule rather than the exception when learning curves differ in stability across configurations.
// Colas2018HowManySeeds — p.7 §3.1 "T-test and Welch's t-test" ("an adaptation of the 2-sample t-test for unequal variances called Welch's t-test should be used"). Uploaded PDF.
Their power analysis shows that small seed counts are badly underpowered. In their worked example a five-seed comparison has a $51 %$ probability of missing a real effect ($beta = 0.51$), which only falls to the conventional $beta = 0.2$ once the budget is raised to ten seeds.
// Colas2018HowManySeeds — p.13 §4.2 "Step 2 - Choosing the sample size" + Fig.4 ("we find β = 0.51 for N = 5 ... To meet the requirement β = 0.2, N should be increased to N = 10 (β = 0.19)"; effect size ε = 1382). Uploaded PDF.
They also show empirically that for small samples the realised false-positive rate exceeds the nominal $alpha = 0.05$. The bootstrap confidence-interval test should not be used with fewer than about twenty samples (its type-I error reaches roughly $10 %$ at small $N$), and Welch's $t$-test, although more conservative, is still slightly inflated.
// Colas2018HowManySeeds — p.16 §5.1 + Fig.6 ("the bootstrap confidence interval test should not be used with small sample sizes (<10) ... probability of type-I error (≈ 10%)"; "around 10% for the bootstrap test and above 5% for the Welch's t-test") and p.18--19 §6 ("A bootstrap test should not be used with less than 20 samples"). Uploaded PDF.
Their final recommendations are therefore to use Welch's $t$-test, to set the significance level below $0.05$, to apply a Bonferroni correction $alpha_"Bon" = alpha / N_E$ when several comparisons share a baseline (since the false-positive rate grows linearly with the number $N_E$ of comparisons), and to choose sample sizes larger than the power analysis prescribes.~@Colas2018HowManySeeds
// Colas2018HowManySeeds — p.5 §2 (Bonferroni correction α_Bon = α/N_E controlling the family-wise error rate; "the false positive rate grows linearly with the number of experiments") and p.19 §6 "Final recommendations". Uploaded PDF.

Even with an adequate seed budget, the way per-run results are aggregated matters. Agarwal _et al._ observe that deep #ac("RL") work usually reports point estimates of aggregate performance (typically the mean or median score across tasks) which ignore the statistical uncertainty implied by a finite number of training runs.
// Agarwal2021Precipice — p.1 Abstract + p.2 §1 ("Most published results on deep RL benchmarks compare point estimates of aggregate performance such as mean and median scores across tasks, ignoring the statistical uncertainty implied by the use of a finite number of training runs"). Uploaded PDF.
They caution that the mean is easily dominated by a few outlier tasks, whereas the median has high variability and is unaffected by zero scores on nearly half the tasks. Instead they advocate reporting interval estimates via stratified bootstrap confidence intervals together with the interquartile mean (IQM), the mean over the middle $50 %$ of the combined runs, which is robust to outliers and attains small uncertainty even with a handful of runs.
// Agarwal2021Precipice — p.2 §1 ("mean can be easily dominated by performance on a few outlier tasks, while median has high variability and zero performance on nearly half of the tasks does not change it"; "interquartile mean, which are not unduly affected by outliers and have small uncertainty even with a handful of runs") and p.7 §4.3 "Robust and Efficient Aggregate Metrics". Uploaded PDF.
To expose the full spread rather than a single number, they additionally propose performance profiles (run-score distributions) and report the probability of improvement and the optimality gap. Their stratified-bootstrap intervals give reliable coverage for as few as roughly ten runs, while three runs underestimate the true interval.~@Agarwal2021Precipice
// Agarwal2021Precipice — p.2 §1 ("performance profiles ... statistically unbiased, more robust to outliers, and require fewer runs for smaller uncertainty") and p.6 §4.2 "Performance Profiles" ("performance profiles are robust to outlier runs and insensitive to small changes in performance across all tasks"); p.3 (report "average probability of improvement and optimality gap"); p.6 Fig.6 ("percentile CIs provide good interval estimates for as few as N = 10 runs"; "with 3 runs, bootstrap CIs underestimate the true 95% CIs"). Uploaded PDF.

Taken together, these guidelines motivate the evaluation protocol used for the reach task in @subsec:reach_performance. Every reported number is aggregated over several independent training seeds and accompanied by a measure of dispersion rather than shown as a single curve, head-to-head comparisons against the baselines use the Welch two-sample $t$-test, and the multiple comparisons sharing a common baseline are read against a Bonferroni-corrected threshold. Because the seed budget in this work ($N = 5$) lies below the sample sizes that Colas _et al._ and Agarwal _et al._ recommend for confirmatory claims, the corresponding comparisons are treated as exploratory and the per-seed distribution is reported alongside the aggregates~@Colas2018HowManySeeds @Agarwal2021Precipice.


// §2.1 Reinforcement Learning for Robotic Manipulation
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/diagrams.typ": rl-agent-env-loop
#import "../../../../shared/formatting/template.typ": faps-figure

== Reinforcement Learning for Robotic Manipulation <sec:rl_manipulation>

Robotic manipulation requires generating control commands that achieve a goal while dealing with uncertainty, partial observability, contacts, and high-dimensional configuration spaces. Classical approaches rely on analytical models, trajectory optimisation, and carefully engineered feedback control. However, accurate models of frictional contact, deformable objects, or cluttered interactions are often difficult to obtain, and manually designing robust policies can be time-consuming. #ac("RL") provides an alternative: it enables learning control policies from interaction data by optimising a task objective, thereby reducing reliance on explicit modelling and hand-crafted strategies. General #ac("RL") background and formalism are treated in standard texts such as Sutton and Barto~@SuttonBarto2018, and robotics-specific perspectives (including practical constraints and common failure modes) are discussed in Kober _et al._~@Kober2013Survey.

#reg-sym("sym:S", "sym:A", "sym:r", "sym:gamma", "sym:theta", "sym:pi", "sym:J", "sym:Ahat", "sym:epsilon")

*MDP viewpoint and partial observability.*
A manipulation task can be modelled as an #ac("MDP") $cal(M) = (cal(S), cal(A), P, r, gamma)$, where $cal(S)$ is the state space, $cal(A)$ the action space, $P(s_(t+1) | s_t, a_t)$ the transition dynamics, $r(s_t, a_t)$ the reward, and $gamma in [0,1)$ the discount factor~@SuttonBarto2018. A policy $pi_theta (a | s)$ (parameterized by $theta$) induces trajectories $(s_0, a_0, r_0, dots, s_T)$ and is trained to maximize the expected discounted return $J(theta) = EE_(pi_theta)[sum_(t=0)^T gamma^t r_t]$~@SuttonBarto2018. In real manipulation systems, the full state $s_t$ is typically not directly accessible; instead, control is based on observations $o_t$ (e.g., proprioception, gripper state, pose estimates, perception features). This partial observability motivates careful observation design and, where appropriate, augmenting policies with memory (e.g., recurrent networks), or stacking observation histories to approximate the Markov property in practice~@Kober2013Survey.

#faps-figure(
  rl-agent-env-loop(),
  caption: [The reinforcement learning interaction loop: the agent selects action~$a_t$ from policy~$pi$, the environment transitions and returns observation~$o_(t+1)$ and reward~$r_t$.],
  short-caption: [RL agent-environment interaction loop],
) <fig:rl_agent_environment_loop>

*Key challenges in manipulation.*
Three aspects commonly make #ac("RL") for manipulation demanding: (i) *continuous, high-dimensional control*, where actions may correspond to joint torques, velocities, or Cartesian increments for many degrees of freedom; (ii) *contact-rich and nonlinear dynamics*, where small modelling errors in friction or contact impulses can lead to qualitatively different outcomes; and (iii) *sparse or delayed feedback*, where success may only be measurable after long sequences of intermediate steps. Further practical constraints include safety, limited real-world interaction budgets, and the need for robustness across object variations and perception noise~@Kober2013Survey. These characteristics influence algorithm choice, reward design, termination conditions, and the role of simulation during training.

*Reward design and task specification.*
In #ac("RL"), the reward function encodes the task objective. For manipulation, rewards are often composed of shaping terms (e.g., distance-to-target, alignment, grasp quality proxies) and sparse success signals (e.g., "object lifted above height"). Well-shaped rewards can accelerate learning but risk biasing the learned behavior toward unintended shortcuts; conversely, sparse rewards better reflect the true objective but can make exploration difficult. Robotics #ac("RL") workflows therefore often iterate on reward definitions and termination criteria to balance learnability and task fidelity~@Kober2013Survey @SuttonBarto2018.

=== Learning objectives and algorithm families <sec:rl_algorithm_families>

For continuous control, modern deep #ac("RL") methods commonly use _policy gradient_ and _actor--critic_ approaches. Policy gradient methods directly optimise the policy parameters by ascending an estimate of $nabla_theta J(theta)$, which naturally supports continuous actions through differentiable policy parameterizations~@SuttonBarto2018. Actor--critic methods introduce a learned critic (value function) that provides lower-variance learning signals for the actor, often improving stability and learning speed in practice~@SuttonBarto2018. A central quantity is the advantage function $A^pi (s_t, a_t)$, which measures how much better an action is compared to the policy's baseline behaviour at a state. In practice, advantage estimates $hat(A)_t$ are computed from value predictions and collected rollouts, and then used to weight policy updates~@SuttonBarto2018.

*Bias--variance trade-off in advantage estimation.*
A practical challenge is balancing low-variance learning signals with accurate credit assignment over time. #ac("GAE") provides a widely used mechanism to interpolate between Monte Carlo returns and bootstrapped temporal-difference estimates by exponentially weighting temporal-difference residuals~@Schulman2015GAE. This improves learning stability in many continuous-control applications and is commonly paired with actor--critic training.

=== Exploration and data collection in manipulation <sec:rl_exploration>

A defining challenge in #ac("RL") is the _exploration--exploitation trade-off_: the agent must explore sufficiently to discover high-reward behaviours while exploiting learned knowledge to improve performance~@SuttonBarto2018. In manipulation, exploration is complicated by long horizons, sparse success signals, and irreversible or unsafe states (in real systems)~@Kober2013Survey. Common practical measures include:

- *Stochastic policies and action sampling*, which induce exploratory behavior while still allowing gradient-based optimization~@SuttonBarto2018.
- *Reward shaping and curricula*, where training starts from simplified conditions (e.g., easier initial object poses) and progressively increases difficulty to maintain a learnable signal~@Kober2013Survey.
- *Episode resets and termination design*, which affect state visitation and can substantially change the data distribution seen by the learner~@Kober2013Survey.

These aspects tie the learning algorithm closely to environment design: even a robust optimiser can fail if exploration is insufficient or if the reward/termination specification induces pathological behavior~@Kober2013Survey @SuttonBarto2018.

=== Simulation-to-reality considerations <sec:rl_sim2real>

#ac("RL") policies are frequently trained in simulation to avoid wear, risk, and slow data collection on real robots~@Kober2013Survey. However, transferring policies to the real world is challenging due to the _reality gap_: discrepancies in geometry, contact parameters, actuator dynamics, sensor noise, and unmodeled effects can degrade performance. Common strategies include *domain randomization*, where visual and physical parameters are randomized during training so that the policy learns behaviors that are robust across a range of plausible conditions~@Tobin2017DomainRand, and *dynamics randomization*, which targets uncertainties in physical parameters (e.g., mass, friction, motor gains, delays) to improve transfer for control policies~@Peng2018DynamicsRand.

=== Proximal Policy Optimization (PPO) <sec:ppo>

Among actor--critic methods, #ac("PPO") is a popular choice due to its robustness and simplicity. #ac("PPO") performs iterative policy improvement by collecting trajectories with the current policy and updating the policy with a surrogate objective that discourages overly large steps in policy space~@Schulman2017PPO. Define the importance ratio $r_t (theta) = pi_theta (a_t | s_t) / pi_(theta_"old") (a_t | s_t)$. #ac("PPO") maximizes the clipped objective

$ cal(L)^"CLIP" (theta) = EE [min(r_t (theta) hat(A)_t, "clip"(r_t (theta), 1 - epsilon, 1 + epsilon) hat(A)_t)] $ <eq:ppo_clip>

where $hat(A)_t$ is an advantage estimate (commonly computed with #ac("GAE"))~@Schulman2015GAE @Schulman2017PPO. Intuitively, clipping limits the incentive to change the policy too aggressively, which helps prevent performance collapse from unstable updates. This is a practical consideration in contact-rich manipulation where learning can be sensitive to abrupt policy changes~@Schulman2017PPO @Kober2013Survey.

=== Why PPO is often used for robotic manipulation <sec:ppo_for_manipulation>

From a robotics perspective, #ac("PPO") has several properties that make it attractive in manipulation pipelines:

- *Stable learning dynamics:* The clipped objective yields conservative and typically reliable updates~@Schulman2017PPO.
- *Compatibility with continuous actions:* #ac("PPO") naturally supports Gaussian policies over continuous controls~@SuttonBarto2018.
- *Practical training loop:* The rollout-and-update structure integrates well with simulator-based training and supports systematic iteration on reward and termination design~@Kober2013Survey.

These considerations motivate #ac("PPO") as a strong baseline for manipulation, especially when training stability and predictable iteration cycles are prioritized.

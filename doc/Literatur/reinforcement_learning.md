# Reinforcement Learning

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Foundations & Textbooks

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [SuttonBarto2018](sources/reinforcement_learning/Books/SuttonBarto2018.pdf) | Sutton, R. S. & Barto, A. G. | *Reinforcement Learning: An Introduction* (2nd ed.) | 2018 | 🟡 | 🟢 | 🟢 | Definitive RL textbook covering MDPs, dynamic programming, policy-gradient, and actor-critic methods. Foundational reference for all RL chapters. [MIT Press](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |
| [Puterman1994](sources/reinforcement_learning/Books/Puterman1994.pdf) | Puterman, M. L. | *Markov Decision Processes: Discrete Stochastic Dynamic Programming* | 1994 | 🔴 | 🟢 | 🟢 | Rigorous mathematical treatment of MDPs. Cited for formal MDP definition in the background chapter. [DOI](https://doi.org/10.1002/9780470316887) |

### Book Chapters

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Spaan2012](sources/reinforcement_learning/Books/Spaan2012.pdf) | Spaan, M. T. J. | "Partially Observable Markov Decision Processes" | 2012 | 🔴 | 🟡 | 🟢 | POMDP theory. Background for discussing partial observability in cloth manipulation tasks. [PDF](https://www.st.ewi.tudelft.nl/mtjspaan/pub/Spaan12pomdp.pdf) |

### Survey Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Kober2013](sources/reinforcement_learning/Reviews/Kober2013.pdf) | Kober, J. et al. | "Reinforcement Learning in Robotics: A Survey" | 2013 | 🔴 | 🟢 | 🟢 | Comprehensive survey of RL applications in robotics. Key background reference for situating the thesis in the RL-robotics landscape. [DOI](https://doi.org/10.1177/0278364913495721) |
| [Liu2021DRLMini](sources/reinforcement_learning/Reviews/Liu2021DRLMini.pdf) | Liu, R. et al. | "Deep Reinforcement Learning for the Control of Robotic Manipulation — A Focussed Mini-Review" | 2021 | 🔴 | 🟢 | 🟢 | Concise focused review of DRL for robotic manipulation. Bridges Kober2013 and newer 2024 surveys. [DOI](https://doi.org/10.3390/robotics10010022) |
| [Tang2024DRLSurvey](sources/reinforcement_learning/Reviews/Tang2024DRLSurvey.pdf) | Tang, C. et al. | "Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes" | 2024 | 🔴 | 🟢 | 🟢 | Up-to-date survey cataloguing real-world DRL deployments in robotics. Core recent-state-of-the-art reference. [arXiv](https://arxiv.org/abs/2408.03539) |
| [Firoozi2023](sources/reinforcement_learning/Reviews/Firoozi2023.pdf) | Firoozi, R. et al. | "Foundation Models in Robotics: Applications, Challenges, and the Future" | 2023 | 🔴 | 🟡 | 🟢 | Survey of foundation-model usage across robotics. Context for large-model-based policies and representation learning. [arXiv](https://arxiv.org/abs/2312.07843) |

---

## Policy Gradient & PPO

### Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Schulman2017PPO](sources/reinforcement_learning/Papers/Schulman2017PPO.pdf) | Schulman, J. et al. | "Proximal Policy Optimization Algorithms" | 2017 | 🟢 | 🟢 | 🟢 | Introduces PPO, the primary RL algorithm used in this project. Core method reference. [arXiv](https://arxiv.org/abs/1707.06347) |
| [Schulman2015GAE](sources/reinforcement_learning/Papers/Schulman2015GAE.pdf) | Schulman, J. et al. | "High-Dimensional Continuous Control Using Generalized Advantage Estimation" | 2015 | 🟢 | 🟢 | 🟢 | GAE for variance reduction in policy gradients. Used in the PPO implementation. [arXiv](https://arxiv.org/abs/1506.02438) |
| [Liu2024adaptive](sources/reinforcement_learning/Papers/Liu2024adaptive.pdf) | Liu, Y. et al. | "Adaptive Reinforcement Learning for Robot Control" | 2024 | 🟡 | 🟡 | 🟢 | Transfer-learning approach for adaptive robot control. Context for sim-to-real adaptation strategies. [arXiv](https://arxiv.org/abs/2404.18713) |

---

## Reproducibility & Statistical Evaluation

### Conference Papers & Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Henderson2018DeepRLMatters](sources/reinforcement_learning/Papers/Henderson2018DeepRLMatters.pdf) | Henderson, P. et al. | "Deep Reinforcement Learning that Matters" | 2018 | 🟢 | 🟢 | 🟢 | AAAI 2018. Seminal reproducibility study for deep RL: documents large run-to-run variance across seeds (two 5-seed splits of one 10-seed run differ significantly) and hyperparameter sensitivity; uses five trials itself and recommends averaging over multiple seeds with significance tests, bootstrap confidence bounds, and power analysis (stops short of prescribing a fixed seed count). Cited in Background §2.1.5 (multi-seed motivation) and Methodology/Results (5-seed protocol). [arXiv](https://arxiv.org/abs/1709.06560) |
| [Colas2018HowManySeeds](sources/reinforcement_learning/Papers/Colas2018HowManySeeds.pdf) | Colas, C., Sigaud, O. & Oudeyer, P.-Y. | "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments" | 2018 | 🟢 | 🟢 | 🟢 | Power-analysis paper. Recommends Welch's *t*-test over bootstrap CIs for small N (β=0.51 at N=5 vs 0.19 at N=10 in their example), shows the bootstrap CI test is unreliable below N≈20, and gives explicit guidance for α<0.05 and Bonferroni correction across multiple comparisons. Cited in Background §2.1.5 (statistical methodology) and Results §5.2 (significance tests). [arXiv](https://arxiv.org/abs/1806.08295) |
| [Agarwal2021Precipice](sources/reinforcement_learning/Papers/Agarwal2021Precipice.pdf) | Agarwal, R. et al. | "Deep Reinforcement Learning at the Edge of the Statistical Precipice" | 2021 | 🟡 | 🟢 | 🟢 | NeurIPS 2021 (Outstanding Paper). Argues point estimates of aggregate performance ignore the uncertainty of finite runs; advocates interval estimates via stratified bootstrap CIs, the robust interquartile mean (IQM), performance profiles, and probability of improvement — reliable for as few as ~10 runs. The `rliable` library. Cited in Background §2.1.5 (robust aggregate reporting). [arXiv](https://arxiv.org/abs/2108.13264) |

---

## Model-Based RL

### Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Clavera2018](sources/reinforcement_learning/Papers/Clavera2018.pdf) | Clavera, I. et al. | "Model-Based Reinforcement Learning via Meta-Policy Optimization" | 2018 | 🔴 | 🟡 | 🟢 | Meta-learning for model-based RL. Background for adaptive dynamics models. [arXiv](https://arxiv.org/abs/1809.05214) |
| [Wang2019MBRL](sources/reinforcement_learning/Papers/Wang2019MBRL.pdf) | Wang, T. et al. | "Benchmarking Model-Based Reinforcement Learning" | 2019 | 🔴 | 🟡 | 🟢 | Unified benchmark for model-based RL evaluation. Context for method comparison. [arXiv](https://arxiv.org/abs/1907.02057) |

---

## Curriculum Learning & Reward Shaping

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Bengio2009](sources/reinforcement_learning/Papers/Bengio2009.pdf) | Bengio, Y. et al. | "Curriculum Learning" | 2009 | 🟢 | 🟢 | 🟢 | Foundational work on curriculum learning. Directly used in the training pipeline progression (reach → pick → sort). [DOI](https://doi.org/10.1145/1553374.1553380) |
| [Ng1999](sources/reinforcement_learning/Papers/Ng1999.pdf) | Ng, A. Y. et al. | "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping" | 1999 | 🟢 | 🟢 | 🟢 | Theoretical foundation for potential-based reward shaping used in the environment reward functions. [Link](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html) |
| [Asmuth2008](sources/reinforcement_learning/Papers/Asmuth2008.pdf) | Asmuth, J. et al. | "Potential-based Shaping in Model-based Reinforcement Learning" | 2008 | 🟡 | 🟡 | 🟢 | Extends reward shaping to model-based RL. Background for reward design. [PDF](https://cdn.aaai.org/AAAI/2008/AAAI08-096.pdf) |
| [Devlin2012](sources/reinforcement_learning/Papers/Devlin2012.pdf) | Devlin, S. & Kudenko, D. | "Dynamic Potential-Based Reward Shaping" | 2012 | 🟡 | 🟢 | 🟢 | Time-dependent potentials and latch-based rewards preserving policy invariance. Extends Ng (1999) to non-stationary settings. |
| [Andrychowicz2017](sources/reinforcement_learning/Papers/Andrychowicz2017.pdf) | Andrychowicz, M., Wolski, F., Ray, A. et al. | "Hindsight Experience Replay" | 2017 | 🟡 | 🟢 | 🟢 | Goal relabeling for sparse binary rewards. HER philosophy: relabel failed trajectories as successes for alternative goals. |
| [Berducci2024](sources/reinforcement_learning/Papers/Berducci2024.pdf) | Berducci, L. et al. | "Hierarchical Potential-Based Reward Shaping" | 2024 | 🟡 | 🟢 | 🟢 | Compositional potentials with strict partial ordering for multi-requirement task decomposition. |
| [Yang2024](sources/reinforcement_learning/Papers/Yang2024.pdf) | Yang, Y. et al. | "Reinforcement Learning with Reward Shaping and Hybrid Exploration in Sparse Reward Scenes" | 2024 | 🟡 | 🟢 | 🟢 | Reward-shaping plus hybrid exploration for sparse-reward industrial tasks. Relevant for the sparse-reward regime in thesis manipulation setups. [DOI](https://doi.org/10.1109/ICIEA61579.2024.10664745) |

### Reward Machines

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [ToroIcarte2022](sources/reinforcement_learning/Papers/ToroIcarte2022.pdf) | Toro Icarte, R., Klassen, T. Q., Valenzano, R. A. & McIlraith, S. A. | "Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning" | 2022 | 🟡 | 🟢 | 🟢 | Formal RM framework (δ_u, δ_r) with four solution algorithms (QRM, CRM, HRM). Core reference for structured reward decomposition. |
| [ToroIcarte2018](sources/reinforcement_learning/Papers/ToroIcarte2018.pdf) | Toro Icarte, R., Klassen, T. Q., Valenzano, R. A. & McIlraith, S. A. | "Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning" | 2018 | 🟡 | 🟢 | 🟢 | Original RM paper: finite-state automata for task phase decomposition with distinct reward functions per phase. |
| [Camacho2021](sources/reinforcement_learning/Papers/Camacho2021.pdf) | Camacho, A., Varley, J., Zeng, A., Jain, D., Iscen, A. & Kalashnikov, D. | "Reward Machines for Vision-Based Robotic Manipulation" | 2021 | 🟡 | 🟡 | 🟢 | DQRM on UR5 PyBullet achieving 6× fewer training steps vs. DQN. Demonstrates RM applicability to robotic manipulation. |
| Camacho2017 | Camacho, A., Icarte, R. T., Sanner, S. & McIlraith, S. A. | "LTL to DFA Compilation for Reward Machine Construction" | 2017 | 🔴 | 🟡 | 🟢 | Compiles linear temporal logic (LTL) specifications to DFA for augmented state space construction. |

### Staged & Composite Rewards

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [VanSeijen2017](sources/reinforcement_learning/Papers/VanSeijen2017.pdf) | Van Seijen, H., Fatemi, M., Romoff, J., Laroche, R., Barnes, T. & Tsang, J. | "Hybrid Reward Architecture for Reinforcement Learning" | 2017 | 🟡 | 🟢 | 🟢 | Decomposed rewards with separate value functions per reward component. Theoretical basis for multi-term reward structures. |
| [Mu2024](sources/reinforcement_learning/Papers/Mu2024.pdf) | Mu, T., Liu, M. & Su, H. | "DrS: Learning Reusable Dense Rewards for Multi-Stage Tasks" | 2024 | 🟡 | 🟡 | 🟢 | Discriminator-based stage rewards evaluated on ManiSkill. Learned dense rewards for sequential manipulation. |

---

## Benchmarks & Frameworks

### Journal Articles & Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [SerranoMunoz2023](sources/reinforcement_learning/Papers/SerranoMunoz2023.pdf) | Serrano-Muñoz, A. et al. | "skrl: Modular and Configurable Library for Reinforcement Learning" | 2023 | 🟢 | 🟢 | 🟢 | The RL training library used in this project. Provides PPO, SAC, and other algorithms with Isaac Lab integration. [JMLR](https://jmlr.org/papers/v24/23-0112.html) |
| [Zhu2020robosuite](sources/reinforcement_learning/Papers/Zhu2020robosuite.pdf) | Zhu, Y. et al. | "robosuite: A Modular Simulation Framework and Benchmark for Robot Learning" | 2020 | 🔴 | 🟡 | 🟢 | Explicit `staged_rewards()` method with max-aggregation across phases. Key manipulation benchmark with structured reward API. |
| [Gu2023maniskill2](sources/reinforcement_learning/Papers/Gu2023maniskill2.pdf) | Gu, J., Xiang, F., Li, X. et al. | "ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills" | 2023 | 🟡 | 🟡 | 🟢 | GPU-accelerated benchmark with per-task `compute_dense_reward()` and multi-stage structure. |
| [Towers2024Gymnasium](sources/reinforcement_learning/Papers/Towers2024Gymnasium.pdf) | Towers, M. et al. | "Gymnasium: A Standard Interface for Reinforcement Learning Environments" | 2024 | 🟢 | 🟡 | 🟢 | Successor to OpenAI Gym providing the canonical RL environment API used by Isaac Lab. [arXiv](https://arxiv.org/abs/2407.17032) |
| [SerranoMunoz2022skrl](sources/reinforcement_learning/Papers/SerranoMunoz2022skrl.pdf) | Serrano-Muñoz, A. et al. | "skrl — Modular and Flexible Library for Reinforcement Learning" (arXiv preprint) | 2022 | 🟢 | 🟢 | 🟢 | Original arXiv preprint of skrl (later JMLR 2023 as `SerranoMunoz2023`). Cited for the implementation library used in this project. [arXiv](https://arxiv.org/abs/2202.03825) |

---

## Multi-Agent & Cooperative RL

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Uchibe2018](sources/reinforcement_learning/Papers/Uchibe2018.pdf) | Uchibe, E. | "Cooperative and Competitive Reinforcement and Imitation Learning for a Mixture of Heterogeneous Learning Modules" | 2018 | 🟡 | 🟢 | 🟢 | Framework for cooperative/competitive multi-agent learning. Relevant for the master thesis multi-agent textile sorting setup. [DOI](https://doi.org/10.3389/fnbot.2018.00061) |

### Dec-POMDP, CTDE, and MARL Algorithms (Master Thesis §2.3)

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [OliehoekAmato2016DecPOMDP](sources/Citations/OliehoekAmato2016DecPOMDP.md) | Oliehoek, F. A. & Amato, C. | *A Concise Introduction to Decentralized POMDPs* | 2016 | 🟡 | 🟢 | 🟢 | Canonical Dec-POMDP formalization used to frame the pick/present/distribute pipeline as a cooperative multi-agent problem. [DOI](https://doi.org/10.1007/978-3-319-28929-8) |
| [Lowe2017MADDPG](sources/Citations/Lowe2017MADDPG.md) | Lowe, R. et al. | "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" | 2017 | 🟡 | 🟢 | 🟢 | Introduces MADDPG and the CTDE paradigm (centralized critic, decentralized actors). [arXiv](https://arxiv.org/abs/1706.02275) |
| [deWitt2020IPPO](sources/Citations/deWitt2020IPPO.md) | de Witt, C. S. et al. | "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?" | 2020 | 🟢 | 🟢 | 🟢 | Shows independent PPO (IPPO) is a strong cooperative-MARL baseline; the algorithm closest to this thesis's current chained single-agent architecture. [arXiv](https://arxiv.org/abs/2011.09533) |
| [Yu2022MAPPO](sources/Citations/Yu2022MAPPO.md) | Yu, C. et al. | "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games" (MAPPO) | 2022 | 🟢 | 🟢 | 🟢 | CTDE-based MAPPO; the joint-training alternative to this thesis's chained-single-agent pipeline. [arXiv](https://arxiv.org/abs/2103.01955) |
| [Gronauer2022MARLSurvey](sources/Citations/Gronauer2022MARLSurvey.md) | Gronauer, S. & Diepold, K. | "Multi-agent deep reinforcement learning: a survey" | 2022 | 🔴 | 🟢 | 🟢 | Broad MARL survey covering CTDE, non-stationarity, and cooperative reward/credit-assignment strategies. [DOI](https://doi.org/10.1007/s10462-021-09996-w) |

---

## Deep RL for Robotics

### Preprints & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Kalashnikov2018](sources/reinforcement_learning/Papers/Kalashnikov2018.pdf) | Kalashnikov, D. et al. | "QT-Opt: Scalable Deep RL for Vision-Based Robotic Manipulation" | 2018 | 🔴 | 🟡 | 🟢 | Distributed off-policy Q-learning for vision-based grasping at scale. Context for large-scale RL approaches. [arXiv](https://arxiv.org/abs/1806.10293) |
| [Shianifar2025](sources/reinforcement_learning/Papers/Shianifar2025.pdf) | Shianifar, J. et al. | "Optimizing Deep RL for Adaptive Robotic Arm Control" | 2025 | 🟡 | 🟡 | 🟡 | Hyper-parameter optimization for deep RL robotic arm controllers. Relevant for training configuration choices. [DOI](https://doi.org/10.1007/978-3-031-73058-0_24) |

---


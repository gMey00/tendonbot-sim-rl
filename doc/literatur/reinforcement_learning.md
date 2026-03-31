# Reinforcement Learning

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Foundations & Textbooks

### Books

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| SuttonBarto2018 | Sutton, R. S. & Barto, A. G. | *Reinforcement Learning: An Introduction* (2nd ed.) | 2018 | 🟡 | 🟢 | 🟢 | Definitive RL textbook covering MDPs, dynamic programming, policy-gradient, and actor-critic methods. Foundational reference for all RL chapters. [MIT Press](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) |
| Puterman1994 | Puterman, M. L. | *Markov Decision Processes: Discrete Stochastic Dynamic Programming* | 1994 | 🔴 | 🟢 | 🟢 | Rigorous mathematical treatment of MDPs. Cited for formal MDP definition in the background chapter. [DOI](https://doi.org/10.1002/9780470316887) |

### Book Chapters

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Spaan2012 | Spaan, M. T. J. | "Partially Observable Markov Decision Processes" | 2012 | 🔴 | 🟡 | 🟢 | POMDP theory. Background for discussing partial observability in cloth manipulation tasks. [PDF](https://www.st.ewi.tudelft.nl/mtjspaan/pub/Spaan12pomdp.pdf) |

### Survey Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Kober2013 | Kober, J. et al. | "Reinforcement Learning in Robotics: A Survey" | 2013 | 🔴 | 🟢 | 🟢 | Comprehensive survey of RL applications in robotics. Key background reference for situating the thesis in the RL-robotics landscape. [DOI](https://doi.org/10.1177/0278364913495721) |

---

## Policy Gradient & PPO

### Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Schulman2017PPO | Schulman, J. et al. | "Proximal Policy Optimization Algorithms" | 2017 | 🟢 | 🟢 | 🟢 | Introduces PPO, the primary RL algorithm used in this project. Core method reference. [arXiv](https://arxiv.org/abs/1707.06347) |
| Schulman2015GAE | Schulman, J. et al. | "High-Dimensional Continuous Control Using Generalized Advantage Estimation" | 2015 | 🟢 | 🟢 | 🟢 | GAE for variance reduction in policy gradients. Used in the PPO implementation. [arXiv](https://arxiv.org/abs/1506.02438) |

---

## Model-Based RL

### Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Nagabandi2017 | Nagabandi, A. et al. | "Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning" | 2017 | 🔴 | 🟡 | 🟢 | Hybrid model-based/model-free RL approach. Background for alternative learning paradigms. [arXiv](https://arxiv.org/abs/1708.02596) |
| Clavera2018 | Clavera, I. et al. | "Model-Based Reinforcement Learning via Meta-Policy Optimization" | 2018 | 🔴 | 🟡 | 🟢 | Meta-learning for model-based RL. Background for adaptive dynamics models. [arXiv](https://arxiv.org/abs/1809.05214) |
| Wang2019MBRL | Wang, T. et al. | "Benchmarking Model-Based Reinforcement Learning" | 2019 | 🔴 | 🟡 | 🟢 | Unified benchmark for model-based RL evaluation. Context for method comparison. [arXiv](https://arxiv.org/abs/1907.02057) |

---

## Curriculum Learning & Reward Shaping

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Bengio2009 | Bengio, Y. et al. | "Curriculum Learning" | 2009 | 🟢 | 🟢 | 🟢 | Foundational work on curriculum learning. Directly used in the training pipeline progression (reach → pick → sort). [DOI](https://doi.org/10.1145/1553374.1553380) |
| Ng1999 | Ng, A. Y. et al. | "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping" | 1999 | 🟢 | 🟢 | 🟢 | Theoretical foundation for potential-based reward shaping used in the environment reward functions. [Link](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html) |
| Asmuth2008 | Asmuth, J. et al. | "Potential-based Shaping in Model-based Reinforcement Learning" | 2008 | 🟡 | 🟡 | 🟢 | Extends reward shaping to model-based RL. Background for reward design. [PDF](https://cdn.aaai.org/AAAI/2008/AAAI08-096.pdf) |

---

## Benchmarks & Frameworks

### Journal Articles & Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Brockman2016 | Brockman, G. et al. | "OpenAI Gym" | 2016 | 🟡 | 🟡 | 🟢 | Standard RL environment interface. Isaac Lab wraps environments in the Gym API. [arXiv](https://arxiv.org/abs/1606.01540) |
| SerranoMunoz2023 | Serrano-Muñoz, A. et al. | "skrl: Modular and Configurable Library for Reinforcement Learning" | 2023 | 🟢 | 🟢 | 🟢 | The RL training library used in this project. Provides PPO, SAC, and other algorithms with Isaac Lab integration. [JMLR](https://jmlr.org/papers/v24/23-0112.html) |
| Mahmood2018 | Mahmood, A. R. et al. | "Benchmarking RL Algorithms on Real-World Robots" | 2018 | 🔴 | 🟡 | 🟢 | First large-scale RL benchmark on physical robots. Highlights the sim-to-real gap. [arXiv](https://arxiv.org/abs/1809.07731) |

---

## Multi-Agent & Cooperative RL

### Journal Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Uchibe2018 | Uchibe, E. | "Cooperative and Competitive Reinforcement and Imitation Learning for a Mixture of Heterogeneous Learning Modules" | 2018 | 🟡 | 🟢 | 🟢 | Framework for cooperative/competitive multi-agent learning. Relevant for the master thesis multi-agent textile sorting setup. [DOI](https://doi.org/10.3389/fnbot.2018.00061) |

---

## Deep RL for Robotics

### Preprints & Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Heess2017 | Heess, N. et al. | "Emergence of Locomotion Behaviours in Rich Environments" | 2017 | 🔴 | 🟡 | 🟢 | Locomotion emergence via deep RL in rich environments. Demonstrates environment complexity driving skill diversity. [arXiv](https://arxiv.org/abs/1707.02286) |
| Kalashnikov2018 | Kalashnikov, D. et al. | "QT-Opt: Scalable Deep RL for Vision-Based Robotic Manipulation" | 2018 | 🔴 | 🟡 | 🟢 | Distributed off-policy Q-learning for vision-based grasping at scale. Context for large-scale RL approaches. [arXiv](https://arxiv.org/abs/1806.10293) |
| OpenAI2019 | OpenAI et al. | "Learning Dexterous In-Hand Manipulation" | 2019 | 🔴 | 🟡 | 🟢 | Domain-randomized sim-to-real for 24-DOF hand manipulation. Benchmark for high-DOF policy transfer. [arXiv](https://arxiv.org/abs/1808.00177) |
| Shianifar2025 | Shianifar, J. et al. | "Optimizing Deep RL for Adaptive Robotic Arm Control" | 2025 | 🟡 | 🟡 | 🟡 | Hyper-parameter optimization for deep RL robotic arm controllers. Relevant for training configuration choices. [DOI](https://doi.org/10.1007/978-3-031-73058-0_24) |

---

## Musculoskeletal Simulation

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Caggiano2022 | Caggiano, V. et al. | "MyoSuite — A Contact-Rich Simulation Suite for Musculoskeletal Motor Control" | 2022 | 🔴 | 🟡 | 🟢 | Musculoskeletal simulation with tendon-like actuation. Related work for tendon-driven robot simulation paradigm comparison. |

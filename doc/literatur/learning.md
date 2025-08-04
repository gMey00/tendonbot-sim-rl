# Annotated Bibliography: Key Works in Reinforcement Learning & Robotics

Below is a curated list of the cited works, each followed by a concise 2–3‑sentence summary highlighting its main contribution and why it matters.

---

### Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*  
**Type:** Book (2nd ed., MIT Press)  
This definitive textbook lays the theoretical and algorithmic foundations of reinforcement learning, covering everything from Markov decision processes and dynamic programming to modern policy‑gradient and actor–critic methods. Widely adopted in academia and industry, it serves as both an entry‑level introduction and a reference for advanced practitioners.

---

### Shianifar, J., Schukat, M., & Mason, K. (2025). “Optimizing Deep Reinforcement Learning for Adaptive Robotic Arm Control”  
**Type:** Book chapter, *PAAMS Collection* (Springer)  
The authors investigate hyper‑parameter and architectural optimizations that boost the sample‑efficiency and real‑time responsiveness of deep RL controllers for robotic arms. Results demonstrate smoother trajectory tracking and faster convergence compared with baseline DDPG and SAC implementations.

---

### Uchibe, E. (2018). “Cooperative and Competitive Reinforcement and Imitation Learning for a Mixture of Heterogeneous Learning Modules”  
**Type:** Journal article, *Frontiers in Neurorobotics* 12:61  
Uchibe proposes a framework in which multiple heterogeneous learning modules—some using reinforcement learning, others imitation—interact cooperatively and competitively to solve complex motor‑control tasks. The study shows that such modular mixtures can speed learning and improve robustness in neurorobotic systems.

---

### Heess, N. et al. (2017). “Emergence of Locomotion Behaviours in Rich Environments”  
**Type:** arXiv preprint (arXiv:1707.02286)  
DeepMind researchers train physically simulated agents in procedurally generated terrains, revealing that diverse locomotion skills (walking, jumping, crawling) emerge without explicit reward shaping. The work underscores the power of rich environments and deep RL to yield naturalistic movement.

---

### Nagabandi, A. et al. (2017). “Neural Network Dynamics for Model‑Based Deep Reinforcement Learning with Model‑Free Fine‑Tuning”  
**Type:** arXiv preprint (arXiv:1708.02596)  
This paper combines learned neural‑network dynamics models with short‑horizon model‑predictive control, then fine‑tunes policies via model‑free RL. The hybrid approach dramatically reduces sample complexity on MuJoCo locomotion tasks while retaining the asymptotic performance of model‑free methods.

---

### Clavera, I. et al. (2018). “Model‑Based Reinforcement Learning via Meta‑Policy Optimization”  
**Type:** arXiv preprint (arXiv:1809.05214)  
The authors introduce a meta‑learning algorithm that lets model‑based RL agents quickly adapt their dynamics models and policies to new tasks with minimal data. Experiments across varied robotic benchmarks illustrate superior transfer and data‑efficiency versus vanilla model‑based or model‑free baselines.

---

### Mahmood, A. R. et al. (2018). “Benchmarking Reinforcement Learning Algorithms on Real‑World Robots”  
**Type:** arXiv preprint (arXiv:1809.07731)  
Providing one of the first large‑scale empirical evaluations of RL algorithms directly on physical robots, this study highlights the sim‑to‑real gap and the importance of wall‑clock training time. It establishes baseline results for TD3, PPO, SAC, and others on tasks such as pushing and reaching.

---

### Kalashnikov, D. et al. (2018). “QT‑Opt: Scalable Deep Reinforcement Learning for Vision‑Based Robotic Manipulation”  
**Type:** arXiv preprint (arXiv:1806.10293)  
QT‑Opt is a distributed, off‑policy Q‑learning framework that leverages massive real‑robot data to learn vision‑based grasping with >90 % success. The paper demonstrates how large‑scale data collection plus deep RL can match—and sometimes surpass—hand‑engineered robotic grasp pipelines.

---

### OpenAI et al. (2019). “Learning Dexterous In‑Hand Manipulation”  
**Type:** arXiv preprint (arXiv:1808.00177)  
Using extensive domain randomization, the team trains policies in simulation that transfer to a 24‑DoF Shadow Dexterous Hand, achieving complex in‑hand object reorientation. The work showcases the viability of RL for high‑degree‑of‑freedom manipulation in the real world.

---

### Wang, T. et al. (2019). “Benchmarking Model‑Based Reinforcement Learning”  
**Type:** arXiv preprint (arXiv:1907.02057)  
Addressing the lack of standards in model‑based RL evaluation, the authors release *MBRL‑Bench*, a unified benchmark suite spanning classical and deep RL models. Their comparative study surfaces trade‑offs in dynamics‑model accuracy, planning horizon, and policy optimization across tasks.


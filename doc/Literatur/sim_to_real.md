# Sim-to-Real Transfer

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Domain Randomization

### Conference Papers & Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Tobin2017](sources/sim_to_real/Papers/Tobin2017.pdf) | Tobin, J. et al. | "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" | 2017 | 🟢 | 🟢 | 🟢 | Introduces domain randomization for sim-to-real transfer. Core technique used in the training pipeline. [DOI](https://doi.org/10.1109/IROS.2017.8202133) / [arXiv](https://arxiv.org/abs/1703.06907) |
| [Peng2018](sources/sim_to_real/Papers/Peng2018.pdf) | Peng, X. B. et al. | "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization" | 2018 | 🟢 | 🟢 | 🟢 | Extends domain randomization to dynamics parameters. Directly applicable to tendon parameter randomization. [arXiv](https://arxiv.org/abs/1710.06537) / [DOI](https://doi.org/10.1109/ICRA.2018.8460528) |
| [Bousmalis2018](sources/sim_to_real/Papers/Bousmalis2018.pdf) | Bousmalis, K. et al. | "Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping" | 2018 | 🔴 | 🟡 | 🟢 | Domain adaptation for grasping. Alternative to randomization for sim-to-real transfer. [DOI](https://doi.org/10.1109/ICRA.2018.8460875) |
| [Sancak2022](sources/sim_to_real/Papers/Sancak2022.pdf) | Sancak, C. et al. | "Sim-to-Real Transfer for Robotic Grasping via Domain Randomization" | 2022 | 🟡 | 🟡 | 🟢 | Empirical study of domain-randomization parameters for grasping transfer. Useful comparator for randomization-budget choices. [DOI](https://doi.org/10.1109/ICRA46639.2022.9811643) |

---

## Reality Gap Analysis

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Jakobi1995](sources/sim_to_real/Papers/Jakobi1995.pdf) | Jakobi, N. et al. | "Noise and the Reality Gap: The Use of Simulation in Evolutionary Robotics" | 1995 | 🔴 | 🟢 | 🟢 | Seminal paper defining the reality gap concept. Historical context for sim-to-real challenges. [DOI](https://doi.org/10.1007/3-540-59496-5_337) |
| [Salvato2021](sources/sim_to_real/Papers/Salvato2021.pdf) | Salvato, E. et al. | "Crossing the Reality Gap: A Survey on Sim-to-Real Transferability of Robot Controllers in RL" | 2021 | 🔴 | 🟢 | 🟢 | Comprehensive survey of sim-to-real methods for RL. Key review for the transfer discussion. [DOI](https://doi.org/10.1109/ACCESS.2021.3126658) |

---

## Policy Distillation

### Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Rusu2015](sources/sim_to_real/Papers/Rusu2015.pdf) | Rusu, A. A. et al. | "Policy Distillation" | 2015 | 🔴 | 🟡 | 🟢 | Compressing RL policies for deployment. Context for future policy optimization. [arXiv](https://arxiv.org/abs/1511.06295) |

---

## Contact-Rich Sim-to-Real Transfer

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Narang2022](sources/sim_to_real/Papers/Narang2022.pdf) | Narang, Y. et al. | "Factory: Fast Contact for Robotic Assembly" | 2022 | 🟡 | 🟢 | 🟢 | GPU-accelerated contact-rich assembly tasks in simulation. Context for contact-heavy task sim-to-real. [DOI](https://doi.org/10.15607/RSS.2022.XVIII.065) |
| [Tang2023](sources/sim_to_real/Papers/Tang2023.pdf) | Tang, B. et al. | "IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality" | 2023 | 🟡 | 🟢 | 🟢 | Successful sim-to-real transfer for contact-rich industrial tasks. Key reference for deployment pathway. [DOI](https://doi.org/10.15607/RSS.2023.XIX.069) |
| [Allshire2022](sources/sim_to_real/Papers/Allshire2022.pdf) | Allshire, A. et al. | "Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger" | 2022 | 🟡 | 🟡 | 🟢 | GPU sim-to-real for dexterous manipulation. Context for high-DOF transfer. [DOI](https://doi.org/10.1109/IROS47612.2022.9981586) |
| [Handa2023](sources/sim_to_real/Papers/Handa2023.pdf) | Handa, A. et al. | "DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality" | 2023 | 🔴 | 🟡 | 🟢 | Aggressive sim-to-real for in-hand manipulation. Demonstrates transfer feasibility for complex tasks. [DOI](https://doi.org/10.1109/ICRA48891.2023.10161417) |
| [Albardaner2024](sources/sim_to_real/Papers/Albardaner2024.pdf) | Albardaner, A. et al. | "Sim-to-Real Gap in RL: Use Case with TIAGo and Isaac Sim/Gym" | 2024 | 🟡 | 🟡 | 🟢 | Concrete sim-to-real case study on a TIAGo robot using Isaac Sim/Gym. Direct comparator for the project's Isaac-Lab pipeline. [DOI](https://doi.org/10.1007/978-3-031-58676-7_25) |

---

## Industrial / Digital-Twin Sim-to-Real

### Conference & Journal Papers / Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Nambiar2025](sources/sim_to_real/Papers/Nambiar2025.pdf) | Nambiar, V. et al. | "Digital Twin-Enabled Adaptive Robotics: Leveraging LLMs in Isaac Sim for Unstructured Environments" | 2025 | 🔴 | 🟡 | 🟢 | LLM-augmented digital-twin pipeline in Isaac Sim. Forward-looking reference for unstructured-environment adaptation. [DOI](https://doi.org/10.3390/machines13070620) |
| [Zhou2023AICPS](sources/sim_to_real/Papers/Zhou2023AICPS.pdf) | Zhou, Z. et al. | "Towards Building AI-CPS with NVIDIA Isaac Sim: An Industrial Benchmark and Case Study for Robotics Manipulation" | 2023 | 🟡 | 🟡 | 🟢 | Industrial AI-CPS benchmark on Isaac Sim. Good reference for Isaac-Sim industrial deployment patterns. [arXiv](https://arxiv.org/abs/2308.00055) |

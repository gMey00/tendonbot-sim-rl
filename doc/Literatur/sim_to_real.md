# Sim-to-Real Transfer

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Domain Randomization

### Conference Papers & Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Tobin2017 | Tobin, J. et al. | "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" | 2017 | 🟢 | 🟢 | 🟢 | Introduces domain randomization for sim-to-real transfer. Core technique used in the training pipeline. [DOI](https://doi.org/10.1109/IROS.2017.8202133) / [arXiv](https://arxiv.org/abs/1703.06907) |
| Peng2018 | Peng, X. B. et al. | "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization" | 2018 | 🟢 | 🟢 | 🟢 | Extends domain randomization to dynamics parameters. Directly applicable to tendon parameter randomization. [arXiv](https://arxiv.org/abs/1710.06537) / [DOI](https://doi.org/10.1109/ICRA.2018.8460528) |
| Bousmalis2018 | Bousmalis, K. et al. | "Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping" | 2018 | 🔴 | 🟡 | 🟢 | Domain adaptation for grasping. Alternative to randomization for sim-to-real transfer. [DOI](https://doi.org/10.1109/ICRA.2018.8460875) |

---

## Reality Gap Analysis

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Jakobi1995 | Jakobi, N. et al. | "Noise and the Reality Gap: The Use of Simulation in Evolutionary Robotics" | 1995 | 🔴 | 🟢 | 🟢 | Seminal paper defining the reality gap concept. Historical context for sim-to-real challenges. [DOI](https://doi.org/10.1007/3-540-59496-5_337) |
| Salvato2021 | Salvato, E. et al. | "Crossing the Reality Gap: A Survey on Sim-to-Real Transferability of Robot Controllers in RL" | 2021 | 🔴 | 🟢 | 🟢 | Comprehensive survey of sim-to-real methods for RL. Key review for the transfer discussion. [DOI](https://doi.org/10.1109/ACCESS.2021.3126658) |

---

## Policy Distillation

### Preprints

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Rusu2015 | Rusu, A. A. et al. | "Policy Distillation" | 2015 | 🔴 | 🟡 | 🟢 | Compressing RL policies for deployment. Context for future policy optimization. [arXiv](https://arxiv.org/abs/1511.06295) |

---

## Contact-Rich Sim-to-Real Transfer

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| Narang2022 | Narang, Y. et al. | "Factory: Fast Contact for Robotic Assembly" | 2022 | 🟡 | 🟢 | 🟢 | GPU-accelerated contact-rich assembly tasks in simulation. Context for contact-heavy task sim-to-real. [DOI](https://doi.org/10.15607/RSS.2022.XVIII.065) |
| Tang2023 | Tang, B. et al. | "IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality" | 2023 | 🟡 | 🟢 | 🟢 | Successful sim-to-real transfer for contact-rich industrial tasks. Key reference for deployment pathway. [DOI](https://doi.org/10.15607/RSS.2023.XIX.069) |
| Allshire2022 | Allshire, A. et al. | "Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger" | 2022 | 🟡 | 🟡 | 🟢 | GPU sim-to-real for dexterous manipulation. Context for high-DOF transfer. [DOI](https://doi.org/10.1109/IROS47612.2022.9981586) |
| Handa2023 | Handa, A. et al. | "DeXtreme: Transfer of Agile In-hand Manipulation from Simulation to Reality" | 2023 | 🔴 | 🟡 | 🟢 | Aggressive sim-to-real for in-hand manipulation. Demonstrates transfer feasibility for complex tasks. [DOI](https://doi.org/10.1109/ICRA48891.2023.10161417) |

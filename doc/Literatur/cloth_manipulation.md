# Cloth & Deformable Object Manipulation

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Surveys & Reviews

### Review Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Longhini2025](sources/cloth_manipulation/Reviews/Longhini2025.pdf) | Longhini, A. et al. | "A Review of Robotic Cloth Manipulation" | 2025 | 🟡 | 🟢 | 🟢 | Comprehensive review of robotic cloth manipulation covering perception, planning, and control. Central review for the master thesis literature chapter. [DOI](https://doi.org/10.1146/annurev-control-022723-033252) |
| [Kadi2023](sources/cloth_manipulation/Reviews/Kadi2023.pdf) | Kadi, H. A. & Terzić, K. | "Data-Driven Robotic Manipulation of Cloth-like Deformable Objects" | 2023 | 🟡 | 🟢 | 🟢 | Survey of data-driven methods for cloth-like deformable object manipulation. Covers DL, RL, and hybrid approaches. [DOI](https://doi.org/10.3390/s23052389) |
| [Gu2023](sources/cloth_manipulation/Reviews/Gu2023.pdf) | Gu, F. et al. | "A Survey on Robotic Manipulation of Deformable Objects: Recent Advances, Open Challenges and New Frontiers" | 2023 | 🔴 | 🟡 | 🟢 | Recent survey covering advances and open challenges. Complements Longhini2025 with a broader scope. [arXiv](https://arxiv.org/abs/2312.10419) |

---

## Garment Benchmarks & Datasets

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Lu2024GarmentLab](sources/cloth_manipulation/Papers/Lu2024GarmentLab.pdf) | Lu, H. et al. | "GarmentLab: A Unified Simulation and Benchmark for Garment Manipulation" | 2024 | 🟡 | 🟢 | 🟢 | Standardized garment manipulation benchmark with parameterized garments and diverse tasks. Key benchmark reference for the master thesis. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/15f80ec0fed53885d2ca6272edb96ede-Paper-Conference.pdf) |
| [Hu2025RGBench](sources/cloth_manipulation/Papers/Hu2025RGBench.pdf) | Hu, W. et al. | "Real Garment Benchmark (RGBench)" | 2025 | 🔴 | 🟡 | 🟢 | High-fidelity garment manipulation benchmark with scalable simulator. Context for benchmark comparison. [arXiv](https://arxiv.org/abs/2511.06434) |
| [GarciaCamacho2020](sources/cloth_manipulation/Papers/GarciaCamacho2020.pdf) | Garcia-Camacho, I. et al. | "Benchmarking Bimanual Cloth Manipulation" | 2020 | 🟡 | 🟢 | 🟢 | Bimanual cloth manipulation benchmark. Relevant for the dual-arm textile sorting setup in the master thesis. [PDF](https://ral-si.github.io/cloth-benchmark/Benchmark_for_cloth_manipulation.pdf) |
| [BlancoMulero2024](sources/cloth_manipulation/Papers/BlancoMulero2024.pdf) | Blanco-Mulero, D. et al. | "Benchmarking the Sim-to-Real Gap in Cloth Manipulation" | 2024 | 🟡 | 🟢 | 🟢 | Directly quantifies the sim-to-real gap for cloth. Critical for the master thesis sim-to-real discussion. [Link](https://upcommons.upc.edu/server/api/core/bitstreams/ee091e2a-883e-4240-9a18-bf2945e56c38/content) |
| [Wang2025DexGarmentLab](sources/cloth_manipulation/Papers/Wang2025DexGarmentLab.pdf) | Wang, X. et al. | "DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy" | 2025 | 🟡 | 🟢 | 🟢 | Dexterous garment manipulation environment with a generalizable policy extending GarmentLab. Directly relevant for policy generalization across garments. [arXiv](https://arxiv.org/abs/2505.11032) |

---

## Simulation Environments for Cloth

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Lin2021SoftGym](sources/cloth_manipulation/Papers/Lin2021SoftGym.pdf) | Lin, X. et al. | "SoftGym: Benchmarking Deep RL for Deformable Object Manipulation" | 2021 | 🔴 | 🟢 | 🟢 | RL benchmark for deformable objects using FleX. Key comparator for simulation environment choice. [PMLR](https://proceedings.mlr.press/v155/lin21a.html) |

---

## Folding, Flattening & Smoothing

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Seita2020](sources/cloth_manipulation/Papers/Seita2020.pdf) | Seita, D. et al. | "Deep Imitation Learning of Sequential Fabric Smoothing From an Algorithmic Supervisor" | 2020 | 🔴 | 🟡 | 🟢 | Imitation learning for fabric smoothing. Alternative to RL for cloth manipulation. [DOI](https://doi.org/10.1109/IROS45743.2020.9341608) |
| [Wu2020Deformable](sources/cloth_manipulation/Papers/Wu2020Deformable.pdf) | Wu, Y. et al. | "Learning to Manipulate Deformable Objects without Demonstrations" | 2020 | 🟡 | 🟡 | 🟢 | Demonstration-free RL for deformable object manipulation. Context for self-supervised cloth manipulation. [arXiv](https://arxiv.org/abs/1910.13439) |

---

## Grasping & Unfolding

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Ha2021FlingBot](sources/cloth_manipulation/Papers/Ha2021FlingBot.pdf) | Ha, H. & Song, S. | "FlingBot: The Unreasonable Effectiveness of Dynamic Manipulation for Cloth Unfolding" | 2021 | 🟡 | 🟢 | 🟢 | Dynamic fling actions for cloth unfolding. Innovative manipulation primitive relevant for the sorting pipeline. [OpenReview](https://openreview.net/pdf?id=0QJeE5hkyFZ) |
| [Zeng2019TossingBot](sources/Citations/Zeng2019TossingBot.md) | Zeng, A. et al. | "TossingBot: Learning to Throw Arbitrary Objects with Residual Physics" | 2019 | 🟢 | 🟢 | 🟢 | Goal-conditioned pick-and-place/throw policy, generalizing across target positions via a learned residual. Direct precedent for shirt_distribute's goal-conditioned bin-selection design (master thesis §2.5). [arXiv](https://arxiv.org/abs/1903.11239) |

---


# Cloth & Deformable Object Manipulation

[← Literature Overview](README.md) · [← Project README](../../README.md)

---

## Surveys & Reviews

### Review Articles

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Longhini2025](sources/cloth_manipulation/Review_Articles/Longhini2025.pdf) | Longhini, A. et al. | "A Review of Robotic Cloth Manipulation" | 2025 | 🟡 | 🟢 | 🟢 | Comprehensive review of robotic cloth manipulation covering perception, planning, and control. Central review for the master thesis literature chapter. [DOI](https://doi.org/10.1146/annurev-control-022723-033252) |
| [Kadi2023](sources/cloth_manipulation/Review_Articles/Kadi2023.pdf) | Kadi, H. A. & Terzić, K. | "Data-Driven Robotic Manipulation of Cloth-like Deformable Objects" | 2023 | 🟡 | 🟢 | 🟢 | Survey of data-driven methods for cloth-like deformable object manipulation. Covers DL, RL, and hybrid approaches. [DOI](https://doi.org/10.3390/s23052389) |
| [Sanchez2018](sources/cloth_manipulation/Review_Articles/Sanchez2018.pdf) | Sanchez, J. et al. | "Robotic Manipulation and Sensing of Deformable Objects: A Survey" | 2018 | 🔴 | 🟢 | 🟢 | Broader survey of deformable object manipulation in domestic and industrial settings. Context for the research gap. [DOI](https://doi.org/10.1177/0278364918779698) |
| [Gu2023](sources/cloth_manipulation/Review_Articles/Gu2023.pdf) | Gu, F. et al. | "A Survey on Robotic Manipulation of Deformable Objects: Recent Advances, Open Challenges and New Frontiers" | 2023 | 🔴 | 🟡 | 🟢 | Recent survey covering advances and open challenges. Complements Longhini2025 with a broader scope. [arXiv](https://arxiv.org/abs/2312.10419) |

---

## Garment Benchmarks & Datasets

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Lu2024GarmentLab](sources/cloth_manipulation/Conference_%26_Journal_Papers/Lu2024GarmentLab.pdf) | Lu, H. et al. | "GarmentLab: A Unified Simulation and Benchmark for Garment Manipulation" | 2024 | 🟡 | 🟢 | 🟢 | Standardized garment manipulation benchmark with parameterized garments and diverse tasks. Key benchmark reference for the master thesis. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/15f80ec0fed53885d2ca6272edb96ede-Paper-Conference.pdf) |
| [Hu2025RGBench](sources/cloth_manipulation/Conference_%26_Journal_Papers/Hu2025RGBench.pdf) | Hu, W. et al. | "Real Garment Benchmark (RGBench)" | 2025 | 🔴 | 🟡 | 🟢 | High-fidelity garment manipulation benchmark with scalable simulator. Context for benchmark comparison. [arXiv](https://arxiv.org/abs/2511.06434) |
| [GarciaCamacho2020](sources/cloth_manipulation/Conference_%26_Journal_Papers/GarciaCamacho2020.pdf) | Garcia-Camacho, I. et al. | "Benchmarking Bimanual Cloth Manipulation" | 2020 | 🟡 | 🟢 | 🟢 | Bimanual cloth manipulation benchmark. Relevant for the dual-arm textile sorting setup in the master thesis. [PDF](https://ral-si.github.io/cloth-benchmark/Benchmark_for_cloth_manipulation.pdf) |
| [DeGusseme2026](sources/cloth_manipulation/Conference_%26_Journal_Papers/DeGusseme2026.pdf) | De Gusseme, V.-L. et al. | "ICRA 2024 Cloth Competition: Dataset and Benchmark for Robotic Cloth Unfolding Grasp Selection" | 2026 | 🔴 | 🟡 | 🟢 | Competition-based benchmark for cloth unfolding grasp selection. Latest benchmark results. [DOI](https://doi.org/10.1177/02783649251414885) |
| [BlancoMulero2024](sources/cloth_manipulation/Conference_%26_Journal_Papers/BlancoMulero2024.pdf) | Blanco-Mulero, D. et al. | "Benchmarking the Sim-to-Real Gap in Cloth Manipulation" | 2024 | 🟡 | 🟢 | 🟢 | Directly quantifies the sim-to-real gap for cloth. Critical for the master thesis sim-to-real discussion. [Link](https://upcommons.upc.edu/server/api/core/bitstreams/ee091e2a-883e-4240-9a18-bf2945e56c38/content) |
| [Wang2025DexGarmentLab](sources/cloth_manipulation/Conference_%26_Journal_Papers/Wang2025DexGarmentLab.pdf) | Wang, X. et al. | "DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy" | 2025 | 🟡 | 🟢 | 🟢 | Dexterous garment manipulation environment with a generalizable policy extending GarmentLab. Directly relevant for policy generalization across garments. [arXiv](https://arxiv.org/abs/2505.11032) |

---

## Simulation Environments for Cloth

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Lin2021SoftGym](sources/cloth_manipulation/Conference_Papers/Lin2021SoftGym.pdf) | Lin, X. et al. | "SoftGym: Benchmarking Deep RL for Deformable Object Manipulation" | 2021 | 🔴 | 🟢 | 🟢 | RL benchmark for deformable objects using FleX. Key comparator for simulation environment choice. [PMLR](https://proceedings.mlr.press/v155/lin21a.html) |

---

## Folding, Flattening & Smoothing

### Conference & Journal Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [MaitinShepard2010](sources/cloth_manipulation/Conference_%26_Journal_Papers/MaitinShepard2010.pdf) | Maitin-Shepard, J. et al. | "Cloth Grasp Point Detection based on Multiple-View Geometric Cues with Application to Robotic Towel Folding" | 2010 | 🔴 | 🟡 | 🟢 | Pioneering work on vision-based cloth grasp point detection for towel folding. Historical context. [PDF](https://people.eecs.berkeley.edu/~pabbeel/papers/Maitin-ShepardCusumano-TownerLeiAbbeel_ICRA2010.pdf) |
| [Miller2012](sources/cloth_manipulation/Conference_%26_Journal_Papers/Miller2012.pdf) | Miller, S. et al. | "A Geometric Approach to Robotic Laundry Folding" | 2012 | 🔴 | 🟡 | 🟢 | Geometric approach to laundry folding. Background for cloth manipulation strategies. [DOI](https://doi.org/10.1177/0278364911430417) |
| [Li2015](sources/cloth_manipulation/Conference_%26_Journal_Papers/Li2015.pdf) | Li, Y. et al. | "Folding Deformable Objects using Predictive Simulation and Trajectory Optimization" | 2015 | 🔴 | 🟡 | 🟢 | Simulation-based trajectory optimization for folding. Context for planning-based approaches. [PDF](https://www.cs.columbia.edu/~yonghao/iros15/li-folding-deformable-objects.pdf) |
| [Doumanoglou2016](sources/cloth_manipulation/Conference_%26_Journal_Papers/Doumanoglou2016.pdf) | Doumanoglou, A. et al. | "Folding Clothes Autonomously: A Complete Pipeline" | 2016 | 🔴 | 🟡 | 🟢 | End-to-end autonomous clothing folding pipeline. Context for complete system design. [PDF](https://labicvl.github.io/docs/pubs/Andreas_TR_2016.pdf) |
| [Petrik2019](sources/cloth_manipulation/Conference_%26_Journal_Papers/Petrik2019.pdf) | Petrík, V. & Kyrki, V. | "Feedback-based Fabric Strip Folding" | 2019 | 🔴 | 🟡 | 🟢 | Feedback-based folding with visual sensing. Context for closed-loop manipulation. [arXiv](https://arxiv.org/abs/1904.01298) |
| [Seita2020](sources/cloth_manipulation/Conference_%26_Journal_Papers/Seita2020.pdf) | Seita, D. et al. | "Deep Imitation Learning of Sequential Fabric Smoothing From an Algorithmic Supervisor" | 2020 | 🔴 | 🟡 | 🟢 | Imitation learning for fabric smoothing. Alternative to RL for cloth manipulation. [DOI](https://doi.org/10.1109/IROS45743.2020.9341608) |
| [Hoque2020](sources/cloth_manipulation/Conference_%26_Journal_Papers/Hoque2020.pdf) | Hoque, R. et al. | "VisuoSpatial Foresight for Multi-Step, Multi-Task Fabric Manipulation" | 2020 | 🔴 | 🟡 | 🟢 | Visual foresight for multi-step fabric manipulation. Context for model-based cloth manipulation. [PDF](https://www.roboticsproceedings.org/rss16/p034.pdf) |
| [Lee2021](sources/cloth_manipulation/Conference_%26_Journal_Papers/Lee2021.pdf) | Lee, R. et al. | "Learning Arbitrary-Goal Fabric Folding with One Hour of Real Robot Experience" | 2021 | 🔴 | 🟡 | 🟢 | Data-efficient fabric folding from real robot data. Context for sample efficiency discussion. [PDF](https://proceedings.mlr.press/v155/lee21a/lee21a.pdf) |
| [Ganapathi2021](sources/cloth_manipulation/Conference_%26_Journal_Papers/Ganapathi2021.pdf) | Ganapathi, A. et al. | "Learning Dense Visual Correspondences in Simulation to Smooth and Fold Real Fabrics" | 2021 | 🔴 | 🟡 | 🟢 | Sim-to-real transfer for fabric manipulation via dense correspondences. Context for transfer approach. [PDF](https://autolab.berkeley.edu/assets/publications/media/2021-ICRA-Ganapathi-Fabric-Descriptors-Camera-Ready.pdf) |
| [Wu2020Deformable](sources/cloth_manipulation/Conference_%26_Journal_Papers/Wu2020Deformable.pdf) | Wu, Y. et al. | "Learning to Manipulate Deformable Objects without Demonstrations" | 2020 | 🟡 | 🟡 | 🟢 | Demonstration-free RL for deformable object manipulation. Context for self-supervised cloth manipulation. [arXiv](https://arxiv.org/abs/1910.13439) |

---

## Grasping & Unfolding

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Ha2021FlingBot](sources/cloth_manipulation/Conference_Papers/Ha2021FlingBot.pdf) | Ha, H. & Song, S. | "FlingBot: The Unreasonable Effectiveness of Dynamic Manipulation for Cloth Unfolding" | 2021 | 🟡 | 🟢 | 🟢 | Dynamic fling actions for cloth unfolding. Innovative manipulation primitive relevant for the sorting pipeline. [OpenReview](https://openreview.net/pdf?id=0QJeE5hkyFZ) |
| [Canberk2022](sources/cloth_manipulation/Conference_Papers/Canberk2022.pdf) | Canberk, A. et al. | "Cloth Funnels: Canonicalized-Alignment for Multi-Purpose Garment Manipulation" | 2022 | 🟡 | 🟡 | 🟢 | Canonicalized garment alignment for multi-purpose manipulation. Context for garment state representation. [arXiv](https://arxiv.org/abs/2210.09347) |

---

## Dressing & Sorting

### Conference Papers

| Ref | Authors | Title | Year | Impl. | Thesis | Cred. | Summary |
|-----|---------|-------|------|-------|--------|-------|---------|
| [Wang2023](sources/cloth_manipulation/Conference_Papers/Wang2023.pdf) | Wang, Y. et al. | "One Policy to Dress Them All: Learning to Dress People with Diverse Poses and Garments" | 2023 | 🔴 | 🟡 | 🟢 | Generalizable dressing policy across poses and garments. Context for garment manipulation generalization. [PDF](https://www.roboticsproceedings.org/rss19/p008.pdf) |

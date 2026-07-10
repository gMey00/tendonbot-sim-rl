# Alex Tier3 Application — Drop-in Field Values

[← Alex Cluster Docs](README.md) · [← Documentation index](../README.md)

**Form:** Application for FAU/Tier3 access to GPGPU cluster Alex
**URL:** https://hpc.fau.de/tier3-access-to-alex/
**Prepared:** 18.06.2026

> Pre-cleared with NHR@FAU support (Martin Mayr, 17.06.2026): submit this standard
> form, select the closest matching GPU type, and describe the RTX 6000 PRO
> requirement in the remarks field. Application to be submitted personally by the
> account holder, not centrally via FAPS.

---

## Your name

```
Georg Meyer
```

## Your email

```
georg.r.meyer@fau.de
```

## Your HPC account

```
iwfa131h
```

## Targeted hardware

Select in dropdown:

```
GPGPU cluster 'Alex' / Nvidia A40 GPGPUs (48 GB, 37 TFlop/s single precision)
```

> The A40 is selected **only because it is the closest available option to the
> actual target**, the new RTX 6000 PRO (Blackwell) partition, which is not yet
> selectable in this form (confirmed too recent by support on 17.06.2026). The
> RTX 6000 PRO is the requested hardware. See justification below.

## Justify the hardware selection (Put into one of the fields below)

```
PRIMARY REQUEST: access to the new RTX 6000 PRO (Blackwell) partition on Alex.

This is the ideal hardware for the workload. NVIDIA officially lists the
RTX PRO 6000 Blackwell (48 GB VRAM) as the "ideal" GPU tier for NVIDIA Isaac
Sim 5.1.0 — the simulation stack this thesis is built on. It provides the
4th-generation RT cores, large VRAM, and Blackwell-generation throughput that
the cloth simulation, multi-agent training, and on-GPU camera rendering of the
master's thesis demand. Since the RTX 6000 PRO is too recent to appear in the
form's dropdown, A40 was selected there as the nearest match only.

Isaac Sim / Isaac Lab requires hardware ray-tracing (RT cores). 
The official requirements explicitly name the A100 (and H100) as UNSUPPORTED 
because they lack RT cores. The A100 partition is therefore unusable for this 
project under any configuration.
```

## Yearly compute time requirements

> Format requested by the form: GPU hours per GPU type; type of usage
> (single GPU / multi-GPU in a node / multi-node); (#GPUs per job) × (walltime per job) × (#jobs)

```
Total request: ~3,000 GPU-hours over the thesis period (Jun–Nov 2026),
on the RTX 6000 PRO partition (or A40 as fallback).

Usage type: single-GPU jobs. Isaac Lab scales across thousands of parallel
environments on ONE GPU. This project is designed around single-GPU runs. 
Typical job: 1 GPU × 6–24 h walltime, batch (headless), via Slurm. 
Independent runs (seeds, ablation variants) are submitted as separate single-GPU jobs.

Breakdown (1 GPU per job throughout):
  - Reward-function development + exploratory training: the cloth-sorting and
    multi-agent reward structures are designed from scratch (no reference
    implementation). Discovering a converging reward requires many shorter,
    mostly-failing partial training runs before the final seeded runs.
                                              ≈ 500 GPU-h
  - Cloth-integrated sorting training (PBD particle cloth, 5 seeds, PD vs.
    tendon): ~10 jobs × ~30 h                 ≈ 300 GPU-h
  - Multi-agent coordination (MAPPO, 2 arms + cloth, 5 seeds):
    ~5 jobs × ~40 h                           ≈ 200 GPU-h
  - Ablation studies (with/without stretching, classification, single vs.
    multi-agent; 3 seeds): ~12 jobs × ~25 h   ≈ 300 GPU-h
  - Domain-randomization / robustness sweep (3 seeds):
    ~9 jobs × ~20 h                           ≈ 180 GPU-h
  - Rigid single-agent sorting refinement (5 seeds):
    ~5 jobs × ~8 h                            ≈  40 GPU-h
  - Damage-classifier (ResNet-18) training on synthetic data:
    ~3 jobs × ~6 h                            ≈  18 GPU-h
  - General debugging / re-runs of working tasks (~40% of productive runs):
                                              ≈ 415 GPU-h
  TOTAL                                       ≈ 2,450 GPU-h  → request ~2,500 GPU-h
```

## Required software / applications

```
- NVIDIA Isaac Sim 5.1.0 (Omniverse Kit) — requires RT cores
- NVIDIA Isaac Lab (latest main) — GPU-parallel RL framework on Isaac Sim
- PhysX 5 (TGS solver; PBD/XPBD particle cloth)
- Python 3.11 (Conda environment)
- PyTorch (CUDA) — policy training
- skrl ≥ 1.4.3 — PPO and MAPPO/IPPO implementations
- CUDA-capable NVIDIA driver (≥ 580.65.06 for Isaac Sim 5.1.0 on Linux)
- Containerized via Apptainer if required by the cluster.

All workloads are GPU-resident, single-GPU, headless batch runs.
```

## Brief description of the planned work

```
Master's thesis at the Institute for Factory Automation and Production Systems
(FAPS, Prof. Franke): "Multi-Agent Reinforcement Learning for Condition-Based
Textile Sorting with a Tendon-Driven and Collaborative Robot System."

Building on the completed project thesis (simulation and RL control of a custom
5-DOF tendon-driven manipulator for conveyor pick-and-place, validated in Isaac
Sim), the master's thesis extends the system to:

1. Selective textile (T-shirt) sorting from a moving conveyor, replacing rigid
   cubes with deformable garments simulated via PhysX PBD particle cloth.
2. A second collaborative arm that stretches garments flat for inspection.
3. A camera-based damage classifier (CNN) assessing garment condition.
4. Multi-agent reinforcement learning (MAPPO/IPPO) coordinating the two robots
   into an end-to-end retrieve–inspect–sort pipeline.
5. Domain randomization and a ROS 2 integration pathway for later sim-to-real
   transfer.

Training uses GPU-parallel PPO/MAPPO across thousands of simultaneous
environments on a single GPU. Cloth simulation, multi-agent coordination, and
on-GPU camera rendering make these workloads substantially more demanding than
the rigid-body project-thesis tasks, which is why the FAPS-internal workstation
is no longer sufficient and Alex's GPU hardware is needed.
```

## Expected outcome

```
- A validated multi-agent RL pipeline for condition-based textile sorting in
  NVIDIA Isaac Sim, with quantitative evaluation (sorting accuracy by damage
  class, false/missed-pick rates, cycle time, coordination efficiency).
- Trained, multi-seed PPO/MAPPO policies with statistical robustness analysis.
- A camera-based garment damage classifier trained on synthetic data with
  domain randomization.
- Ablation studies isolating the contribution of cloth stretching,
  classification, and multi-agent coordination.
- A documented ROS 2 architecture and sim-to-real readiness assessment.
- A completed master's thesis (30 ECTS) and, where results warrant, a potential
  publication. Resource usage will be acknowledged per NHR@FAU guidelines.
```

## Export Control Regulations

Select:

```
B.Sc./M.Sc. Thesis
```

## Other remarks

```
Submitting on the advice of NHR@FAU support (Martin Mayr, ticket reply
17.06.2026), who confirmed: (a) use this standard Tier3 form, (b) select the
closest matching GPU type, (c) describe the RTX 6000 requirement here, and
(d) submit personally as the account holder.

KEY REQUEST: please grant access to the new RTX 6000 PRO (Blackwell) partition
on Alex, which is not yet selectable in the hardware dropdown above.

Hard technical constraint: my stack (NVIDIA Isaac Sim 5.1.0 / Isaac Lab) requires
GPUs WITH RT cores. NVIDIA's official requirements explicitly list A100 and H100
as unsupported because they lack RT cores.

Account context: my HPC account (iwfa131h) belongs to the project "Studentische
Abschlussarbeiten Tier3 Grundversorgung LS Fertigungsautomatisierung und
Produktionssystematik (Prof. Franke)".
```

---

### Pre-submission checklist

- [ ] Confirm the anti-spam sum from the live form.
- [ ] Double-check that "A40" is still the closest selectable option (in case the
      RTX 6000 PRO option has since been added — if so, select it directly).

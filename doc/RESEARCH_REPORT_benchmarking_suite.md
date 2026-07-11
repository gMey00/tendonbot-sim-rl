# Benchmarking Methodology & Phased Work Plan: Computational Performance of a Simulation-Based RL Pipeline for Robotic Cloth (T-Shirt) Sorting on NVIDIA Isaac Lab

## TL;DR

- **Report two throughputs and two "ideal env counts," never one.** Measure raw simulation throughput (zero-action `env.step`, aggregate env·steps/s after warmup exclusion) *separately* from end-to-end training throughput; the N that maximizes throughput (a systems knee driven by occupancy/bandwidth/contact-buffer overflow) is distinct from the N that maximizes learning progress per wall-clock hour (governed by gradient-noise-scale / critical-batch-size theory, McCandlish et al. 2018). Your Stage-0 32–128 knee must be re-derived with ≥5 repeats, tight confidence intervals, and a physics-sanity gate — and *explained*, not assumed.
- **Treat cloth as a first-class non-rigid scaling axis and benchmark both backends.** Run a vertex-count × env-count 2-D grid for BOTH the PBD particle-cloth and XPBD/FEM surface-deformable backends, holding *physical* garment behavior constant (via SoftGym-style normalized coverage) while varying resolution so you measure compute, not a different garment. The "contact buffer overflow" regime is explainable: PhysX 5 GPU simulation pre-allocates fixed buffers that **cannot** dynamically grow, so at high N collisions are silently dropped unless `gpu_*` buffers are sized up — any throughput number measured in that regime is invalid.
- **Isolate morphology/controller cost by additive decomposition, and make it reproducible.** Fix task + cloth, vary only robot/controller; ground expectations in Featherstone O(n) articulated-body complexity, Khatib operational-space control, and (for the novel tensegrity-wrist target) the super-linear cost of closed-loop/tendon constraints. Follow MLPerf time-to-target reporting and rliable/Henderson statistics: fixed seeds, ≥5 repeats, medians + IQM + bootstrap CIs, full version capture, machine-tagged artifacts comparable across the RTX A6000 (48 GB, 768 GB/s) and Alex RTX PRO 6000 Blackwell (96 GB, 1,792 GB/s).

---

## 1. Executive Summary

This report specifies a rigorous, citable methodology to benchmark the *computational* performance (not task success) of a GPU-parallel RL pipeline for cloth sorting built on Isaac Lab (Isaac Sim 5.1 / PhysX 5). The strategy rests on five pillars:

1. **Separate simulation throughput from end-to-end training throughput.** Report both, but measure them in isolation. The canonical metric is aggregate environment-steps per second (env·steps/s), reported after warmup exclusion over a fixed steady-state window, with a physics-sanity gate. This mirrors the reporting conventions of Isaac Gym (Makoviychuk et al. 2021), Brax (Freeman et al. 2021), and Madrona (Shacklett et al. 2023).
2. **Find two distinct "ideal env counts."** The throughput-maximizing N (a knee/saturation curve) is *not* the same as the learning-progress-maximizing N. The former is a systems question (occupancy, memory bandwidth, contact-buffer growth — framed by roofline/Amdahl); the latter is governed by the gradient-noise-scale / critical-batch-size theory (McCandlish et al. 2018).
3. **Treat cloth as a first-class, non-rigid scaling axis.** Sweep particle/vertex count × env count as a 2-D grid, for BOTH the PBD particle-cloth backend and the XPBD/FEM surface-deformable backend, so the backend choice is motivated by data. Cloth solvers do not scale like rigid bodies; the "contact buffer overflow" regime must be re-derived and *explained* using PhysX's documented fixed-GPU-buffer behavior.
4. **Isolate morphology and controller cost.** Use an additive-decomposition design (fix task + cloth, vary only robot/controller) to attribute per-step cost to DOF count, added tendon-driven tensegrity wrist, and action-space controller (joint vs differential-IK vs absolute-IK vs OSC). Ground expectations in Featherstone O(n) articulated-body complexity and Khatib operational-space control.
5. **Make everything reproducible and cross-hardware comparable.** Follow MLPerf-style time-to-target reporting and Henderson et al. / Agarwal et al. (rliable) statistical practice: fixed seeds, many repetitions, medians + interval estimates (IQM, bootstrap CIs), full environment/version capture, and machine-tagged artifacts on both the A6000 workstation and the Alex RTX PRO 6000 cluster.

The work plan is phased: (Phase 0) harden the measurement harness and lock the physics-sanity gate; (Phase 1) the cheap, high-value throughput-vs-N and vertex×N grids plus rigid baseline; (Phase 2) morphology/controller cost matrices; (Phase 3) optimization-lever ablations and the tensegrity-wrist novelty study; (Phase 4) end-to-end training-efficiency and cross-GPU comparison. Go/no-go checkpoints gate the expensive training runs behind validated cheap throughput measurements.

---

## 2. Metric Catalogue

Legend for "Serves": A–F map to the research-question groups (Throughput/N, Cloth scaling, Morphology/controller, Additional metrics, Optimization, Cross-HW).

| # | Metric | Precise definition | How measured | Unit | Serves | Primary citation(s) |
|---|--------|--------------------|--------------|------|--------|---------------------|
| M1 | **Aggregate throughput (env·steps/s)** | steps/s × N: number of environment transitions (s,a,r,s′) produced per wall-clock second across all N envs, at steady state | Time a fixed window of `env.step` calls after warmup; multiply per-env rate by N | env·steps/s | A,B,C,E,F | Makoviychuk et al. 2021 (Isaac Gym); Freeman et al. 2021 (Brax); Shacklett et al. 2023 (Madrona) |
| M2 | **Per-env step rate (steps/s / FPS)** | Wall-clock rate of simulation stepping normalized per environment | Same window as M1, not multiplied by N | steps/s | A,B,C | Freeman et al. 2021; Mattson et al. 2020 (MLPerf) |
| M3 | **Construction wall time** | One-off cost: scene build + cloth authoring + PhysX init + pre-settle | Timestamp around startup to first steady step; one process per N | s | A,D,E | Farrell et al. 2021 (MLPerf HPC staging rule) |
| M4 | **reset() wall time** | Wall-clock for a full vectorized `env.reset()` | Time reset call, median over repeats | ms | A,D | Mittal et al. 2023 (ORBIT/Isaac Lab) |
| M5 | **Peak & steady-state GPU memory** | torch CUDA allocated & reserved bytes, AND driver-level used bytes (NVML) | `torch.cuda.max_memory_allocated/reserved`; NVML `used`; report all three | GiB | A,B,D,F | PyTorch caching-allocator docs; NVIDIA NVML |
| M6 | **Host RAM** | Peak resident set size of the process | `/proc` RSS or `resource.getrusage` | GiB | D | Author's own (standard OS metric) |
| M7 | **CPU utilization** | Mean/peak CPU% during steady state | psutil sampling | % | D | Author's own (standard OS metric) |
| M8 | **GPU utilization / SM occupancy** | Fraction of time GPU kernels active; achieved occupancy | NVML utilization; Nsight Compute for occupancy | % | A,D,E | Williams et al. 2009 (roofline); NVIDIA Nsight |
| M9 | **GPU power draw** | Instantaneous board power under sustained load | NVML `power.draw` sampled ≥1 Hz | W | D | Henderson et al. 2020 (systematic energy reporting) |
| M10 | **Energy per million env-steps** | Integral of power over time ÷ (env·steps / 1e6) | ∫power dt from NVML, divide by env·steps | kJ / Msteps (or Wh) | D,E,F | Henderson et al. 2020; Schwartz et al. 2020 (Green AI); Strubell et al. 2019 |
| M11 | **Time-to-target-return** | Wall-clock (and env-steps) to reach a fixed return threshold | Log return vs steps/time; interpolate crossing | s, steps | A,D,E | Mattson et al. 2020 (MLPerf time-to-train); Agarwal et al. 2021 |
| M12 | **Sample efficiency** | Return achieved per env-step (or steps to threshold) | Learning curve area / threshold crossing | return/step | A,C,D,E | Henderson et al. 2018; Agarwal et al. 2021 |
| M13 | **Determinism/variance across seeds** | Dispersion of throughput and return across pinned seeds | Repeat runs; report median, IQR, bootstrap CI, IQM | dimensionless | D,F | Agarwal et al. 2021 (rliable); Henderson et al. 2018 |
| M14 | **Physics-sanity gate** | Boolean: no NaNs, cloth AABB bounded, no PhysX overflow warnings | NaN check on tensors; bounding-box check; grep PhysX log | bool | A,B,E | NVIDIA Isaac Lab PhysxCfg / PhysX 5 GPU buffer docs |
| M15 | **Per-step controller cost** | Marginal ms/step attributable to the action-space controller | Differential timing: controller-on minus joint-position baseline | ms/step | C,E | Khatib 1987 (OSC); Isaac Lab controller docs |
| M16 | **Tendon/tensegrity-wrist overhead** | Marginal ms/step and memory of added compliant DOF + TendonEffortAction vs plain arm | Additive decomposition; hold task+cloth fixed | ms/step, GiB | C,E | Featherstone 2008 (ABA O(n)); Sathya et al. (cABA loop-closure cost) |
| M17 | **Compile / shader-cache time** | First-run JIT/shader-compile overhead vs warm-cache run | Cold vs warm process timing | s | D,F | Freeman et al. 2021 (JIT); Mattson et al. 2020 |
| M18 | **GPU-hours & $/run per training run** | Wall-clock × device count; convert to GPU-hours and cloud-equivalent cost | Log job duration; multiply | GPU-h, $ | D,F | Schwartz et al. 2020; Mattson et al. 2020 |
| M19 | **Fidelity-vs-speed cloth metric** | Quantified physical-behavior deviation vs a high-iteration reference at reduced solver cost | Compare cloth state (drape height, coverage) to gold reference | normalized error | B,E | Macklin et al. 2016 (XPBD iteration-dependence); Lin et al. 2020 (SoftGym coverage/normalized score) |

**Additional-metric recommendation (Question D).** M6–M10, M13, M17, M18, and M19 are the metrics the student explicitly asked me to recommend beyond throughput and GPU memory. My recommended *core* additional set, in priority order:

1. **Energy per million env-steps (M10)** — the single most defensible sustainability/cost metric and increasingly expected in ML systems work (Henderson et al. 2020; Schwartz et al. 2020; Strubell et al. 2019 pioneered quantifying training-time energy in ML).
2. **Determinism/variance across seeds (M13)** — without it, single-number throughput claims are not credible.
3. **Honest three-way GPU memory accounting (M5)** — because a PyTorch caching allocator makes "reserved" ≥ "allocated" ≥ live tensors, and driver-level "used" (NVML) includes the CUDA context, so reporting only one number is misleading.
4. **Fidelity-vs-speed cloth metric (M19)** — the crux of the PBD-vs-XPBD backend decision.

These are cheap to add to the existing harness and materially raise the rigor of a thesis chapter.

---

## 3. Experimental Design

Global conventions (apply to all experiments): read all swept parameters from the project's versioned config objects (never hard-code); one OS process per env-count N (Isaac Sim cannot rebuild the stage in-process); warmup exclusion before every timed window; the physics-sanity gate (M14) must pass or the datapoint is flagged invalid; emit raw + summary artifacts with full machine/version tags (Section 4).

### (a) Throughput-vs-env-count knee
- **Purpose:** Locate the throughput-maximizing N and characterize the saturation-then-decline curve; establish the "ideal env count for throughput."
- **Independent variable:** N ∈ {16, 32, 48, 64, 96, 128, 192, 256, 384, 512} (denser sampling around the preliminary 32–128 knee; extend to 512 on the 96 GB Blackwell card). Rationale: the Stage-0 preliminary result peaks at 32–64 and falls beyond ~128 on the A6000; a log-plus-refined grid resolves the knee without wasting runs.
- **Controlled:** identical task (shirt_pick), backend (PBD first), cloth resolution (~11k particles), solver iterations, dt, decimation, GPU clocks, zero-action policy.
- **Metrics:** M1, M2, M3, M5, M8, M9, M10, M14.
- **Repetitions:** ≥5 process-repeats per N (median + IQR).
- **Runtime:** cheap; minutes per point → a few hours total per machine.
- **Framing:** attribute the knee via roofline (Williams et al. 2009) — is the decline memory-bandwidth-bound (bandwidth roof) or compute/occupancy-bound (compute roof)? Contrast with Brax, which reports near-linear env·steps/s scaling until memory-bandwidth bottlenecks around 10,000 environments (Freeman et al. 2021) — cloth's far earlier knee is the phenomenon to explain.

### (b) Vertex-count × env-count 2-D grid (BOTH backends)
- **Purpose:** Quantify how cloth resolution and parallelism jointly bound throughput and memory; detect the quality-degradation (overflow) regime.
- **Independent variables:** particle/vertex count per garment ∈ roughly {¼×, ½×, 1× (≈11k), 2×, 4×} authored resolutions × N from experiment (a). Run the full grid separately for **PBD particle cloth** and **XPBD/FEM surface deformable**.
- **Hold physical behavior ~constant while varying resolution:** re-tune per-particle mass, stretch/bend/shear stiffness, and solver iterations so the macroscopic drape/coverage of a single garment matches a reference within tolerance (verified via M19) — so we measure *compute*, not a different garment. This is the SoftGym normalized-coverage discipline (Lin et al. 2020) applied as a control, not a task metric.
- **Metrics:** M1, M5 (all three memory numbers), M8, M14, M19.
- **Repetitions:** ≥3 per cell (grid is large).
- **Runtime:** moderate; the dominant Phase-1 cost.

### (c) Simulate-vs-train shirt-count thresholds (BOTH backends)
- **Purpose:** Establish two distinct ceilings — (i) the physics-stability/memory ceiling (max shirts you can *simulate* validly) and (ii) the env count that maximizes learning progress per wall-clock hour (max shirts you should *train* with).
- **Method (i):** from (a)+(b), the largest N (and resolution) at which M14 passes and memory fits.
- **Method (ii):** short PPO training runs at a subset of N; measure time-to-target-return (M11) and sample efficiency (M12); locate the N beyond which added parallelism stops improving return-per-hour. Interpret via gradient-noise-scale / critical-batch-size theory (McCandlish et al. 2018): beyond the critical batch size, more parallel envs buy diminishing sample-efficiency returns. Note the caveat that gradient-noise-scale is an *approximate* predictor of the critical batch size, not an exact one.
- **Metrics:** M11, M12, M13, M1, M10.
- **Repetitions:** ≥5 seeds per N (learning variance is high).
- **Runtime:** expensive (training) — gate behind Phase-1 go/no-go.

### (d) Action-space cost matrix
- **Purpose:** Isolate per-step controller compute across joint-position, differential-IK (relative), absolute-IK (pose), and OSC.
- **Independent variable:** 4 action spaces; reuse the existing reach 6-arm × 4-space = 24-config grid as the clean, cloth-free probe, then confirm on one cloth task.
- **Controlled:** same robot, same task, same N, same policy update; vary only the action term.
- **Method:** differential timing — subtract the joint-position baseline per-step cost to obtain the marginal controller cost (M15). Expect IK (Jacobian pseudo-inverse / damped-least-squares) and OSC (mass-matrix/dynamics solve) to add cost per Khatib 1987 and the Isaac Lab controller implementation (the Isaac Lab docs cite Khatib 1987 directly for their OSC; the differential-IK controller offers `pinv`, `svd`, `trans`, `dls` Jacobian-inverse methods).
- **Metrics:** M15, M1, M2, M12 (learning difficulty differs by action space).
- **Repetitions:** ≥5.
- **Runtime:** cheap for sim-cost; moderate if training-efficiency arm included.

### (e) Tensegrity-wrist / tendon overhead (NOVEL TARGET)
- **Purpose:** Quantify the compute overhead the added tendon-driven tensegrity wrist imposes on a standard arm ("Frankenstein" vs plain), and the cost of pure tendon-actuated tensegrity manipulators (3-DOF, 5-DOF) including the custom TendonEffortAction (Jacobian-transpose tendon→torque mapping each control step) and the heavier "physical tendon" variant.
- **Independent variables:** {plain UR5e/UR10/Kinova} vs {same arm + tensegrity wrist}; {3-DOF, 5-DOF pure tensegrity}; {kinematic tendon vs physical-tendon}.
- **Controlled (critical, to avoid confounds):** same task, same cloth, same N, same control frequency; decompose additively so the marginal costs of (extra DOF), (extra bodies/joints/closed-loop constraints), and (TendonEffortAction per-step compute) are separated.
- **Theoretical framing (flag: the literature is largely silent on this exact artifact — the following is the author's synthesis of the closest analogues):** Featherstone's articulated-body algorithm is O(n) in DOF n (Featherstone 2008, Ch. 6: "computational complexity of O(n) as compared with O(n³) for the composite-rigid-body method"), so extra DOF *alone* should add roughly linear per-step cost. But closed-loop/tendon constraints are **not** representable in the reduced-coordinate tree and are solved as added equality/constraint rows, whose cost scales super-linearly in the number of constraint rows m. Primary analogues: the constrained-ABA formulation gives O(n + m²d + m³) for loop-closure-constrained dynamics, and MuJoCo's joint-space approach is O(nd² + m²d + d²m), where m is the number of constraint rows and d the tree depth (Sathya et al., arXiv:2310.00688 and cABA — flag: arXiv preprints, peer-review status to verify). MuJoCo's official docs confirm tendons and "loop joints" are modeled as solver constraints, not tree elements. Hence the tensegrity wrist is expected to cost *more than* its DOF increment implies. Where no citable number exists, the deliverable is the *measurement protocol* above, not an invented figure. The closest empirical analogues for tensegrity-in-sim cost are MuJoCo-based tensegrity RL studies (e.g., differentiable/graph-network tensegrity dynamics work modeling cables as stiff springs), which report cable stiffness/damping parameters but not per-step solver-cost decompositions — a genuine gap this thesis can fill.
- **Metrics:** M16, M15, M1, M5, M12.
- **Repetitions:** ≥5.
- **Runtime:** moderate.

### (f) DOF/robot matrix (train & eval)
- **Purpose:** Attribute performance to joint/DOF count across UR5e (6), UR10 (6), Kinova Gen3 (7), tensegrity 3/5-DOF, at both training and evaluation time; separate simulation cost from learning-system cost (DOF changes obs/action dim → network size → PPO cost).
- **Independent variable:** robot (hence DOF, link count, obs/action dim).
- **Method:** measure sim-only per-step cost (zero-action) AND end-to-end training cost; the difference isolates the learning-system contribution. Hold the network-architecture rule fixed (or record it) so obs/action dimensionality effects are attributable.
- **Metrics:** M1, M2, M11, M12, M13, M5.
- **Repetitions:** ≥5 seeds (training).
- **Runtime:** expensive (training arm) — gate behind Phase-1.

### (g) Optimization-lever ablations
- **Purpose:** Measure (not assert) each lever's effect and flag speed-vs-fidelity/quality trade-offs.
- **Levers (each a one-factor-at-a-time sweep from a fixed baseline config):** solver position/velocity iteration count; substeps; control decimation; physics dt; fp32 vs tf32/fp16; PhysX GPU buffer/aggregate sizing (parameters and documented defaults below); rendering off vs on; contact/rest offsets; PPO rollout length × env-count trade-off; network size; obs/action dimensionality; cloth backend (PBD vs XPBD).
- **PhysX GPU buffer parameters and documented defaults (from Isaac Lab `PhysxCfg`):** `gpu_max_rigid_contact_count = 2²³`, `gpu_max_rigid_patch_count = 5·2¹⁵`, `gpu_found_lost_pairs_capacity = 2²¹`, `gpu_found_lost_aggregate_pairs_capacity = 2²⁵`, `gpu_total_aggregate_pairs_capacity = 2²¹`, `gpu_collision_stack_size = 2²⁶`, `gpu_heap_capacity = 2²⁶`, `gpu_temp_buffer_capacity = 2²⁴`, `gpu_max_particle_contacts = 2²⁰`, `gpu_max_soft_body_contacts = 2²⁰`, `gpu_max_num_partitions = 8`. The `gpu_heap_capacity` docstring notes it *can* grow ("additional memory will be allocated if more memory is required"), but the contact/pair/collision-stack buffers cannot — that distinction is exactly why the overflow regime appears.
- **Trade-off detection:** every lever that can degrade physics or policy must be paired with M19 (cloth fidelity) and/or M12 (sample efficiency); a speedup that fails the M14 gate or worsens M19/M12 is reported as a trade, not a win. Solver-iteration reduction is the archetypal fidelity trade — XPBD's compliance is iteration-count-dependent by construction (Macklin et al. 2016 exists precisely to address iteration-dependent stiffness in PBD).
- **Metrics:** M1, M5, M14, M19, M12 depending on lever.
- **Repetitions:** ≥3–5.
- **Runtime:** cheap-to-moderate per lever.

### (h) Cross-GPU comparison (A6000 Ampere 48 GB vs RTX PRO 6000 Blackwell 96 GB)
- **Purpose:** Fair, citable device comparison.
- **Hardware under test:** NVIDIA RTX A6000 (Ampere): 48 GB GDDR6 ECC, 384-bit bus, 768 GB/s, 10,752 CUDA cores. NVIDIA RTX PRO 6000 Blackwell Workstation Edition: 96 GB GDDR7 (28 Gbps), 1,792 GB/s peak bandwidth, 24,064 CUDA cores / 188 SMs, 600 W. Blackwell offers ≈2.33× the memory bandwidth of the A6000 (1,792 vs 768 GB/s) and 2× the memory capacity — directly relevant because cloth throughput is expected to be bandwidth-sensitive.
- **Method:** hold task/config/seed/container constant; run experiments (a),(b),(d),(g)-subset on both; control clocks/thermals/driver/CUDA/container versions, ECC, MIG-off, background load; report medians + dispersion + bootstrap CI; present standard scaling plots (throughput vs N per device on one axis) and tables.
- **Metrics:** M1, M5, M8, M9, M10, M13, M17.
- **Repetitions:** ≥5 per point per machine.
- **Runtime:** moderate; parallelizable via the Alex sbatch seed-array wrapper.

### (i) Cloth-free rigid baseline (reference)
- **Purpose:** Provide the rigid-body scaling reference (reach / cube_place / cube_sort) so cloth-specific costs are interpretable as a delta over rigid.
- **Method:** run experiment (a) throughput-vs-N with rigid tasks; expect near-linear scaling to much higher N than cloth. For calibration of the expected rigid ceiling: Isaac Gym (Makoviychuk et al. 2021) demonstrated a 21-DOF humanoid trained with 4,096 parallel environments at simulation rates near 200,000 steps/s on a single GPU — rigid bodies scale to thousands of envs, whereas your cloth solver knees out below ~128.
- **Metrics:** M1, M5, M8, M14.
- **Repetitions:** ≥5.
- **Runtime:** cheap.

---

## 4. Statistical & Reproducibility Protocol

**Warmup & steady state.** Every timed window excludes an initial warmup (JIT/shader compile, allocator warmup, cloth pre-settle). Report the steady-state window length and warmup length. Distinguish cold-cache (M17) from warm-cache runs explicitly.

**Repetitions & statistics.** Follow rliable (Agarwal et al. 2021) and Henderson et al. (2018): never report a single run. For throughput, report median and IQR over ≥5 process-repeats. For training outcomes, report interquartile mean (IQM) and stratified bootstrap 95% confidence intervals over ≥5 seeds (ideally more). Agarwal et al. recommend pooling runs across tasks to strengthen interval estimates — e.g., "evaluating 3 runs each on Atari 100k, which contains 26 tasks, results in 78 sample scores for uncertainty estimation"; your analogue is pooling across the several sorting tasks/robots. Use performance profiles where comparing configs. Report medians rather than means for skewed timing distributions.

**Seeding & determinism.** Pin all RNG seeds (Python, NumPy, PyTorch, simulator). Document deterministic vs stochastic modes: PhysX GPU simulation is not bitwise-deterministic across all settings. Isaac Lab's `PhysxCfg` exposes `enable_enhanced_determinism: bool = False` ("Enable/disable improved determinism at the expense of performance. Defaults to False") — record its state for every run. Report seed-to-seed variance (M13) as a first-class result, not noise to be hidden.

**Environment/version capture (every run emits):** GPU model, driver version (580.x on the workstation), CUDA/cuDNN, PhysX/Isaac Sim/Isaac Lab versions, container image digest, conda env hash, git commit of the suite and of the config files, host CPU/RAM, ECC/MIG state, GPU clock/power caps, and the full resolved config object. **Version note:** Isaac Lab 2.3.0 is built on Isaac Sim 5.1, but the Isaac Lab `PhysxCfg` determinism documentation links to PhysX 5.4.1 — capture the exact PhysX point release, as buffer behavior can differ across releases.

**Artifact schema (comparable across machines).** Each run emits a raw per-step CSV and a summary JSON. Minimum columns/fields: `machine_tag, gpu_model, driver, cuda, physx_ver, isaacsim_ver, isaaclab_ver, container_digest, git_commit, task, backend, robot, action_space, N, cloth_particle_count, solver_pos_iters, solver_vel_iters, substeps, decimation, dt, precision, seed, enhanced_determinism, warmup_steps, window_steps, steps_per_s, env_steps_per_s, construct_s, reset_ms, mem_alloc_gib, mem_reserved_gib, mem_driver_gib, host_ram_gib, cpu_pct, gpu_util_pct, sm_occupancy, power_w_mean, energy_kj, energy_per_msteps, physics_ok, physx_overflow_warns, notes`. Machine-tagged outputs let the A6000 and Alex results be merged and compared directly.

---

## 5. Prioritized Work Plan

**Phase 0 — Harness hardening (do first; cheap; foundational).**
- Extend the existing single-env-count script to emit the full artifact schema (Section 4) and the additional metrics (M5 three-way memory, M8–M10 energy, M13 variance, M17 cache).
- Lock the physics-sanity gate (M14): NaN check, cloth-AABB bound, PhysX overflow-warning grep — and wire the PhysX `gpu_*` buffer parameters through the config so overflow is *diagnosable*, not silent.
- Validate the sbatch seed-array wrapper stages assets to `$TMPDIR` and runs headless on Alex (this mirrors MLPerf HPC's rule of including data staging in measured time).
- **Go/no-go:** harness reproduces a known throughput number within seed variance on the A6000 and emits an identical schema on Alex.

**Phase 1 — Cheap, high-value core (the thesis backbone).**
- Experiment (a) throughput-vs-N knee (PBD), (i) rigid baseline, and (b) vertex×N grid for BOTH backends.
- This directly answers "ideal env count for throughput," "how many shirts can I simulate," and motivates the backend choice — with minutes-to-hours-scale runs.
- **Go/no-go:** knee and overflow regime reproduced with tight CIs on both GPUs before committing to any training runs.

**Phase 2 — Morphology & controller cost (cheap sim-cost arm).**
- Experiment (d) action-space matrix (reuse reach 24-config grid), (f) DOF matrix sim-only arm, and (e) tensegrity-wrist overhead sim-only arm.
- All measurable with zero-action stepping → cheap. This delivers the novel tensegrity-overhead result early.

**Phase 3 — Optimization levers (cheap-to-moderate).**
- Experiment (g) one-factor ablations, each paired with fidelity/quality trade-off detection.
- Feeds directly into the recommended production config.

**Phase 4 — Expensive training-efficiency & cross-GPU (gated).**
- Training arms of (c) simulate-vs-train thresholds, (f) DOF matrix, (d)/(e) learning-difficulty arms; full (h) cross-GPU comparison.
- Only launch after Phases 1/2 confirm configs are stable and worth the GPU-hours.

**Minimal-viable subset (if time-constrained):** Phase 0 + Phase 1 (a,b,i) + the sim-only arms of (d),(e),(f) + a single small training run per backend for (c) method (ii). This yields the ideal-env-count answer, the backend comparison, the controller/DOF/tensegrity sim-cost deltas, and a first simulate-vs-train threshold — a complete, defensible chapter.

**Cheap/high-value vs expensive/marginal:** Cheap + high-value: (a),(b),(i),(d-sim),(e-sim),(f-sim),(g). Expensive: all training arms (c),(f-train),(h-train). Marginal: extreme high-resolution cloth cells that always fail M14 (report the boundary, don't over-sample beyond it).

**Benchmarks that would change the plan:** if the PBD knee turns out to sit far above 128 once buffers are sized correctly (i.e., the Stage-0 result was a buffer-overflow artifact, not a true saturation), skip the aggressive high-N XPBD grid and reallocate to training-efficiency runs. If XPBD fidelity (M19) is not materially better than well-tuned PBD at equal cost, drop XPBD from the production recommendation and note it. If seed variance (M13) exceeds ~10–15% of the median throughput, increase repetitions before drawing any cross-GPU conclusions.

---

## 6. Threats to Validity

1. **Silent physics degradation at high N (most dangerous).** PhysX 5 GPU simulation pre-allocates fixed buffers that *cannot* dynamically grow (per Isaac Lab PhysxCfg docs: "Unlike CPU PhysX, the GPU simulation feature is unable to dynamically grow all the buffers"); on overflow it emits errors such as "increase PxgDynamicsMemoryConfig::foundLostPairsCapacity … otherwise the simulation will miss interactions," and the analogous `gpu_collision_stack_size` / collisionStackSize overflow. A throughput number measured while collisions are being dropped is meaningless. *Mitigation:* M14 gate + explicit buffer-parameter logging; any datapoint with overflow warnings is flagged invalid, and the buffer-sized-up rerun is reported separately.
2. **Caching-allocator memory illusion.** PyTorch reserves ≥ allocated ≥ live; reporting one number misleads. *Mitigation:* M5 reports allocated, reserved, and driver-used.
3. **Warmup/compile contamination.** JIT/shader-cache and cloth pre-settle inflate early steps. *Mitigation:* warmup exclusion + separate cold/warm (M17).
4. **Conflating sim throughput with training throughput.** Policy inference, PPO minibatching, host-device transfer, and logging are not simulation. *Mitigation:* measure zero-action sim throughput separately from end-to-end (experiments a vs c/f-train).
5. **Confounding morphology with learning-system cost.** More DOF changes obs/action dim → network size → PPO cost. *Mitigation:* additive decomposition (sim-only vs end-to-end) in (e),(f).
6. **Cross-GPU confounds.** Clocks, thermals, driver/CUDA/container drift, ECC, MIG, background load. *Mitigation:* hold constant, log all, MIG-off, ≥5 repeats + CIs.
7. **Single-seed overclaiming.** *Mitigation:* rliable IQM + bootstrap CIs, report variance (M13).
8. **Changing the garment while changing resolution.** Higher mesh resolution can silently alter physical behavior. *Mitigation:* re-tune material/iterations to match reference drape/coverage (M19) so only compute varies.
9. **Backend apples-to-oranges.** PBD and XPBD have different stability/iteration regimes; equal iteration counts ≠ equal fidelity. *Mitigation:* match fidelity (M19) first, then compare cost.
10. **Version drift.** Isaac Lab 2.3.0 is built on Isaac Sim 5.1 but its PhysxCfg links to PhysX 5.4.1; behavior can differ across point releases. *Mitigation:* full version capture and a single pinned stack per study.

---

## 7. Annotated Bibliography

- **Makoviychuk et al. 2021, "Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning" (NeurIPS Datasets & Benchmarks; arXiv:2108.10470).** Foundational GPU-native RL simulator; establishes env·steps/s reporting, the end-to-end-on-GPU paradigm, and the rigid-scaling reference (4,096-env humanoid). Supports M1/M2 and the rigid baseline (i).
- **Mittal et al. 2023, "ORBIT" (IEEE RA-L, DOI 10.1109/LRA.2023.3270034); NVIDIA 2025, "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning" (arXiv:2511.04831).** The framework under test; documents PhysX-5 deformables/cloth, closed-loop chains, and coupled rigid–deformable solvers. Grounds the pipeline and backend descriptions.
- **Freeman et al. 2021, "Brax" (arXiv:2106.13281).** GPU/TPU RL engine; explicitly reports env·steps/s scaling with N and the memory-bandwidth bottleneck around 10,000 envs — the template for the throughput-vs-N knee (experiment a) and its roofline interpretation.
- **Shacklett et al. 2023, "An Extensible, Data-Oriented Architecture for High-Performance Batch Simulation" (SIGGRAPH; Madrona).** Batch-simulator throughput methodology; the aggregate-throughput-over-latency framing supports M1 and the occupancy discussion.
- **Williams, Waterman & Patterson 2009, "Roofline: An Insightful Visual Performance Model" (CACM).** Compute-vs-memory-bound framing used to *attribute* the throughput knee (experiment a, M8).
- **McCandlish et al. 2018, "An Empirical Model of Large-Batch Training" (arXiv:1812.06162).** Gradient-noise-scale / critical-batch-size theory; basis for distinguishing throughput-optimal N from learning-optimal N (experiment c). Caveat: gradient-noise-scale is an approximate, not exact, predictor of the critical batch size.
- **Schulman et al. 2017, "Proximal Policy Optimization" (arXiv:1707.06347).** The on-policy algorithm in use (skrl PPO/IPPO/MAPPO); rollout-length × env-count trade-off (lever g).
- **Müller et al. 2007, "Position Based Dynamics."** Foundational PBD; grounds the PBD particle-cloth backend.
- **Macklin, Müller & Chentanez 2016, "XPBD: Position-Based Simulation of Compliant Constrained Dynamics" (MIG).** Grounds the XPBD backend AND the iteration-count-dependent-stiffness fidelity trade-off (M19, lever g).
- **Lin, Wang, Olkin & Held 2020, "SoftGym: Benchmarking Deep RL for Deformable Object Manipulation" (CoRL).** Cloth-manipulation-in-sim benchmark; source of the normalized-coverage / normalized-score discipline reused as a *control* (M19) to hold garment behavior constant across resolution.
- **Lu et al. 2024, "GarmentLab: A Unified Simulation and Benchmark for Garment Manipulation" (NeurIPS).** Isaac-Sim-based garment benchmark using BOTH FEM and PBD; closest analogue for a dual-backend garment methodology and for GPU-parallel garment-RL data collection.
- **Antonova et al. / DeformableGym (DFKI) and related 3D-deformable-grasping benchmarks.** Provide RL-evaluation patterns for deformable grasping (success-rate + IQM reporting), corroborating the metric choices for the training arms.
- **Khatib 1987, "A Unified Approach for Motion and Force Control … The Operational Space Formulation" (IEEE J. Robotics & Automation, 3(1):43–53).** The OSC reference cited by Isaac Lab's own controller docs; grounds the controller-cost expectation (M15, experiment d).
- **Featherstone 2008, Rigid Body Dynamics Algorithms (Springer), Ch. 6.** Articulated-body algorithm is O(n) in DOF (vs O(n³) composite-rigid-body); grounds the DOF-cost expectation (M16, experiments e/f). Also underpins PhysX's own reduced-coordinate articulation solver.
- **Sathya et al., constrained/LQR-formulation dynamics (arXiv:2310.00688) and cABA.** Closed-loop/tendon constraints add super-linear cost in constraint-row count m (cABA O(n + m²d + m³); MuJoCo joint-space O(nd² + m²d + d²m)); grounds the expectation that the tensegrity wrist costs more than its DOF increment (experiment e). *Flag: arXiv preprints; peer-review status to verify.*
- **Tensegrity-in-simulation RL literature (e.g., MuJoCo-based tensegrity locomotion; differentiable/graph-network tensegrity dynamics modeling cables as stiff springs).** Closest empirical analogues for the novel tensegrity-wrist study; they report cable stiffness/damping parameters but not per-step solver-cost decompositions — the gap this thesis fills.
- **Mattson et al. 2020, "MLPerf Training Benchmark" (MLSys); Farrell et al. 2021, "MLPerf HPC" (arXiv:2110.11466).** Time-to-target-quality metric and the data-staging-in-measured-time rule; ground M11, M18 and the Alex `$TMPDIR` staging discipline.
- **Henderson et al. 2018, "Deep Reinforcement Learning that Matters" (AAAI).** RL reproducibility crisis; motivates seeds, variance reporting, and cautious throughput claims (M12, M13).
- **Agarwal et al. 2021, "Deep RL at the Edge of the Statistical Precipice" (NeurIPS; rliable).** IQM + stratified bootstrap CIs + performance profiles; the statistical backbone of Section 4.
- **Henderson et al. 2020, "Towards the Systematic Reporting of the Energy and Carbon Footprints of ML" (JMLR); Schwartz et al. 2020, "Green AI" (CACM); Strubell et al. 2019, "Energy and Policy Considerations for Deep Learning in NLP" (ACL).** Energy-per-experiment reporting practice; ground M9, M10, M18.
- **NVIDIA Isaac Lab `PhysxCfg` documentation & PhysX 5 SDK (`PxgDynamicsMemoryConfig`).** Primary source for the fixed-GPU-buffer behavior, the exact `gpu_*` parameter names/defaults (Section 3g), the `enable_enhanced_determinism` flag, and the overflow errors; grounds M14, lever g, and threat #1.
- **NVIDIA RTX A6000 datasheet and RTX PRO 6000 Blackwell architecture whitepaper.** Primary source for the cross-GPU hardware specs (48 GB/768 GB/s vs 96 GB/1,792 GB/s); ground experiment (h).
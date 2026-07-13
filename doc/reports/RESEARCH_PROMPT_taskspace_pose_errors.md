# Research prompt — Validating an RL action-space comparison for full-pose reaching and fixing the task-space controllers' pose errors

## Your role and deliverable

You are a research assistant with web/literature access. Investigate the three
questions in §5 about the robot-learning study described below and produce a
structured research report.

**Citation requirements (mandatory):**
- Every non-trivial claim must carry an inline citation to a verifiable source
  (author, year, title, venue, and a DOI or arXiv link).
- Prefer peer-reviewed venues (ICRA, IROS, CoRL, RSS, T-RO, IJRR, RA-L, NeurIPS,
  ICML) and widely-cited arXiv papers; official documentation (NVIDIA Isaac
  Lab / Isaac Sim, PhysX) is acceptable for framework-specific behaviour.
  Blog posts / forums only as a last resort and marked as such.
- Do not fabricate references. If you cannot find a source for a claim, say so
  explicitly and mark the claim as your own reasoning.
- End the report with a bibliography of the **most important sources** (the
  5–15 that most strongly support your conclusions), ranked by relevance, each
  with 1–2 sentences on what it contributes.

---

## 1. Study context

**Goal.** A reach task is used as a *controller-hardening* baseline: for each of
six robot arms, train the same PPO agent under four action spaces and determine,
per robot, which action space / control mechanism works best (this is NOT a
robot-vs-robot benchmark). The chosen controllers are reused later in harder
contact-rich cloth-manipulation tasks. A special interest is a compliant 2-DOF
"tensegrity wrist" retrofitted between arm flange and gripper ("Frankenstein"
variants) and whether/where it helps.

**Simulator / learning stack.**
- Isaac Sim 5.1 / Isaac Lab (manager-based RL env), PhysX GPU simulation.
- Physics 60 Hz, control decimation 2 → 30 Hz agent steps; episode 6 s
  = 180 steps; 4096 parallel envs; gravity disabled for the arm; scene props
  penetrable; robot self-collision disabled in physics (a *kinematic* sampling
  filter handles it, see §2).
- PPO (skrl): Gaussian policy, MLP 64×64 ELU (separate=False), rollouts 24,
  learning_epochs 5, mini_batches 4, γ=0.99, GAE λ=0.95, lr 1e-3 with
  KL-adaptive scheduler (threshold 0.01), grad-norm clip 1.0, ratio/value clip
  0.2, entropy 0.01, running standard scalers on obs and value,
  initial_log_std 0.0 (except IK-Abs: −1.0), 100k timesteps ≈ 36 min/run.
- Observations: joint positions (sin/cos encoded), joint velocities, the 7-D
  pose command (base frame), previous action; ±0.01 uniform obs noise.
- Reward (IsaacLab reference reach): position error L2 weight −0.2,
  fine-grained position tanh (std 0.1) weight +0.1, orientation quaternion
  shortest-path error weight −0.1, action-rate and joint-velocity penalties
  −1e-4 ramped to −0.005/−0.001 after 4500 steps (curriculum).
- Metrics (fraction of steps, converged = last 10 % of training):
  `pos%` = position error < 5 cm; `pose%` = position < 5 cm AND orientation
  error < 0.3 rad; mean position error in cm.

**Robots (6).** UR5e (6-DOF, reach 0.85 m), UR10 (6-DOF, 1.30 m), Kinova Gen3
(7-DOF, 0.90 m, four *continuous* joints), each in two forms: rigid with a
Robotiq 2F-140 gripper ("F140") and "Frankenstein" (same arm + compliant 2-DOF
tensegrity wrist between flange and gripper; wrist joints ±0.873 rad, implicit
PD stiffness 400 / damping 20, effort limit 3.5 N·m, near-zero reflected
inertia). EE frame for control/reward = the gripper base link
(`robotiq_base_link`). Floor-standing at z = 0.75 m; collidable ground at z = 0.

**Action spaces (4).**
1. **Joint**: EMA-smoothed joint-position-to-limits action, α = 0.2 (action =
   normalized joint targets over the controlled joints; 6–9 D).
2. **IK-Rel**: Isaac Lab `DifferentialInverseKinematicsAction`, damped
   least-squares (DLS), relative mode: policy outputs a 6-D pose delta
   (Δpos + rot-vec), scale 0.5; arm PD stiffened to 800/160 for tracking.
3. **IK-Abs**: same DLS controller, absolute mode: policy outputs the full 7-D
   target pose (position + quaternion) in the robot base frame, solved in one
   shot per step; PD 800/160; PPO initial_log_std −1.0. *Note: the raw Gaussian
   policy output is used as a quaternion without explicit normalization by the
   policy (the controller consumes it).*
4. **OSC**: Isaac Lab `OperationalSpaceControllerAction`,
   `target_types=["pose_abs"]`, `impedance_mode="variable_kp"` (action = 7-D
   pose + 6-D task-space stiffness, scaled ×100, clamped 50–200), damping ratio
   1.0 (critical), **partial** inertial dynamics decoupling (translational /
   rotational blocks; full Λ = (J M⁻¹ Jᵀ)⁻¹ was ill-conditioned on these
   arm+gripper articulations), no gravity compensation (gravity off),
   null-space posture control toward the default configuration only for
   redundant arms (> 6 OSC joints); arm actuators run pure-effort (PD zeroed).
   The compliant tensegrity wrist is **excluded** from the OSC Jacobian and
   held by its own PD (including it pure-effort removed all damping from a
   near-massless 3.5 N·m joint → oscillation → whole-arm divergence).

**Targets.** One target per 6-s episode. Two distributions were studied:
- *Uniform box* (legacy): fixed position box in the base frame with a fixed
  "gripper-down, free-yaw" orientation — later shown geometrically unreachable
  for the 6-DOF arms over most of the box (a GPU diagnostic found ~0 % coverage
  for UR5e), making reach a position-only task there.
- *FK-sampled full poses* (current, the hardening distribution): teleport the
  arm to random joint configurations, read the EE pose via forward kinematics →
  targets are reachable by construction. Sampling: per-joint uniform within
  joint limits intersected with default ± 1.5 rad, inner 80 % (10 % margin per
  end); rejection filters (≤ 8 rounds): kinematic self-collision check
  (capsule model, radius 5 cm, auto-calibrated exclusion of rigidly-stacked
  frame pairs) and floor clearance (all link origins ≥ 5 cm, EE ≥ 30 cm above
  the ground plane, world frame). Acceptance measured ≈ 100 % within 8 rounds.

---

## 2. Key empirical results

### A. Multi-seed box-grid finding (5 seeds: {0,1,2,42,123})

With the fixed gripper-down box orientation, OSC was **bimodal on all four
6-DOF UR variants** (some seeds ≈ 2–5 cm, others diverge to 20–60 cm with
collapsed episodes; e.g. ur5e_f140_osc per-seed: 19.3 / 28.1 / 26.8 / 2.1 /
2.6 cm), while both 7-DOF Kinova OSC variants were tight (2.6–2.7 ± 0.8 cm).
Joint / IK-Rel / IK-Abs were seed-stable everywhere (1.9–5.8 cm). Lowering
stiffness (kp 50) and/or fixing the impedance shifted *which* seeds diverge
(best: fixed impedance + kp 50 → 4/5 converge at 1.6–2.1 cm but a previously
good seed flipped to 24 cm) — the divergence basin shrank but never closed.

### B. Switching to FK-sampled (reachable) targets

- Eliminated the catastrophic OSC divergence entirely (no collapsed runs in
  120 runs; worst single seed 17.7 cm).
- Unlocked orientation on the 6-DOF arms: UR5e **joint** control reaches
  **86 % pose** (pos < 5 cm AND ori < 0.3 rad) vs 0 % on the box.

### C. FK full-pose grid (24 variants × 5 seeds, fixed sampler) — the headline

Cell = mean position error cm / pos% / pose%:

| Arm | Joint | IK-Rel | IK-Abs | OSC |
|---|---|---|---|---|
| Kinova-F140 (7-DOF) | **5.9 / 79 / 22** | 8.9 / 50 / 11 | 9.4 / 46 / 12 | 8.2 / 44 / 7 |
| Kinova-Frankenstein (9-DOF) | **9.8 / 49 / 16** | 9.9 / 46 / 15 | 12.9 / 32 / 11 | 13.3 / 16 / 2 |
| UR5e-F140 (6-DOF) | **3.4 / 91 / 86** | 8.3 / 51 / 21 | 7.8 / 56 / 22 | 8.3 / 48 / 18 |
| UR5e-Frankenstein (8-DOF) | **5.3 / 79 / 60** | 8.7 / 55 / 29 | 6.9 / 68 / 32 | 9.0 / 40 / 13 |
| UR10-F140 (6-DOF) | **5.2 / 87 / 84** | 12.4 / 44 / 20 | 14.3 / 34 / 11 | 9.7 / 51 / 19 |
| UR10-Frankenstein (8-DOF) | **6.2 / 80 / 64** | 13.6 / 37 / 18 | 10.0 / 58 / 34 | 14.4 / 19 / 6 |

**Joint control wins on every robot — often by 2–4×.** Task-space controllers
(diff-IK and OSC) sit at 7–15 cm / 6–34 % pose on the identical targets, and
their *orientation* tracking is much worse than joint control's.

### D. Probes (all negative)

On UR5e-F140, FK targets, 2 seeds each:
- IK-Abs with 12 s episodes (double time for large reorientation): 11.1 cm /
  27 % — **worse** than 6 s (7.8 cm / 56 %).
- IK-Rel with 12 s episodes: 12.0 cm — worse (8.3 cm at 6 s).
- IK-Rel with per-step delta scale doubled to 1.0: 11.6 cm — worse.
- Tensegrity wrist included in the OSC Jacobian as damped-compliant (stiffness
  0, damping 20 kept): stable (no divergence) but strictly worse than
  exclusion (15.1 cm / 12 % vs 9.0 cm / 40 %).

### E. Wrist matched-pair A/B (UR5e-Frankenstein, joint control, FK targets)

| targets × wrist | pos cm | pose% |
|---|---|---|
| base-arm-reachable × wrist active | 4.7 | 71 |
| base-arm-reachable × wrist locked | 4.3 | 68 |
| wrist-requiring × wrist active | 4.6 | 60 |
| wrist-requiring × wrist locked | 10.3 (bimodal) | 22 |

→ The wrist adds *capability* (extends the reachable pose set), not
*redundancy assistance* on already-reachable poses.

---

## 3. Questions to research

### Q1 — Are these results sensible and expected?

Evaluate against the literature:
1. Is "joint-space actions beat task-space (differential-IK / operational-space)
   actions for RL on wide-workspace, full-pose reaching" consistent with
   published action-space comparisons for manipulation RL? Which papers support
   or contradict it, and under what conditions (task type, workspace size,
   contact vs free-space, orientation demands)? Note that much of the
   literature (e.g. variable-impedance-in-task-space work) argues the opposite
   for *contact-rich* tasks — reconcile.
2. Is the OSC seed-bimodality on non-redundant arms chasing an *infeasible*
   orientation target a known failure mode (task-space controllers under
   unreachable/singular references)? Is our diagnosis (removing the infeasible
   orientation removes the divergence) consistent with control theory?
3. Are the probe outcomes expected — i.e. longer episodes reducing sample
   diversity at fixed total steps hurting PPO, and larger per-step IK deltas
   not helping?
4. Is "compliant wrist = capability extension but useless/harmful under
   torque-level OSC" plausible given the wrist's properties (near-zero
   reflected inertia, 3.5 N·m limit, PD 400/20)?

### Q2 — Are the configs correct?

Audit the configuration choices above (§1) for correctness and known pitfalls,
citing framework documentation or literature where possible. Specifically:
1. **IK-Abs quaternion actions:** a Gaussian policy emitting raw 4-D quaternion
   components (unnormalized, sign-ambiguous, not a chart of SO(3)) — is this a
   known anti-pattern for learning orientations? What do papers on rotation
   representations for learning (e.g. continuity of rotation representations)
   recommend, and does this plausibly explain IK-Abs's poor *orientation*
   tracking specifically?
2. **IK-Rel rotation deltas** (axis-angle / rot-vec scaled by 0.5 at 30 Hz):
   reasonable? Known better parameterizations?
3. **OSC**: partial vs full inertial decoupling on arm+gripper chains;
   variable-kp as policy action (stiffness range 50–200, is the ×100 scaling
   sane?); damping ratio 1; nullspace posture control only when redundant; PD
   zeroed on arm actuators. Any misconfiguration that would explain mediocre
   full-pose tracking?
4. **PD gains 800/160 for IK tracking** on UR/Kinova-class arms at 30 Hz
   commands, gravity off — adequate? Known rules of thumb?
5. **PPO settings** (64×64 net, lr 1e-3 KL-adaptive, 24-step rollouts at 4096
   envs, 100k steps) for these action spaces — anything that systematically
   disadvantages task-space actions (e.g. action-dim 13 for OSC vs 6–9 for
   joint; initial_log_std −1 only for IK-Abs)?
6. The FK target sampler design (rejection filters, ±1.5 rad sector) — any
   known-better practice for generating reachable full-pose target
   distributions for reach RL?

### Q3 — What can fix the task-space controllers' pose errors?

Propose concrete, literature-backed interventions to close the 2–4× gap in
position error and the larger gap in pose% for IK-Rel / IK-Abs / OSC, e.g.:
- rotation action representations (6-D continuous representation, rot-vec,
  quaternion with normalization/canonicalization) and their measured impact;
- residual / hybrid schemes (policy corrects a scripted or model-based
  task-space baseline);
- action-space shaping: EE-frame deltas vs base-frame, per-axis scales,
  smoothing/EMA on task-space actions (we EMA the joint action but not the
  task-space ones — relevant?);
- controller-side: more DLS iterations per step, damping/λ tuning, joint-limit
  aware IK (e.g. SVD clamping), nullspace objectives for the non-redundant case;
- curriculum over orientation difficulty or target distance;
- exploration/std tuning per action dimension (position vs orientation vs
  stiffness dims);
- anything specific to Isaac Lab's DifferentialIK / OSC implementations
  (documented quirks, recommended settings, known issues).

Rank the proposals by (expected impact × implementation cost), state for each
what evidence supports it, and flag which are directly testable in our setup
(one 100k-step run ≈ 36 min on one GPU).

---

## 4. Report format

1. **Executive summary** (≤ 1 page): verdicts on Q1 (expected? yes/no/partly),
   Q2 (config issues found, ranked by severity), Q3 (top 3 recommended fixes).
2. **Q1 analysis** with citations.
3. **Q2 config audit** — a table: config item → assessment (OK / suspicious /
   wrong) → evidence/source.
4. **Q3 recommendations** — ranked list with evidence, expected effect size,
   and concrete parameter suggestions.
5. **Bibliography** — the most important sources, ranked, with 1–2 sentence
   annotations (see citation requirements above).

Be honest about uncertainty: if the literature is thin or contradictory on a
point, say so rather than overclaiming.

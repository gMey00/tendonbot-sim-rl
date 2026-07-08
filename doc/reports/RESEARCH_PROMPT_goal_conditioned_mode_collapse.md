# Research Prompt — Escaping Goal-Conditioned Mode Collapse in On-Policy PPO

> Hand this entire document to a literature-research agent. It is self-contained:
> everything needed to understand the problem is below. The goal is a
> **literature-grounded, citation-backed** solution to a specific, reproducible
> RL failure, with concrete recommendations implementable in on-policy PPO.

---

## 0. Your task (read first)

We have a **goal-conditioned reinforcement-learning** control problem in which
on-policy **PPO** reliably **collapses to a subset of the commanded goals**
instead of learning all of them. We have already isolated the failure and ruled
out the obvious alternative explanations (details in §2–§4). We need you to:

1. **Diagnose** the failure against the peer-reviewed literature: what is this
   phenomenon called, what is its mechanism, and which papers establish it?
2. **Survey and rank candidate solutions** that are compatible with our setting
   (on-policy PPO, a *small number of discrete goals*, expensive rollouts).
   For each: the mechanism, the evidence it works, its assumptions/limitations,
   how well it fits our constraints (§5), and an implementation sketch.
3. **Recommend** a concrete, prioritized experiment plan.

**Every non-trivial claim must cite the literature** — author(s), year, title,
venue (conference/journal or arXiv id). Prefer primary sources; include both
seminal and recent (roughly 2015–2025) work. Be critical: explicitly flag
methods that are *natively off-policy* or designed for *many/continuous goals*
and explain whether/how they transfer to on-policy PPO with few discrete goals.
Do **not** invent citations; if you are unsure a reference exists, say so.

---

## 1. System under study

- **Task.** A 6-DOF robot arm (Universal Robots UR5e) with a parallel-jaw
  gripper holds a deformable object (a simulated T-shirt, PBD particle cloth)
  and must **drop it into one of three fixed bins** ("reusable", "recyclable",
  "trash") arranged around the robot. The scene is simulated in NVIDIA Isaac
  Lab / Isaac Sim.
- **Goal-conditioning (this is the crux).** Each episode a **target bin is
  sampled uniformly** from the three. The policy observes the *commanded* bin's
  position relative to the object and to the end-effector (a 3-D vector each,
  i.e. the goal enters as a **relative-position vector**, not a one-hot index).
  The success metric requires the cloth to be **released** and settle inside the
  **commanded** bin — so "nearest bin ≠ correct bin", and a policy cannot ignore
  the goal. This is the TossingBot-style formulation (Zeng et al., 2020) where
  the target is part of the observation.
- **Episode.** ~8 s at 60 Hz (≈480 steps). The episode starts with the object
  already grasped; the policy must carry it over the commanded bin and open the
  gripper (a binary action) to release it.

## 2. RL setup (exact, so you can judge method compatibility)

- **Algorithm:** PPO (clipped surrogate, GAE-λ), **on-policy**, implemented with
  the **skrl** library on top of Isaac Lab.
- **Actor–critic:** a **single shared Gaussian policy** MLP `[256, 128, 64]`
  ELU, and a **separate value MLP** of the same width (not a shared trunk).
  Both take the **same observation** (no privileged/asymmetric critic).
- **Preprocessing:** `RunningStandardScaler` on observations and on value
  targets (i.e. value/return normalization of the *aggregate* signal is already
  on — but it is **not per-goal**).
- **Key hyperparameters:** 64 parallel environments; rollout 48 steps;
  discount 0.99; GAE-λ 0.95; KL-adaptive learning rate (~1e-4);
  5 epochs × 8 minibatches per update; entropy coefficient 0.0–0.003 (see §4);
  Gaussian action std is a **state-independent, learned log-std** annealed by the
  optimizer. Training budget ~96k policy updates-worth of timesteps
  (each run ≈ hours on one GPU because the cloth sim caps throughput at
  ~5 steps/s).
- **Observation** ≈45 dims: arm joint pos/vel, end-effector pose, cloth
  centroid + lowest-point state, **goal (bin-relative vectors, 6 dims)**,
  gripper closure, grasp flag, previous action.
- **Reward (per-goal symmetric):** dense shaping toward the commanded bin
  (tanh distance, faded once positioned to prevent hovering), a **graded
  one-shot release bonus** when the gripper opens over the commanded bin
  (scaled by centering × clearance), a **release-required** per-step success
  term for cloth inside the commanded bin, a penalty for releasing over the
  wrong bin, and standard smoothness/безопасность regularizers. Reward
  structure is **identical across the three goals** (nothing favors any bin).

## 3. The failure (reproducible)

Trained policies exhibit **per-seed goal specialization**: each random seed
learns to place into **exactly two of the three bins and abandons the third**
(deterministic success ≈ 0 on the dropped bin). *Which* bin is dropped is
**seed-dependent**. Representative deterministic-evaluation results
(mean actions, 96 episodes/seed, three independent seeds of the strongest
configuration):

| Seed | bin 0 | bin 1 | bin 2 | overall |
|------|-------|-------|-------|---------|
| A | 0.91 | 0.61 | **0.09** | 0.52 |
| B | 0.86 | **0.00** | 0.81 | 0.51 |
| C | 1.00 | 0.92 | **0.00** | 0.63 |

- One bin ("bin 0") is consistently the **easiest** and is essentially always
  learned; the policy sacrifices one of the other two.
- The **best single checkpoint that covers all three** reaches only
  0.77 / 0.42 / 0.60 (overall 0.60).
- **Target:** ≥ **0.85 success on every bin** simultaneously.

## 4. What we have already ruled out (do not re-propose these as the fix)

- **Reachability / kinematics:** a **scripted** inverse-kinematics baseline
  places into **all three bins at 0.88** overall — every bin is reachable and
  the MDP is correct. The gap is a *learning/optimization* failure, not a
  physical one.
- **Exploration temperature:** raising the initial action std **and** adding an
  entropy bonus (0 → 0.003) did **not** break the collapse — σ stayed higher
  throughout yet each seed still dropped a bin. (This is our strongest evidence
  that the cause is **not** an exploration-schedule problem.)
- **Action-space pathologies, reward-shaping bugs, a spurious
  termination "cliff", release-exploration collapse, and a train/eval
  initial-state-distribution mismatch** — all found and fixed earlier; the
  collapse persists after all of them are corrected.
- **Determinism gap:** we already select checkpoints by deterministic
  (mean-action) evaluation, not by training-reward.

## 5. Constraints the solution must respect

- **On-policy PPO** is the incumbent (Isaac Lab + skrl). Solutions that are
  *natively off-policy* (e.g. replay-buffer methods) are acceptable to discuss
  but must come with an explicit assessment of how they'd work — or not — under
  on-policy PPO, or what it would cost to switch frameworks.
- **Few, discrete goals (exactly 3)** — not the "many/continuous goals" regime
  much of the goal-conditioned literature targets. Note where a method's
  benefit depends on goal cardinality/continuity.
- **Expensive rollouts** (~5 env-steps/s; hours per run, one GPU). Favor
  sample-efficient, low-run-count interventions; note compute cost.
- **Minimal-invasive preferred** but not required: changing the value-function
  head, observation/goal encoding, minibatch composition, reward normalization,
  auxiliary losses, or the goal-sampling curriculum are all in scope. A larger
  architecture change is acceptable if the evidence is strong.

## 6. Research questions

1. **Naming & mechanism.** What is the established term(s) for "on-policy /
   actor-critic RL converging to a subset of goals/tasks and ignoring others"?
   Candidate framings to verify against the literature (do not assume any is
   correct): multi-task **negative transfer** / **gradient conflict**;
   **value-function (critic) interference** across tasks; **winner-take-all /
   mode collapse** in multimodal or multi-goal policies; **primacy bias** /
   loss of plasticity; **capacity/representation bottleneck** in a shared
   network. Which of these are documented, with what evidence, and which best
   matches the symptom "a *shared critic* devalues the hardest-so-far goal so
   its advantage goes negative and the policy is pushed away from it"?
2. **Solution families.** For each of the following, report the key papers, the
   mechanism, empirical evidence, assumptions, and fit to §5. Add any families
   we've missed.
   - **Per-task/per-goal value or return normalization** (e.g. PopArt-style
     adaptive normalization) and **multi-head / separate critics per goal**.
   - **Multi-task gradient methods**: gradient surgery / projecting conflicting
     gradients, conflict-averse gradient descent, loss/gradient balancing.
   - **Goal representation & conditioning**: universal value function
     approximators; goal one-hot vs relative-vector vs learned embedding;
     feature-wise conditioning (FiLM-style) of the policy/value on the goal.
   - **Relabeling**: hindsight experience replay and its variants — and
     critically, **whether/how relabeling is usable in on-policy PPO** (e.g.
     on-policy or importance-weighted hindsight variants).
   - **Curriculum / adaptive goal sampling**: sampling goals by learning
     progress or difficulty; automatic curricula; whether non-uniform goal
     sampling provably counteracts winner-take-all.
   - **Policy factorization / distillation**: per-goal experts distilled into
     one conditioned policy; mixture/multimodal policies.
   - **Optimization/plasticity**: entropy schedules for multimodal coverage,
     resetting/regularization against primacy bias, larger networks.
3. **Direct-symptom evidence.** Is the specific pattern — *independent seeds
   each solving a **different** subset of goals* — described anywhere as a
   diagnostic of a particular cause (e.g. symmetry-breaking into local optima,
   critic interference)? Cite it.
4. **Ranked recommendation.** Given §5, which 2–4 interventions are the highest
   expected value, in what order, and why? For the top pick, give a concrete
   implementation sketch for PPO (what to change in the critic/loss/sampling),
   the expected mechanism of action, and how to measure success (per-goal
   success curves, per-goal advantage/return statistics, gradient-conflict
   metrics).

## 7. Deliverable format

A structured report:

- **§A Diagnosis** — the phenomenon named and mechanistically explained, with
  citations; an explicit statement of which documented cause best fits our
  symptom and why.
- **§B Solution survey** — one subsection per solution family from §6.2, each
  with: mechanism · key citations · evidence · assumptions/limits · fit to our
  constraints (on-policy PPO, 3 discrete goals, expensive rollouts) · rough
  implementation effort.
- **§C Ranked recommendation & experiment plan** — top 2–4 interventions
  ordered by expected value, with an implementation sketch for the #1 pick and
  the metrics that would confirm the mechanism.
- **§D References** — full bibliography (author, year, title, venue/arXiv id).
  Distinguish clearly between claims you are confident are supported and any you
  could not verify.

**Tone:** critical and evidence-weighted, not a listicle. Where the literature
is mixed or a popular method would *not* transfer to our setting, say so plainly
and cite the reason.

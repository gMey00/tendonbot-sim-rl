# Research Prompt — From Three Stage-1 RL Tasks to the Camera-Realistic Multi-Agent Shirt-Sorting Pipeline (Isaac Lab / Isaac Sim 5.1)

> **How to use this file.** Paste everything below the horizontal rule into a
> deep-research AI (one that can search and cite primary sources). It is written
> to be self-contained: the target AI has **no access to the code repository**,
> so all necessary context — including every relevant measured result — is
> included inline. The expected output is a *design + annotated roadmap with
> citations* that a coding agent (Claude Code, full repo access) will then
> implement task by task.

---

## ROLE

You are a research assistant with expertise in (a) deformable-object (cloth)
manipulation with reinforcement learning, (b) multi-agent / bimanual robot
learning (MAPPO, IPPO, centralized-critic PPO, single-policy bimanual control),
(c) sim-to-real transfer for cloth manipulation and vision-based garment
perception, and (d) GPU-parallel simulation (NVIDIA Isaac Sim 5.1 / Isaac Lab /
PhysX 5 PBD particle cloth). You are helping design the **second half of a
master's thesis**: the evolution of three individually-trained RL tasks into one
coherent, camera-realistic, potentially multi-agent shirt-sorting pipeline.
Every non-trivial claim, architecture recommendation, and design pattern you
propose **must be backed by a citation to a primary source** (peer-reviewed
papers, official NVIDIA/Isaac Lab documentation, published benchmark suites).
The student cites your sources directly in the thesis — source quality and
traceability are the top priority. Where the literature is thin (e.g., MARL for
bimanual cloth stretching), say so explicitly and reason from the nearest
precedents instead of overclaiming.

## OVERALL GOAL

Produce **the ideal design and a staged, decision-gated roadmap** for turning
the existing three-task pipeline (all three tasks already train and reach
their stage-1 bars, all measured numbers below) into the best achievable
version of: *robot 1 picks a crumpled shirt from a trash conveyor and holds it
up; robot 2 grasps a second point and stretches the shirt into a maximally
inspectable presentation for a fixed front + back camera pair; robot 2 then
drops the shirt into the condition-matched drum.* The five research goals in
§RESEARCH QUESTIONS structure your deliverable. The design must be directly
implementable by a coding agent in the described codebase — concrete MDP
changes, algorithm choices with library-level feasibility notes, and an
ordered plan with success criteria and abort conditions.

---

## PROJECT CONTEXT (ground truth — treat as fixed, measured facts)

### Hardware, scene, and robots

- **Scene** (one lane, all tasks share it): a trash conveyor belt (surface
  height 0.80 m); a gantry-mounted 5-DOF **tensegrity manipulator** ("retriever",
  robot 1) hanging above the belt at mount (0.15, 0, 2.30) — 2 prismatic base
  DOF (y, z) + tendon-driven elbow + 2-DOF wrist; **its base cannot translate in
  x** (x is only weakly controllable through the arm); a **UR5e + Robotiq 2F-140**
  (robot 2) on a pedestal at (0.75, 1.0, 0.75); three open drums (bins) at
  (0.15, 1.0), (1.35, 1.0), (0.75, 1.6), rim height 0.88 m; a fixed **inspection
  camera pair**: front camera at (0.15, 2.0, 1.25) looking along −Y, and a
  mirrored back view — the image plane is the world XZ plane. **Only these two
  camera views exist; adding more cameras is out of scope (user constraint).**
- The presentation anchor (where robot 1 holds the shirt between tasks) is
  (0.15, 0.90, 1.60).
- **Only two robot arms are available (user constraint)** — no third
  simultaneous grasp, no fixtures beyond the drums/belt.
- A Kinova Gen3-7DOF + F140 variant exists for every robot-2 task as a
  registered fallback; UR5e is the primary (chosen by a reach-task grid:
  best-variant 1.8 cm / 96 % vs Kinova).

### Simulation stack and performance envelope

- Isaac Sim 5.1 + Isaac Lab manager-based RL envs; PhysX 5 **PBD particle
  cloth** (t-shirt: 11 048 particles, ~5 mm spacing, one-sided flat area
  0.294 m²), 60 Hz / decimation 1, 24 PBD solver iterations. skrl 2.1.0 PPO;
  the repo's `train.py` also exposes **IPPO and MAPPO** (skrl multi-agent) and
  a `multi_agent_to_single_agent` wrapper.
- **Throughput knee (measured):** cloth env stepping peaks at **64 envs
  (~400 env·steps/s)** on the NVIDIA RTX A6000 (48 GB) and *falls* above 128
  envs. Training runs use 64–128 envs; an HPC cluster (multi-GPU, SLURM) is
  available for seed sweeps. One Isaac process per GPU.
- **Caveat (unverified):** whether installed skrl 2.1.0 truly separates policy
  vs critic observation groups for asymmetric actor–critic is an open repo
  TODO; RSL-RL is the designated fallback if it aliases them.
- Isaac Lab structural note: the manager-based env API used by all three tasks
  is **single-agent**; Isaac Lab's multi-agent support (`DirectMARLEnv`) lives
  in the direct-workflow API. Any MARL recommendation must state how to bridge
  this (rewrite as direct env, one joint-action single agent in the
  manager-based API, or wrapper-level splitting).

### The grasp model (this is what "gripper realism" currently means)

All cloth grasping is a **deterministic attachment**, not contact physics:
when the (virtual) finger tip is within **0.10 m** of the designated grasp
point AND the finger close command exceeds a threshold, all cloth particles
within a **0.07 m radius** of the tip are kinematically anchored to the tip
("ANCHOR" attachment, union-composed masses); the anchor is re-pinned to the
moving tip every step; commanding the gripper open detaches. Two independent
attachment slots exist (robot 1's hold + robot 2's grasp); a two-attachment
stretch is validated stable to tautness ratio ≤ 1.15. Consequences:

- Grasps **cannot fail, slip, or grab the wrong layer**; grasp quality is
  binary and perfect. The policy's "choose where and when to close" is real,
  but everything after closure is idealized.
- The Robotiq 2F-140 is articulated and physically collides (main drive
  stiffness 11.25, effort 10 N, mimic finger joints; identical config on all
  arms), but its contact forces play **no role in holding the cloth**. The
  finger tip is a *virtual frame* (tool_link_0 + calibrated offset; the
  gripper's body frames are co-located at the flange — a known asset quirk).
- PBD particle cloth ↔ rigid body contact exists (the shirt rests on the belt,
  falls into drums), but pinch-grasping a 2-layer cloth between fingertips via
  friction was never attempted in this project.

### Task 1 — `shirt_pick` (tensegrity retriever, DONE to stage-1 bar)

- **MDP:** pick the crumpled shirt from the belt and hold it at the
  presentation anchor. Observations (~35-dim): controlled joint pos/vel,
  EE position, shirt-centroid relative position, **grasp-target (highest
  point) relative to finger tip**, shirt centroid velocity, presentation-pose
  relative position, gripper closure, grasp-active flag, last action. Actions:
  joint-position deltas + binary gripper. Rewards: sequential
  reach→grasp→lift→transport→presented (per-step, grasp-gated transport) +
  one-shot drop penalty. Success gate: grasp point near pose + **cloth
  centroid speed** < 0.20 m/s windowed (≥ 80 % of last 1 s) — EE-speed gates
  are unharvestable (arm sway 0.24–0.27 m/s at the raised posture).
- **Initial states:** a cached bank of 288 pre-settled crumpled states
  ("trash-toss" protocol: piles ⌀ 0.10 m, footprint 0.72× flat), restored
  bit-exact in 1.2 ms with yaw + mirror augmentation.
- **The pick heuristic is highest-point** — chosen deliberately as the
  depth-camera-trivial grasp (literature-standard bootstrap; semantic
  keypoints are known-unreliable on crumpled cloth).
- **Trained result:** deterministic present **1.000 / grasp 1.000** (96
  episodes × 2 seeds; scripted baseline 0.125). Known blemish: 5–7 % of
  episodes slip the grasp *after* the success latch. A terminal-state
  snapshot hook exists but the Task-1→2 terminal bank has not been generated.

### Task 2 — `shirt_present` (UR5e-F140, the pipeline's weak link)

- **Current geometry (after a 2026-07-08 redesign):** robot 1 is a *passive,
  IK-posed scenery arm* rigidly holding the shirt at the anchor; the shirt
  hangs from a **random bank particle** (356-state hanging bank, yaw+mirror
  augmented). Robot 2 must grasp a **hem corner** and pull it **horizontally
  along camera-plane x at the holder's height** (the study-validated
  presentation, below), holding a windowed present latch (coverage +
  tautness + stillness gates).
- Observations: joint pos/vel, EE pos, shirt centroid rel, **targeted hem
  corner rel to finger tip**, **horizontal-pull goal rel to tip**, shirt
  velocity, both-grasps-active flags, tautness ratio, camera-plane coverage,
  gripper closure, last action. (All designed to be "camera-derivable in
  principle", but currently read from privileged particle state.)
- Rewards: reach shaping, one-shot grasp bonus (200), grasp hold, pull-target
  shaping, tautness-band progress, coverage, per-step presented (30),
  overstretch penalty from 1.10, one-shot drop, small camera-occlusion
  penalty, action regularizers on a curriculum.
- **Trained results:** the older *naive lowest-point* variant reached
  deterministic present **0.927 @ final coverage 0.68** (its env is now
  superseded; checkpoint archived). The current **hem↔hem variant reaches only
  present 0.380 @ coverage 0.637, grasp 0.73**. The two measured limiters:
  1. **The high second-hem-corner grasp is RL-unlearnable so far** (grasp
     rate < 0.1 when the target corner hangs high); only the low/accessible
     corner is learned.
  2. **A random first grasp caps coverage at ~0.64**: the holder's grasp point
     is whatever bank particle the episode restored, and most first-grasp
     regions cannot reach the 0.73–0.82 coverage of the study's best pairs.
  Together these motivate the central question of this prompt: *the first
  robot's grasp choice and (possibly) active cooperation determine the
  achievable presentation — one-sided learning by robot 2 appears
  insufficient for anything beyond lowest-point.*
- Hard-won reward lessons (do not contradict): never penalize "gripper closed
  while far" (it teaches never-close, grasp rate → 0); tautness must be
  normalized by flat-rest distance (geodesic over-reads wrap-around pairs);
  one-shot rewards need ~60× per-step weights (Isaac Lab multiplies rewards
  by dt = 1/60); entropy 0 + tight log-std bounds fix a stochastic/
  deterministic play gap; select checkpoints only by deterministic eval on
  unseen seeds.

### Task 3 — `shirt_distribute` (UR5e-F140, DONE to stage-1 bar on 2 of 3 bins)

- Goal-conditioned drop into the commanded drum, initialized "shirt already in
  gripper" from a cached holding-pose bank. A goal-conditioned **mode collapse**
  (every PPO seed zeroing one bin) was diagnosed as cross-goal critic
  interference and solved with per-goal value/advantage normalization,
  a per-goal multi-head critic, and **FiLM goal-gating + per-goal actor heads**:
  mean success 0.596 → **0.882** (bins 0.910 / 0.899 / 0.832). Bin 2 (behind
  the pedestal arm) sits at a reachability ceiling ≈ 0.83 (scripted baseline
  itself 0.88). Open: bin-layout randomization (nearest ≠ correct), and the
  Task-2→3 seam (currently a synthetic holding-pose bank, not real Task-2
  terminal states).

### The presentation-heuristics study (robot-free; the geometry ground truth)

A 15 000+-trial scripted study (no robots — attachment anchors only) measured
presentation quality as **camera-plane silhouette coverage** (rasterized
projected silhouette ÷ flat one-sided area; front+back cameras see the same
silhouette). Key calibrated facts your design must build on:

- **The horizontal presentation stretch** (second grasp raised to the first
  grasp's height, pulled along camera-plane x; gravity extends the garment
  below the taut chord) **self-aligns the garment to the camera plane**
  (yaw-sweep gap ≈ 0.003) — no orientation control is needed.
- **hem_corner ↔ hem_corner is the winning grasp pair: 0.820 median coverage
  @ tautness 1.05** (p90 0.95) — it *matches* the realizable oracle bound (the
  best cell of an 11 072-trial brute-force pair map, 0.82). The naive
  lowest-point second grasp reaches 0.679; free hang 0.567;
  shoulder↔shoulder (the human grip) FAILS at 0.493. Tautness beyond 1.05 is
  not significant (and > 1.0 is partly PBD overstretch).
- **The best second grasp is a hem corner / bottom-centre for EVERY first-grasp
  region** (stratified map, ≥128 trials/cell, opposite-side sampling). An
  oracle region→region policy from an *arbitrary* first grasp reaches 0.713
  (+0.034 over lowest-point). A JSON policy table (first-grasp region → best
  second-grasp region) exists.
- **Regrasping during presentation HURTS with either target rule** (paired
  −0.031 lowest-point / −0.051 oracle-target, sign-test p ≈ 1.5×10⁻⁵): once
  robot 1 releases, the new holder is a garment extremity and the
  extremity↔extremity chord fills poorly. **Robot 1 must keep holding.**
  (Note: this is a *single regrasp inside the presentation procedure*; the
  literature's iterative-regrasp *unfolding* to reach known corners — e.g.
  Doumanoglou 2014 — is a different, multi-cycle process and was NOT tested.)
- **Grasping near a garment opening (border) is a distinct quality lever**
  (+0.08 monotone for the moved grasp; the anchor grasp's border distance is
  irrelevant), and **12 symmetric region-landmark keypoints** exist on the
  garment: a keypoint-restricted chooser reaches the oracle (best cells
  side→hem_corner 0.87, hem↔hem 0.80) — i.e., **perception-friendly keypoints
  are sufficient grasp targets on the hanging shirt**.
- A **tautness pulse** (overstretch to 1.10, relax to 1.05) adds no coverage
  but **halves the settle time** — overshoot-then-relax is legitimate policy
  behaviour.
- An **occlusion-aware companion metric** (two-sided visible fraction from
  exactly the front+back cameras; per-raster-cell depth-layer clustering)
  confirms the coverage ranking is not distorted by hidden layers (hem↔hem
  0.990 visible fraction; all two-point presentations ≥ 0.95).
- **Recommended success threshold: coverage ≥ 0.65** to start (the current RL
  gate is 0.50 — a known pending re-threshold), 0.75 as the curriculum end.
- Limits of the study: static anchors (no arm kinematics/occlusion except a
  cosmetic term), one garment (one 11 k-particle ClothesNet t-shirt), grasp =
  7 cm attachment patch, stretch speed 0.45 m/s quasi-static.

### Skill chaining infrastructure

Cached particle-state banks connect the tasks (crumpled bank → Task 1;
hanging bank → Task 2; holding-pose bank → Task 3). Terminal-state snapshot
hooks exist for Task 1 and Task 2 but the *real* chained banks (Task-1
terminals feeding Task 2, Task-2 terminals feeding Task 3) have not been
generated — chaining distribution shift is the project's #1 flagged risk.

---

## RESEARCH QUESTIONS (structure your deliverable around exactly these five)

### Q1 — How should the three tasks be adapted, in their current single-agent form, to maximize presentation quality end-to-end?

The study proves presentation quality is decided by the **grasp pair**, and the
first grasp is chosen in Task 1 (currently: highest point of a crumpled shirt —
camera-trivial but region-random). Address concretely:

1. Should Task 1 be re-targeted to produce hem-corner-region first grasps —
   and is that even perceivable on a crumpled shirt (the literature says
   semantic keypoints are unreliable when crumpled)? Compare: (a) keep
   highest-point pick + oracle second-grasp policy in Task 2 (calibrated
   ceiling ≈ 0.713); (b) pick-then-rehang cycles on robot 1 until a hem
   region is held (iterative unfolding literature — cycle cost vs +0.11
   coverage); (c) a learned Task-1 grasp-point policy trained against
   downstream presentation reward (grasp-point value maps / affordance
   learning on cloth); (d) two-stage Task 2 where robot 2 first brings the
   shirt into a canonical state. Quantify expected coverage and cite
   precedents for each.
2. Task 2 fixes independent of the pair question: curriculum designs that make
   the high hem corner learnable (grasp rate < 0.1 today) — e.g., bank
   filtering so the target corner starts low/near (the hanging bank stores the
   anchor particle and drape; hem-corner-anchored states can be selected),
   reach-assist resets, or target-conditioned grasp curricula. Also: raising
   the success gate 0.50 → 0.65, and whether the reward should adopt the
   study's pulse/settle-time findings.
3. Task 3: is anything presentation-relevant left (e.g., preserving the
   inspected state until the drop decision), or is it purely the bin-2
   reachability + layout-randomization backlog?
4. The chaining question: in what order should the real terminal banks be
   generated and the tasks re-trained against them, given the measured 5–7 %
   post-latch slips in Task 1?

### Q2 — Should Task 2 be reformulated as a multi-agent task (MAPPO or otherwise), with robot 1 an active participant?

The evidence motivating this: one-sided learning by robot 2 has only reached
0.927 with the lowest-point grasp and 0.380 with hem↔hem; the study says the
*pair* — and therefore robot 1's behaviour — is the dominant lever, and robot 1
must keep holding (so its cooperation is holding-pose control, not release).
Address:

1. The formulation space, with literature precedent for each: (a) **MAPPO/IPPO**
   (two agents = two arms; skrl provides both, but Isaac Lab manager-based
   envs are single-agent — state the concrete integration path, e.g.
   DirectMARLEnv rewrite vs `multi_agent_to_single_agent`); (b) **one joint
   policy over both arms' actions** (bimanual cloth works — FlingBot,
   SpeedFolding-style — typically use ONE policy; is CTDE actually needed when
   both arms are simulated centrally and there is no communication
   constraint?); (c) **hierarchical/sequenced**: a grasp-selection decision
   (where robot 1 holds / where robot 2 grasps) + a continuous stretch
   controller, e.g. an affordance/value-map chooser feeding a low-level
   policy; (d) keep robot 1 scripted but *movable* (holder repositions to a
   study-informed pose — note the tensegrity cannot translate in x).
2. A clear recommendation with decision criteria: what measurable evidence
   (e.g., hem↔hem grasp-rate after the Q1 curricula, coverage ceiling vs
   holder pose) should trigger moving from single-agent to the recommended
   cooperative formulation?
3. Feasibility constraints to respect: robot 1 is the 5-DOF tensegrity with
   weak wrist torques and no x-translation; both-arm cloth stretching is
   validated only to tautness 1.15; 64-env throughput bounds sample budgets;
   the two grasps are attachment slots (no handover physics needed).
4. Credit assignment and reward design for the cooperative case (shared
   presentation reward vs per-arm shaping), and known MARL pathologies
   (lazy-agent, non-stationarity) in this 2-agent, fully-observable,
   centrally-simulated setting.

### Q3 — How should observations be redesigned so the policy inputs are only things obtainable from real cameras/sensors — and should camera/sensor simulation be included directly?

Current observations are privileged particle-state reads labelled
"camera-derivable in principle". Deliver:

1. A per-task **observation contract**: for each observation term, the real
   perception source that would produce it (depth highest point; segmented
   point cloud down-sampled to N points; segmentation-mask area → coverage;
   keypoint detector + per-keypoint visibility flags on hanging/stretched
   cloth; grasp-state estimation — how does a real system know "grasp
   active"?; force/torque or gripper-current sensing on the F140), with
   accuracy/noise figures from the literature so noise models can be
   calibrated. Note the study's 12 symmetric region keypoints and the
   border-distance feature as candidate detector outputs.
2. The **strategy question**: state-based training with calibrated noise +
   domain randomization now, camera simulation later (teacher–student
   distillation, e.g. DAgger-style, into a vision policy) — versus tiled-
   -rendering cameras in the training loop directly. Include Isaac Lab tiled
   rendering throughput evidence and the project's 64-env knee; the user's
   stated preference is to keep the RL loop camera-free first and keep the
   door open — challenge or confirm this with evidence.
3. Asymmetric actor–critic design: what goes in the privileged critic
   (full particle field, true tautness/coverage, attachment state) vs the
   camera-realistic actor, and the skrl-2.1.0 caveat (verify or switch to
   RSL-RL).
4. Which observation gaps are *load-bearing for sim2real* (e.g., grasp-active
   flags and exact hem-corner positions are free in sim but hard on a real
   system) and in what order to close them.

### Q4 — What existing sim-to-real work implements a comparable pipeline, and what transfers?

Survey and synthesize (this feeds the thesis's related-work chapter as well):

1. Dual-arm garment inspection/unfolding/sorting pipelines: Maitin-Shepard
   (towel folding), Doumanoglou et al. 2014 (dual-arm unfolding via hanging +
   keypoints — the closest ancestor of this pipeline), ICRA 2024 Cloth
   Competition (in-air unfolding, coverage metrics), SpeedFolding, FlingBot,
   Cloth Funnels, UniFolding, GarmentLab / ClothesNet ecosystems; industrial
   deployments (e.g. sewts VELUM — crumpled textiles from conveyors — and
   similar laundry-automation systems).
2. For each: what was trained in sim vs engineered, what perception stack, how
   the grasp was modelled in sim vs executed on hardware, and the reported
   sim2real gap mechanisms for PBD/particle cloth (PhysX cloth parameter
   identification, real2sim system ID for garments).
3. Specifically: is there ANY published case of a policy trained with
   *attachment-based* (kinematic) cloth grasps transferring to a physical
   gripper — or is a contact-physics or grasp-success-model intermediate
   always used? This directly gates Q5.

### Q5 — How should the gripper implementation be made physically more accurate, within the simulation constraints?

Current state: deterministic attachment (described above), articulated F140
whose contacts are cosmetic for grasping. The user wants higher physical
fidelity *without* breaking the working pipeline. Evaluate, with feasibility
evidence for PhysX 5 particle cloth:

1. **Contact/friction pinch grasping** of PBD cloth between the F140 pads:
   is two-sided friction pinch of a multi-layer particle cloth stable in
   PhysX 5 at 60 Hz / 24 iterations, at all? (PhysX particle-rigid adhesion
   and friction parameters; GarmentLab and RFUniverse design choices; SoftGym
   picker precedent — most sim benchmarks also use attach-style pickers, why?)
2. **Intermediate fidelity models**, ordered by cost: (a) keep attachment but
   gate success on *physically plausible pre-conditions* (approach alignment,
   finger clearance, single- vs multi-layer pinch detection at the grasp
   point); (b) stochastic grasp-failure/slip models calibrated from published
   cloth-grasp success statistics (grasp point on seam/hem vs mid-panel,
   layer count, pull force vs slip); (c) force-limited attachments (detach
   when the anchor constraint force exceeds a calibrated pinch capacity —
   note the two-attachment stretch already operates near 1.15 tautness);
   (d) full contact pinch only for a validation study, not training.
3. What the 2F-140's real cloth-grasping behaviour looks like (pad geometry,
   85–140 mm strokes, grip force 10–125 N, published cloth picking studies
   with parallel-jaw grippers) and which sim knobs (mimic-joint drives,
   effort limits, virtual-tip calibration) matter for matching it.
4. A recommendation: which fidelity level should the *training* grasp use,
   which the *evaluation* grasp, and what experiment validates that the gap
   between them does not change policy ranking (the same logic as the
   study's occlusion-metric check).

---

## CONSTRAINTS AND PRIORITIES (user-set, treat as fixed)

- Front + back inspection cameras only; two arms only.
- The observation design flows **heuristics-study → RL task** (the study's
  findings define what the tasks should perceive and reward), not the other
  way around.
- Camera/vision models stay a black box for now: observations must be
  *realistically obtainable*, but no camera pipeline in the training loop
  unless your evidence says the cost is worth it; keep the door open.
- Do not regress the stage-1 results (pick 1.000, distribute 0.882, present
  0.927-naive / 0.380-hem↔hem) without a measured reason; every redesign needs
  a decision gate ("adopt if X beats Y on Z").
- Budget realism: A6000 48 GB primary (64–128 cloth envs, ~400 env·steps/s),
  HPC cluster for seed sweeps; single-GPU training runs of 20 k–100 k+
  trainer steps are the norm; the remaining thesis time is a few months —
  the roadmap must fit that and mark cut-lines.

## DELIVERABLE (structure your answer exactly like this)

1. **Executive summary** — the recommended pipeline architecture in one page:
   which tasks change, which formulation Task 2 adopts, target metrics.
2. **Q1–Q5 answers** — each with: evidence review (cited), the concrete
   recommendation, the MDP/implementation change list (observation terms,
   reward terms, curriculum, algorithm/config), and the decision gate that
   would reverse the recommendation.
3. **Roadmap** — ordered stages with: goal, tasks, measurable success
   criterion, estimated effort (agent-days of implementation + GPU-days of
   training), dependencies, and abort/fallback per stage. Mark which stages
   are thesis-critical vs stretch.
4. **Risk register** — top risks (chaining distribution shift, MARL
   integration with manager-based envs, grasp-fidelity rabbit hole, skrl
   asymmetric-AC uncertainty, throughput ceilings) with mitigations.
5. **Annotated bibliography** — every cited source with 1–2 lines on what it
   contributes and how much to trust it for this design.

## KNOWN UNCERTAINTIES TO FLAG IN YOUR ANSWER

- All coverage numbers above come from *this project's* metric (rasterized
  silhouette ÷ flat area) on one garment; cross-paper coverage numbers use
  different normalizations — align them before comparing.
- The heuristics study used static anchors, not arms: arm kinematics,
  reachability, and camera occlusion by the arm may shave the calibrated
  ceilings (the 2.9 cm holder-IK residual and the x-translation-free
  tensegrity base are real examples).
- skrl MAPPO/IPPO exist in the training script but have never been run in
  this repo; treat their maturity as unverified.
- The PBD cloth is one t-shirt at one resolution; garment generalization is
  out of scope for the thesis but note where the design would break.

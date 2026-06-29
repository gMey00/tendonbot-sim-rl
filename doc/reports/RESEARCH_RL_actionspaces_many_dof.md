# RL Action Spaces for Redundant & Tendon-Wrist Manipulators in Isaac Lab / skrl PPO: Diagnosing the Kinova Gen3 Pose-Reach Failure

## TL;DR
- **The Kinova failure is overwhelmingly a target-generation problem, not an action-space problem.** Your FK-sampling spans nearly all of SO(3), but the stock Isaac Lab reach benchmark that the UR arms "win" on samples a *constrained* orientation set (roll=0, pitch=π/2 fixed, yaw=±π) — so you are unknowingly giving Kinova a vastly harder full-SO(3) orientation task while giving UR an effectively single-DOF (yaw-only) orientation task. Fix the target distribution first.
- **Switch the action space to task-space (differential IK-rel for reach→pick, OSC/task-space impedance for cloth/contact), encode the 4 continuous joints as sin/cos in observations, and raise the orientation reward weight** (currently -0.1 with geodesic `quat_error_magnitude`, dwarfed by the -0.2 position term plus a +0.1 fine-grained position bonus that orientation lacks entirely). These three changes together should move Kinova from 0–19% to UR-comparable success.
- **For the roadmap (reach→pick→cloth→multi-arm), standardize on a relative task-space action in the EE/root frame.** It handles redundancy implicitly, transfers cleanly across 6- and 7-DOF arms, and scales to decentralized multi-agent PPO. Use null-space posture control (OSC) to tame the redundant elbow and the tendon wrist.

## Key Findings

**1. The smoking gun: your benchmark is not the same task for UR and Kinova.** Isaac Lab's stock manager-based reach command (`UniformPoseCommandCfg`) for the UR arms samples `pos_x=(0.35, 0.65), pos_y=(-0.2, 0.2), pos_z=(0.15, 0.5), roll=(0.0, 0.0), pitch=(math.pi/2, math.pi/2), yaw=(-3.14, 3.14)` — i.e., orientation is effectively a single free DOF (yaw), with roll/pitch pinned (confirmed verbatim from the NVIDIA "Getting Started With Isaac Lab" reach tutorial and the `ur_10/joint_pos_env_cfg.py` source, which overrides only pitch). Your Kinova target generator instead reads FK from random joint configs, so its target orientations cover essentially all of SO(3). A policy that only has to learn a 1-parameter orientation family will look like it "learns orientation" (your UR arms' 4–11° error); a policy that must hit arbitrary SO(3) orientations through a redundant arm with a weak orientation reward will plateau (your Kinova's ~27–35°). This single difference explains the bulk of the gap.

**2. The orientation reward is structurally under-weighted.** The Isaac Lab reach reward uses `orientation_command_error`, which is implemented (confirmed verbatim from `reach/mdp/rewards.py`) as `quat_error_magnitude(curr_quat_w, des_quat_w)` — the geodesic/shortest-path quaternion distance — at weight **-0.1**. Position uses `position_command_error` (L2 norm) at **-0.2** *plus* a `position_command_error_tanh` fine-grained bonus `1 - tanh(distance/std)` at **+0.1** (std 0.1). The NVIDIA tutorial explicitly notes the tanh term "produces larger gradients … amplifies weight updates for small mistakes" — orientation has no such fine-grained term. So position has a dense, well-shaped basin with a positive "homing" gradient near the goal, while orientation has only a single coarse linear penalty. At ~30° error (0.52 rad) the orientation term contributes ≈ -0.052/step — negligible against position shaping. The policy rationally ignores orientation. This matches published practice: Jiang et al.'s loco-manipulation work (IEEE RA-L 10(2):1481–1488, 2025; arXiv:2412.03012) states "Only when the target pose is within the workspace of the arm does orientation tracking make sense … there is an inner priority between position tracking and orientation tracking," and resolves it with multiplicative Reward Prioritization that "establishes priorities among rewards, simply through multiplication."

**3. Continuous joints (J1/J3/J5/J7) hurt twice — once in target generation, once in observation encoding.** Per the Kinova Gen3 datasheet, the arm has "Infinite rotation on all joints" (7 DoF, 902 mm max reach, 4 kg continuous payload, 1 kHz low-level closed-loop control). Two distinct issues arise: (a) FK-sampling from continuous joints produces wild orientation coverage (Finding #1); (b) feeding the raw joint angle to the policy is discontinuous at ±π, so two near-identical configurations can present as maximally distant inputs. The standard fix (DeepMind control suite, Furuta pendulum, many MuJoCo tasks) is sin/cos encoding of any wrap-around joint. There is also a PhysX subtlety you flagged: revolute limits are only valid in [-2π, 2π], PhysX articulation revolute joints auto-wrap position at ±2π (`eREVOLUTE` vs `eREVOLUTE_UNWRAPPED`), and finite-limit writes on USD-continuous joints do not reliably "stick" — so trying to clamp these joints to a finite range in config is fragile.

**4. Task-space / OSC action spaces are the literature-favored choice for the harder downstream tasks.** Martín-Martín et al. (VICES, IROS 2019; arXiv:1906.08880) directly compared joint position, joint velocity, joint torque, variable joint impedance, and end-effector-space impedance on Path Following, Door Opening, and Surface Wiping, and found Variable Impedance Control in End-Effector Space "improves sample efficiency, maintains low energy consumption, and ensures safety across all three experimental setups … RL policies learned with VICES can transfer across different robot models in simulation, and from simulation to real." Isaac Lab ships an `OperationalSpaceControllerActionCfg` / `OperationalSpaceControllerAction` (PR #913, ozhanozen) with optional null-space posture control for redundant arms, plus the `Isaac-Reach-Franka-OSC-v0` env.

**5. Self-collision / unreachability is a secondary but real contributor.** Because you sample joint configs and read FK, every target is by construction reachable by *at least one* configuration (the one you sampled). So pure unreachability is NOT your main cause. But you teleport into self-colliding configs without checking, and you never verify a *collision-free* IK branch exists from a sensible start — so some targets require the policy to pass through or end in awkward/near-singular postures. Best practice (reachability/capability maps, RM4D, manipulability filtering) is to reject ill-conditioned and self-colliding samples.

**6. There is a working reference implementation.** `louislelay/kinova_isaaclab_sim2real` trains a Gen3 7-DOF reach in Isaac Lab (rsl_rl/rl_games PPO) and deploys sim2real over ROS2 — it is built on the j3soon UR10 Reacher lineage and demonstrates a learnable Gen3 reach, which is strong evidence your failure is configuration-specific (target/reward/encoding), not a fundamental Kinova limitation.

## Details

### (1) Action-Space Comparison Table (7-DOF arm)

| Action space | Action dim (7-DOF) | Redundancy handling | Contact-rich | Cloth/deformable | Multi-agent | Sim2real | reach → pick → cloth rank |
|---|---|---|---|---|---|---|---|
| **Joint position (abs, PD)** | 7 (+gripper) | None (policy must learn null-space implicitly) | Poor–fair (stiff, no compliance) | Poor | Good (simple, per-agent) | Good (Gen3 takes joint targets natively) | reach: **1** / pick: 3 / cloth: 5 |
| Delta/relative joint position | 7 | None, but smoother exploration | Fair | Poor | Good | Good | reach: 2 / pick: 3 / cloth: 5 |
| Joint velocity | 7 | None | Fair | Poor | Good | Fair (needs good PD/gravity comp; Isaac Lab Franka vel control unstable w/ gravity OOTB, Issue #2807) | reach: 4 / pick: 4 / cloth: 5 |
| Joint torque/effort | 7 | None (full dynamics exposed) | Good (compliant) | Fair | Fair (harder credit assignment) | Hard (model/latency sensitive) | reach: 5 / pick: 5 / cloth: 4 |
| **Differential IK absolute (DLS)** | 6 (pose) or 3 (pos) | Implicit via DLS pseudo-inverse; redundancy resolved by controller, not policy | Fair (position-controlled, stiff) | Fair | Fair (world/root-frame targets) | Good (controller is deterministic) | reach: 2 / pick: 2 / cloth: 4 |
| **Differential IK relative (DLS)** | 6 | Implicit via DLS; relative deltas ease exploration | Fair–good | Fair–good | **Good** (frame-relative deltas compose across arms) | Good | reach: **1–2** / pick: **1** / cloth: 3 |
| **OSC / task-space impedance / VICES** | 6 (+ up to 6 stiffness for variable impedance, + null-space target) | **Explicit null-space posture control** (Isaac Lab OSC supports it) | **Best** (compliance, force control) | **Best** | Good | **Best** (operational-space transfers across dynamics) | reach: 3 / pick: 2 / cloth: **1** |
| Reduced / learned action space (latent) | <7 | Learned manifold can encode redundancy resolution | Task-dependent | Good | Good (CLAS: central latent action space) | Task-dependent | reach: 4 / pick: 3 / cloth: 2 |
| Residual RL on top of controller | 6 (residual) | Inherited from base controller | **Good** (base handles gross motion, RL handles contacts) | Good | Good | **Best** (base controller bounds exploration) | reach: 3 / pick: **1–2** / cloth: 2 |
| Relative-to-EE vs world/root frame | (frame variant of above) | — | EE-frame better for contact; world/root better for absolute goals | EE-frame for grasp-relative cloth moves | **Root/world frame for multi-arm shared targets** | EE-frame more robust to base placement | — |

**Recommended per task:** reach → **joint position (to debug) then differential IK-relative**; pick-and-place → **differential IK-relative or residual RL on an IK/pick controller**; cloth → **OSC / task-space impedance (VICES-style)**.

**Why redundancy changes the ranking.** With joint-position actions on a 7-DOF arm, the policy must implicitly discover a redundancy resolution (which elbow posture to adopt for each pose) — the network is solving IK *and* null-space optimization *and* the task simultaneously. Task-space actions (IK/OSC) offload IK to a deterministic controller, collapsing the policy's job to "emit a 6-DOF pose/wrench," which is identical for 6- and 7-DOF arms — that is exactly why your UR arms (where joint-space *is* effectively task-space-dimensioned) learn easily and Kinova does not.

### (2) Prioritized Experiment Plan (reach, predictive of harder tasks)

1. **Constrain the orientation target distribution to match the UR benchmark** (roll=0, pitch=π/2, yaw=±π). *Proves* the SO(3)-coverage hypothesis: if Kinova jumps to UR-like success, target generation is the cause. (Keep current joint-position action.) — Transfers fine to pick (grasps are usually near-vertical); for cloth you will re-widen later.
2. **Switch observation encoding of J1/J3/J5/J7 to sin/cos** (keep finite joints raw or also sin/cos for uniformity). *Proves* the discontinuity hypothesis. Cheap, always-on improvement. — Transfers to all tasks.
3. **Raise orientation reward weight and add a fine-grained orientation term** (e.g., orientation weight -0.5 to -1.0, plus `1 - tanh(d_θ/std)` bonus, or multiplicative prioritization `r = r_pos · r_ori`). *Proves* the reward-balance hypothesis. — Transfers to pick (grasp orientation matters); for cloth, orientation matters less than position/velocity.
4. **Swap to `DifferentialInverseKinematicsActionCfg` (IK-rel, DLS), `body_name="end_effector_link"`, scale 0.5.** *Proves* the action-space hypothesis and removes the implicit-IK burden. Pin to PhysX (Newton lacks batched Jacobian for IK). — This is your bridge to pick.
5. **Add OSC (`OperationalSpaceControllerActionCfg`) with null-space posture = joint centers.** *Proves* whether explicit redundancy/posture control matters and sets up contact-rich/cloth. — Direct path to cloth.
6. **Reachability/manipulability-filtered target sampling** (reject self-colliding and near-singular configs). *Proves* the ill-conditioned-target contribution. — Transfers to pick/cloth (filters bad grasps).
7. **Residual RL on top of IK-abs smoke-test controller.** Since IK-abs-as-action already drives EE to 1.1 cm, learning a small residual is trivial and de-risks pick/insertion. — Best transfer to contact-rich.

**Flag — transfers poorly from reach to cloth:** any *static-pose* absolute action (IK-abs, joint-pos-abs) is a weak proxy for cloth, where velocity/delta actions and compliance dominate (EquiBot/geometry-aware RL output velocities for folding; deformable-RL work uses Δpose/Δθ/Δgripper). Don't over-tune reach on absolute-pose actions and assume cloth follows.

### (3) Kinova Root-Cause Verdict + Ranked Fixes

**Verdict:** The dominant cause is a **mismatched, full-SO(3) target distribution generated by FK from continuous joints, combined with an orientation reward too weak to drive SO(3) matching**, compounded by **raw (discontinuous) continuous-joint observations** and a **joint-space action that forces the policy to solve redundancy resolution itself**. The IK smoke test (1.1 cm) proves controller/target geometry is fine; PPO simply can't learn the SO(3) map under these conditions. It is a learning/representation failure, not stability.

**Ranked fixes:**

1. **Fix target generation (highest impact).** Replace FK-from-random-config with the canonical `UniformPoseCommandCfg` (sample EE pose directly in root frame) and start with a *constrained* orientation set (roll=0, pitch=π/2, yaw=±π), then *curriculum-widen* roll/pitch toward full SO(3) only as success climbs. Trade-off: coverage vs learnability — wide SO(3) coverage is only worth it if downstream tasks need it (cloth/twisting yes; top-down pick no). *Expected effect:* the single largest jump, plausibly to UR-class success. *Validation:* success vs orientation-range-width sweep; orientation error should fall to <10°.
2. **Rebalance the reward for full-6-DOF tracking.** Keep geodesic `quat_error_magnitude`, but raise weight to ~-0.5…-1.0 and add a fine-grained orientation bonus `1 - tanh(d_θ/0.1)`, or use multiplicative prioritization `r_pos·r_ori` (per Jiang et al. 2025) so orientation only competes once position is close. *Expected:* orientation error converges; sigma drops. *Validation:* orientation/position error curves; policy sigma → <0.1.
3. **Encode continuous joints as sin/cos in observations.** Map J1/J3/J5/J7 (and optionally all) to (sin θ, cos θ). *Expected:* smoother value landscape, faster/lower-variance convergence. *Validation:* learning-curve variance and final sigma vs raw-angle ablation.
4. **Choose a redundancy-aware action space.** Move from joint-EMA to **IK-relative (reach/pick)** and **OSC with null-space posture control (cloth/contact)**. Set OSC `nullspace_control="position"` targeting joint centers to keep the elbow well-conditioned and away from limits. *Expected:* removes implicit-IK burden; better-conditioned motion. *Validation:* compare success and manipulability across action spaces at fixed budget.
5. **Handle the continuous joints in sampling/limits.** Do NOT rely on writing finite limits to USD-continuous joints (PhysX wrap at ±2π; finite writes unreliable). Instead constrain via target distribution + reward + (optionally) a soft joint-centering penalty. *Expected:* avoids silent limit failures, controlled posture. *Validation:* inspect realized joint ranges in rollouts.
6. **Tendon 2-DOF wrist participation.** The passive-ish tensegrity wrist (k=400/d=20, ±50°) should NOT be in the controlled-joint set for the IK/OSC Jacobian if it is meant to be compliant — it adds 2 uncontrolled-but-actuated DOFs the policy must fight. Options: (a) treat it as passive (exclude from action, include its state in observation), letting compliance absorb contact; or (b) include it as a low-authority residual action with its own small reward. *Expected:* cleaner credit assignment; compliance benefits cloth/contact. *Validation:* compare reach/insertion with wrist-passive vs wrist-actuated; measure contact-force smoothness.

### (4) FK-Sampling / Self-Collision Verdict

- **(a) Are unreachable / only-self-colliding-reachable targets a real cause?** Unreachability: largely **no** — FK-from-sampled-config guarantees at least one reaching configuration exists. Self-collision-only reachability: **partially yes** — you don't verify a *collision-free* path/branch from a sensible start, so some targets demand awkward/near-singular end postures, which raises orientation error and policy variance. It is a secondary contributor, not the primary cause.
- **(b) Best practice for sampling reachable, well-conditioned targets:** Use rejection sampling with (i) self-collision check, (ii) manipulability/condition-number filtering (Yoshikawa index) to avoid near-singular targets that congest around singularities, and (iii) a precomputed **reachability/capability map** (sample joints → FK → voxel, store max-manipulability and IK-solution count per 6D cell; RM4D, arXiv:2410.06968, reduces to 4D for common 6/7-axis arms; see also `pearl-robot-lab/sampled_reachability_maps`). Sample targets only from high-reachability, collision-free, well-conditioned cells. FK sampling is recommended over IK sampling for speed.
- **(c) Does another collision-free IK solution make self-collision a non-issue?** **No, not automatically.** Redundancy gives a self-motion manifold of solutions, but (i) the policy may not discover the collision-free branch, (ii) moving between branches can require crossing singularities or self-collisions, and (iii) your reward doesn't credit posture. Redundancy *helps* only if you actively exploit it (null-space control toward a safe posture, or reward/curriculum that guides branch selection). Otherwise the existence of a good solution is irrelevant to whether PPO finds it.

### (5) Continuous-Joint Strategy

- **Observation:** encode each continuous joint as **(sin θ, cos θ)** to remove the ±π discontinuity (standard in DeepMind control-suite and pendulum tasks; raw angle creates artificial max-distance jumps between near-identical states). Velocities can stay raw.
- **Sampling:** for targets, sample EE poses (root frame) directly rather than FK-from-continuous-config; if you must sample joints, sample continuous joints over a *bounded* range (your default ±0.75 rad fallback is reasonable) to control orientation coverage.
- **Limits:** don't fight PhysX — `eREVOLUTE` wraps at ±2π and finite-limit writes on USD-continuous joints are unreliable. Manage "limits" behaviorally (target distribution, joint-centering penalty, OSC null-space), not via hard USD clamps.
- **Reward:** if you penalize joint travel, compute it on the wrapped (shortest-arc) difference, not raw angle, so the penalty is continuous.
- **When unbounded rotation is a real advantage:** continuous re-orientation, screwing/winding/bolt-driving, cloth-twisting/wringing, and avoiding wrist-roll unwinding / wrist singularities. For these, *keep* the continuous joint unbounded, encode sin/cos in observation, and let the *action* be a delta (velocity or Δangle) so the policy can wind indefinitely without hitting a representation wall. This is precisely where Kinova's infinite-rotation wrists beat the UR arms — but only if the task rewards it.

### (6) Literature / Implementation References (tagged)

**Directly addresses Kinova redundancy / continuous joints / Isaac Lab:**
- `louislelay/kinova_isaaclab_sim2real` (github.com/louislelay/kinova_isaaclab_sim2real) — Gen3 7-DOF reach in Isaac Lab + ROS2 sim2real; pretrained reach models. **[Directly relevant]**
- `louislelay/isaaclab_ur_reach_sim2real` — sibling UR reach sim2real. **[Relevant baseline]**
- `j3soon/OmniIsaacGymEnvs-UR10Reacher` (+ Dofbot/Kuka/Hiwin Reacher forks) — UR10 reacher RL sim2real; the lineage the Gen3 repo builds on; uses `set_joint_efforts`-style control and reduced joint limits for safety. **[Relevant baseline]**
- Isaac Lab `Isaac-Reach-Franka-OSC-v0` + OSC PR #913 (ozhanozen) — OSC action term with null-space control for redundant arms. **[Directly relevant: redundancy]**
- Isaac Lab Issue #911 (IK-Abs with rotated base) — confirms IK-abs frame subtleties. **[Relevant: your IK-abs path]**

**Action-space / contact-rich / OSC:**
- Martín-Martín et al., "Variable Impedance Control in End-Effector Space (VICES)," IROS 2019 (arXiv:1906.08880) — canonical action-space comparison; advocates EE-space variable impedance for contact-rich; shows cross-robot and sim2real transfer. **[Directly relevant: action space]**
- Khatib, "A unified approach for motion and force control … the operational space formulation," IEEE J. Robotics & Automation 3(1):43–53, 1987 — OSC foundation. **[Foundational]**
- "Variable Impedance Skill Learning for Contact-Rich Manipulation," IEEE RA-L — Cartesian variable-impedance RL on Franka peg-in-hole. **[Relevant: contact/cloth]**

**Redundancy / null-space RL & reachability:**
- Shen et al., "RL-Based Reactive Obstacle Avoidance for Redundant Manipulators," Entropy 24(2):279, 2022 (10.3390/e24020279) — RL in the Jacobian null space with manipulability in the reward. **[Directly relevant: redundancy]**
- "Learning Task Execution Hierarchies for Redundant Robots," arXiv:2508.10780. **[Relevant]**
- Dietrich/Ott/Albu-Schäffer, "An overview of null-space projections for redundant, torque-controlled robots," IJRR 34(11):1385–1400, 2015. **[Foundational]**
- RM4D (arXiv:2410.06968) and `pearl-robot-lab/sampled_reachability_maps` — capability/reachability maps via FK sampling + manipulability. **[Directly relevant: target sampling]**
- "The Dexterity Capability Map for a 7-DOF Manipulator," Machines 10(11):1038, 2022. **[Relevant: 7-DOF reachability]**

**Residual RL:**
- Johannink et al., "Residual RL for Robot Control," 2018 (arXiv:1812.03201); Silver et al., "Residual Policy Learning," 2018 (arXiv:1812.06298). **[Relevant: pick/insertion]**

**Cloth / deformable RL:**
- "Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects," arXiv:2502.07005 — RL benchmark with rope/cloth, velocity actions, multiple end-effectors. **[Directly relevant: cloth + multi-arm]**
- Matas et al., "Sim-to-Real RL for Deformable Object Manipulation," CoRL 2018 (arXiv:1806.07851) — 4-DOF EE-velocity + grip action for folding/draping. **[Relevant: cloth action space]**
- "Dynamic Cloth Manipulation with Deep RL" (IRI/UPC). **[Relevant]**

**Reward shaping (SO(3)):**
- "Geometric RL for Robotic Manipulation," arXiv:2210.08126 — quaternion-distance orientation reward. **[Relevant: orientation reward]**
- Jiang et al., "Learning Whole-Body Loco-Manipulation … Task-Space Pose Tracking," IEEE RA-L 10(2):1481–1488, 2025 (arXiv:2412.03012) — multiplicative reward prioritization `r_pos·r_ori`, geodesic SO(3) distance. **[Directly relevant: your orientation-weight problem]**

**Multi-agent / multi-arm:**
- Yu et al., MAPPO; "JointPPO" (arXiv:2404.11831); CLAS "Coordinating Multi-Robot Manipulation with Central Latent Action Spaces" (arXiv:2211.15824); SpaceOctopus multi-arm MAPPO (arXiv:2403.08219). **[Relevant: multi-arm action spaces]**

### (7) Multi-Agent (multi-arm) Notes

- **Scales cleanly:** **per-agent task-space relative actions in each arm's root/EE frame.** Identical action semantics per arm (6-DOF Δpose) make the policy permutation-friendly and let you share weights across arms — the standard CTDE/MAPPO setup. Normalize actions to [-1,1] per arm with a fixed scale so magnitudes are comparable across agents.
- **Coordinate frames:** use a **shared world/root frame for the task target** (e.g., cloth corners) but **per-agent EE-frame for the agent's own deltas**; this keeps each agent's action interpretable and avoids one agent's frame contaminating another's credit.
- **Creates problems:** raw **joint-torque** actions and **centralized joint-position** actions both worsen non-stationarity and credit assignment for bimanual/cloth tasks — torque because contact forces couple the arms through the cloth (one agent's torque changes the other's optimal action), and centralized joint-position because the joint action space grows multiplicatively (the CTCE scalability blowup noted in JointPPO). Task-space relative actions localize each agent's effect, easing credit assignment. For tightly-coupled bimanual moves, a **central latent action space (CLAS)** or fully-centralized single-agent controller can outperform decentralized — choose centralized when coupling is high (rigid co-manipulation) and decentralized when arms are quasi-independent (separate cloth corners).

### (8) Single Recommended End-to-End Recipe (reach → pick → cloth)

1. **Robot/controller:** Gen3 7-DOF with a stiff PD config analogous to `FRANKA_PANDA_HIGH_PD_CFG` (the high-PD variant raises arm gains and disables gravity; the base `FRANKA_PANDA_CFG` is stiffness 80 / damping 4, per Isaac Lab Issue #2807, and the IsaacGymEnvs Franka cabinet config uses stiffness 400 / damping 80 on the 7 arm joints — use these as your Gen3 starting point for stiff IK tracking). Tendon wrist = **passive** (excluded from action, included in observation) for v1.
2. **Action:** `DifferentialInverseKinematicsActionCfg`, IK-**relative**, DLS, `joint_names=["joint_.*"]`, `body_name="end_effector_link"`, `scale=0.5`. (Pin to PhysX; Newton lacks the batched Jacobian for IK.) Plan to swap to `OperationalSpaceControllerActionCfg` with null-space posture = joint centers when you reach cloth/contact.
3. **Observation:** EE pose error in root frame, target pose, joint pos/vel with **continuous joints encoded sin/cos**, last action.
4. **Targets:** sample EE pose directly in root frame via `UniformPoseCommandCfg`; **curriculum** from constrained orientation (roll=0, pitch=π/2, yaw=±π) → progressively wider roll/pitch toward full SO(3). Reject self-colliding / near-singular targets via a manipulability + collision filter (reachability map).
5. **Reward:** position L2 (-0.2) + position tanh fine-grained (+0.1, std 0.1) + **orientation geodesic at -0.5…-1.0 with a fine-grained `1-tanh(d_θ/0.1)` bonus** (or multiplicative `r_pos·r_ori`); action-rate (-0.0001) and joint-vel (-0.0001) penalties; small joint-centering penalty to keep the redundant elbow conditioned.
6. **PPO (skrl):** keep 4096 envs; widen MLP to [256,128] (the [64,64] net is thin for SO(3)+7-DOF+redundancy); extend horizon beyond 48k timesteps once orientation reward is rebalanced; monitor policy sigma as your convergence gauge.
7. **Transfer:** for pick, keep IK-rel and add a binary gripper action + grasp/lift reward; for cloth, switch the action to OSC/task-space impedance (compliance), keep velocity/delta actions, re-widen orientation coverage (twisting), and move to per-agent root-frame task-space actions for multi-arm with MAPPO/CTDE.

## Recommendations

**Stage 1 (today — isolate the cause, ~1 day):** Run Experiment 1 (constrain orientation range) + Experiment 2 (sin/cos encoding) on the *existing* joint-position action. **Threshold:** if Kinova success rises to ≥50% and orientation error drops below ~12°, the root cause is confirmed as target-distribution + encoding, and you proceed to reward rebalancing. If not, escalate to action-space change immediately.

**Stage 2 (reward + action space, ~2–3 days):** Apply Fix 2 (orientation weight -0.5…-1.0 + fine-grained term) and switch to IK-relative (Experiment 4). **Threshold:** target UR-class success (≥67%) and sigma convergence <0.15. If sigma still won't converge, add the manipulability/collision target filter (Experiment 6).

**Stage 3 (redundancy + downstream, ~1 week):** Introduce OSC with null-space posture control; validate on pick (cube_place) with IK-rel + gripper. **Threshold:** pick success ≥60% before touching cloth.

**Stage 4 (cloth + multi-arm):** Move to OSC/task-space impedance, velocity/delta actions, per-agent root-frame actions under MAPPO. **Benchmark that would change the plan:** if cloth fails under OSC despite good pick, the bottleneck is deformable dynamics/observation (point-cloud encoders), not action space — pivot research there.

**What would change these recommendations:** If constraining orientation does *not* recover Kinova (Stage 1 fails), the problem is deeper (e.g., a controller/asset bug or actuator gains), and you should re-run the IK-abs smoke test under the *training* action pipeline, not a separate script. If the tendon wrist destabilizes contact tasks even when passive, model it as an explicit low-gain residual action.

## Caveats
- The exact `FRANKA_PANDA_HIGH_PD_CFG` arm gains were confirmed indirectly: the base config is stiffness 80 / damping 4 (Isaac Lab Issue #2807, which also confirms HIGH_PD "disables gravity by default"), and the IsaacGymEnvs Franka cabinet config uses stiffness 400 / damping 80 on the 7 arm joints. Verify the precise numbers in `isaaclab_assets/robots/franka.py` before relying on them for Gen3 tuning.
- Success-rate figures for UR vs Kinova are from your own runs; the SO(3)-coverage explanation is an inference strongly supported by the confirmed UR command ranges (roll/pitch fixed at 0 and π/2), but you should confirm by directly logging your Kinova target-orientation distribution.
- "Best for cloth/contact = OSC/VICES" is well-supported for contact-rich *rigid* tasks (VICES) and variable-impedance peg-in-hole; its superiority specifically for *cloth* is an extrapolation — cloth RL literature more often uses EE velocity/delta actions than impedance, so validate empirically.
- Multi-agent guidance is synthesized from MAPPO/JointPPO/CLAS and dual-arm RL papers, not from a Kinova-specific multi-arm study.
- The PhysX continuous-joint limit behavior (±2π wrap, unreliable finite-limit writes) is documented for PhysX articulations generally; behavior may vary slightly across Isaac Sim 5.1 / PhysX versions — verify on your build.
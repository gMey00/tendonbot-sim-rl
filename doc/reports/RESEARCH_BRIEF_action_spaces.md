# Research Brief — RL Action Spaces for Redundant + Tendon-Wrist Manipulators
### (Isaac Lab / skrl PPO; reach → pick-and-place → cloth; multi-agent)

This document has two parts:
- **Part A — Prompt for the research AI** (directives, questions, required output format).
- **Part B — Context package** (my stack, robots, MDP, the three action-space
  implementations I already tried, the FK target-sampling code, joint-limit
  facts, measured results, and my diagnosis).

Give Part A to the research AI together with the whole of Part B as context.

---

# PART A — PROMPT FOR THE RESEARCH AI

You are a robotics-RL research assistant. I am training manipulators in
**Isaac Lab (manager-based RL env, Isaac Sim 5.1) with skrl PPO**. My immediate
benchmark is an **end-effector pose-reach** task, but the real downstream goals
are **pick-and-place** and **deformable/cloth manipulation**, eventually with
**multi-agent (multi-arm) training**. Read the full context package (Part B)
first — it contains my robots, MDP, the three action spaces I already
implemented and benchmarked, the FK target-sampling code, and my results.

## What I need from you (be concise and concrete; cite sources)

1. **Action-space taxonomy & ranking.** Enumerate the action spaces used for RL
   manipulation (joint position / delta-joint-position / joint velocity / joint
   torque-effort; differential IK absolute & relative; operational-space /
   task-space impedance / OSC; reduced or learned action spaces; relative-to-EE
   vs world-frame; residual RL on top of a controller). For **each**, give: what
   it is, action dim for a 7-DOF arm, pros/cons, sim-to-real implications, and
   **how well it handles kinematic redundancy and contact-rich tasks**. Then
   **rank them** specifically for (a) pose-reach, (b) pick-and-place, (c) cloth,
   and note where the best choice *changes* between tasks.

2. **Which should I try next, and in what order?** Give a short prioritized
   experiment list for the reach task that is predictive of the harder tasks.
   Flag any that are known to transfer poorly from reach to contact-rich/cloth.

3. **Kinova fix (highest priority).** My Kinova Gen3 (7-DOF, **4 continuous
   `[-inf,inf]` joints: J1,J3,J5,J7**) fails to learn the pose-reach under *all
   three* action spaces I tried (joint-EMA, IK-rel, IK-abs: 0–11 % success),
   while the 6-DOF UR arms learn well (76–87 %). The persistent failure
   signature is a **~27–35° orientation error** (UR arms: 4–11°). I want the
   Kinova to **reach as much of its reachable/usable workspace as possible** —
   not a crippled subset. Give concrete, ranked fix proposals covering:
   - target generation (orientation diversity vs coverage trade-off),
   - action space / controller choice for a redundant arm,
   - reward shaping for full-6-DOF pose tracking,
   - handling the continuous joints (see #5),
   - and the **tendon 2-DOF wrist** ("Frankenstein" variants, J `wrist_x/wrist_y`,
     ±50°, low-gain) — how should the wrist participate in the action/IK?

4. **Is this a forward-kinematics target-sampling problem?** I generate reach
   targets by sampling random joint configs and reading EE FK (code in Part B).
   Targets are reachable *by construction by the sampling config*, but **I do
   NOT check self-collision or that a collision-free IK solution exists**.
   Determine: (a) Are unreachable/only-self-colliding-reachable targets a real
   cause of the Kinova failure? (b) Best practice for sampling *reachable,
   collision-free, well-conditioned* pose targets for a redundant arm (reject
   sampling? manipulability/condition-number filtering? collision checking?
   reachability maps / capability maps?). (c) Does the redundant arm having
   *another* collision-free IK solution to the same pose make this a non-issue,
   or not?

5. **Exploit the infinite-rotation joints.** Kinova's continuous joints are
   usually treated as a nuisance (I currently clamp them to ±π for the
   joint-space action). I want to **explore their benefits** for manipulation
   (e.g. continuous re-orientation, screwing/winding, cloth-twisting, avoiding
   wrist singularities/unwinding). How should continuous joints be represented
   to a policy (raw angle is discontinuous at ±π — use sin/cos encoding?),
   sampled, limited (or not), and rewarded? Are there tasks where unbounded
   rotation is a real advantage, and how to encode it in observation/action?

6. **Prior art / "is there an ideal known solution?"** Find existing
   research and open-source implementations of RL (or RL+controller) for:
   - **Kinova Gen3 7-DOF** reach / pick-place in Isaac Lab / Isaac Gym / MuJoCo
     (e.g. community configs, sim2real repos), incl. how they handle the
     continuous joints, redundancy (null-space), and action space.
   - **UR5e / UR10** RL manipulation baselines (action space, reward).
   - Redundancy resolution + null-space RL, capability/reachability maps, and
     **task-space / OSC RL** for contact-rich and deformable tasks.
   - Multi-arm / multi-agent manipulation action-space conventions.
   For each, state what they use and whether it would solve my Kinova case.

7. **Multi-agent (multi-arm) lookahead.** Which action-space choices scale
   cleanly to decentralized/centralized multi-agent PPO (shared vs per-agent
   action spaces, coordinate frames, action normalization), and which create
   credit-assignment or non-stationarity problems for cloth/bimanual tasks.

## Required output format

Produce a single structured response with these sections:
- **(1) Action-space comparison table** (rows = action spaces; columns = dim,
  redundancy handling, contact suitability, cloth suitability, multi-agent
  suitability, sim2real, reach→pick→cloth rank). Bold the recommended choice
  per task.
- **(2) Prioritized experiment plan** for my reach task (ordered list, each
  with the exact Isaac Lab `ActionCfg` to use and what it would prove).
- **(3) Kinova root-cause verdict + ranked fixes**, each fix written as a
  concrete change to *my* code (name the file/field from Part B, e.g.
  `FKSampledPoseCommandCfg.joint_range_margin`, the reward weights, the
  `clamp_infinite_joint_limits` event, or a new action cfg), with expected
  effect and a quick way to validate it.
- **(4) FK-sampling / self-collision verdict** (yes/no it's a cause) + the
  recommended target-sampling procedure.
- **(5) Continuous-joint strategy** (representation, sampling, reward, when to
  exploit unbounded rotation).
- **(6) Literature/implementation references** (with links), each tagged with
  whether it directly addresses my Kinova redundancy/continuous-joint problem.
- **(7) Multi-agent notes.**
- **(8) A single recommended end-to-end recipe** I should adopt for reach that
  will transfer to pick-place and cloth.

Be concrete enough that I can implement your recommendations directly against
the code in Part B. Prefer primary sources (papers, official Isaac Lab code,
maintained repos) and say when something is your inference vs. a cited result.

---

# PART B — CONTEXT PACKAGE (my implementation)

## B.0 Stack & training setup
- **Isaac Lab 0.54.3** (manager-based RL), **Isaac Sim 5.1**, PhysX GPU.
- **skrl PPO**, MLP policy+value `[64, 64]` elu, GaussianMixin policy /
  Deterministic value, `RunningStandardScaler` on obs & value.
  rollouts 24, learning_epochs 5, mini_batches 4, γ 0.99, λ 0.95,
  lr 1e-3 with `KLAdaptiveLR` (kl_threshold 0.01), entropy 0.01.
  **4096 parallel envs, 48 000 trainer timesteps** (≈ joint-space converges).
- Reach episode: `episode_length_s = 6.0` (→ 180 control steps), decimation 2,
  sim dt 1/60. **Gravity is disabled** for the reach task (kinematic focus);
  self-collisions disabled on the articulation.

## B.1 Robots (all mounted **upright** at env-local `(0.75, 1.0, 0.75)`)
- **UR10 / UR5e**: 6-DOF, all-revolute, **finite** limits (the ±2π wrists get
  clamped to ±1.5 rad for RL). + Robotiq 2F-140 gripper. EE tracked =
  `robotiq_base_link`. These **learn the reach well**.
- **Kinova Gen3**: **7-DOF, redundant**. Joints **J1,J3,J5,J7 are continuous
  `[-inf,inf]`**; J2 ±138°, J4 ±152°, J6 ±128°. + Robotiq 2F-140. **Fails to
  learn the reach** under all action spaces tried.
- **"Frankenstein" variants** (each base arm): a passive-ish **tensegrity 2-DOF
  wrist** (`wrist_x_joint`, `wrist_y_joint`, ±50°≈±0.873 rad, low-gain
  k=400/d=20 in USD) is spliced between the arm flange and the gripper; those
  two joints are added to the controlled-joint set so the wrist is part of the
  task. EE still = `robotiq_base_link` (rigidly past the wrist).
- Downstream tasks already scaffolded in the same repo: **`cube_place`**
  (pick-and-place) and **`shirt_place`** (cloth/deformable). Goal is to pick the
  action space that transfers to these and to **multi-agent** (multi-arm).

### Kinova soft joint limits *as seen by the policy at runtime* (margin 0.4, clamp fallback 2π, no max_range):
```
joint_1  [-inf, +inf]      (continuous)        default  0.000
joint_2  [-2.41, +2.41]    (USD ±138°)         default  0.262
joint_3  [-inf, +inf]      (continuous)        default  3.142
joint_4  [-2.66, +2.66]    (USD ±152°)         default -2.269
joint_5  [-inf, +inf]      (continuous)        default  0.000
joint_6  [-2.23, +2.23]    (USD ±128°)         default  0.960
joint_7  [-inf, +inf]      (continuous)        default  3.142
```
**PhysX note:** `setLimitParams()` only accepts revolute limit angles in
`[-2π, 2π]`, and for these USD-continuous joints a finite limit write below ~2π
does **not** reliably "stick" — i.e. I cannot tightly bound J1/3/5/7 below ±π
via the articulation. (See `clamp_infinite_joint_limits` in B.4.)

## B.2 The reach MDP (`reach_env_cfg.py`, abridged)
```python
# Command: a reachable EE pose target sampled by forward kinematics (see B.3).
ee_pose = mdp.FKSampledPoseCommandCfg(
    asset_name="robot", body_name=MISSING, joint_names=MISSING,
    resampling_time_range=(1e9, 1e9),   # one fixed target per episode
    success_threshold=0.05, debug_vis=True,
)

# Observations (policy): joint_pos_rel, joint_vel_rel, generated_commands(ee_pose), last_action
# Actions: arm_action = MISSING (set per-variant: joint-EMA OR diff-IK), gripper_action=None

# Rewards (weights):
end_effector_position_tracking      = position_command_error        weight=-0.2
end_effector_position_tracking_fine = position_command_error_tanh   weight=+0.1  (std set to 0.5)
end_effector_orientation_tracking   = orientation_command_error     weight=-0.1
position_reached(<5cm)  / orientation_reached(<0.3rad) / pose_reached  weight=1e-6  (logging only)
action_rate_l2  weight -1e-4 → curriculum -5e-3 @4500 steps
joint_vel_l2    weight -1e-4 → curriculum -1e-3 @4500 steps
# Terminations: time_out; joint_vel_out_of_manual_limit(max 100 rad/s) on controlled joints.
# Reset: reset_joints_by_offset(±0.125 rad) on controlled joints.
```

## B.3 FK target sampling — `mdp/fk_sampled_pose_command.py` (the key suspect)
Samples a random joint config within (margin-trimmed) soft limits, teleports the
robot, reads the EE body pose via PhysX FK, expresses it in the root frame, and
uses that as the per-episode target. **No self-collision / IK-feasibility check.**
For continuous joints whose soft limit is `[-inf,inf]`, sampling falls back to
`default ± 0.75 rad`.
```python
def _resample_via_self(self, env_ids):
    joint_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]
    lower, upper = joint_limits[:,0].clone(), joint_limits[:,1].clone()
    default = self.robot.data.default_joint_pos[0, self.joint_ids]
    invalid = (~torch.isfinite(lower) | ~torch.isfinite(upper)
               | ((upper-lower) <= 1e-5) | ((upper-lower) > 8*math.pi))
    lower = torch.where(invalid, default - 0.75, lower)   # continuous-joint fallback
    upper = torch.where(invalid, default + 0.75, upper)
    margin = (upper-lower) * self.cfg.joint_range_margin   # trim both ends
    lower += margin; upper -= margin
    q = lower + (upper-lower) * torch.rand(len(env_ids), len(self.joint_ids), device=self.device)
    # ... write q to sim, update kinematics, read EE body pose (pos+quat) ...
    pos_b, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    self.pose_command_b[env_ids,:3] = pos_b
    self.pose_command_b[env_ids,3:] = quat_b   # full-6-DOF target (position + orientation)
    # ... restore the robot's joints ...
```
Config knob: `FKSampledPoseCommandCfg.joint_range_margin` (fraction trimmed per
joint; 0.4 used for Kinova → samples inner 20 % of each joint's range).

## B.4 Joint-limit clamp event — `mdp/events.py`
```python
def clamp_infinite_joint_limits(env, env_ids, asset_cfg, fallback_range=6.2832, max_range=None):
    limits = asset.data.joint_pos_limits.clone(); defaults = asset.data.default_joint_pos
    lower, upper = limits[...,0], limits[...,1]
    threshold = max_range if max_range is not None else 4*pi
    bad = ~isfinite(lower) | ~isfinite(upper) | ((upper-lower) > threshold)
    half = fallback_range/2.0
    limits[...,0] = where(bad, defaults-half, lower)
    limits[...,1] = where(bad, defaults+half, upper)
    asset.write_joint_position_limit_to_sim(limits)
# UR uses fallback=3.0,max=3.0 (→ all joints ±1.5). Kinova uses fallback=2π,max=None
# (→ continuous joints ±π, finite joints kept). Tighter writes on continuous joints
# do NOT stick in PhysX (see B.1).
```

## B.5 Action space #1 — joint-space (EMA-to-limits)  [`f140_reach_common.py`]
```python
env.actions.arm_action = mdp.EMAJointPositionToLimitsActionCfg(
    asset_name="robot", joint_names=controlled_joints, alpha=0.2,
)   # action dim = n_controlled (6 UR / 7 Kinova / +2 wrist for Frankenstein)
# maps policy output [-1,1] -> joint soft limits, exponential-moving-average smoothed.
```

## B.6 Action space #2 & #3 — Differential IK  [`ik_reach_common.py`]
```python
IK_ARM_STIFFNESS, IK_ARM_DAMPING = 800.0, 160.0   # stiffened PD for IK tracking
# RELATIVE (action dim 6 = pos3 + axis-angle3, scaled):
DifferentialInverseKinematicsActionCfg(
    asset_name="robot", joint_names=controlled_joints, body_name="robotiq_base_link",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.5,
)
# ABSOLUTE (action dim 7 = pos3 + quat4):
DifferentialInverseKinematicsActionCfg(
    asset_name="robot", joint_names=controlled_joints, body_name="robotiq_base_link",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
)
# (For IK, all non-gripper actuator groups are overridden to stiffness 800 / damping 160.)
```
Modeled on Isaac Lab's franka sample
`source/isaaclab_tasks/.../manipulation/reach/config/franka/{ik_abs,ik_rel}_env_cfg.py`
(which switches to `FRANKA_PANDA_HIGH_PD_CFG`, stiffness 400 / damping 80).

## B.7 Kinova per-variant reach cfg (current)  [`config/kinova_f140/joint_pos_env_cfg.py`]
```python
configure_f140_reach(self, KINOVA_GEN3_GRIPPER_CFG, list(CONTROLLED_JOINT_NAMES),
                     joint_range_margin=0.4, clamp_fallback_range=6.2832, clamp_max_range=None)
# Kinova arm actuators (USD/cfg): shoulder k5000/d220, elbow & wrist k2500/d160;
# large actuators J1-4 39 N·m, small J5-7 9 N·m.
```

## B.8 Measured results (final, 48k-step PPO; "pose" = within 5 cm AND 0.3 rad; std = converged policy σ)
```
variant               joint-space (EMA)      IK-Rel (DLS)          IK-Abs (DLS)
ur10_f140             86% / 4.5cm / σ.05     69% / 6.2cm / σ.16    67% / 7.2cm / σ.16
ur10_frankenstein     81% / 4.8cm / σ.08     27% /10.5cm / σ.19    80% / 5.3cm / σ.09
ur5e_f140             87% / 3.6cm / σ.04     32% / 8.4cm / σ.14    68% / 5.1cm / σ.07
ur5e_frankenstein     76% / 4.8cm / σ.10     19% /10.2cm / σ.15    69% / 5.8cm / σ.09
kinova_f140           11% /17.8cm / σ.35      5% /18.6cm / σ.44     9% /19.0cm / σ.58
kinova_frankenstein   19% /14.3cm / σ.20      0% /33.1cm / σ.33     0% /30.0cm / σ.23
```
Notes / diagnosis so far:
- **Kinova fails under every action space**; UR arms learn. Failure signature is
  **orientation error 27–35°** (UR: 4–11°) and **non-converging policy σ** (0.2–0.58).
- Episodes run the **full 180 steps** (no physics blow-up / early termination):
  it's a **learning** failure, not instability.
- **IK-Abs smoke test:** feeding the *commanded* target pose directly as the
  action drives the EE to **1.1 cm** of target → the IK controller and target are
  fine; PPO just fails to *learn* to output it for Kinova.
- **Hypothesis:** Kinova's continuous joints → FK targets span ~all of SO(3);
  matching arbitrary orientation is the bottleneck. UR finite joints → constrained
  target orientations → learnable. (Needs confirmation; tie to the self-collision
  question #4.)
- The EE body (`robotiq_base_link`) is verified **rigidly** attached (rel-pose σ=0),
  so EE-body choice is not the cause.

## B.9 What "good" looks like / constraints
- Want Kinova to reach **as much of its real reachable, collision-free workspace
  as possible** (not a hand-crippled subset), full 6-DOF pose, and ideally
  **leverage** the continuous joints rather than just clamp them.
- The chosen approach must **transfer to pick-and-place and cloth** (contact-rich,
  deformable) and to **multi-agent multi-arm** PPO.
- Stack is fixed: Isaac Lab manager-based env + skrl PPO. Solutions should be
  expressible as Isaac Lab `ActionCfg` / command / reward / event changes, or a
  small custom action/command term.
```

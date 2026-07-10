# Physical Tendon Variant — Reach Task Rework (2026-07-09)

This report documents the changes made to get the *physical* (body-force,
antiparallelogram) tendon variant of the reach task to a trainable state, per
the recommendations in [reach_evaluation_findings.md](reach_evaluation_findings.md)
(§9) and the project-thesis discussion (§6, "the physical tendon variant calls
primarily for an intermediate actuation layer").

## 1 Problem

The 2026-05 evaluation left the physical variant at **31.4 % ± 14.4 %** reach
success (vs. ~98 % for PD and sim-tendon).  Visual inspection showed the root
cause was not just hard dynamics but a **reward hack**: the policy applied
large sustained elbow tensions until the PhysX loop-closure joint
(`coupler_right_joint`, a maximal-coordinate constraint outside the
articulation solver) drifted apart — *breaking the four-bar linkage* — and
then steered the dangling forearm toward the goal with the linear base.

Contributing model weaknesses:

1. **No cable length limits.** The body-force action applied any commanded
   tension regardless of the attachment-point distance.  A real cable is
   inextensible and has a finite spooled length; without these bounds the
   policy could drive the linkage arbitrarily far past its geometric range.
2. **Unphysical tension saturation.** A single `max_tension = 500 N` for all
   five tendons — 6× the hardware limit for the wrist cables (80 N
   direct-drive) and slightly above the elbow's 480 N (160 N motor-side × 3:1
   pulley).
3. **Misleading joint observations.** The policy saw the three raw linkage
   joint angles (`rod_left`, `rod_right`, `coupler_left`).  The physically
   meaningful lower-arm rotation is the **sum** of two linkage angles, and the
   raw values become meaningless once the closure drifts.
4. **No penalty for breaking the mechanism.** Episodes continued after the
   linkage tore open, so the hack was profitable.

## 2 Verification of the existing model (what was already correct)

Before changing anything, the existing implementation was re-derived and
cross-checked:

- **Closure equation & joint limits** — the antiparallelogram closure equation
  in `joint_pos_env_cfg_physical.py` and the USD joint limits
  (rod_left [−39.45°, +30.55°], rod_right mirrored, coupler ±39.45°) were
  validated numerically against a continuation solve of the four-bar
  constraint: predictions match to machine precision, and the limits map to
  exactly ±70° elbow.  `elbow = rod_left + rod_right` and
  `coupler_left = rod_right` hold as documented.
- **Wrench application** — the manual `τ = r × F` torque about the link origin
  is correct: the PhysX tensor API applies forces **at the link transform**
  when no position is passed (verified in
  `omni/physics/tensors/impl/api.py::apply_forces_and_torques_at_position`),
  so force + torque-about-origin is the exact equivalent wrench of the force
  acting at the attachment point.  The `_link_poses_updated` cache
  invalidation before `set_external_force_and_torque(is_global=True)` is also
  correct with the Isaac Lab 0.54 wrench composer.
- **Attachment geometry** — root (0, ±0.0725, −0.34) / forearm local
  (0, ±0.0725, −0.02) offsets are mutually consistent between
  `tendon_actuator.py`, `scripts/model_validation/common.py`, and the USD
  build constants.
- **Hardware parameters** — effort limits (35 / 3.5 N·m), velocity limits
  (2.5 / 9.0 rad/s), spool radius, pulley ratio: all consistent with
  `doc/open_questions.md` §1–4 (all resolved; nothing to change there).

No bugs were found in the force-application math; the failures were the four
model-security gaps listed in §1.

## 3 Changes

### 3.1 Cable min/max length limits (`robots/tendon_actuator.py`)

The elbow cable length between the attachment points was computed over the
full ±70° elbow workspace from the four-bar closure + attachment geometry
(numeric continuation, branch-tracked from the rest pose):

| Elbow angle | Cable T0 | Cable T1 |
|---|---|---|
| −70° | 0.2577 m | 0.0913 m |
| 0°   | 0.1775 m | 0.1775 m |
| +70° | 0.0913 m | 0.2577 m |

`ELBOW_CABLE_LENGTH_RANGE = (0.0913, 0.2577)` is now enforced in
`PhysicalTendonEffortAction` (new cfg fields on
`PhysicalTendonEffortActionCfg`):

- **Wind-up stop (min length):** commanded tension fades linearly to zero over
  `cable_slack_ramp = 1 cm` above the minimum — the spool cannot wind the
  cable shorter, so there is no active pull below the limit.
- **Stretch stop (max length):** a passive spring–damper tension
  (`cable_stop_stiffness = 20 kN/m`, `cable_stop_damping = 200 N·s/m`, capped
  at `cable_stop_max_tension = 1 kN`) engages when a cable exceeds its maximum
  free length, modelling cable inextensibility.  Because the *long* cable
  reaches its max exactly when the elbow reaches ±70°, this passively confines
  the linkage to its physical workspace even under adversarial commands.

Cable length **rates** are computed from the attachment-point velocities
(`v = v_link + ω × r`), so the stop damping acts on the physical extension
speed.

### 3.2 Per-tendon hardware tension saturation

Both tendon actions now accept `max_tension: float | list[float]`.
`HW_TENDON_MAX_TENSIONS = [480, 480, 80, 80, 80] N` (Klein 2023 §3.2.5/§3.2.6)
is the new **default for the physical action** — closing TODO M2 for this
variant.  The sim-tendon task cfgs keep their scalar 500 N until their
retrain, so existing checkpoints stay valid.

### 3.3 Linkage-integrity penalty (`reach/mdp/terminations.py` + `rewards.py`)

`linkage_closure_broken` detects either of two measured failure modes of the
closed chain; `linkage_integrity_penalty` applies it as a **−1/step reward
penalty**:

- **Closure gap** — world-space distance between the two anchor points of the
  loop-closure joint (rod_right tip vs. forearm coupler pivot, local offsets
  from the USD build constants) above **3 cm**.  Calibration (measured at
  120 Hz): 0 at rest, ≲1 mm settled under a sustained 480 N tension, 1–3 cm
  during violent transients; the exploit opens it much further.
- **Branch flip** — during violent transients the chain can re-close in the
  *parallelogram* branch, which satisfies the closure constraint (gap ≈ 0!)
  but decouples the forearm from the rods.  Detector: in the correct branch
  the effective elbow angle equals `rod_left + rod_right` exactly; a mismatch
  above **0.3 rad** flags the flipped branch (normal elastic drift stays
  below ~0.15 rad even at full tension).

Together with §3.1 the "break the linkage, steer with the base" exploit is
both physically suppressed and unprofitable.

> **Penalty, not termination (retrain finding, 2026-07-09):** the first
> retrain attempt used the detector as a failure *termination*. Because the
> reach reward is net-negative per step, terminating early is itself a reward
> hack — within 2 k timesteps both variants learned to break the linkage as a
> *suicide exit* (mean episode length collapsed 180 → 5 steps while the
> logged reward "improved"). As a −1/step penalty (vs. max +0.1/step task
> reward) breaking is strictly unprofitable and there is no exit to escape
> to; the cable stops keep the state recoverable so episodes continue
> meaningfully. The detector remains available as a termination term for
> non-RL use.

### 3.4 Actuator dynamics: first-order tension lag

`tension_time_constant = 50 ms` first-order lag on the commanded elbow
tensions and wrist efforts, applied at the physics rate.  A real Maxon EC60 +
spool cannot step the cable force 0→480 N in one physics step; without the
lag, such steps produce constraint-impulse spikes through the closed chain
(and through the near-massless `wrist_intermediate` frame link) that
destabilise the solver and immediately re-terminate episodes.

### 3.5 Simulation-fidelity fixes found during verification

- **120 Hz physics** for the physical variants (`sim.dt = 1/120`,
  `decimation = 4` → policy still 30 Hz).  At 60 Hz the maximal-coordinate
  closure joint yields centimetres under a sustained 480 N tension and the
  drive wrench leaks into constraint violation instead of rotating the
  mechanism; at 120 Hz the settled gap is <1 mm.  This also matches the
  validated step-response setup (which ran at 120 Hz).
  `enable_external_forces_every_iteration = True` and
  `min_velocity_iteration_count = 2` further improve constraint accuracy
  under the large body forces.
- **PhysX sleep disabled** — the arm is driven purely by external body
  forces, and a *sleeping body silently ignores tensor-API forces* (they do
  not wake it), so the arm freezes mid-episode.  Zeroing the sleep /
  stabilization thresholds at the **articulation root** (via
  `ArticulationRootPropertiesCfg`) proved **insufficient** on Isaac Lab 0.54 /
  PhysX GPU; the effective fix is a `*_awake.usd` asset bake that zeroes
  `physxRigidBody:sleepThreshold` / `stabilizationThreshold` on **every rigid
  body** under the robot prim (`tools/make_physical_awake_usd.py`; the
  robot configs point at the `*_awake` assets and keep the root setting as
  belt-and-braces).  The loop-closure joint is kept enabled — the awake bake
  changes only sleep behaviour, nothing kinematic.
- **`joint_vel_diverged` threshold 100 → 500 rad/s** (physical variants
  only): the 1-gram `wrist_intermediate` dummy link produces harmless solver
  velocity spikes >100 rad/s during reset transients; at 100 the env
  re-terminated in a *silent* reset loop (the term is a truncation) and never
  accumulated an episode — a plausible contributor to the original training
  instability.
- **Elbow gravity-compensation recalibrated** for the hierarchical controller:
  the 3-DOF validation constant (m·g·l = 6.35 N·m) over-compensates ~2× on
  the assembled 5-DOF robot; the measured static balance (30 N tension holds
  43.7°) gives ≈ 2.8–3.2 N·m → `elbow_grav_arm = 0.115 m`.

### 3.4 Correct lower-arm rotation measurement (`reach/mdp/observations.py`)

New shared helper `compute_lower_arm_angle_and_rate` (robots package):
the effective elbow angle is the forearm body's **twist about the upper-arm X
axis**, `θ = 2·atan2(q_rel.x, q_rel.w)` with `q_rel = q_root⁻¹ ⊗ q_forearm`
(exact, since the linkage constrains the forearm to rotate about X), and the
rate is the relative angular velocity projected onto that axis.  This replaces
the three raw linkage joint angles in the policy observation:

- `joint_pos` / `joint_vel` obs now cover only base + wrist joints,
- new obs `lower_arm_angle`, `lower_arm_ang_vel`,
- new obs `cable_lengths`, `cable_length_rates` (read from the action term) —
  implements evaluation-report §9 open issue #1 (expose per-cable state to the
  policy).

The same body-quaternion measurement (not a joint angle) is what the validated
step-response controller already used (`common.compute_elbow_angle_from_body_quat`);
the reach task now measures at the same, correct points.

### 3.6 Direct force-control variant (updated)

`Template-Reach-Tensegrity-Physical-Tendon-v0` keeps its 5-D tension action
(2 elbow body-force tendons + 3 wrist tendons via J^T) but now runs with
§3.1–§3.4 applied.

### 3.7 Hierarchical controller variant (new)

`Template-Reach-Tensegrity-Physical-Hierarchical-v0` (+ `-Play-v0`) —
`HierarchicalPhysicalTendonAction` in `robots/tendon_controllers.py`, a
vectorized (torch) port of the controller heuristic validated in the
step-response study (`scripts/model_validation/common.py` /
`run_step_response_tendon.py --variant physical`), including its tuned gains:

| Stage | Details |
|---|---|
| Action | 3-D set-points: effective elbow angle (±70°) + wrist_y/x (±50°), mapped from [−1, 1]; base stays a separate 2-D position action |
| Inner loop | PID at physics rate (60 Hz): elbow kp=75, ki=6, kd=3; wrist kp=10, ki=1.5, kd=0.6; derivative-on-measurement with EMA α=0.5; integral clamp ±50; wrist conditional-integration zone 0.175 rad |
| Gravity comp | τ_g = m·g·l_c·sin(θ): elbow m·g·l ≈ 2.8 N·m (recalibrated to the measured 5-DOF static balance, see §3.5), wrist 0.40 kg × 0.068 m |
| Distribution | Elbow: antagonistic pair (pre-tension 6 N, lever 0.0725 m); wrist: 2×3 pseudo-inverse + null-space bias toward previous tensions (pre-tension 5 N) |
| Output | Same physical channel as the direct variant — body forces at the attachment points with the cable-length limits of §3.1 |

Everything else (scene, FK target sampling from the hidden PD reference robot,
rewards, terminations, observations) is identical to the direct variant, so a
training comparison isolates the action-interface effect — exactly the
experiment the thesis recommends.

Note: the wrist now saturates at the hardware 80 N (the validation study used
600 N), so wrist transients are slower than the study's; the elbow gains carry
over unchanged.

### 3.8 Registration / infra

- New agents cfg `skrl_ppo_cfg_physical_hier.yaml` (same PPO profile, separate
  run directory `reach/tensegrity_physical_hier`).
- The scripted heuristic and its GUI viewer
  (`scripts/skrl/heuristic_physical_ik.py`,
  `scripts/diagnostics/play_physical_heuristic.py`) double as the reusable
  controller-validation tool; per-run health is enforced by the pipeline's
  sanity gate.

## 4 Verification

All checks were run via bounded diagnostic probes (a random-action smoke
test plus tension-staircase and constant-set-point probes), 2026-07-09,
RTX A6000, Isaac Sim 5.1 / Isaac Lab 0.54.  These were one-off development
scripts and are not retained; the surviving reusable checks are the pipeline's
sanity gate (`run_physical_reach_pipeline.sh`) and the GUI viewer
(`scripts/diagnostics/play_physical_heuristic.py`).

**Mechanism physics (tension staircase, terminations off, quasi-static):**

| T0 [N] | 30 | 60 | 120 | 240 | 480 | ↓240 | ↓120 | ↓60 | ↓30 |
|---|---|---|---|---|---|---|---|---|---|
| elbow [°] | 43.7 | 65.4 | 66.7 | 68.8 | 69.5 | 68.3 | 67.9 | 63.8 | 25.0 |

Smooth monotone response saturating at the ~70° workspace limit, mild
hysteresis, `elbow ≡ rod_left + rod_right` and `coupler_left ≡ rod_right`
throughout — the four-bar and the body-force channel behave physically.

**Security (direct variant, `Template-Reach-Tensegrity-Physical-Tendon-v0`):**

- *Max-tension probe* (sustained T0 = 480 N — exactly the configuration that
  used to tear the linkage open): the elbow drives to ~69° and **parks at the
  mechanical limit** with the closure gap settled at **0.8–0.9 mm**; the
  wind-up stop throttles the applied tension to the ~30–100 N needed to hold
  against gravity, and the antagonist's stretch stop engages (≈25 N at 71°).
- *120 N hold from rest*: reaches ~67° in 20 policy steps, holds; gap <1 mm.
- *Training-realistic run* (600 continuous random-action steps, 8 envs,
  12 natural terminations with partial resets): no NaNs, **zero
  frozen-dynamics windows**, elbow covers the full [−78°, +80°] range, cable
  lengths bounded to [0.077, 0.261] m (≈ the geometric [0.0913, 0.2577] m
  plus brief transient overshoot), closure gap max 29.7 mm (transient; the
  3 cm termination fired on 12 of 4800 env-steps ≈ 0.25 %).

**Hierarchical variant (`Template-Reach-Tensegrity-Physical-Hierarchical-v0`):**

- Obs (26) / act (5) spaces build; same 600-step random-phase health as the
  direct variant (14 terminations, no stuck windows).
- *Set-point hold* (elbow 35°, wrist ±15°, fresh env): elbow settles at
  **34.2–35.6°** (<1° steady-state error after a single ~50° overshoot,
  ~1.3 s settle), wrist tracks within ~3° with mild cross-coupling sway.

**Known diagnostic artifact:** a *full* `env.reset()` mid-run (as the probe
phases of the check script do) can leave some envs' closed-chain dynamics
frozen on the GPU pipeline.  Training is unaffected (skrl full-resets only
once at startup; per-env partial resets are exercised by the random phase and
show no freezes), but probe read-outs after such a reset are advisory only —
the script docstring documents this.

## 5 What deliberately did NOT change

- **PD and sim-tendon variants** — untouched (solved baselines; regression
  reference).  Their `TendonEffortActionCfg` gained per-tendon *support* but
  keeps the scalar 500 N default.
- **Reward shaping / PPO hyper-parameters** — per evaluation-report
  implication #3, the investment went into the actuator model and
  observations, not reward tuning.
- **USD assets** — the linkage joint limits and the closure joint were
  verified correct; no rebuild needed.

## 6 Next steps

1. Retrain both physical variants (5 seeds each) with
   `run_reach_pipeline.sh`-style seed aggregation; compare against the 31.4 %
   baseline and the ~98 % sim-tendon reference.
2. Re-run the physical step-response validation with the 80 N wrist
   saturation.
3. Sensitivity sweep over the cable-stop parameters if training shows
   stiffness-related instability.

---

# Addendum — 2026-07-10: GUI review findings, controller retune, PhysX forensics

The user's GUI review of the seed-0 policies and the first scripted heuristic
("oscillates around the target, sometimes does not approach at all, occasional
linkage breaks; heuristic worse than RL") triggered a second, deeper
investigation.  Chronology and outcomes:

## A1 Inner-PID retune (hierarchical action term)

| Change | Before | After | Reason |
|---|---|---|---|
| Set-point slew limiting | none (raw jumps) | 1.2 / 5.0 / 5.0 rad/s (elbow/wrists) | kp × O(1 rad) jumps slam tensions into saturation; ramping at the arm's own top speed still kept T > 400 N on 27 % of steps — rates sit ~50 % under the hardware velocity limits |
| Integrator clamps | 50 (≈ 300 N·m!) | per-joint (2.0, 0.8, 0.8) rad·s | windup limit-cycles around far/unreachable targets |
| Elbow integral zone | none | 0.35 rad | anti-windup (wrist zone 0.175 kept) |
| Set-point ranges | ±70° / ±50° | **±60° / ±40°** | riding the geometric boundary grinds the stretch stop (elbow observed at +84° with ~520 N stop tension); the wrist is under-actuated near ±50° with the 1 kg gripper (≤ ~1.5 N·m available vs ~2 N·m needed) and chatters |
| Cable stop cap | 1000 N | 500 N | bound stop impulses near the hardware maximum |
| Target sampling margin | 0.01 | **0.10** | sample targets from the *achievable* workspace; unreachable targets put the controller/policy into permanent-saturation limit cycles |

Scripted IK+PID heuristic (`scripts/skrl/heuristic_physical_ik.py`, DLS-IK on
the FK reference robot + command ramping): reach-rate 0.47 → 0.56, best
position error 0.16 → 0.12 m across the retune steps.

## A2 PhysX closed-chain forensics (root-cause hunt)

Systematically measured, each with a dedicated probe:

1. **Loop-closure elastic drift** — the maximal-coordinate closure joint
   absorbs cable torque as constraint strain (9.7 mm gap ≈ 9° of "stolen"
   elbow angle); quasi-static tensions creep while dynamic ones act fully.
2. **Absorbing branch flips** — under transients the chain re-closes in the
   parallelogram branch (gap ≈ 0, forearm decoupled); 13–16 of 96 heuristic
   episodes flipped and never recovered.
3. **Quasi-static breakaway (~10–20 N·m)** — present with the closure joint,
   without it, at zero gravity, with sleep/stabilization thresholds zeroed at
   articulation *and* body level, with wrist pre-tension, and with authored
   friction verified zero.  A ±1e-3 rad/s external joint-velocity jiggle
   partially unlocks it; the same write inside the action path does not
   reproduce the effect reliably (kept as the EXPERIMENTAL `solver_dither`
   cfg, default off).  Full physics-prim inventory of the composed USD showed
   no hidden springs/tendons — this is a GPU-solver-level artifact, open.
4. **Two dead ends, tried and reverted**: (a) kinematic closure via stiff
   mimic drives — a servo-held coupler provably collapses the cable moment
   along the remaining free DOF (quasi-static stall by construction); (b) a
   rigid single-elbow rebase — rejected: the four-bar IS the thesis model.

**Resulting model state (kept, per user directive: the physical four-bar):**
dynamic actuation is exact and strong (fresh-state 120 N ⇒ 67° in 0.7 s;
set-point sweep tracks ±60° within 6–11°); quasi-static response exhibits the
documented solver breakaway; the `*_awake.usd` bakes (per-body sleep
thresholds zeroed) are in use.  Item 3 is the highest-priority open issue for
the master thesis (candidates: CPU-pipeline comparison, PhysX version sweep,
NVIDIA bug report).

## A3 RL learning fixes (both physical variants)

Seed-0 diagnosis: policy σ pinned at 1.3–1.6 for the whole run (exploration
noise never collapses) because the 50 ms tension lag low-pass filters action
dither — the environment under-penalises noise while the entropy bonus keeps
it alive; the action-rate curriculum then dominates the reward (−0.1/step).

1. `entropy_loss_scale` 0.01 → 0, `max_log_std` 2 → 0 (σ ≤ 1),
   `initial_log_std` 0 → −0.5 — the shirt-task profile that fixed the same
   signature.
2. New observation `applied_tensions` (lag-filtered, stop-adjusted elbow
   tensions, scale 1/480) — actuator state the policy cannot infer from its
   own last action.
3. The A1 slew limiting also converts policy set-point jumps into ramps.
4. `linkage_integrity_penalty` unchanged (−1/step; suicide-exploit analysis
   in §3.3 still applies).

## A4 Answer to "is the antiparallelogram inherently hard to learn?"

No — the evidence points at the actuator *interface* and the simulation
artifacts, not the kinematics: (i) the sim-tendon variant (identical
kinematics, idealised actuator) reaches ~98 %; (ii) the scripted IK+PID
heuristic reaches every target the mechanism can physically hold and tracks
set-points within degrees; (iii) the RL failure signature (σ never collapses,
oscillation) matches lag-masked action noise plus the quasi-static breakaway
randomly nullifying small corrective actions.  The fixes above address the
learnable part; the solver breakaway remains as sim-fidelity noise that the
policy must (and the heuristic does) overpower dynamically.

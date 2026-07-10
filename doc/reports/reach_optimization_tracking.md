# Reach Task Optimization Tracking

Change log for the reach task reward structure, PPO hyperparameters, and
training pipeline.  Each iteration records _what_ was changed, _why_, and the
observed training outcome.

---

## Baseline (2026-03-25) — Pre-optimization

**Status**: All 4 variants diverged to NaN during training.

| Variant | First reward | NaN at step | Root cause |
|---|---|---|---|
| Tensegrity PD | −245.97 | 720 | Immediate physics divergence — massive L2 penalty |
| Tensegrity Tendon | −759.90 | 720 | Same, worse due to passive actuator dynamics |
| UR10e | −6.42 | 26 640 | KL-adaptive LR cranked to 5.8e-3 → gradient explosion |
| Kinova | −4.05 | 22 320 | KL-adaptive LR cranked to 7.4e-3 → gradient explosion |

### Diagnosed root causes

1. **Regularization 10× too strong** — `action_rate` initial weight −0.001 and
   `joint_vel` −0.0005 vs Isaac Lab reference −0.0001 each.  Dominant penalties
   make "do nothing" the optimal policy, preventing exploration.
2. **Curriculum final weights too aggressive** — `action_rate` ramped to −0.01
   (reference: −0.005) and `joint_vel` to −0.005 (reference: −0.001) over
   12 000 steps (reference: 4 500).
3. **Exploration suppressed** — `min_log_std = −5.0` floors the policy std at
   ≈ 0.007, far below the reference −20.0 (effectively unclamped).
4. **Network oversized** — [256, 128] (49 k+ params) vs reference [64, 64]
   (8 k).  Harder to train with limited data budget.
5. **Learning rate too low** — 5e-4 vs reference 1e-3.  KL-adaptive scheduler
   compensated by _increasing_ LR aggressively, causing instability.
6. **Proximity tanh too narrow** — std = 0.03 gives near-zero signal beyond
   ~0.1 m, creating a "dead zone" for distant targets.

### Plotting issue

The plotting script itself was correct — TensorBoard tags matched exactly.
The empty plots were caused by NaN data: matplotlib does not render NaN line
segments, so the `smooth()` function propagated NaN to every subsequent point.
Fixed `smooth()` to be NaN-aware (skip NaN, resume smoothing on next finite
value).

---

## Iteration 1 (2026-03-26) — Align with Isaac Lab reference

**Goal**: Stable convergence for all 4 variants within 30 min each.

**Reference**: Isaac Lab `source/isaaclab_tasks/.../manipulation/reach/`
(Franka variant, SKRL PPO).

### Changes

#### Reward weights (`reach_env_cfg.py`)

| Term | Before | After | Reference |
|---|---|---|---|
| `action_rate` (initial) | −0.001 | **−0.0001** | −0.0001 |
| `joint_vel` (initial) | −0.0005 | **−0.0001** | −0.0001 |
| `proximity` tanh std | 0.03 | **0.05** | N/A |
| Other task rewards | unchanged | unchanged | same |

#### Curriculum (`reach_env_cfg.py`)

| Term | Before (weight / steps) | After | Reference |
|---|---|---|---|
| `action_rate` | −0.01 / 12 000 | **−0.005 / 4 500** | −0.005 / 4 500 |
| `joint_vel` | −0.005 / 12 000 | **−0.001 / 4 500** | −0.001 / 4 500 |

#### PPO hyperparameters (all 4 `skrl_ppo_cfg.yaml`)

| Parameter | Before | After | Reference |
|---|---|---|---|
| Network layers | [256, 128] | **[128, 64]** | [64, 64] |
| `min_log_std` | −5.0 | **−20.0** | −20.0 |
| `learning_rate` | 5e-4 | **1e-3** | 1e-3 |
| `rollouts` | 32 | **24** | 24 |
| `timesteps` | 36 000 | **48 000** | 24 000 |
| `learning_epochs` | 8 | 8 (kept) | 5 |
| `separate` (networks) | True | True (kept) | False |
| `entropy_loss_scale` | 0.01 | 0.01 (kept) | 0.01 |

#### Plotting (`plot_reach_training_results.py`)

- `smooth()` now skips NaN values instead of propagating them.
- `DEFAULT_CURRICULUM_STEP` updated from 12 000 → 4 500.

### Rationale

The core strategy is: keep the user's improved reward terms (FK-sampled
commands, proximity bonus, goal bonus, clamped joint velocities) but align the
_regularization budget_ and _PPO tuning_ with the battle-tested Isaac Lab
reference config.  The reference uses lighter initial penalties that let the
agent explore freely at first, then ramps them up half-way through training.
The larger network [128, 64] is a middle ground — more capacity than reference
[64, 64] for the harder multi-robot setup, but less than the original
[256, 128] which was hard to train in 48 k steps.

### Training results

Covered in Iteration 2 — this round was superseded by further simplification.

---

## Iteration 2 (2026-03-27) — Eliminate NaN: Root cause & fix

**Status**: First stable training! Mean reward improved from −0.6 to +0.48
in 24k steps. Extended 96k run in progress.

### Problem

All training runs (Runs 1–7) failed with NaN gradients at the first
TensorBoard checkpoint (~step 240), regardless of reward tuning or PPO config.

### Root cause chain

1. **Passive Robotiq gripper joints** (8 joints) originally had stiffness=0,
   damping=0 (`gripper_passive` actuator group in `robotiq_2f140_gripper_cfg.py`).
2. Random arm motions (base_y stiffness=8000, effort_limit=300 Nm) create
   large inertial forces on the lightweight gripper links.
3. With no restoring torque, gripper joint velocities **diverge exponentially**:
   1.7 → 359 → 4 594 → 1.7 M → 10^15 rad/s over 500 steps.
4. PhysX solver becomes **numerically unstable** when any joint velocity
   overflows, corrupting ALL joint data in the articulation (including
   controlled joints 0–4).
5. Reward terms read corrupted joint data → catastrophic reward values
   (min = −7.4×10²²) → NaN PPO gradients → training collapse.

### Changes (cumulative)

#### Reward simplification (`reach_env_cfg.py`)

Stripped to match NVIDIA reference exactly (5 terms):

| Term | Weight | Source |
|---|---|---|
| `end_effector_position_tracking` | −0.2 | NVIDIA reference |
| `end_effector_position_tracking_fine_grained` | 0.1 (std=0.1) | NVIDIA reference |
| `end_effector_orientation_tracking` | −0.1 | NVIDIA reference |
| `action_rate` | −0.0001 | NVIDIA reference |
| `joint_vel` | −0.0001 | NVIDIA reference |

Removed: `proximity`, `goal_reached`, `pose_goal_reached`.

#### PPO alignment (all `skrl_ppo_cfg.yaml`)

| Parameter | Iter 1 | Iter 2 | Reference |
|---|---|---|---|
| Network layers | [128, 64] | **[64, 64]** | [64, 64] |
| `separate` | True | **False** | False |
| `learning_epochs` | 8 | **5** | 5 |
| `initial_log_std` | −1.0 | **0.0** (std=1) | 0.0 |
| `timesteps` | 48 000 | **96 000** | 24 000 |

#### Action space (`config/tensegrity/joint_pos_env_cfg.py`)

Switched from `JointPositionActionCfg` (scale=0.125) to
**`JointPositionToLimitsActionCfg`** — maps policy output [−1, 1] directly to
joint limits, preventing out-of-range position targets.

#### Observation filtering (all variant configs)

Observations reduced from (38,) → **(22,)**: only controlled joint
positions/velocities. Passive gripper joints excluded via
`SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)`.

#### Reward & termination filtering (all variant configs)

`joint_vel` reward and `joint_vel_diverged` termination now use
`SceneEntityCfg` restricted to controlled joint names only.

#### Gripper stabilization — THE KEY FIX (all variant configs)

All 3 gripper actuator groups overridden with:

```python
ImplicitActuatorCfg(
    stiffness=100.0,
    damping=100.0,
    effort_limit_sim=1000.0,   # Allow full PD restoring torque
    velocity_limit_sim=1.0,
    armature=10.0,             # Add reflected inertia to resist acceleration
)
```

**Critical insight**: `effort_limit_sim=10` (original) capped PD output at
10 Nm, completely negating damping=100 (which wanted 10,000 Nm at 100 rad/s).
Even with uncapped effort, stiffness/damping alone couldn't counteract inertial
forces from the arm's 8000 N·m/rad base actuators. The `armature=10.0`
parameter adds 10 kg·m² of reflected inertia to each gripper joint, making them
resistant to external acceleration. Result: gripper velocity stays at
0.1–0.8 rad/s (was trillions before).

### Training Run 8 results (24k steps, seed 42)

| Metric | Start (step 240) | End (step 24000) |
|---|---|---|
| Mean reward | −0.607 | **+0.482** |
| Position tracking | −0.030 | **−0.009** |
| Fine-grained tracking | 0.000 | **0.072** |
| Orientation tracking | −0.037 | **−0.026** |
| Policy std | 0.991 | **0.084** |
| Episode length | 81 | **360** (full) |
| NaN entries | **0/100** | **0/100** |

### Training Run 9 — FINAL (96k steps, seed 42)

| Metric | Start (step 960) | End (step 96000) | Last 10% mean |
|---|---|---|---|
| Mean reward | −1.272 | **+0.480** | **+0.463 ± 0.029** |
| Position tracking | −0.041 | **−0.008** | −0.009 |
| Fine-grained tracking | 0.006 | **0.077** | 0.075 |
| Orientation tracking | −0.045 | **−0.023** | −0.024 |
| Action rate penalty | −0.001 | −0.004 | −0.004 |
| Joint velocity penalty | −0.002 | −0.001 | −0.001 |
| Policy std | 0.917 | **0.066** | — |
| Learning rate | 6.6e-3 | **1.5e-4** | — |
| Episode length | 266 | **360** (full) | 360 |
| NaN entries | **0/100** | **0/100** | — |
| Best reward | **0.524** (step 86,400) | | |

### Plotting updates (`plot_reach_training_results.py`)

- `plot_task_success`: Replaced goal_reached/proximity with fine-grained
  tracking + episode length.
- `plot_reward_decomposition`: Added guard for empty reward terms.
- `write_report`: Removed proximity/goal_reached metrics.

---

## Iteration 3 (2026-03-28) — Success metrics, unified training, timestep tuning

**Goal**: Add explicit binary success metrics visible in plots, retrain all
variants under the current reward structure with identical hyperparameters,
and determine appropriate timestep budgets.

### Changes

#### Success metrics (`reach_env_cfg.py` + `rewards.py`)

Added 3 new reward terms with weight=1×10⁻⁶ (near-zero, logged to
TensorBoard but negligible effect on the reward signal):

| Term | Function | Threshold |
|---|---|---|
| `position_reached` | `goal_reached()` | Position error < 0.02 m (2 cm) |
| `orientation_reached` | `orientation_reached()` (new) | Orientation error < 0.1 rad (5.7°) |
| `pose_reached` | `pose_goal_reached()` | Both position **and** orientation met |

New function `orientation_reached()` added to `rewards.py` — binary 1.0 when
quaternion error magnitude < threshold.

**Design note**: `weight=0.0` was tried first but produces zero TensorBoard
output because Isaac Lab logs `weight × func()` summed over the episode.
Using `1e-6` ensures values are recorded while contributing < 0.001 to the
episode return (max 360 steps × 1 × 1e-6 = 3.6e-4).

All 4 variant config files updated with `body_names` overrides for the 3 new
terms.

#### Plotting updates (`plot_reach_training_results.py`)

- `plot_task_success()`: Now shows 3 success-rate curves (position, orientation,
  full pose) on the left axis instead of the fine-grained tracking reward.
  Fallback to older tags for backward compatibility.
- `plot_reward_decomposition()`: Added position_reached, orientation_reached,
  pose_reached rows.
- `plot_converged_summary()`: Added success metrics to the reward panel.
- `write_report()`: Added position/orientation/pose reached metrics.

#### Timestep optimisation

| Variant | Before | After | Rationale |
|---|---|---|---|
| Tensegrity PD | 96 000 | **24 000** | Converges by ~12k steps; 24k provides margin |
| Tensegrity Tendon | 24 000 | 24 000 | Already correct |
| UR10e | 24 000 | **96 000** | Did not converge at 24k; still fails at 96k |
| Kinova | 24 000 | **96 000** | Did not converge at 24k; still fails at 96k |

### Training results

All runs: seed 42, 4096 parallel envs, 2026-03-28.

| Variant | Steps | Run ID | Total Reward | Pos. Error | Fine-Grained | Orient. Error | Pos. Reached† | Orient. Reached† | Pose Reached† | Policy σ | LR (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tensegrity PD | 24k | `04-04-46` | **+0.47** | −0.008 | 0.075 | −0.024 | 0.7 | 0.4 | 0.4 | 0.076 | 4.6e-4 |
| Tensegrity Tendon | 24k | `04-22-38` | **+0.61** | −0.008 | 0.076 | −0.013 | 0.7 | 0.7 | 0.5 | 0.078 | 3.4e-4 |
| UR10e | 96k | `05-23-35` | **−4.37** | −0.165 | 0.000 | −0.187 | 0.0 | 0.0 | 0.0 | 0.125 | 7.4e-5 |
| Kinova | 96k | `06-28-02` | **−2.25** | −0.088 | 0.002 | −0.098 | 0.0 | 0.0 | 0.0 | 0.058 | 1.5e-4 |

†Success metric = average episode sum of binary (1 if threshold met, 0 otherwise), extracted from TensorBoard by dividing the logged weighted value by the weight (1e-6).

### Analysis

**Tensegrity variants converge well at 24k steps.**  Both reach positive
rewards with sub-5 cm position accuracy and non-zero success rates.  The
tendon variant outperforms PD (reward +0.61 vs +0.47) — likely due to
better force distribution from the Jacobian-transpose tendon actuation.

**UR10e and Kinova fail to converge even at 96k steps.**  Both maintain
high position errors (17 cm and 9 cm respectively) and zero success rates.
Kinova's policy std collapsed to 0.058 without learning useful behavior,
indicating premature convergence to a sub-optimal policy.  Both robots
are mounted at `(0.15, 0.0, 2.5)` with rotation `(0, 0, 1, 0)` (180° yaw)
— the workspace configuration may need investigation.  Possible root causes:

1. `scale=0.125` in `JointPositionActionCfg` may be too restrictive for
   the FK-sampled workspace range
2. The 2.5 m mounting height (hardcoded) may place the robots too far from
   the conveyor / target volume
3. The 180° yaw rotation may create a workspace mismatch with the
   FK-sampled pose generator

**Next steps**: Investigate UR10e/Kinova mounting, action scale, and FK
workspace overlap before increasing timesteps further.

---

## Iterations 4–7 (2026-03-28 → 2026-03-29) — FK sampler bug fix, EMA actions, UR10e/Kinova investigation

**Note**: These iterations were performed rapidly during debugging and are
summarised here retroactively.

### Key findings

1. **FK sampler cache bug (critical)**: The `body_link_pose_w` TimestampedBuffer
   cache was not invalidated after `write_joint_position_to_sim()` during FK
   sampling. This meant FK-sampled targets were always the _current_ EE pose,
   making the task trivial ("stay still"). **All results from iterations 0–3
   were measured against the wrong targets.**
2. **Fix**: Bypassed cache entirely using direct `update_articulations_kinematic()`
   + `get_link_transforms()` PhysX calls. Confirmed via diagnostic: 30–50 cm EE
   displacement for tensegrity, 131–189 cm for UR10e.
3. **FK sampler joint restoration**: Original code only saved/restored controlled
   joints during FK sampling. `update_articulations_kinematic()` modifies _all_
   joint positions in the articulation. Fixed to save/restore ALL joint positions
   (including passive gripper joints).
4. **EMA-smoothed actions for UR10e/Kinova**: `EMAJointPositionToLimitsActionCfg`
   with `alpha=0.2` — maps policy output [−1, 1] to joint limits with exponential
   moving-average smoothing. Prevents rapid target oscillation that PD controllers
   can't track. Raw `JointPositionToLimitsAction` was too coarse for PPO.
5. **UR10e/Kinova gripper stabilization**: Overrode all gripper actuator groups
   with K=100, D=100, armature=10 to prevent freewheeling passive joints from
   accumulating velocity drift.
6. **UR10e/Kinova disable_gravity + disabled self-collisions**: Ceiling-mounted
   robots; gravity disabled for consistency. Self-collisions disabled to reduce
   solver work (not needed for reach task).

---

## Iteration 8 (2026-03-29) — Pose error observation, min_log_std floor

**Goal**: Improve convergence by giving the agent direct pose-error information
and preventing premature exploration collapse.

### Changes

1. **`pose_error` observation** (6D: position error + orientation error in world
   frame) added to base `reach_env_cfg.py` with `body_names=MISSING` (overridden
   per variant).
2. **`min_log_std: -2.0`** (was −20.0) in all 4 PPO YAML configs. Floors policy
   std at ~0.135, preventing premature convergence to deterministic sub-optimal
   policies (observed in iter 7b where std collapsed to 0.055 without learning).
3. **FK margins**: Tensegrity 0.25 (inner 50%), UR10e/Kinova 0.40 (inner 20%)
   to keep FK-sampled targets within reach of the fine-grained tanh reward.
4. **Timesteps**: Tensegrity/Tendon 96k, UR10e/Kinova 192k.

### Training results (2026-03-29)

| Variant | Steps | Run ID | Pos. Error (cm) | Status |
|---|---|---|---|---|
| Tensegrity PD | 96k | `09-00-54` | **2.4** | Learning, ~2cm threshold |
| Tensegrity Tendon | 96k | `10-15-39` | **2.15** | Learning, close to threshold |
| UR10e | 192k | `11-30-35` | **18.1** | Learning slowly |
| Kinova | 192k | `13-39-00` | **NaN** | All rewards NaN from step 0 |

### Critical issue: Kinova NaN

Kinova produced NaN at step 0. Extensive investigation (8 debugging phases):

1. Self-collisions disabled → still NaN
2. Gravity disabled → still NaN
3. Diagnostic script: joint positions valid after reset, ALL NaN after first step
4. FK sampler save/restore all joints → still NaN
5. FK sampler bypassed entirely → **still NaN** (FK not the cause)
6. Gripper stabilization removed → still NaN
7. Cube-place task with same robot → **SUCCESS** (no NaN in 20 steps)
8. Joint limits inspection → **ROOT CAUSE FOUND**

**Root cause**: `EMAJointPositionToLimitsAction` calls `unscale_transform(actions,
lower, upper)` which computes `(action+1)/2 × (upper−lower) + lower`. Kinova
joints 1, 3, 5, 7 are **continuous** with limits `[-inf, inf]`. The unscale
computation: `0.5 × (inf − (−inf)) + (−inf) = inf − inf = NaN`. This corrupted
ALL joint data in the articulation on the very first step.

**Fix**: Created `clamp_infinite_joint_limits()` reset event in
`reach/mdp/events.py`. Detects joints with infinite or excessively large (>4π)
limits and replaces them with `default_pos ± fallback_range/2` (default 2π).
Added to Kinova config as a `mode="reset"` event (idempotent).

**Verified**: 50 steps with zero actions, no NaN, finite rewards (−0.0127 to
−0.0077).

**Why UR10e works**: All 6 UR10e joints have finite limits (±2π), so
`unscale_transform` never encounters infinite values.

**Why Kinova cube-place works**: Uses `JointPositionActionCfg` (delta actions),
not `JointPositionToLimitsAction`, so never calls `unscale_transform` with limits.

---

## Iteration 9 (2026-03-29) — Extended training after Kinova NaN fix

**Goal**: Evaluate all 4 variants with increased training budgets now that
Kinova NaN is fixed. Target: >80% pose_reached for all variants.

### Changes

Only timesteps increased (reward structure unchanged):

| Variant | Before | After | Rationale |
|---|---|---|---|
| Tensegrity PD | 96k | **192k** | 2.4cm at 96k → close to 2cm threshold |
| Tensegrity Tendon | 96k | **192k** | 2.15cm at 96k → very close to threshold |
| UR10e | 192k | **384k** | 18.1cm at 192k → needs substantially more training |
| Kinova | 192k | **384k** | First real training with working config |

### Training results

_Skipped — superseded by Iteration 10 (simplification + root-cause fixes)._

---

## Iteration 10 (2026-03-30) — Simplification + configuration fixes

**Goal**: Simplify the reach task back to basics with LOCKED thresholds
(5 cm / 0.3 rad) and LOCKED timesteps (48 000). Diagnose and fix why UR10e
and Kinova don't converge while tensegrity does.

### Phase 1 — Simplification + table mount fix

| Change | Before | After | Rationale |
|---|---|---|---|
| Thresholds (pos/ori) | 10 cm / 0.5 rad | **5 cm / 0.3 rad** | More meaningful; locked from now on |
| Timesteps | 96–384 k | **48 000** (all) | Locked; fast iteration budget |
| `pose_error` obs | present | **removed** | Simplification |
| Reset | `reset_joints_by_offset` | **`reset_joints_by_scale`** | Revert to simpler default |
| FK margin | per-variant overrides | **0.10** (default, all) | Revert to default |
| `min_log_std` | −2.0 | **−20.0** | Revert to skrl default |
| UR10e mounting | ceiling (z=2.5 m, 180° Y-flip) | **table (z=1.40 m, identity rot)** | **Critical fix**: robot was upside-down |
| Kinova mounting | ceiling (z=2.5 m, 180° Y-flip) | **table (z=1.40 m, identity rot)** | Same ceiling-mount error |
| UR10e `fix_root_link` | missing | **True** | Root drifted under reaction forces |

#### Phase 1 results (table mount only)

| Variant | Pos success | Pos error | Ori error | Status |
|---|---|---|---|---|
| **Tensegrity** | **83.1 %** | **4.0 cm** | **0.24 rad** | ✓ Converged |
| UR10e | 0.0 % | 77.2 cm | 1.92 rad | ✗ Not converging |
| Kinova | 1.2 % | 38.1 cm | 1.32 rad | ✗ Not converging |
| Tendon | — | — | — | Skipped (under reconstruction) |

#### Root-cause analysis (UR10e / Kinova)

Investigation revealed UR10e and Kinova's large joint ranges (±2π ≈ ±6.28 rad)
create a workspace 10× larger than tensegrity's. Combined with:

1. **FK targets span entire workspace** — `joint_range_margin=0.10` means
   targets sample from inner 80 % of ±6.28 rad = ±5.03 rad per joint.
2. **`reset_joints_by_scale` broken for zero-default joints** — shoulder_pan
   (default=0.0) and wrist_3 (default=0.0) always reset to 0.0×scale=0.0,
   giving zero starting diversity for those joints.
3. **Tanh reward saturated** — `std=0.1` provides zero gradient beyond ~30 cm;
   average starting error of ~50–77 cm means the fine-grained reward has no
   useful signal.

### Phase 2 — Target/reset/reward fixes for UR10e and Kinova

| Change | Before | After | Rationale |
|---|---|---|---|
| FK `joint_range_margin` | 0.10 | **0.40** (Kinova) / **0.20** (UR10e) | Restrict targets within reachable startup region |
| Reset function | `reset_joints_by_scale` | **`reset_joints_by_offset` (±0.125 rad)** | Fix zero-default joints; consistent diversity |
| Tanh reward `std` | 0.1 | **0.5** | Provide gradient up to ~1.5 m distance |
| UR10e joint limits | ±2π (12.6 rad/joint) | **clamped to ±1.5 rad from defaults** | 4× workspace reduction; finer action mapping |

The UR10e required the most aggressive fix: clamping joint limits from ±6.28 rad
to ±1.5 rad around defaults using `clamp_infinite_joint_limits` with
`max_range=3.0`. This both narrows the FK target workspace and creates a finer
[-1,1]→limits action mapping.

#### Phase 2 results (final)

| Variant | Pos success | Pos error | Ori success | Ori error | Pose success |
|---|---|---|---|---|---|
| **Tensegrity** | **83.1 %** | **4.0 cm** | **73.8 %** | **0.23 rad** | **69.7 %** |
| **Kinova** | **76.8 %** | **7.0 cm** | **84.5 %** | **0.19 rad** | **73.8 %** |
| UR10e | 12.4 % | 16.4 cm | 55.8 % | 0.36 rad | 9.7 % |
| Tendon | — | — | — | — | — (under reconstruction) |

#### UR10e optimisation history

| Run | margin | reset | std | clamp | Pos % | Pos err |
|---|---|---|---|---|---|---|
| v10a | 0.10 | scale | 0.1 | — | 0.0 % | 77.2 cm |
| v10b | 0.35 | offset±0.125 | 0.1 | — | 1.1 % | 50.4 cm |
| v10c | 0.40 | offset±0.125 | 0.5 | — | 7.2 % | 28.6 cm |
| **v10d** | **0.20** | **offset±0.125** | **0.5** | **±1.5** | **12.4 %** | **16.4 cm** |
| v10e | 0.10 | offset±0.125 | 0.5 | ±1.0 | 2.5 % | 25.5 cm |

### Summary

Tensegrity and Kinova converge well (>70 % pose success) with the simplified
config.  UR10e shows learning (0 % → 12.4 % in 48 k steps) but its 6-DOF
symmetric joint limits make the exploration problem inherently harder; the
joint-limit clamping and reward shaping provide meaningful improvement but
further gains would require more training budget (locked at 48 k).

The key finding: robots with large symmetric joint ranges (UR10e ±2π) require
explicit workspace restriction (limit clamping + FK margin) and reward shaping
(wider tanh std) to produce useful gradients within limited training budgets.
Kinova benefits naturally from its clamped infinite joints and narrower
finite-range joints.

**LOCKED parameters** (do not change):
- `position_threshold`: 0.05 m (5 cm)
- `orientation_threshold`: 0.3 rad
- `timesteps`: 48 000

---

## Iteration 11 — Penetrable scene, FK coupling, tighter margins (2026-04-06)

**Scope:** Tensegrity variants only (PD, Tendon, Physical Tendon).

### Issues addressed

1. **Robot colliding with conveyor/drum** — During reach training the robot's
   links collide with scene props (conveyors, drum) which introduces undesired
   contact forces.  Fix: disable collisions on all scene props in
   `ReachSceneCfg.__post_init__` via
   `CollisionPropertiesCfg(collision_enabled=False)`.  Props remain visible as
   a reference for downstream tasks.

2. **Physical elbow FK sampling broken** — The 4-bar antiparallelogram linkage
   has 3 joints (rod_left, rod_right, coupler_left) that were sampled
   independently, violating the closure constraint.  Fix: added
   `joint_coupling_fn` parameter to `FKSampledPoseCommandCfg` and implemented
   `antiparallelogram_joint_coupling()` that enforces rod_right = rod_left and
   computes coupler_left from the closure equation:
   `φ = θ + 2·atan2(−k_e·cos θ, l_e − k_e·sin θ)` (with USD offset
   correction).

3. **FK joint margin too large** — PD/Tendon used the default 0.10 (80% of
   range), Physical used 0.25 (50% of range), producing easy goals.  Fix:
   reduce all to 0.01 (98% of range) to test harder poses.

### Changes

| Change | File | Before | After |
|---|---|---|---|
| Penetrable scene | `reach_env_cfg.py` | Props have default collisions | `collision_enabled=False` on conveyor, conveyor_upstream, drum_target |
| FK coupling API | `fk_sampled_pose_command.py` | No coupling support | Added `joint_coupling_fn: Callable \| None = None` field; called after sampling |
| Physical coupling | `joint_pos_env_cfg_physical.py` | Independent 7-joint sampling | `antiparallelogram_joint_coupling()` enforces 4-bar closure |
| PD margin | `joint_pos_env_cfg.py` (tensegrity) | 0.10 (default) | 0.01 |
| Tendon margin | `joint_pos_env_cfg.py` (tensegrity_tendon) | 0.10 (default) | 0.01 |
| Physical margin | `joint_pos_env_cfg_physical.py` | 0.25 | 0.01 |

### Results

| Variant | Reward (Iter 10) | Reward (Iter 11) | Δ | Pos. Error | Orient. Error | Fine-Grained |
|---|---|---|---|---|---|---|
| Tensegrity PD | +0.52 | **+0.77** | +0.25 ↑ | −0.006 | −0.009 | 0.087 |
| Tensegrity Tendon | +0.64 | **+0.75** | +0.11 ↑ | −0.007 | −0.012 | 0.085 |
| Physical Tendon | +0.67 | **−1.64** | −2.31 ↓ | −0.055 | −0.092 | 0.025 |

### Analysis

- **PD and Tendon improved significantly.** Removing conveyor collisions and
  using the full 98% joint range produced better convergence.  PD reward
  improved from +0.52 to +0.77 and Tendon from +0.64 to +0.75.

- **Physical Tendon regressed severely.** The combination of three changes hit
  this variant hardest: (a) margin dropped from 0.25 → 0.01 (25× harder goal
  distribution), (b) FK coupling now correctly constrains the elbow instead of
  sampling invalid independent configurations, and (c) the body-force tendon
  actuation makes convergence inherently slower.  The policy std deviation
  (0.15 vs 0.02/0.03 for PD/Tendon) indicates the policy has not converged.

- **Next steps for Physical Tendon:** Try intermediate margin (e.g. 0.10),
  increase training budget (96k+), or investigate whether the coupling
  function produces out-of-limit coupler values that PhysX clamps.

---

## Iteration 12 (2026-04-07) — Physical Tendon optimisation (coupling fix + margin sweep)

**Scope:** Physical Tendon variant only (PD and Tendon are LOCKED at their
Iter 11 values).

### Issues addressed

1. **Coupling function bug** — Iter 11 set `rod_right = rod_left` in the
   coupling function.  This is wrong: the antiparallelogram closure requires
   `rod_right = coupler_left = φ + θ₀` where φ is the closure angle computed
   from `rod_left`.  Fixed in `antiparallelogram_joint_coupling()`.

2. **Margin too aggressive** — Iter 11 dropped the margin from 0.25 → 0.01,
   exposing the policy to near-limit targets that body-force actuation cannot
   reach precisely.  Systematically swept margins 0.01–0.40 to find the
   optimum.

### Experiments

| Run | Margin | Coupling | Network | Timesteps | Reward | σ |
|---|---|---|---|---|---|---|
| baseline (Iter 11) | 0.01 | broken | [64,64] | 48 000 | −1.64 | 0.151 |
| 22-42-20 | 0.01 | fixed | [64,64] | 48 000 | −1.64 | 0.151 |
| 23-16-00 | 0.05 | fixed | [64,64] | 48 000 | −1.61 | 0.141 |
| 23-44-19 | 0.25 | fixed | [64,64] | 48 000 | −0.71 | 0.120 |
| 00-16-30 | 0.05 | fixed | [64,64] | 96 000 | −1.44 | 0.133 |
| 01-13-48 | 0.05 | fixed | [256,128,64] | 96 000 | −1.40 | 0.073 |
| **03-34-31** | **0.35** | **fixed** | **[64,64]** | **48 000** | **−0.32** | **0.064** |
| 04-02-59 | 0.40 | fixed | [64,64] | 48 000 | −0.84 | 0.070 |
| 04-32-16 | 0.35 | fixed | [64,64] | 96 000 | −0.33 | 0.055 |
| 05-29-15 | 0.35 | fixed | [256,128,64] | 48 000 | −1.19 | 0.103 |

### Changes (final)

| Change | File | Before | After |
|---|---|---|---|
| Coupling function | `joint_pos_env_cfg_physical.py` | `rod_right = rod_left` | `rod_right = coupler_left = closure(rod_left)` |
| FK margin | `joint_pos_env_cfg_physical.py` | 0.01 | **0.35** |

### Results

| Variant | Reward (Iter 11) | Reward (Iter 12) | Δ | Pos. Track | Ori. Track | Fine-Grained | σ |
|---|---|---|---|---|---|---|---|
| Physical Tendon | −1.64 | **−0.32** | +1.32 ↑ | −0.032 | −0.046 | 0.066 | 0.064 |

### Analysis

- **Major improvement from Iter 11.** Reward improved by +1.32 points, driven
  primarily by the margin increase (0.01 → 0.35) which restricts FK targets to
  the inner 30 % of each joint range.  The coupling fix alone had no measurable
  impact at margin 0.01 (identical −1.64).

- **Margin sweet spot at 0.35.** Below 0.25, body-force actuation cannot
  track orientation targets precisely enough (−0.71 at 0.25).  Above 0.35,
  the workspace becomes too narrow for meaningful exploration (−0.84 at 0.40).

- **Larger networks do not help.** The [256,128,64] network underperformed
  [64,64] at every margin tested, likely due to insufficient data for the
  extra parameters within 48k timesteps.  Extended training (96k) yielded
  negligible gains (−0.33 vs −0.32).

- **Gap to PD/Tendon is expected.** The physical variant (−0.32) still lags
  behind PD (+0.77) and Tendon (+0.75).  This is a fundamental consequence of
  body-force actuation vs direct position/effort control — cable forces
  applied at physical attachment points have inherently lower positioning
  precision than the idealised actuators used by PD and Tendon.  The policy
  converged (σ = 0.064) but cannot overcome the actuator bandwidth limitation.

- **Recommended final configuration:** margin = 0.35, coupling ON, [64,64]
  network, 48 000 timesteps, collision disable ON.

---

## Iteration 13 (2026-04-08) — FK Reference Robot & Convergence

**Status**: Physical Tendon variant **converged** at **−0.35** (margin = 0.01).
PD (+0.77) and Tendon (+0.75) variants remain **LOCKED**.

### Problem

Iteration 12's FK target sampling used the physical robot itself.  During
training, the 4-bar linkage breaks (rod joints lose tracking), and the broken
kinematics feed back into FK sampling.  The result: after the first epoch,
targets degenerate to poses the broken robot can trivially reach, preventing
any useful learning.

### Architecture: Dual-Robot FK Sampling

Added a **separate FK reference robot** (the PD-approximated variant) to the
scene.  It is used exclusively for FK target sampling, while the physical
robot handles control.  This decouples target generation from the fragile
physical dynamics.

**Key design decisions:**

1. **FK reference robot** — `TENS_5DOF_GRIPPER_CFG` (PD variant) added as a
   hidden, collision-disabled articulation at the same position as the
   physical robot.  Its stable single-`elbow_joint` provides clean FK.
2. **Gripper stabilisation** — Reference robot's gripper joints need
   `stiffness=100, damping=100, armature=10` to prevent PhysX NaN corruption
   during `get_link_transforms()`.
3. **No coupling function** — The PD robot uses simple revolute joints, so the
   antiparallelogram coupling is not needed for FK sampling.
4. **Margin = 0.01** — With a clean PD reference, the full PD workspace
   (98 % of joint range) can be sampled safely.  This is a harder target
   distribution than Iter 12's margin = 0.35 (30 % of range).

### Files changed

| File | Change |
|---|---|
| `mdp/fk_sampled_pose_command.py` | Added `fk_reference_asset_name`, `fk_reference_body_name`, `fk_reference_joint_names` config fields.  `_resample_command()` dispatches to `_resample_via_reference()` (new) or `_resample_via_self()` (original, for PD/Tendon).  Reference sampling teleports the FK robot, reads EE pose, and expresses the target relative to the main robot's root frame. |
| `config/tensegrity_tendon/joint_pos_env_cfg_physical.py` | Added PD reference robot (`scene.fk_reference_robot`), gripper stabilisation for both robots, updated FK command config (margin=0.01, reference asset name, no coupling fn). |
| `config/tensegrity_tendon/agents/skrl_ppo_cfg_physical.yaml` | `rollouts: 48` (was 24), `timesteps: 150000` (was 48000). |

### Experiments

| Run | Network | Rollouts | Steps | Margin | Best Reward | Final | Notes |
|---|---|---|---|---|---|---|---|
| 18-29-20 | [64,64] | 24 | 48 000 | 0.01 | −1.70 | −1.94 | First FK-ref test; confirmed no NaN |
| 19-30-57 | [64,64] | 24 | 150 000 | 0.01 | −0.39 | −0.67 | Plateaued at −0.55 ± 0.10, large oscillation |
| 22-38-41 | [128,128] | 24 | 200 000 | 0.01 | −1.60 | killed@43K | Larger net slower, plateaued worse |
| **23-32-28** | **[64,64]** | **48** | **150 000** | **0.01** | **−0.35** | **−0.46** | **Best: stable convergence, low variance** |

### Results

| Metric | Iter 12 (margin=0.35) | Iter 13 (margin=0.01) |
|---|---|---|
| **Best reward** | −0.32 | **−0.35** |
| Position tracking | −0.032 | −0.020 |
| Orientation tracking | −0.046 | −0.044 |
| Fine-grained | 0.066 | 0.041 |
| Total reward (max) | N/A | +0.416 |
| Margin | 0.35 (30% range) | 0.01 (98% range) |
| Network | [64,64] | [64,64] |
| Rollouts | 24 | 48 |
| Timesteps | 48 000 | 150 000 |

### Analysis

- **Dual-robot architecture solves the broken-FK problem.** The physical
  robot's linkage can now break during training without corrupting FK target
  sampling.  Targets are always sampled from clean PD kinematics.

- **Rollouts = 48 is critical for stability.** With rollouts = 24, the reward
  oscillated ±0.15 around a −0.55 plateau.  Doubling the rollout buffer halved
  the oscillation and improved the best reward from −0.39 → −0.35.

- **Larger networks do not help (again).** The [128,128] network converged
  slower and to a worse plateau (−1.60), confirming Iter 12's finding that
  [64,64] is optimal for this problem.

- **Comparable performance despite 3× harder workspace.** Iter 13's
  margin = 0.01 samples from 98 % of the PD joint range (vs Iter 12's 30 %).
  The absolute best reward (−0.35 vs −0.32) is similar, but the position
  tracking is better (−0.020 vs −0.032), indicating the policy generalises
  across a much wider workspace.

- **Fundamental control gap remains.** The physical variant (−0.35) trails
  PD (+0.77) by ~1.12 reward points.  This is inherent to the control problem:
  the physical robot uses 5 effort-based cable tensions (2 elbow + 3 wrist) to
  drive the 4-bar linkage, while the PD variant uses 5 direct position
  commands.  The effort-to-position indirection limits positioning precision.

- **Some environments reach positive rewards.** The max episode reward of
  +0.416 shows the robot CAN reach PD-sampled targets — it just cannot do so
  consistently across the full workspace.

### Recommended configuration

| Parameter | Value |
|---|---|
| FK reference robot | PD variant (hidden, collision-off) |
| Margin | 0.01 |
| Network | [64, 64] |
| Rollouts | 48 |
| Timesteps | 150 000 |
| Collision disable | ON |

---

## Iteration 14 (2026-06-28 → 2026-07-01) — F140 comparison grid, box targets, reference reward

**Scope:** New 6-arm × 4-action-space comparison grid (24 variants), built for
the action-space study (see `doc/reports/RESEARCH_BRIEF_action_spaces.md`).
Full context and open problems were handed over in `doc/reach_task_handoff.md`;
this entry summarises it for the change log.

### Setup

- **Arms (×6):** `UR5e-F140`, `UR10-F140`, `Kinova-F140` (rigid Robotiq 2F-140)
  and their "Frankenstein" counterparts (same arm + compliant 2-DOF tensegrity
  wrist spliced between flange and gripper).  All floor-standing at
  `(0.75, 1.0, 0.75)`, EE body `robotiq_base_link`, shared setup funnelled
  through `reach/config/f140_reach_common.py`.
- **Action spaces (×4):** joint (EMA joint-position-to-limits, α=0.2),
  Differential-IK relative (`_ik`), Differential-IK absolute (`_ikabs`),
  Operational-Space Control (`_osc`).
- **Targets:** switched from FK-sampled full-SO(3) poses to a **uniform
  position box** in the robot base frame (`uniform_ranges` mode of
  `FKSampledPoseCommand`): x (0.30, 0.50), y (−0.20, 0.20), z (0.25, 0.50),
  orientation gripper-down `pitch=π` with free yaw — the IsaacLab UR10-reach
  recipe, sized to fit every arm's envelope (farthest corner ≈ 86 % of UR5e
  reach).  One target per episode; episode 6 s = 180 steps @ 30 Hz.
- **Success metric thresholds:** position 5 cm.

### Key findings (P1/P2 of the handoff)

1. **Heavy orientation reward wrecked position (P1).**  An orientation weight
   of −0.5 (+ extra fine-grained orientation term, position std widened to
   0.5) made the 6-DOF UR arms trade position for orientation: 20–75 cm
   position error.  **Fix: revert to the IsaacLab reference reward exactly**
   (position −0.2, fine-grained tanh 0.1 @ std 0.1, orientation −0.1,
   action_rate/joint_vel −1e-4 with the 4 500-step curriculum).
   UR5e-F140: 55 cm → **1.8 cm**, 96 % of steps within 5 cm.
2. **A fixed orientation target is geometrically unreachable for 6-DOF arms
   (P2).**  A GPU diagnostic (`scripts/diagnose_box_orientation.py`) showed the
   best possible *single* fixed orientation covers ~0 % of the box for UR5e
   (~2–9 % for the 7-DOF Kinova).  Orientation is therefore **loose by
   design**; reach is a position task.  Do not re-add orientation weight.

### Status at end of iteration (seed 42, 100k timesteps)

20/24 variants at 2–9 cm / 73–96 %.  By action space: joint 3.1 cm / 91 %,
ik 3.4 cm / 88 %, ikabs 4.5 cm / 85 %, **osc 21.3 cm / 38 %**.  Open problems:
OSC diverges on the tensegrity wrist (P3: `ur10_frankenstein_osc` 65 cm / 0 %,
`ur5e_frankenstein_osc` 41 cm / 0 %), `kinova_frankenstein_osc` collapsed
(P4: logs 0.8 cm but 0 % success), and `ur5e_f140_osc` / two ikabs variants
mediocre (P5).

---

## Iteration 15 (2026-07-02) — OSC stability & controller tuning: full grid passes

**Goal:** Fix the OSC divergences (P3/P4) and tighten the weak controller
configs (P5) while keeping the comparison fair — same reward, targets and
observations everywhere; only per-action-space controller/PPO settings differ,
identically across arms.

### Root cause of the OSC+Frankenstein divergence (P3/P4)

`apply_osc_action` zeroed the implicit PD of **all** non-gripper actuators —
including the compliant tensegrity wrist (`wrist_x/y_joint`: effort limit
3.5 N·m, PD 400/20, near-zero reflected inertia) — and commanded them with
pure-effort OSC.  Undamped compliant joints oscillate, the EE error feeds back
into the task-space loop, and the arm diverges.  P4 confirmed via episode
length: `kinova_frankenstein_osc` episodes were killed by `joint_vel_diverged`
after **3.0 of 180 steps** on average — its logged "0.8 cm / 2°" was an
artifact of frozen 3-step episodes, exactly the degenerate pattern the handoff
warned about (§8).

### Changes

1. **OSC wrist fix** (`reach/config/osc_reach_common.py`,
   `TENSEGRITY_WRIST_JOINTS`): the `tendon_wrist` actuator keeps its PD (holds
   neutral, like the gripper groups) and `wrist_x/y_joint` are excluded from
   the OSC action's joint set / Jacobian.  OSC corrects wrist deflection with
   the rigid arm joints.  Redundancy for null-space posture control is
   detected from the OSC joint count (Kinova 7-DOF: redundant; URs: not).
2. **OSC partial inertial dynamics decoupling** (`osc_reach_common.py`):
   `inertial_dynamics_decoupling=True` + `partial_inertial_dynamics_decoupling=True`.
   Full decoupling (franka sample) is ill-conditioned on these arm+gripper
   articulations, but the partial (translational/rotational block) form is
   robust and scales task-space gains by the arm's inertia, so the same
   kp/ζ behave consistently across light (UR5e) and heavy (UR10) arms.
3. **ikabs PPO exploration** (all six `agents/skrl_ppo_ikabs_cfg.yaml`):
   `initial_log_std` 0.0 → **−1.0**.  IK-abs actions are an absolute EE pose
   in metres; std=1 Gaussian exploration swamps the 0.2–0.4 m target box.

### Experiments that lost (kept for the record)

| Hypothesis | Variant tested | Result | Verdict |
|---|---|---|---|
| OSC damping ratio ζ=2 (under-damping suspected) | ur5e_f140_osc | 11.9 → 44.0 cm, 0 % (over-damped crawl) | rejected |
| OSC stiffness ceiling 200 → 400 | kinova_f140_osc | 4.1/74 % → 4.3/72 % | neutral, rejected |

### Results (seed 42, 100k timesteps, §8 metric conventions of the handoff)

Full 24-variant grid retrained 2026-07-02 (runs `10-33` – `11-00`).  Joint and
IK-rel variants reproduced their Iteration-14 numbers bit-exactly (configs
untouched, deterministic training) — sanity check that only the intended knobs
changed.

| Variant | Iter 14 (cm / %succ) | Iter 15 (cm / %succ) | Driver |
|---|---|---|---|
| ur5e_f140_osc | 11.9 / 67 | **2.1 / 93** | decoupling |
| ur10_f140_osc | 4.8 / 85 | **3.1 / 91** | decoupling |
| kinova_f140_osc | 4.1 / 74 | **3.7 / 87** | decoupling |
| ur5e_frankenstein_osc | 41.0 / 0 | **3.8 / 91** | wrist fix + decoupling |
| ur10_frankenstein_osc | 65.1 / 0 | 20.1 / 58 † | wrist fix + decoupling |
| kinova_frankenstein_osc | collapsed (P4) | **3.0 / 90** | wrist fix + decoupling |
| ur10_frankenstein_ikabs | 8.9 / 73 | **5.1 / 83** | ikabs std |
| kinova_f140_ikabs | 5.5 / 73 | 5.8 / 67 | ~neutral |
| ur10_f140_ikabs | 2.8 / 94 | 4.7 / 85 | slightly worse |
| ur5e_f140_ikabs | 2.6 / 92 | 2.1 / 94 | ikabs std |
| ur5e_frankenstein_ikabs | 4.4 / 85 | 3.8 / 87 | ikabs std |
| kinova_frankenstein_ikabs | 2.6 / 92 | 2.4 / 90 | ~neutral |
| joint / ik variants (12) | 1.8–4.9 / 83–96 | unchanged (bit-exact) | — |

By action space (mean position error / mean success):

| Space | Iter 14 | Iter 15 |
|---|---|---|
| joint | 3.1 cm / 91 % | 3.1 cm / 91 % |
| ik (rel) | 3.4 cm / 88 % | 3.4 cm / 88 % |
| ikabs | 4.5 cm / 85 % | 4.0 cm / 84 % |
| **osc** | **21.3 cm / 38 %** | **7.0 cm / 87 %** (3.3 / 90 % excl. †) |

### († ) `ur10_frankenstein_osc` — seed sensitivity, not divergence

The seed-42 run converges to a *stable* local optimum (full 180-step episodes,
plateau from ~iteration 16) that trades position for orientation (its 75°
orientation error beats healthy ur10_f140_osc's 87°).  Same config, different
seeds: **seed 7 → 4.1 cm / 90 %, seed 123 → 6.5 cm / 76 %** — both pass.  No
per-variant config change was made (it would break the fair comparison);
recommendation: report the study multi-seed (`tools/train_alex.sh -s "…"`), or
footnote this variant.

### Acceptance status (handoff §9)

- No divergent or collapsed runs; every run completes full episodes.
- 23/24 variants ≤ 5.8 cm and ≥ 67 % success (bar: ≤ ~8 cm, > ~60 %); the one
  miss is the documented seed-42 outlier above.
- Reward (IsaacLab reference), target box, observation structure: unchanged.

---

## Iteration 16 (2026-07-09) — Physical variant rework: cable limits, linkage integrity, hierarchical control

**Scope:** Physical tendon variant + NEW physical hierarchical variant.
PD and Tendon remain LOCKED. Full change log with measurements:
[physical_variant_fix_report.md](physical_variant_fix_report.md).

### Problem

The 2026-05 seed study ([reach_evaluation_findings.md](reach_evaluation_findings.md))
left the physical variant at 31.4 % ± 14.4 % success. Visual inspection showed
a reward hack: sustained max tension tears the PhysX loop-closure joint open
(maximal-coordinate constraint) and the policy steers the dangling forearm
with the linear base. A subtler mode was found during verification: the
four-bar can re-close in the *parallelogram branch* (closure satisfied, but
the forearm decoupled from the rods).

### Changes (model security + sim fidelity)

| Change | Where | Detail |
|---|---|---|
| Cable length limits | `robots/tendon_actuator.py` | [0.0913, 0.2577] m between attachment points (±70° workspace geometry); wind-up stop + 20 kN/m / 200 N·s/m stretch stop |
| Per-tendon saturation | 〃 | `[480, 480, 80, 80, 80]` N hardware limits (was scalar 500 N) — M2 for this variant |
| Motor/spool lag | 〃 | 50 ms first-order filter on tensions/efforts (kills constraint-impulse spikes from 0→480 N action steps) |
| Linkage integrity **penalty** | `reach/mdp/terminations.py` + `rewards.py` | closure gap > 3 cm OR branch flip \|elbow − (rod_L+rod_R)\| > 0.3 rad → **−1/step reward penalty**. First retrain attempt used a failure *termination*: with the net-negative reach reward this is a suicide exit — both variants collapsed to 5-step episodes within 2 k timesteps. Sanity gate now also asserts episode length > 150. |
| Correct elbow measurement | `reach/mdp/observations.py` | `lower_arm_angle`/`_ang_vel` from forearm body twist (elbow = rod_L + rod_R, not any single joint); + `cable_lengths`/`cable_length_rates` obs (eval report §9 #1); raw linkage joints removed from obs |
| 120 Hz physics | `joint_pos_env_cfg_physical.py` | closure drifts cm under 480 N at 60 Hz; decimation 4 keeps control at 30 Hz; `enable_external_forces_every_iteration`, min 2 velocity iterations |
| PhysX sleep disabled | `robots/tendon_robot_cfg.py` | body-force-driven arm freezes when asleep (tensor-API forces don't wake it) |
| joint_vel_diverged 100 → 500 rad/s | env cfg | 1-gram wrist dummy link spikes on reset transients → silent truncation reset-loops at 100 (likely a contributor to the 2026-05 instability) |
| Hierarchical variant (NEW) | `robots/tendon_controllers.py` | `Template-Reach-Tensegrity-Physical-Hierarchical-v0`: 3 joint set-points tracked by the step-response-validated PID (75/6/3, 10/1.5/0.6) + gravity comp (recalibrated to measured 5-DOF static balance: m·g·l 6.35 → ≈ 2.8 N·m) + block-wise tension distribution → same body-force channel |

Verified before training (see fix report §4): sustained 480 N now parks the
elbow at the ~70° stop with the closure intact (< 1 mm gap); 600-step random
run with partial resets shows no NaN, no frozen dynamics, full ±78° coverage;
hierarchical set-point tracking < 1° steady-state elbow error.

Also fixed: `evaluate_reach.py` picked the "latest" checkpoint from a run dir
by **lexicographic** sort — the 2026-05 evaluation actually scored
`agent_9600` (PD/Tendon, of 48 k) and `agent_90000` (Physical, of 150 k)
instead of the final checkpoints. Now numeric.

### Training setup

| Parameter | 2026-05 (physical) | Iter 16 (both physical variants) |
|---|---|---|
| Timesteps | 150 000 | **48 000** (standard reach budget — the secured task should no longer need 3× budget) |
| Physics / control | 60 Hz / 30 Hz | **120 Hz** / 30 Hz |
| Seeds | 0–4 | 0–4 |
| Everything else | rollouts 48, [64, 64], margin 0.01, FK reference robot | unchanged |

Pipeline: [run_physical_reach_pipeline.sh](../../src/tensegrity_pick/scripts/training/run_physical_reach_pipeline.sh)
(resumable train → eval → plots; eval figures now scripted in
[plot_reach_eval_results.py](../../src/tensegrity_pick/scripts/plotting/plot_reach_eval_results.py)).

### Results (local, partial — full 5-seed matrix moved to the cluster)

Local training was stopped after seeds 0–2 (direct) / 0–1 (hierarchical) for
GUI review; full training + evaluation runs on the cluster (see the pipeline
script and the cluster section of the reach README).

Seed 0 (48 k steps, before the 2026-07-10 controller/RL fixes below):

| Metric (last 10 %) | Direct | Hierarchical |
|---|---:|---:|
| Total reward | −1.01 | −1.25 |
| position_reached (per-step) | 0.20 | 0.09 |
| Episode length | 180 (full) | 180 (full) |
| linkage_broken fraction | 0.018 | 0.029 |
| Policy σ (final) | 1.28 | 1.57 |

The σ plateau (exploration noise never collapsing) + the GUI review
("oscillates around the target, sometimes does not approach") triggered
iteration 16b.

### Iteration 16b (2026-07-10) — controller retune, PhysX forensics, RL profile

Full detail: [fix report addendum](physical_variant_fix_report.md#addendum--2026-07-10-gui-review-findings-controller-retune-physx-forensics).
Summary of applied changes:

- **Inner PID**: set-point slew limiting (1.2/5/5 rad/s), per-joint integrator
  clamps (2.0/0.8/0.8), elbow integral zone 0.35 rad; set-point ranges
  clamped to the *achievable* workspace (elbow ±60°, wrist ±40°); target
  sampling margin 0.01 → 0.10.  Set-point sweep tracks ±60° within 6–11°.
- **Scripted IK+PID heuristic** (`heuristic_physical_ik.py` + `--agent
  heuristic` for the hierarchical Play task): DLS-IK on the FK reference
  robot + command ramping.  Local: reach 0.56, best-err 0.12 m over 128
  episodes.  Wired into the eval pipeline (5 seeds on the cluster).
- **PhysX forensics**: three measured closed-chain artifacts (closure elastic
  drift, absorbing branch flips, ~10–20 N·m quasi-static solver breakaway —
  the latter unresolved, `solver_dither` knob experimental).  Two dead ends
  (mimic-drive closure, rigid-elbow rebase) tried and reverted — the model
  stays the true four-bar per user directive.  `*_awake.usd` bakes in use.
- **RL profile**: entropy 0, max_log_std 0, initial_log_std −0.5 (σ-plateau
  fix, shirt-task profile); new `applied_tensions` observation (obs 28 → 30
  direct, 26 → 28 hierarchical); random-phase smoke: hierarchical shows ZERO
  linkage flags under random set-points (slew-limited inner loop).
- **Harness fixes**: run-dir checkpoint selection now numeric (the 2026-05
  study actually evaluated agent_9600/agent_90000, not the final
  checkpoints); eval inference-tensor crash fixed (`prev_actions` clone) —
  `action_rate_mean`/`success_held` fields become populated for the first
  time.

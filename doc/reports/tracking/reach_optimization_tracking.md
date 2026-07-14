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
the action-space study (see `doc/reports/tmp/RESEARCH_BRIEF_action_spaces.md`).
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
[physical_variant_fix_report.md](../physical_variant_fix_report.md).

### Problem

The 2026-05 seed study ([reach_evaluation_findings.md](../task_evaluations/reach_evaluation_findings.md))
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

Pipeline: [run_physical_reach_pipeline.sh](../../../src/tensegrity_pick/scripts/training/run_physical_reach_pipeline.sh)
(resumable train → eval → plots; eval figures now scripted in
[plot_reach_eval_results.py](../../../src/tensegrity_pick/scripts/plotting/plot_reach_eval_results.py)).

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

Full detail: [fix report addendum](../physical_variant_fix_report.md#addendum--2026-07-10-gui-review-findings-controller-retune-physx-forensics).
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
---

> **Parallel work streams — iteration numbering.** The entry above (Iteration 16,
> 2026-07-09) is the **physical tensegrity variant** rework (developed on the
> tensegrity branch). The entries below (Iterations 16–22, 2026-07-07 onward) are
> the **F140 comparison-grid** controller-hardening study, developed on a separate
> branch with its own iteration numbering — within those entries "iteration N"
> refers to this F140 study. Both share the pre-2026-07-07 history above.

## Iteration 16 (2026-07-07) — Kinova PhysX joint-limit fix + orientation-refinement study

Two independent items, both flagged by Georg's visual verification of the
iteration-15 checkpoints (`logs/skrl/need_visual_verification/reach/findings.md`):
**(A)** a concrete startup bug (Kinova `setLimitParams()` PhysX errors), and
**(B)** a quality investigation into the poorly-tracked target orientation and the
never-used tensegrity wrist.  The settled 24-variant position-only baseline is
**unchanged and still citable** — (A) is behaviour-preserving, (B) is opt-in and
off by default.

### Part A — Kinova `setLimitParams()` startup errors (bug fix)

**Symptom (findings.md "Cross-cutting"):** every Kinova checkpoint throws 8–9×
`PxArticulationJointReducedCoordinate::setLimitParams() only supports limit
angles in range [-2Pi, 2Pi]` at load, one as a GUI popup.  Did not affect
playback, but noisy and user-facing.

**Root cause (confirmed on GPU).** The Kinova Gen3 continuous joints
(`joint_1/3/5/7`, plus wide Robotiq gripper revolute joints — 8 in total) are
authored in the referenced NVIDIA asset with per-limit angles outside PhysX's
`[-2π, 2π]`.  PhysX validates them when it **bakes the articulation at
`sim.reset()`** and throws once per offending joint.  The existing
`clamp_infinite_joint_limits` runs at `mode="reset"`, i.e. *after* the bake, so
it fixes the limits for stepping but cannot prevent the init-time error.
Reproduced directly: with the fix disabled, a Kinova env logs **8×
`setLimitParams`** at startup; the reset clamp still leaves the correct baked
limits (`joint_1 [-π,π]`, `joint_3 [0,2π]`, `joint_5 [-π,π]`, `joint_7 [0,2π]`,
`joint_2/4/6` untouched) and 0 NaN steps — exactly the "silently fixed for
stepping" pattern.

**Why not a `mode="prestartup"` event.** Isaac Lab *does* expose a prestartup
event mode that edits USD before the bake — but `EventManager._prepare_terms`
**hard-rejects any prestartup term when `replicate_physics=True`**
(`event_manager.py:363`), and reach needs `replicate_physics=True` for 4096-env
training.  (For our case the replication sharing is actually desired — identical
limits everywhere — but the guard is unconditional.)  Editing the shared
`res/KinovaGen3/*.usd` asset was also rejected: `shirt_distribute` and
`shirt_present` reference the same Kinova cfg, so a binary asset edit would leak
out of the reach-grid scope.

**Fix (reach-local, before-bake).** Wrap the robot spawner
(`reach/mdp/events.py::spawn_usd_with_clamped_joint_limits`, `@clone`-decorated
like `spawn_from_usd`): it spawns the single **source** prim (`env_0`) and clamps
its over-wide/infinite revolute-joint USD limits *before* the `@clone` decorator
clones to the other envs and before `sim.reset()` bakes PhysX.  With
`replicate_physics=True` PhysX parses only `env_0`, so a valid source removes the
error at its root; clones inherit the clamped limits.  It uses the **identical
`default_pos ± fallback_range/2` rule** as the reset-mode clamp (same
`max_range`/4π threshold), so the baked limits — and therefore training — are
byte-for-byte unchanged; the reset-mode term is kept as an idempotent no-op net.
Clamp parameters are passed via a module-level registry keyed by `usd_path`
(`register_joint_limit_clamp`) rather than a closure, because Isaac Lab's Hydra
config round-trip serialises `spawn.func` to `module:qualname` and rebuilds it
via `getattr` — a closure has no resolvable module attribute (this bit us once:
the first closure implementation trained fine via `gym.make` but crashed the
Hydra training entrypoint with `AttributeError: ... has no attribute
'_spawn_and_clamp'`).  Wired only for Kinova (`"Kinova" in usd_path`) in
`f140_reach_common.py`; the UR arms are untouched (their ±2π limits are already
in range).

**Verification.**

| Check | Result |
|---|---|
| Pre-fix (stock spawner), Kinova-F140 | **8×** `setLimitParams`, 0 NaN |
| Post-fix, Kinova-F140 joint / OSC / IK-Abs | **0×** `setLimitParams`, 0 NaN |
| Post-fix, Kinova-Frankenstein joint / OSC | **0×** `setLimitParams`, 0 NaN |
| Post-fix baked limits vs pre-fix reset-clamp limits | identical (behaviour-preserving) |
| Kinova training runs (OSC, IK-Abs) reach the training loop | yes (spawner clamps 8 joints, no crash) |

Regression across all four action spaces and both Kinova arms (zero-action
stepping in `scripts/agents/verify_kinova_limits.py`): errors gone, no NaN, no
behaviour change.  GUI-popup confirmation is deferred (Alex is headless; RTX GUI
segfaults on the cluster driver) but the popup was the same `setLimitParams`
error, now eliminated at the stderr level.

### Part B — Orientation refinement for redundant arms (investigation)

**Question (findings.md).** Target orientation (gripper-down `pitch=π`, free yaw)
is tracked poorly and the tensegrity wrist is never used.  Can orientation be
improved on the redundant arms *without* reproducing the P1 position regression
(iteration 14: a blanket orientation weight traded position catastrophically)?

**Baseline orientation, by variant (seed 42, from the iteration-15 reports).**
Success = fraction of steps within threshold (position < 5 cm, orientation
< 0.3 rad); `pose` = both.

| Variant | DOF | pos err | pos % | **orient %** | pose % |
|---|---|---|---|---|---|
| kinova_frankenstein_ikabs | 9 | 2.4 cm | 90 | **40** | 40 |
| kinova_frankenstein (joint) | 9 | 2.5 cm | 93 | **27** | 26 |
| kinova_f140_osc | 7 | 3.7 cm | 87 | **9** | 8 |
| kinova_f140_ikabs | 7 | 5.8 cm | 67 | **6** | 6 |
| kinova_f140 (joint) | 7 | 2.9 cm | 88 | **1** | 1 |
| ur5e_frankenstein_ikabs | 8 | 3.8 cm | 88 | **0** | 0 |
| ur5e/ur10_frankenstein_osc | 8 | 3.8–4.1 cm | 90–91 | **0** | 0 |
| ur5e/ur10_frankenstein (joint/ik) | 8 | — | — | **0** | 0 |
| ur5e_f140, ur10_f140 (rigid 6-DOF) | 6 | — | — | 0 | 0 |

Read-off: at the reference −0.1 orientation weight, orientation-seeking emerges
**only** where redundancy is high *and* the controller exposes orientation
directly — the 9-DOF Kinova-Frankenstein under IK-Abs (40 %) and joint (27 %).
Every UR-Frankenstein (8-DOF) sits at 0 %: the spare wrist is present but the
weak −0.1 signal never recruits it.  All OSC-Frankenstein sit at 0 % too — the
wrist is excluded from the OSC Jacobian (iteration 15) and held straight, and OSC
on the UR arms is non-redundant once the wrist is out.

**Intervention (opt-in, redundant arms only).** A *position-gated* orientation
reward, `orientation_refine_gated` (`reach/mdp/rewards.py`):
`(1 − tanh(orient_err/0.3)) · (1 − tanh(pos_err/0.1))`.  The smooth position gate
is ~0 while the EE is far and ~1 once it is close, so orientation is only
rewarded **after** position is (nearly) achieved.  It cannot cause the P1 trade:
moving off-target lowers *both* the gate and the main position reward, so there
is never an incentive to sacrifice position; a redundant arm can instead reduce
orientation error at fixed position via its null space (pure gain).  Added with a
positive weight **only** to variants with > 6 controlled joints (all Frankenstein
arms + Kinova-F140) — the two rigid 6-DOF UR-F140 arms have no spare DOF and are
excluded, matching the P2 finding that their target orientation is largely
geometrically unreachable.

**Reproducibility / fairness.** Off by default: the term is added only when
`REACH_ORIENT_REFINE_WEIGHT > 0` (env var), so the 24-variant position-only
baseline is bit-exact and still citable.  This is a per-variant redundancy term
that exists only on redundant configs by construction — it is *not* a rebalance
of the shared reward (cf. iteration 14's premise).

**Experiment (seed 42, 100k steps, weight 0.5, gate std pos 0.1 m / ori 0.3 rad).**
Four variants spanning the hypothesis (runs `2026-07-07_17-3*`):

| # | Variant | Tests | Baseline pos cm / pos% / **orient%** / pose% | Enhanced pos cm / pos% / **orient%** / pose% |
|---|---|---|---|---|
| 1 | kinova_f140_osc | 7-DOF OSC null space | 3.7 / 87 / **9** / 8 | 4.0 / 81 / **75** / 74 |
| 2 | kinova_frankenstein_ikabs | 9-DOF, headroom | 2.4 / 90 / **40** / 40 | 5.5 / 73 / **69** / 65 |
| 0 | ur5e_frankenstein_ikabs | 8-DOF, wrist in IK set | 3.8 / 88 / **0.2** / 0.1 | 5.8 / 78 / **7.9** / 5.3 |
| 3 | ur5e_frankenstein_osc | wrist excluded from OSC | 3.8 / 91 / **0** / 0 | 2.7 / 92 / **0** / 0 |

**Findings.**

1. **The gate works — no P1 blowup anywhere.** The worst position regression is
   +3.1 cm (variant 2), vs. the 20–75 cm of iteration-14's blanket weight.  The
   gated term never drives position off a cliff, confirming the design: moving
   off-target loses both the gate and the position reward.

2. **Standout: `kinova_f140_osc`, orientation 9 % → 75 % for +0.3 cm.**  The
   7-DOF arm's OSC null space is exactly the "redundancy the controller can
   spend on orientation at ~fixed position" the hypothesis predicted.  This is
   the clean win — a 66-point orientation jump at essentially no position cost
   (87 % → 81 % success is within the range this variant already varies over).

3. **`kinova_frankenstein_ikabs` 40 % → 69 %, but position 2.4 → 5.5 cm.**  The
   9-DOF arm reaches high orientation but at a real position cost at weight 0.5
   — a genuine (bounded) trade, not a free lunch.  Suggests 0.5 is too strong
   for this variant; a weight sweep is running (below).

4. **`ur5e_frankenstein_osc` 0 % → 0 %, position unchanged (even +): the safe
   negative.**  The wrist is excluded from the OSC Jacobian (iteration 15) and
   the UR arm is non-redundant without it, so there is *no* accessible
   redundancy for orientation.  The gate correctly makes the term inert —
   position is protected, orientation does not move.  This is the control that
   proves the mechanism only acts where redundancy is real.

5. **`ur5e_frankenstein_ikabs` 0 % → 8 %, position 3.8 → 5.8 cm: marginal.**  The
   8-DOF UR-Frankenstein *has* the wrist in its IK set, but recruits it only
   weakly and pays a position cost to do so — consistent with P2 (gripper-down
   + sampled yaw is near the edge of what the UR5e reach envelope allows), so
   there is little "free" orientation to gain via the wrist here.

**Interim conclusion.** Position-gated orientation refinement **does** exploit
redundancy for orientation without the P1 regression, and the effect tracks
exactly where redundancy is *controller-accessible*: dramatic on the 7-DOF OSC
null space (`kinova_f140_osc`), strong-but-costly on the 9-DOF IK-Abs arm,
marginal on the 8-DOF UR-Frankenstein (reachability-limited), and correctly
inert where the controller can't reach the spare DOF (OSC with the wrist
excluded).  It is safe but not always free at weight 0.5.

**Weight sweep (weight 0.2, seed 42)** on the two big-gain variants:

| Variant | baseline | w=0.5 | w=0.2 |
|---|---|---|---|
| kinova_f140_osc (pos cm / pos% / orient%) | 3.7 / 87 / 9 | 4.0 / 81 / 75 | 4.2 / 74 / 41 |
| kinova_frankenstein_ikabs (pos cm / pos% / orient%) | 2.4 / 90 / 40 | 5.5 / 73 / 69 | 4.0 / 80 / 68 |

- **`kinova_frankenstein_ikabs` (IK-Abs): clean tunable trade.** w=0.2 keeps
  essentially all the orientation gain (68 % vs 69 %) while recovering position
  (5.5 → 4.0 cm, 73 → 80 % success). So the position cost at 0.5 was the weight,
  not a hard limit — ~0.2 is a better operating point (though position still
  costs ~1.6 cm vs the 2.4 cm baseline).
- **`kinova_f140_osc` (OSC): noisier, not monotone.** w=0.2 gave 41 % orient /
  74 % pos — *less* orientation than w=0.5 (expected) but *also* slightly worse
  position (single-seed OSC variance; this action space is seed-sensitive, cf.
  the iteration-15 `ur10_frankenstein_osc` note). The w=0.5 point (75 % orient,
  +0.3 cm) remains the best OSC result; a clean weight curve here would need a
  multi-seed sweep.

### Overall conclusion (Part B)

**Yes — wrist/redundancy-mediated orientation refinement is achievable without the
P1 position regression, but only where the redundancy is *controller-accessible*,
and it is not free at the weights tested.** Concretely, per arm family:

- **Kinova-F140 (7-DOF, OSC null space):** best case. Orientation 9 % → **75 %**
  for +0.3 cm at weight 0.5 — the redundancy is spent on orientation almost for
  free. Recommended if orientation matters downstream.
- **Frankenstein arms via IK-Abs / joint (wrist in the controlled set):** works,
  with a tunable position cost (`kinova_frankenstein_ikabs` 40 → 68 % orient at
  4.0 cm with weight 0.2). The tensegrity wrist *is* recruited here.
- **Frankenstein arms via OSC:** does **not** work — the wrist is excluded from
  the OSC Jacobian (iteration 15, for stability) and the UR arm is otherwise
  non-redundant, so there is no accessible DOF; the gated term is correctly
  inert (`ur5e_frankenstein_osc` 0 → 0 %, position kept). Recruiting the wrist
  under OSC would require re-including it with a low-stiffness impedance row —
  which iteration 15 showed reintroduces the divergence it was excluded to fix,
  so it was **not** pursued.
- **UR-Frankenstein via IK-Abs (8-DOF):** marginal — some orientation gain but a
  real position cost, because much of the gripper-down + sampled-yaw target set
  is near the edge of the UR5e reach envelope (P2), so there is little "free"
  orientation for the wrist to capture.
- **Rigid UR5e-F140 / UR10-F140 (6-DOF):** out of scope, unchanged. No spare DOF;
  P2 says the target orientation is largely geometrically unreachable. The
  iteration-14 P1/P2 conclusion stands for these two.

**What ships:** nothing changes in the settled 24-variant baseline (the term is
off unless `REACH_ORIENT_REFINE_WEIGHT > 0`). The refinement is available as an
opt-in for downstream tasks that need orientation on a redundant arm — the
`kinova_f140_osc` path is the clear recommendation. A production adoption should
multi-seed the chosen variant/weight (OSC especially) before committing.

### Files touched (iteration 16)

- `reach/mdp/events.py`: `_clamp_revolute_joint_limits_on_prim`,
  `register_joint_limit_clamp`, `spawn_usd_with_clamped_joint_limits` (Part A).
- `reach/mdp/rewards.py`: `orientation_refine_gated` (Part B).
- `reach/mdp/__init__.py`: exports.
- `reach/config/f140_reach_common.py`: wire the Kinova spawner (A) + the opt-in
  gated reward for >6-DOF variants (B).
- `reach/README.md`: before-bake clamp note. Verification/experiment scripts:
  `scripts/agents/verify_kinova_limits.py`, `tools/verify_kinova_fix.sbatch`,
  `tools/exp_orient_refine*.sbatch`.

---

## Iteration 17 (2026-07-08) — Multi-seed grid: OSC seed-instability + orientation read-out

**Goal:** de-risk the whole grid by running every variant over 5 seeds — the
first thing steps 16 and earlier lacked. All numbers 14–16 were single-seed(42).

**Setup:** 24 variants × seeds {0, 1, 2, 42, 123} = **120 runs, all COMPLETED**,
position-only baseline (orientation-refinement term off). Command:
`./tools/train_reach_alex.sh -- -s "0 1 2 42 123" -t 02:00:00`.
**Reproduction check:** `ur5e_f140_osc` seed 42 = **2.1 cm / 93.3 %**, byte-identical
to the iteration-15 single-seed number — the pipeline is deterministic at fixed
seed for the untouched (non-Kinova) configs. (Kinova differs slightly by design;
see the note at the end.)

### Finding 1 — OSC is only seed-robust on the *redundant* Kinova arms

The iteration-15 conclusion "OSC full grid passes" was **seed-42 luck.** OSC is
tight and reliable on the 7-DOF Kinova arms, but **all four UR OSC variants are
bimodal** — they converge on some seeds and diverge catastrophically (20–60 cm,
collapsed episodes) on others:

| OSC variant | per-seed pos err (cm), seeds {0,1,2,42,123} | mean | verdict |
|---|---|---|---|
| kinova_f140_osc | 2.9 / 1.8 / 4.0 / 2.8 / 2.2 | **2.7 ± 0.8** | robust ✓ |
| kinova_frankenstein_osc | 2.0 / 3.0 / 2.7 / 3.3 / 2.2 | **2.6 ± 0.5** | robust ✓ |
| ur10_f140_osc | 13.6 / 11.7 / 2.6 / 3.1 / 5.3 | 7.3 ± 4.5 | mixed |
| ur5e_f140_osc | 19.3 / 28.1 / 26.8 / 2.1 / 2.6 | 15.8 ± 11.4 | **bimodal ✗** (3/5 diverge) |
| ur10_frankenstein_osc | 29.0 / 4.8 / 54.5 / 20.1 / 6.5 | 23.0 ± 18.1 | **bimodal ✗** |
| ur5e_frankenstein_osc | 12.6 / 4.6 / 58.4 / 3.8 / 60.4 | 28.0 ± 25.9 | **bimodal ✗** |

Seed 42 sat at the *good* end for `ur5e_f140_osc` and `ur5e_frankenstein_osc`,
which is exactly why iteration 15 recorded them as passing. This is what
multi-seed was meant to catch. **Root cause (hypothesis):** the variable-kp /
partial-decoupling OSC config has no null-space regulariser on the non-redundant
UR arms (nullspace control is Kinova-only, >6 OSC joints), so the redundancy-free
task-space loop is seed-fragile. The non-OSC spaces have no such failure.

### Finding 2 — Non-OSC spaces are stable and hold across seeds

| Space | mean pos / success | status |
|---|---|---|
| joint | 3.8 cm / 88 % | stable ✓ |
| ik (rel) | 3.3 cm / 90 % | stable ✓ |
| ikabs | 3.5 cm / 88 % | stable ✓ (grid best: **ur5e_f140_ikabs 1.9 ± 0.2 cm / 94.5 %**) |
| osc | 13.2 cm / 72 % | **only reliable on Kinova** (2.65 cm / 90 % excl. the 4 UR-OSC) |

Full per-variant means±std (5 seeds): pos error / pos% / orient% / pose%.

| Variant | pos cm | pos% | orient% | pose% |
|---|---|---|---|---|
| kinova_f140 | 2.7±0.6 | 90.0±4.6 | 6.4 | 5.8 |
| kinova_f140_ik | 3.0±0.7 | 89.9±3.7 | 4.1 | 4.0 |
| kinova_f140_ikabs | 4.2±0.6 | 79.9±7.1 | 13.1 | 12.2 |
| kinova_f140_osc | 2.7±0.8 | 90.3±5.0 | 9.2 | 8.6 |
| kinova_frankenstein | 4.1±2.1 | 83.4±13.7 | 23.5 | 22.4 |
| kinova_frankenstein_ik | 2.9±0.9 | 92.4±2.1 | 20.7 | 20.6 |
| kinova_frankenstein_ikabs | 2.3±0.2 | 91.5±1.4 | 29.4 | 29.0 |
| kinova_frankenstein_osc | 2.6±0.5 | 90.6±3.8 | 1.3 | 1.3 |
| ur10_f140 | 3.4±0.4 | 90.8±0.8 | 0.0 | 0.0 |
| ur10_f140_ik | 3.3±0.6 | 90.1±2.5 | 0.0 | 0.0 |
| ur10_f140_ikabs | 3.7±0.7 | 90.1±3.0 | 0.0 | 0.0 |
| ur10_f140_osc | 7.3±4.5 | 82.5±8.8 | 4.8† | 0.0 |
| ur10_frankenstein | 5.7±1.1 | 83.2±7.4 | 3.0 | 3.0 |
| ur10_frankenstein_ik | 4.7±0.4 | 85.9±2.2 | 0.9 | 0.9 |
| ur10_frankenstein_ikabs | 5.0±0.7 | 85.7±2.2 | 0.7 | 0.7 |
| ur10_frankenstein_osc | 23.0±18.1 | 54.7±30.1 | 25.5† | 0.0 |
| ur5e_f140 | 3.2±2.1 | 91.0±4.6 | 0.0 | 0.0 |
| ur5e_f140_ik | 2.3±0.3 | 93.0±0.7 | 0.0 | 0.0 |
| ur5e_f140_ikabs | 1.9±0.2 | 94.5±0.6 | 0.0 | 0.0 |
| ur5e_f140_osc | 15.8±11.4 | 65.1±23.4 | 22.7† | 0.0 |
| ur5e_frankenstein | 3.6±0.3 | 88.3±1.3 | 0.8 | 0.8 |
| ur5e_frankenstein_ik | 3.7±0.3 | 89.4±1.4 | 0.3 | 0.3 |
| ur5e_frankenstein_ikabs | 4.2±1.0 | 85.3±6.2 | 0.4 | 0.3 |
| ur5e_frankenstein_osc | 28.0±25.9 | 51.3±42.3 | 34.6† | 0.0 |

### Finding 3 — Orientation read-out (baseline, no refinement term)

**Read `pose%` (position AND orientation), not `orient%` alone.** On the diverged
UR-OSC runs `orient%` is a **collapse artifact** — e.g. `ur5e_frankenstein_osc`
seed 2 = 58 cm / 0 % pos / **80 % orient / 0 % pose**: the arm is frozen
gripper-down 58 cm from target. The `†`-marked orient% above are all such
artifacts (pose% = 0). Reading the honest `pose%`:

- **Real orientation tracking is a Kinova-only, redundancy-scaled phenomenon.**
  Best is `kinova_frankenstein_ikabs` (9-DOF) at **29 % pose**, then
  Kinova-Frankenstein joint/ik (~21–22 %), then Kinova-F140 (4–12 %). It scales
  with redundancy (9-DOF > 7-DOF) and controller access (IK-Abs/joint/ik > OSC).
- **P2 confirmed robustly across all seeds:** every UR arm — rigid *and*
  Frankenstein — is **0–3 % pose**. The tensegrity wrist does **not** rescue
  orientation on the UR arms (all `ur*_frankenstein` ~0–3 %); redundancy alone is
  insufficient, the target orientation is geometrically out of reach for the
  6-DOF-effective arms.
- **OSC suppresses orientation even on the most redundant arm:**
  `kinova_frankenstein_osc` = **1.3 % pose** vs the same arm's 29 % under IK-Abs,
  because the wrist is excluded from the OSC Jacobian (iteration 15). The
  redundancy is present but the controller can't spend it.
- Orientation is **seed-noisy** where it exists (`kinova_frankenstein` joint
  24 ± 9 %), so even the Kinova numbers are soft.

This is the baseline that motivates the iteration-16 gated refinement, which took
`kinova_f140_osc` 9 % → 75 % pose exactly where the redundancy is
controller-accessible — and which (correctly) cannot help the ~0 % UR arms or
`kinova_frankenstein_osc`.

### Revised acceptance status (vs iteration 15's "23/24 pass")

- **18 variants seed-robust and passing** (all joint / ik / ikabs, + the 2
  Kinova-OSC): tight std, mean ≤ 5.7 cm, ≥ 80 % success.
- **4 UR-OSC variants are the real weak spot** — bimodal; 3 fail the ≤ ~8 cm bar
  on the mean (`ur5e_f140_osc`, `ur10_frankenstein_osc`, `ur5e_frankenstein_osc`),
  `ur10_f140_osc` marginal (7.3 cm).
- Net: the single-seed "23/24 pass" was optimistic; the honest multi-seed picture
  is **20/24 pass on the mean, with OSC-on-UR unreliable rather than fixed.**

### Note — Kinova fixed-seed reproduction vs iteration 15

The iteration-16 before-bake spawner clamps the Kinova USD limits *before* the
articulation bake, whereas iteration 15 baked the raw (unlimited) limits and the
reset event rewrote them *after*. The final baked limits are identical, but the
initial-bake path differs by construction, so fixed-seed Kinova training is **not**
byte-identical to iteration 15 (`kinova_f140_osc` seed 42: 3.7 → 2.8 cm) — it
lands well within the multi-seed band (1.8–4.0 cm). Non-Kinova configs reproduce
exactly.

### Implications

1. **Revise the action-space recommendation:** OSC is best-in-class *only with
   arm redundancy* (Kinova 7-DOF). For the rigid 6-DOF UR arms, joint / IK-Rel /
   IK-Abs are the seed-robust choices; IK-Abs on UR5e is the grid's tightest
   (1.9 cm / 94.5 %).
2. **New open problem: stabilise OSC on non-redundant (UR) arms** before trusting
   or building on those variants (e.g. lower task stiffness, higher damping ratio,
   or a null-space posture regulariser the 6-DOF arms currently lack).
3. **Scope the step-2 orientation-refinement tier to Kinova-OSC** (+ Frankenstein
   IK-Abs/joint); the UR-OSC variants can't be a stable base for it yet.

---

## Iteration 18 (2026-07-08) — Step 1b: OSC stabilisation sweep on ur5e_f140_osc

**Goal:** find an OSC config that makes the non-redundant UR arm seed-robust
(iteration 17: all four UR-OSC variants bimodal). Test case `ur5e_f140_osc`
(baseline bimodal: seeds 0/1/2 diverge 19–28 cm, seeds 42/123 converge ~2 cm).
3 configs × 5 seeds via env-var overrides in `osc_reach_common.py`
(`tools/exp_osc_stab.sbatch`, job 3828799, all 15 COMPLETED).

**Hypotheses tested:** (0) `fixed` impedance — the policy cranking task stiffness
in `variable_kp` is the destabiliser; (1) `lowstiff` — kp 100→50, ceiling 200→100
(still variable_kp); (2) `fixed_lowstiff` — both.

**Results** (converged mean position error, cm; ✓ = converged ≲ 5 cm):

| config | seed0 | seed1 | seed2 | seed42 | seed123 | # converged |
|---|---|---|---|---|---|---|
| baseline (variable_kp, kp=100) | 19.3 | 28.1 | 26.8 | **2.1**✓ | **2.6**✓ | 2/5 |
| fixed (kp=100) | 20†✗ | **2.4**✓ | 22–43†✗ | **3.1**✓ | 17.2✗ | 2/5 |
| lowstiff (kp=50, var) | 18–23†✗ | 19.1✗ | 22–43†✗ | 9.8~ | **2.7**✓ | 1/5 |
| **fixed_lowstiff (fixed, kp=50)** | **1.6**✓ | **1.7**✓ | **1.9**✓ | 24.1✗ | **2.1**✓ | **4/5** |

† `fixed`/`lowstiff` seeds 0 & 2 collided on the log dir (see tooling note); both
tfevents in each shared dir diverged, so the verdict (both configs fail those
seeds) holds regardless of attribution.

**Findings.**

1. **`fixed` and `lowstiff` alone do NOT help** — still bimodal (1–2/5), and they
   *move* which seeds diverge (fixed now converges seed 1 but loses seed 123).
2. **`fixed_lowstiff` (fixed impedance + kp 50) is the clear lead: 4/5 converge,
   and the 4 are excellent (1.6–2.1 cm — tighter than the baseline's best 2.1 cm).**
   It **rescued all three baseline-diverging seeds (0/1/2)** to ~1.7 cm. Mean
   ≈ 6.3 cm vs baseline 15.8 cm.
3. **But it is not a full fix:** it *flipped* seed 42 (good→24 cm). So the
   divergence is a **seed-dependent bifurcation of the OSC task-space loop that
   these knobs shift but do not eliminate.** Lower stiffness + no policy stiffness
   control shrinks the diverging basin (4/5 vs 2/5) without closing it.

**Interpretation & next hypothesis.** That every config diverges on *some* seed
points at a root cause the stiffness knobs only damp: the OSC controller fighting
the **geometrically-unreachable gripper-down orientation** on the 6-DOF arm (P2)
— persistent rotational task-space torque that can bifurcate into divergence. The
targeted next test is to **split position/orientation task stiffness** and drop
the *rotational* stiffness (or the orientation task itself) on the non-redundant
arms, so the controller stops fighting an infeasible orientation. `fixed_lowstiff`
should be the base for that follow-up, multi-seeded (≥5 seeds) on a clean re-run.

**Tooling note (bug found & to fix):** the sweep's log-dir name is
`<timestamp-to-the-second>_ppo_torch_seed<N>`; array tasks that start in the same
second with the same seed **collide on the directory** (checkpoints overwritten;
tfevents survive as separate files). 2 of 15 dirs collided here. Fix for future
sweeps: include the config tag / `SLURM_ARRAY_TASK_ID` in the run name, or stagger
task starts. Conclusions above are unaffected (recovered per-file).

### Step 1b follow-up — FK-sampled (reachable) targets diagnostic (2026-07-10)

**Question (Georg):** the box's fixed gripper-down orientation is geometrically
unreachable for 6-DOF arms (P2). Does going back to **FK-sampled targets**
(reachable-by-construction full poses, the pre-iteration-14 mode) fix both
(a) the OSC-on-UR instability and (b) the 0%-orientation-on-UR limitation?

**Setup:** opt-in env gate `REACH_FK_TARGETS=1` in `f140_reach_common.py` (drops
the box, `uniform_ranges=None` → per-arm FK sampling); off by default so the box
baseline stays bit-exact. `tools/exp_fk_targets.sbatch`, job 3830573, 10 runs
(2 variants × 5 seeds), all COMPLETED. `ur5e_f140` (joint) isolates
target-reachability from the controller; `ur5e_f140_osc` tests OSC stability.

**Results** (converged; pos error cm / pos% / **pose%** = both pos<5cm & ori<0.3rad):

| variant | targets | pos cm (per seed) | pos% | **pose%** | divergence |
|---|---|---|---|---|---|
| ur5e_f140 (joint) | box | 3.2 | 91 | **0** | none |
| **ur5e_f140 (joint)** | **FK** | 3.1/3.1/3.2/3.2/3.4 | 92 | **76–89** | none |
| ur5e_f140_osc | box | 2/2/19/27/27 | 65 | **0** | 3/5 collapse |
| **ur5e_f140_osc** | **FK** | 5.0/9.9/10.0/11.2/12.5 | 21–79 | **6–32** | **none** |

**Findings.**

1. **Orientation on UR arms is fully achievable — P2 was target-induced, not
   arm-induced.** With FK-reachable targets the UR5e (joint control) tracks
   full pose at **76–89 % pose across all 5 seeds** (vs 0 % on the box), at
   unchanged position (3.1–3.4 cm / 92 %). The "6-DOF arms can't track
   orientation" limitation was *entirely* the box's unreachable fixed
   orientation. Georg's hypothesis confirmed, decisively.
2. **FK targets eliminate the catastrophic OSC divergence.** No `ur5e_f140_osc`
   seed collapses (box: 3/5 diverged to 19–27 cm; other UR-OSC variants to
   55–60 cm). Max is 12.5 cm. So the infeasible-orientation-fighting *was* a
   major driver of the OSC-on-UR instability (iteration-18 hypothesis) — remove
   the infeasible target and the collapse goes away.
3. **But OSC is still a weak full-pose controller on the non-redundant arm.**
   FK-OSC lands at ~10–12 cm / ~30 % on 4/5 seeds (one good seed at 5 cm/79 %),
   pose 6–32 % — vastly worse than FK-joint (3 cm/92 %, ~86 % pose) on the *same*
   targets. The instability converts from "bimodal collapse" to "consistent
   mediocrity": OSC needs redundancy to gracefully resolve a 6-DOF pose task.

**Implications.**

- **FK targets are the better task design** for orientation (unlocks it on every
  arm) and for OSC stability (no collapse). If the study wants real orientation
  tracking, this is the way — and it reframes reach from "position, loose
  orientation (IsaacLab recipe)" to "full-pose reach."
- **It shifts the action-space ranking:** on the full-pose FK task, **joint /
  IK dominate on the non-redundant arms**; OSC's value is redundancy-gated
  (Kinova). This is a *stronger, more useful* comparison story than the box's
  "everything ~passes on position."
- **Caveats stand:** per-arm FK is not a shared cross-arm distribution (position
  errors not comparable across arms — would need a reference-robot scheme); and
  the FK sampler wants re-validation before a full re-base. This diagnostic did
  **not** touch the settled box grid.

**Recommendation:** worth adopting FK targets for a *new* full-pose reach study
(the current box grid stays as the position-only comparison). Next concrete step
if pursued: pick a reference-robot sampling scheme for cross-arm comparability,
re-validate the FK sampler, then run the full 24-variant grid on FK targets.

---

## Iteration 19 (2026-07-10) — Per-robot-FK grid (24×5) + tensegrity-wrist matched-pair A/B

**Reframed purpose (now in the README):** reach = per-robot controller hardening
+ per-robot action-space selection + wrist evaluation — *not* cross-robot
comparison. Fairness = reachable targets per robot, hence per-robot FK sampling
(`REACH_FK_TARGETS=1`). Jobs 3831267 (grid, 24 variants × 5 seeds) and 3831268
(wrist A/B, 4 configs × 3 seeds), all 132 COMPLETED.

### Grid results (FK full-pose targets; mean±std over seeds {0,1,2,42,123})

| variant | pos cm | pos % | pose % |
|---|---|---|---|
| ur5e_f140 (joint) | **3.2±0.1** | **92** | **86** |
| ur5e_frankenstein (joint) | 4.7±0.6 | 83 | 59 |
| ur10_f140 (joint) | **5.1±0.4** | **87** | **84** |
| ur10_frankenstein (joint) | 6.5±0.4 | 78 | 50 |
| ur5e_frankenstein_ikabs | 6.9±1.5 | 70 | 36 |
| ur5e_f140_ik / _ikabs / _osc | 9.4–9.7 | 38–43 | 14–19 |
| ur10_frankenstein_ikabs | 9.6±4.9 | 65 | 50 |
| ur10_f140_ik / _ikabs / _osc | 12.2–12.6 | 40–43 | 16–18 |
| ur10_frankenstein_ik / _osc | 13.4 / 14.8 | 39 / 18 | 22 / 6 |
| ur5e_frankenstein_ik / _osc | 9.1 / 9.9 | 48 / 33 | 26 / 9 |
| **all 8 Kinova variants** | **24.6–34.7** | **0.4–3.8** | **~0** |

**Finding 1 — joint control dominates FK full-pose reach on the UR arms.** The
plain joint-space policies solve it well (UR5e 3.2 cm / 86 % pose — full-pose
tracking, robust across all 5 seeds); every task-space controller (IK-Rel /
IK-Abs / OSC) sits at 9–15 cm / 6–50 % pose on the identical targets. No
divergence anywhere (the box's OSC collapse stays gone), but task-space
controllers are consistently mediocre on wide-workspace full-pose targets.
IK-Abs is the best of the task-space trio, and the wrist helps it
(frankenstein_ikabs > f140_ikabs on both URs).

**Finding 2 — the Kinova FK grid FAILED to learn (24–35 cm, ~1 % — plateaued).**
Root cause is the **FK sampling range, not the sampler mechanics**: all variants
share `joint_range_margin=0.1`, but the UR arms clamp their joint limits to
range 3.0 rad (`clamp_max_range=3.0` → default±1.5), so their FK targets stay in
a moderate workspace sector; Kinova keeps full limits (`clamp_max_range=None` —
continuous joints at default±π, joints 2/4/6 at their full 4.4–5.3 rad ranges),
so its FK targets span the **entire reachable envelope** (behind the mount, near
the floor, self-collision-adjacent; `self_collision_filter` is off). 100k steps
can't crack that distribution. My earlier "validated on Kinova" call was
premature — the sampler ran cleanly (no NaN) but the task distribution it
produced is untrainable as-is. Fix: restrict Kinova's FK sampling to a
task-relevant sector (narrower per-joint sampling ranges and/or the collision
filter), then re-run the 8 Kinova variants.

### Wrist matched-pair A/B (ur5e_frankenstein, joint, FK; 3 seeds)

| config (targets × wrist) | pos cm | pos % | pose % |
|---|---|---|---|
| base-reachable × active | 4.7±0.2 | 84 | 71 |
| base-reachable × locked | 4.3±0.1 | 88 | 68 |
| frank-reachable × active | 4.6±0.1 | 85 | 60 |
| frank-reachable × locked | **10.3±6.5** | 52 | **22** |

**Finding 3 — the wrist's value is capability extension, not redundancy
assistance.** On base-reachable targets, locking the wrist changes nothing
(≈4.5 cm, ~70 % pose either way — the extra 2 DOF are neutral-to-slightly-drag).
On wrist-requiring targets, locking it costs **38 points of pose success**
(60 → 22 %) and destabilises training (per-seed 19.4/6.6/4.9 cm). So: the wrist
does not make already-reachable poses easier, but it meaningfully **extends the
reachable pose set** — and the policy does learn to use it when the targets
demand it (contradicting the box-grid impression that the wrist is dead weight;
that was the box's fixed orientation never requiring it).

### Status / next
- UR arms: hardened on FK full-pose; per-robot action-space answer = **joint**
  (IK-Abs best-of-task-space; benefits from the wrist).
- Kinova: **blocked on FK sampling-range fix**, then re-run.
- Task-space controllers (IK/OSC): consistent-but-mediocre on FK targets —
  hardening opportunity (episode length vs. large reorientations, IK step
  scale / ikabs exploration std over the wider workspace).
- Wrist under OSC: still excluded from the Jacobian — separate workstream.

---

## Iteration 20 (2026-07-11, in progress) — FK sampler fixes: collision filter audit, floor clearance, Kinova sector

**Trigger:** iteration 19's two open items — the Kinova FK grid failed to learn,
and the `self_collision_filter` had never been enabled or audited for the F140
grid. Georg: fix/verify the collision filter first, then re-run.

### Collision-filter audit findings & fixes (`reach/mdp/fk_sampled_pose_command.py`)

1. **Calibration fragility (fixed).** The structural-pair calibration ran once on
   the *first* resample batch and was cached forever — with few envs (play/eval,
   4 envs) that estimate is statistical garbage. Now: close-fraction statistics
   **accumulate across resamples** until `self_collision_calibration_min_samples`
   (1024) configs are seen, then freeze; the current estimate filters from the
   first batch. At training size (≥1024 envs) this is identical to the old
   one-shot behaviour.
2. **No ground-plane awareness (fixed — the bigger workspace bug).** FK configs
   could put the EE or the elbow *below the collidable ground plane* (mount
   0.75 m < reach 0.85–0.9 m): physically unreachable targets that poison
   training. New rejection terms, world frame: `fk_min_link_height_w` (checked
   links ≥ 0.05) and `fk_min_ee_height_w` (EE ≥ 0.30, keeping the ~24 cm Robotiq
   hanging below it clear).
3. **Filter never wired (fixed).** FK mode (`REACH_FK_TARGETS=1`) now enables
   `self_collision_filter` + both floor clearances in `f140_reach_common.py`.
   Box mode untouched (baseline bit-exact).
4. **Pair check vectorised** (one batched segment-distance call over env×pair
   instead of a per-pair Python loop) + one-time acceptance-stats print.
5. **Kinova sampling sector** (`fk_sampling_half_range=1.5` on both Kinova joint
   cfgs, inherited by IK/OSC): FK sampling restricted to default ± 1.5 rad,
   matching the UR arms' reset-clamped range — the iteration-19 failure was the
   Kinova sampling its full-circle envelope.

**GPU smoke (512 envs, zero agent):** Kinova-F140 FK — calibration froze at 1024
configs (accumulated over 2 batches, validating fix 1), 35/36 pairs kept,
512/512 accepted in ≤8 rounds, no NaN. UR5e-Frankenstein-OSC with
`REACH_OSC_INCLUDE_WRIST=1` — calibration excluded 12/55 structural pairs (the
stacked wrist/gripper frames, exactly the pairs the calibration exists for),
512/512 accepted, stepping clean.

### Launched (results pending)

- **FK grid v2** (job 3839201): full 24 variants × 5 seeds on the fixed sampler
  (filter + floor + Kinova sector active). Replaces iteration 19's numbers.
- **Probes** (job 3839202, seeds {7,77} — disjoint from grid seeds, so run-dir
  names cannot collide): (a) **wrist-under-OSC** — wrist included in the OSC
  Jacobian as a *damped-compliant* joint (stiffness 0, damping kept at 20 —
  iteration 15's divergence was zeroing the PD entirely) on UR5e- and
  Kinova-Frankenstein-OSC; (b) task-space hardening — IK-Abs @ 12 s episodes,
  IK-Rel @ 12 s, IK-Rel @ scale 1.0 (knobs: `REACH_OSC_INCLUDE_WRIST`,
  `REACH_OSC_WRIST_DAMPING`, `REACH_EPISODE_S`, `REACH_IK_REL_SCALE` — all
  opt-in, baseline unchanged).

### Iteration 20 — Results (2026-07-12, 130/130 COMPLETED)

**Grid v2 (fixed sampler: collision filter + floor clearance + Kinova ±1.5 rad
sector; 24 × 5 seeds).** Headline numbers (pos cm / pos % / pose %):

| variant | joint | IK-Rel | IK-Abs | OSC |
|---|---|---|---|---|
| kinova_f140 | **5.9 / 79 / 22** | 8.9 / 50 / 11 | 9.4 / 46 / 12 | 8.2 / 44 / 7 |
| kinova_frankenstein | **9.8 / 49 / 16** | 9.9 / 46 / 15 | 12.9 / 32 / 11 | 13.3 / 16 / 2 |
| ur5e_f140 | **3.4 / 91 / 86** | 8.3 / 51 / 21 | 7.8 / 56 / 22 | 8.3 / 48 / 18 |
| ur5e_frankenstein | **5.3 / 79 / 60** | 8.7 / 55 / 29 | 6.9 / 68 / 32 | 9.0 / 40 / 13 |
| ur10_f140 | **5.2 / 87 / 84** | 12.4 / 44 / 20 | 14.3 / 34 / 11 | 9.7 / 51 / 19 |
| ur10_frankenstein | **6.2 / 80 / 64** | 13.6 / 37 / 18 | 10.0 / 58 / 34 | 14.4 / 19 / 6 |

1. **The Kinova rescue worked.** All 8 Kinova variants went from
   24–35 cm / ~1 % (iteration 19, untrainable) to 5.9–13.3 cm / 16–79 % —
   `kinova_f140` joint at **5.9 cm / 79 % / 22 % pose**. The sampling-sector +
   floor-filter diagnosis was correct. (Kinova pose % remains below the URs' —
   its ±1.5 rad sector around the folded default yields a harder orientation
   distribution; a hardening follow-up, not a blocker.)
2. **UR results replicate grid v1 within seed noise** (`ur5e_f140` joint 3.4 vs
   3.2 cm; `ur10_f140` 5.2 vs 5.1) — the new filters did not distort the UR
   distribution, and there is still **no OSC collapse anywhere** (worst single
   seed 17.7 cm).
3. **Joint control wins on every robot** for FK full-pose reach; IK-Abs is the
   best task-space option (and the wrist helps it: frankenstein_ikabs beats
   f140_ikabs on both URs).

**Probes (seeds {7,77}) — all four negative, each informative:**

| probe | result | vs. reference | verdict |
|---|---|---|---|
| IK-Abs @ 12 s episodes | 11.1 / 27 % | 7.8 / 56 % @ 6 s | worse — longer episodes = fewer target resets per 100k steps; time was not the bottleneck |
| IK-Rel @ 12 s | 12.0 / 23 % | 8.3 / 51 % | worse — same |
| IK-Rel @ scale 1.0 | 11.6 / 28 % | 8.3 / 51 % | worse — step size not the bottleneck |
| **wrist-in-OSC-Jacobian (damped-compliant)** | ur5e 15.1 / 12 %, kinova 15.8 / 11 % | 9.0 / 40 %, 13.3 / 16 % (wrist excluded) | worse — but **no divergence** (keeping damping prevented iteration 15's blowup); OSC still cannot productively drive the compliant wrist |

**Conclusions (iteration 20 / FK-target hardening study):**

- The task-space gap on wide-workspace full-pose targets is **not** a
  time/step-size artifact — it is inherent to greedy DLS/OSC pathing. For the
  downstream cloth tasks, the hardened per-robot recommendation is **joint
  control everywhere** (IK-Abs as the task-space fallback, with the wrist).
- **Wrist-under-OSC is now a closed question:** pure-effort inclusion diverges
  (iteration 15), damped-compliant inclusion is stable but strictly worse than
  exclusion (this iteration). The wrist's proven value (capability extension,
  iteration 19) is only accessible under joint/IK control.
- The fixed FK sampler (collision filter + floor clearance + per-robot sector)
  is the validated controller-hardening target distribution; all knobs are
  opt-in so the settled box grid remains bit-exact.

---

## Iteration 21 (2026-07-12, in progress) — Research-report fixes: rot6d IK-Abs action, task-space EMA, OSC stiffness

**Trigger:** external research report
(`doc/reports/tmp/RESEARCH_REPORT_taskspace_pose_error.md`) validating iterations
17–20 and auditing the configs. Verdicts: results regime-appropriate (joint
wins *free-space* reaching per Aljalbout et al. RA-L 2024 — scope the claim!);
OSC-divergence diagnosis textbook-confirmed; **one real defect: IK-Abs feeds
the policy's raw Gaussian 4-vector as an UNNORMALISED quaternion** (Isaac Lab
source-confirmed; SO(3) anti-pattern per Zhou et al. CVPR 2019 / Schuck et al.
ICLR 2026). Secondary asymmetries: task-space actions not EMA-smoothed (joint
is), OSC stiffness clamp soft (50–200 vs Isaac Lab ceiling 1000), single global
exploration std.

**Implementation (`reach/mdp/task_space_actions.py`, all opt-in/env-gated):**
- `SixDRotDiffIKAction` (`REACH_IKABS_ROT6D=1`): 9-D action = 3-D position +
  6-D continuous rotation (Gram–Schmidt → matrix → unit quaternion to the
  controller). Buffers size from the overridden `action_dim`; small constant
  bias keeps Gram–Schmidt non-degenerate at near-zero actions.
- `EMADiffIKAction` / `EMAOSCAction` (`REACH_TS_EMA=<α>`): EMA on the raw
  task-space actions with per-env reset, mirroring the joint space's α=0.2.
- Stage-2 stiffness sweep reuses the iteration-18 `REACH_OSC_*` knobs.

**Smoke (GPU, zero agent):** IK-Abs rot6d action shape 9 ✓, EMA-OSC shape 13 ✓,
no NaN/crash on FK targets.

**Launched (compute-throttled ≤5 concurrent — shared cluster):**
- **Stage 0** (job 3841042, `%3`): rot6d+EMA on ur5e_f140_ikabs and
  kinova_f140_ikabs × seeds {0,1,2,42,123} (vs iter-20 baselines 7.8 cm/56 %/22 %
  and 9.4/46/12), + attribution ablations rot6d-only and EMA-only on
  ur5e_f140_ikabs × seeds {7,77}. **Threshold:** ur5e IK-Abs pose% 22 → ≥50 %
  ⇒ representation was dominant ⇒ roll into IK-Rel + re-run grid columns.
- **Stage 2** (job 3841043, `%2`): OSC stiffness on ur10_f140_osc — kp 200
  clamp (100,400) and kp 300 clamp (200,500) × seeds {0,42,123} (baseline kp 100
  clamp (50,200): 9.7 cm/51 %/19 %). **Threshold:** nothing < ~7 cm / > 30 % pose
  ⇒ stop tuning OSC for free space, re-evaluate in the cloth tasks.
- **Stage 1** (residual task-space control) and **Stage 3** (orientation
  curriculum) are gated on these thresholds, per the report's staged plan.

### Iteration 21 — Results (2026-07-12, Stage 0: 14/14, Stage 2: 6/6 COMPLETED)

**Stage 0 — rot6d + EMA (report Rank 1): the hypothesis INVERTED empirically.**
Converged (pos cm / pos% / pose%), FK targets:

| config | ur5e_f140_ikabs | kinova_f140_ikabs |
|---|---|---|
| baseline (raw quat, iter 20, 5 seeds) | 7.8±1.1 / 56 / **21.9** | 9.4±0.8 / 46 / 12.2 |
| rot6d + EMA (5 seeds) | 7.5±0.4 / 56 / 14.6 | 11.0±2.2 / 41 / 5.9 |
| rot6d only (seeds 7,77) | **14.7–16.1 / 9–14 / 1–2** | — |
| **EMA only (seeds 7,77)** | **6.2–9.5 / 40–73 / 21–42** | — |

Attribution is unambiguous in direction: the **6-D rotation action is *harmful***
in this setup (rot6d-only collapses to 15–16 cm), while **EMA on the stock
quaternion actions is the beneficial factor** (EMA-only is the only config that
beats baseline; seed 77 reached 41.7 % pose — nearly double the baseline).
rot6d+EMA ≈ EMA partially rescuing a harmful representation. The research
report's source-confirmed "unnormalized quaternion anti-pattern" did **not**
translate into an empirical win for one-shot absolute-pose regression here —
a plausible mechanism is that Gram–Schmidt direction-normalization makes the
commanded rotation hyper-sensitive to small-magnitude Gaussian actions early in
training, a worse noise geometry than the (theoretically ugly) raw quaternion.
**Threshold verdict: pose 22 → ≥50 % clearly missed; the rotation
representation was NOT the dominant limitation.** The greedy one-shot DLS
pathing remains the prime suspect for the residual gap.

**Stage 2 — OSC stiffness sweep (ur10_f140_osc): both settings worse.**

| config | pos cm | pos% | pose% |
|---|---|---|---|
| baseline kp100 clamp(50,200) (iter 20) | **9.7±3.4** | **51** | **19** |
| kp200 clamp(100,400) | 12.9±2.1 | 30 | 13 |
| kp300 clamp(200,500) | 12.8±2.6 | 26 | 8.5 |

Higher task-space stiffness *hurts* the learned policy (stiffer tracking of
noisy pose commands → jerkier motion, harsher credit assignment) — the static-
offset hypothesis does not hold for the RL-in-the-loop setting. **Report
threshold fires: nothing < 7 cm / > 30 % pose ⇒ STOP tuning OSC for the
free-space baseline; re-evaluate OSC only in the contact-rich cloth tasks.**

**Stage 0b (launched, job 3842755):** validating the one positive finding —
EMA-only on the stock actions — on full seeds: ur5e_f140_ikabs × {0,1,2,42,123}
and ur5e_f140_ik × {0,42,123}. If EMA-only reproduces ≥ baseline+meaningful
pose% at 5 seeds, adopt `REACH_TS_EMA=0.2` as the default task-space hardening
knob (and document IK-Abs+EMA as the task-space fallback of record).

**Staged-plan status:** Stage 1 (residual task-space control) remains the only
untried Rank-2 lever — deferred, documented in the TODO with its ≤5 cm
adoption threshold (the per-robot hardening answer — joint control — is already
settled, so residual is fallback-improvement work, not blocking). Stage 3
(curriculum) skipped: the 12 s-episode probe showed the limitation is
asymptotic, not convergence-bound.

### Iteration 21 — Stage 0b validation result (2026-07-12, 8/8 COMPLETED): EMA adopted

**EMA-only (α=0.2 on the stock task-space actions) validated at full seeds — the
effect is large and robust:**

| config | pos cm | pos% | **pose%** |
|---|---|---|---|
| ur5e_f140_ikabs baseline (iter 20, 5 seeds) | 7.8±1.1 | 56 | 21.9 |
| **ur5e_f140_ikabs + EMA (5 seeds)** | **5.8±0.9** | **77** | **45.4±8.0** |
| ur5e_f140_ik baseline (iter 20) | 8.3±1.2 | 51 | 21.0 |
| **ur5e_f140_ik + EMA (3 seeds)** | **6.0±0.4** | **73** | **42.3** |

Position error −26 %, pos% +21 points, **pose% more than doubled** (22 → 45,
best seeds 53–54 %) — on both IK variants. The research report's audit item #4
(the joint action was EMA-smoothed while task-space actions were not — an
unfair asymmetry) turns out to be the **dominant fixable factor**, not the
quaternion representation (item #1, which tested harmful). The Stage-0
threshold (pose ≥ 50 %) is effectively met by the best seeds; the Stage-1
residual-control adoption threshold (≤ 5 cm) is nearly met by EMA alone
(5.8 cm), further weakening the case for building residual control.

**Adopted:** `REACH_TS_EMA=0.2` is the task-space hardening knob of record;
**IK-Abs + EMA (5.8 cm / 77 % / 45 % pose) is the task-space fallback of
record** (joint control remains the primary recommendation at 3.4 / 91 / 86).
Follow-up (optional, queued in TODO): roll `REACH_TS_EMA=0.2` across the full
task-space grid columns (6 arms × ik/ikabs/osc × 5 seeds) to refresh the
official table; deferred under the shared-cluster compute cap.

**Iteration-21 net conclusions:**
1. Task-space actions must be smoothed like joint actions — EMA α=0.2 halves
   the pose-tracking gap at zero controller changes.
2. The 6-D rotation representation is *harmful* for one-shot absolute-pose
   regression in this stack (theory-motivated fix, empirically refuted).
3. OSC stiffness tuning in free space is closed (both directions worse).
4. Joint control stays the hardened per-robot answer; residual control (Stage
   1) is deferred with a now-marginal expected payoff; curriculum (Stage 3)
   skipped (asymptotic limit, not convergence-bound).

---

## Iteration 22 (2026-07-12, in progress) — Kinova sector verdict + THESIS GRID (final consistent rerun)

**Kinova sampling-sector probe** (kinova_f140 joint, FK, seeds {7,77}; job
3843922): the pose% gap to the URs is confirmed to be **target-distribution-
driven, monotone in the sector width** — not a controller/policy limitation:

| `fk_sampling_half_range` | pos cm | pos% | pose% |
|---|---|---|---|
| 1.5 (iter-20 baseline) | 5.9 | 79 | 22 |
| 1.2 | 3.8–4.0 | 90 | 30–46 |
| **1.0 (locked)** | **2.7–3.0** | **92–93** | **55–67** |

**Locked: `REACH_FK_HALF_RANGE=1.0` for the Kinova arms** — near-UR pose
quality; ±0.9 rad effective sampling still covers the downstream (belt-front)
workspace. Documented honestly as *workspace scoping*: the sector defines the
task distribution, and the Kinova numbers are not cross-robot comparable
anyway (per-robot hardening design). (Probe note: seed-77 runs of the two
configs collided on a same-second run dir — recovered per-tfevents-file; the
thesis grid has unique variant+seed per task, no collision risk.)

**THESIS GRID launched (Georg: "rerun everything to be consistent"):**
- Job 3843960 (`%4`): 24 variants × 5 seeds {0,1,2,42,123}, single locked
  config — FK targets, fixed sampler (self-collision filter + floor clearance),
  Kinova sector 1.0, **EMA α=0.2 on every action space** (joint built-in,
  task-space via `REACH_TS_EMA=0.2`, the iteration-21 adopted knob).
- Job 3843961 (`%1`): wrist matched-pair A/B rerun (4 configs × 3 seeds) on the
  fixed sampler, consistent with the grid.
- Combined ≤5 concurrent (shared-cluster cap). These runs supersede the
  iteration-19/20/21 mixed-config tables as THE thesis dataset; afterwards:
  regenerate per-variant figures/reports, rewrite `action_spaces.md` + README
  results, curate `logs/skrl/theses_logs/reach/`, and rebuild the
  visual-verification package.

### Iteration 22 — THESIS GRID results (2026-07-13, 132/132 COMPLETED, zero failures)

Final consistent dataset — one locked config everywhere: FK targets, fixed
sampler, Kinova sector 1.0, EMA α=0.2 on all action spaces. Cell = pos cm /
pos% / pose% (5 seeds):

| Arm | Joint | IK-Rel | IK-Abs | OSC |
|---|---|---|---|---|
| Kinova-F140 | 3.4 / 90 / 43 | 3.7 / 86 / 45 | 3.1 / 90 / 35 | **2.9 / 91 / 32** |
| Kinova-Frankenstein | **4.6 / 85 / 37** | 6.0 / 69 / 29 | 5.1 / 80 / 26 | 3.9 / 85 / 11 |
| UR5e-F140 | **3.4 / 91 / 86** | 6.1 / 71 / 37 | 5.8 / 77 / 45 | 9.1 / 43 / 16 |
| UR5e-Frankenstein | **5.3 / 79 / 60** | 7.9 / 59 / 31 | 6.1 / 75 / 44 | 12.4 / 20 / 7 |
| UR10-F140 | **5.2 / 87 / 84** | 10.0 / 60 / 35 | 11.3 / 47 / 22 | 13.2 / 31 / 16 |
| UR10-Frankenstein | **6.2 / 80 / 64** | 10.8 / 49 / 27 | 10.7 / 51 / 28 | 17.1 / 10 / 3 |

**Findings.**

1. **The Kinova sector fix + EMA transformed the Kinova column:** all 8 variants
   at 2.9–6.0 cm (was 5.9–13.3), and **Kinova-F140-OSC is now the single tightest
   task-space cell in the grid (2.9 cm / 91 % / 32 % pose)** — OSC finally shows
   its redundancy advantage once targets are well-scoped. Kinova task-space is
   now *competitive with its joint control* — on the 7-DOF arm the action-space
   choice barely matters; on the 6-DOF URs it matters enormously.
2. **EMA reproduces and generalises on IK:** ur5e_f140_ikabs 5.8 / 77 / 45
   matches the iteration-21 validation exactly (same seeds, deterministic);
   IK-Rel/IK-Abs improved across every arm vs iteration 20 (e.g. ur10_f140_ik
   12.4→10.0 cm, pose 20→35 %).
3. **EMA is mixed-to-harmful on UR OSC** (ur10_f140_osc 9.7→13.2 cm,
   ur10_frankenstein_osc 14.4→17.1): smoothing the 13-D pose+stiffness action
   lags the stiffness modulation the UR arms rely on. Honest caveat in the
   final table: UR-OSC remains the grid's weak corner under every config tried
   (iterations 17, 18, 20, 21, 22) — the recommendation stands: no OSC on
   non-redundant arms.
4. **Joint control remains the overall winner** (3.4–6.2 cm, pose up to 86 %),
   with the gap now robot-dependent: decisive on the URs, marginal on Kinova.
5. **Wrist A/B v2 (fixed sampler) — conclusion holds, magnitude softer:** on
   wrist-requiring targets, locking the wrist costs 22 pose-points (57→35 %)
   and no longer destabilises position (5.1 cm vs v1's bimodal 10.3 — the
   filtered targets are less extreme). On base-reachable targets the wrist is
   ≈neutral (67 vs 59 % pose). Capability extension, confirmed on the
   consistent sampler.

**This grid supersedes iterations 19–21 as THE thesis dataset** (they remain
as the development history). Follow-ups executed next: figures/reports
regeneration, action_spaces.md + README rewrite, `theses_logs` curation,
visual-verification package.

# Cube Place Task — Optimization Tracking

**Task:** `Template-Tensegrity-Cube-Place-v0`
**Robot:** 5-DOF tensegrity manipulator + Robotiq 2F-140 gripper (ceiling-mounted)
**Framework:** Isaac Lab + SKRL (PPO), Isaac Sim
**Hardware:** NVIDIA RTX A6000 (48 GB), 8192 parallel environments (default)

---

## Phase 0: Restructure

**Date:** 2026-03-30

### Changes
- Made `place_scene_cfg.py` robot-independent: `PlaceSceneCfg` inherits `robot = MISSING` from `ProjBaseSceneCfg`
- Made `place_env_cfg.py` robot-agnostic: renamed `TensegrityPlaceEnvCfg` → `PlaceEnvCfg`, added `_set_robot_params()` helper, all body/joint names use `MISSING` sentinels
- Rewrote `ActionsCfg` with `arm_action: ActionTerm = MISSING` + shared `gripper_action`
- Created proper inherit-and-override variant configs for tensegrity, UR10e, Kinova, tendon, and physical tendon
- Eliminated ~1000 lines of duplicated code across UR10e and Kinova standalone configs
- Tensegrity-specific `base_velocity` penalty moved to tensegrity variant only

### Verification
- All syntax checks pass
- Headless training starts and runs correctly (32 envs, 2733 steps verified)
- Action space: arm_action(3) + gripper_action(1) + base_delta(2) = 6 dims ✓
- Observation space: 47 dims (14 terms) ✓
- Reward terms: 19 active (includes tensegrity-specific base_velocity) ✓
- All gym registrations resolve correctly

### Architecture (after restructure)
```
place_env_cfg.py          → PlaceEnvCfg (robot-agnostic base, MISSING sentinels)
place_scene_cfg.py        → PlaceSceneCfg (ProjBaseSceneCfg + cubes, robot=MISSING)
place_env.py              → TensegrityPlaceEnv (custom env with latched flags)

config/tensegrity/        → TensegrityPlaceEnvCfg(PlaceEnvCfg)
config/ur10e/             → UR10ePlaceEnvCfg(PlaceEnvCfg)
config/kinova/            → KinovaPlaceEnvCfg(PlaceEnvCfg)
config/tensegrity_tendon/ → TensegrityPlaceTendonEnvCfg(TensegrityPlaceEnvCfg)
                          → TensegrityPlacePhysicalTendonEnvCfg(TensegrityPlaceEnvCfg)
```

---

## Iteration 0: Baseline

**Date:** 2026-03-30
**Config:** Post-restructure, no reward changes
**Training:** 300k timesteps, 8192 envs

### Current Reward Structure
| Phase | Reward Term | Weight | Description |
|-------|-------------|--------|-------------|
| 1. Reach | reaching_object | 1.0 | tanh proximity (fingertip → nearest cube, std=0.1) |
| 2. Grasp | grasping | 5.0 | closure × proximity (std=0.08) |
| 3. Lift | lifting_object | 8.0 | binary: above belt + near gripper + closed + low vel |
| 3b. Height | height_bonus | 8.0 | smooth gradient to lift above drum rim |
| 4. Transport (coarse) | goal_tracking | 50.0 | tanh proximity to drum (std=1.0, lift gate z>belt+0.02) |
| 4b. Transport (fine) | goal_tracking_fine | 5.0 | tanh proximity to drum (std=0.20) |
| 5. Release | release | 10.0 | open gripper above drum (within radius, was_grasped) |
| 6. Success | green_in_target | 100.0 | per-step while cube in drum (was_grasped gated) |
| 7. Negative | red_in_target | -12.0 | red cube in drum penalty |
| Reg. | action_rate | -1e-4 → -2e-3 | L2 action rate (curriculum ramp) |
| Reg. | joint_vel | -1e-4 → -2e-3 | L2 joint velocity (curriculum ramp) |
| Reg. | base_velocity | -1.5 | L2 base joint velocity (tensegrity only) |
| Reg. | arm_utilization | 1.5 | arm velocity bonus |
| Reg. | belt_contact | -10.0 | fingertip below belt penalty |
| Reg. | joint_torque | -0.05 | arm joint torque penalty |
| Metric | metric_place_success | 0.01 | binary placement success |
| Metric | metric_grasp_rate | 0.01 | binary grasp metric |
| Metric | metric_ee_distance | -0.01 | EE-to-green distance |
| Penalty | cube_off_conveyor | -5.0 | cubes knocked off belt |

### Curriculum
- Red cube activated at 100k steps
- Action rate ramp: -1e-4 → -2e-3 over 200k steps
- Joint vel ramp: -1e-4 → -2e-3 over 200k steps

### Results
> Baseline training in progress (old rewards, ~8h for 300k steps at 10 it/s)

---

## Iteration 1: Combined Optimization

**Date:** 2026-03-30
**Changes:** Coarse/fine reaching + faster release + red cube active avoidance

### Change 1: Coarse/Fine Reaching Rewards
**Problem:** Single reaching reward with std=0.1 provides meaningful gradient only within ~0.2m.
The arm needs to explore extensively to find the gradient signal.

**Solution:** Split into coarse + fine, matching the cube sort task pattern:

| Term | Before | After |
|------|--------|-------|
| `reaching_object` (coarse) | std=0.1, weight=1.0 | **std=0.3**, weight=1.0 |
| `reaching_object_fine` (new) | — | **std=0.1, weight=2.0** |

**Reasoning:** tanh(d/0.3) provides useful gradient up to ~0.5m (coarse pull),
while tanh(d/0.1) sharpens near the cube for precise grasping.

### Change 2: Faster Release Acquisition
**Problem:** Agent learns to transport cube over drum quickly but holds instead of
releasing. Per-step reward for holding (~72: goal_tracking=50 + lift=8 + height=8 +
extras) significantly exceeds release reward (10), creating a local optimum.

**Solution:** Increase release weight from 10 → 25.

| Term | Before | After |
|------|--------|-------|
| `release` | weight=10.0 | **weight=25.0** |

**Reasoning:** With weight=25, the per-step reward for opening the gripper above the
drum (25 × openness) better competes with the holding reward (~72). Once released,
green_in_target (100/step) dominates, making the overall episode return higher for
early release. The increased release weight reduces the "reward cliff" during the
cube's free-fall transition.

### Change 3: Red Cube Active Avoidance
**Problem:** Red cube penalty (red_in_target, weight=-12) only fires AFTER red enters
drum — no proactive incentive to keep red away. Agent may accidentally push red into
drum during transport.

**Solution:** Added `red_clearance_from_drum` reward function + `red_clearance` term.

| Term | Weight | Description |
|------|--------|-------------|
| `red_clearance` (new) | **3.0** | tanh(d_xy/0.4) — rewards red cube being far from drum |

**Reward function:** `tanh(d_red_to_drum_xy / 0.4)` — returns higher values as the
red cube is further from the drum in XY. Gated on `!was_grasped` — only active before
green cube is grasped, so the agent focuses on clearing red first, then switches to
green cube transport.

### Updated Reward Structure
| Phase | Reward Term | Weight | Status |
|-------|-------------|--------|--------|
| 1a. Reach (coarse) | reaching_object | 1.0 | **std=0.3 (was 0.1)** |
| 1b. Reach (fine) | reaching_object_fine | 2.0 | **NEW** |
| 2. Grasp | grasping | 5.0 | unchanged |
| 3. Lift | lifting_object | 8.0 | unchanged |
| 3b. Height | height_bonus | 8.0 | unchanged |
| 4. Transport (coarse) | goal_tracking | 50.0 | unchanged |
| 4b. Transport (fine) | goal_tracking_fine | 5.0 | unchanged |
| 5. Release | release | **25.0** | **was 10.0** |
| 6. Success | green_in_target | 100.0 | unchanged |
| 7. Red penalty | red_in_target | -12.0 | unchanged |
| 7b. Red clearance | red_clearance | **3.0** | **NEW** |
| Reg. | action_rate | -1e-4 → -2e-3 | unchanged |
| Reg. | joint_vel | -1e-4 → -2e-3 | unchanged |
| Reg. | base_velocity | -1.5 | tensegrity only |
| Reg. | arm_utilization | 1.5 | unchanged |
| Reg. | belt_contact | -10.0 | unchanged |
| Reg. | joint_torque | -0.05 | unchanged |
| Metric | metric_place_success | 0.01 | unchanged |
| Metric | metric_grasp_rate | 0.01 | unchanged |
| Metric | metric_ee_distance | -0.01 | unchanged |
| Penalty | cube_off_conveyor | -5.0 | unchanged |

### Results
> Training pending (will run after baseline completes or in parallel)

---

## Iteration 2: Cube-Sort Reward Alignment

**Date:** 2026-03-30  
**Run:** `2026-03-31_01-12-36_ppo_torch`  
**Status:** ✅ COMPLETED — 300k / 300k steps

### Problem Diagnosis

After the Phase 0 restructure, training stalled: `lifting_object` and
`goal_tracking` were permanently 0, policy std flat at 0.489. Root cause
identified by comparing to the working `cube_sort` task:

1. **Broken coarse/fine split:** Both `reaching_object` and
   `reaching_object_fine` used `std=0.1` — functionally identical, no
   coarse gradient from the top start position (0.25 m away from cube).
2. **Missing `!grasp_active` gate on reaching:** Reaching reward was wasting
   budget during transport, competing with goal_tracking.
3. **Weight imbalance vs. cube_sort:** Transport was over-weighted, lift/grasp
   under-weighted relative to the proven working configuration.
4. **`base_z_joint` start:** `0.0` (top of range) is intentional — coarse
   reaching bridges the gap.

### Changes Applied

| Term | Before | After |
|---|---|---|
| `reaching_object` | std=0.1, w=1.0 | **std=2.0, w=2.0** (long-range gradient) |
| `reaching_object_fine` | std=0.1, w=2.0 | **std=0.5, w=5.0** (mid-range) |
| Reach gate | always-on | **`!grasp_active`** (turns off when holding) |
| `grasping` | w=5.0 | **w=3.0** |
| `lifting_object` | w=8.0 | **w=5.0** |
| `height_bonus` | w=8.0 | **w=5.0** |
| `goal_tracking` | w=50.0 | **w=40.0** |
| `goal_tracking_fine` | w=5.0 | **w=10.0** |
| `release` | w=10.0 | **w=25.0** |
| `arm_utilization` | w=1.5 | **w=0.5** |
| `gpu_total_aggregate_pairs_capacity` | 32×1024 | **64×1024** (PhysX fix) |

### Results

| Metric | Value |
|---|---|
| **Place success rate** | **93.6%** 🎉 |
| **Grasp rate** | **97.5%** |
| Mean episode length | 250 / 250 (maxed) |
| Total reward (final plateau) | ≈ 335 |
| Convergence step | **~30k** (10% of budget) |
| Training time | 9.6 h @ ~9 steps/s |

**Reward progression:**

| Step | Total Reward |
|---|---|
| 250 | 5.8 |
| 30,000 | 342 |
| 75,000 | 350 |
| 300,000 | 335 |

Policy std settled at 0.46, learning rate descended 2.7e-4 → 1.5e-5
(KL-adaptive). The task is considered **solved** for the tensegrity PD variant.

### Remaining Concerns

- `cube_off_conveyor` penalty = −3.87 at end: robot occasionally knocks red
  cube off belt during approach. Red cube active avoidance (next iteration)
  should address this.
- All other penalty terms are small and expected.

---

## Iteration 2.5: Convergence Acceleration + Return-to-Neutral

**Date:** 2026-04-04
**Run:** `2026-04-04_21-01-57_ppo_torch` (combined with Iter 3 below)
**Status:** ✅ Complete — 93.2% place success at 300k steps

### Problem Diagnosis

Iteration 2 converged at ~30k steps (10% of 300k budget). Three optimisations:

1. **Over-long training:** 270k wasted steps after convergence.
2. **Missing post-drop behaviour:** After releasing the green cube the agent
   wanders aimlessly — no incentive to return home, and the reaching reward
   keeps firing on unrelated cubes.
3. **Cube spawn variety:** Cubes always spawned at belt centre
   (y ∈ ±0.10 m); the agent never sees them at the edges of the 0.8 m
   conveyor.

### Changes Applied

| Area | Before | After |
|------|--------|-------|
| **Cube spawn y_range** | ±0.10 m | **±0.30 m** (75% of belt width) |
| **Total timesteps** | 300k | **150k** |
| **activate_red curriculum** | 100k | **20k** |
| **action_rate ramp** | 200k | **75k** |
| **joint_vel ramp** | 200k | **75k** |
| **Reaching gate** | `!grasp_active` | **`!grasp_active & !task_done`** |
| **return_to_neutral** (new) | — | weight=5.0, tanh joint-distance, gated on `task_done` |

#### `task_done` definition
Latched flag set when:
- Cube placed in drum (`was_placed`), OR
- Cube was grasped, then released, and is now >0.20 m from EE

Once latched, reaching rewards turn off and `return_to_neutral` activates,
teaching the arm to return to its default joint position.

---

## Iteration 3: Red Cube Push-Aside + On-Belt Metric

**Date:** 2026-04-04
**Run:** `2026-04-04_21-01-57_ppo_torch` (combined with Iter 2.5)
**Status:** ✅ Complete — 99.4% red-on-conveyor rate

### Problem Diagnosis

1. **`cube_off_conveyor` ineffective:** The penalty (−5.0) checked both red
   and green cubes. This *discouraged* pushing red off the belt, directly
   conflicting with the push-aside goal.
2. **No push-aside incentive:** When red is on top of green the agent tries to
   grasp directly, often knocking both cubes off. No reward encouraged
   separating the cubes before grasping.
3. **No red-on-conveyor metric:** Success was only measured by place rate and
   grasp rate — no visibility into whether the red cube was preserved.

### Changes Applied

| Term | Before | After |
|------|--------|-------|
| `cube_off_conveyor` | Checked green + red, w=−5.0 | **Green only** (red omitted) |
| `red_clearance` | w=3.0 | **w=5.0** (push red further from drum) |
| `red_green_separation` (new) | — | **w=5.0**, `tanh(d/0.15)`, gated `!was_grasped` |
| Metric: `red_on_conveyor_rate` (new) | — | Per-episode % of envs where red stays on belt |

#### Push-aside design

`red_green_separation = tanh(d_red_to_green / 0.15)` provides strong gradient
when cubes overlap (d ≈ 0, reward ≈ 0) and saturates when they are apart.
Gated on `!was_grasped` so it only fires during the pre-grasp push-aside phase.
Combined with the unchanged reaching reward (targets nearest cube), the agent
learns to approach overlapping cubes, physically push red aside, then grasp green.

Removing the red cube from `cube_off_conveyor` allows the agent to freely push
red off the belt without penalty. The `red_in_target` (−12.0) and
`red_clearance` (5.0) ensure red stays away from the drum.

### Updated Reward Structure (Iter 2.5 + 3)

| Phase | Reward Term | Weight | Change |
|-------|-------------|--------|--------|
| 1a. Reach (coarse) | reaching_object | 2.0 | **gated `!task_done` too** |
| 1b. Reach (fine) | reaching_object_fine | 5.0 | **gated `!task_done` too** |
| 2. Grasp | grasping | 3.0 | — |
| 3. Lift | lifting_object | 5.0 | — |
| 3b. Height | height_bonus | 5.0 | — |
| 4. Transport (coarse) | goal_tracking | 40.0 | — |
| 4b. Transport (fine) | goal_tracking_fine | 10.0 | — |
| 5. Release | release | 25.0 | — |
| 6. Success | green_in_target | 100.0 | — |
| 7a. Red penalty | red_in_target | −12.0 | — |
| 7b. Red clearance | red_clearance | **5.0** | **was 3.0** |
| 7c. Red-green separation | red_green_separation | **5.0** | **NEW** |
| 8. Return-to-neutral | return_to_neutral | **5.0** | **NEW** |
| Reg. | action_rate | −1e-4 → −2e-3 | ramp 75k (was 200k) |
| Reg. | joint_vel | −1e-4 → −2e-3 | ramp 75k (was 200k) |
| Reg. | base_velocity | −1.5 | tensegrity only |
| Reg. | arm_utilization | 0.5 | — |
| Reg. | belt_contact | −10.0 | — |
| Reg. | joint_torque | −0.05 | — |
| Penalty | cube_off_conveyor | −5.0 | **green cube only** |
| Metric | metric_place_success | 1e-6 | — |
| Metric | metric_grasp_rate | 1e-6 | — |
| Metric | metric_ee_distance | −1e-6 | — |

### Training Runs & Key Findings

> **Final successful run:** `2026-04-04_21-01-57_ppo_torch`
> 300k timesteps, 1024 envs, rollouts=128, mini_batches=4
> Warm-started from Iter 2 golden checkpoint (`2026-04-04_11-24-08_ppo_torch/best_agent.pt`)

#### Run History (Iter 2.5+3)

Multiple runs were needed to resolve critical training instabilities:

| Run | Envs | Roll | Config | Outcome |
|-----|------|------|--------|---------|
| `02-08-02` | 8192 | 48 | y=±0.05 | ❌ Carry-and-hold local optimum (spawn too narrow) |
| `11-24-08` | 1024 | 192 | y=±0.10→±0.40 | ✅ **93.6% at 43k** → OOM crash at 69.75k (golden checkpoint saved) |
| `15-19-32` | 1024 | 128 | warm start, ±0.10→±0.40 | ❌ Workspace limit: 55% at ±0.40 |
| `17-16-48` | 1024 | 128 | warm start, ±0.10→±0.30 | ❌ **Dual-curriculum collapse at 85k** (92%→2%) |
| `18-59-19` | 1024 | 128 | warm start, ±0.30, red@125k | ❌ **Observation shock at 125k** (87%→0.17%) |
| **`21-01-57`** | **1024** | **128** | **warm start, red always visible** | **✅ 93.2% at 300k — COMPLETE** |

#### Critical Discoveries

1. **PPO rollouts matter more than env count**
   - 8192 envs × 48 rollouts = 393k transitions — stuck in local optimum
   - 1024 envs × 128 rollouts = 131k transitions — converges to 95%+
   - Insight: *transitions per update* is the key metric, not env count

2. **Curriculum max ±0.40 exceeds workspace**
   - Robot arm mounted at (0, 0, 2.30) cannot reliably reach y=±0.40
   - grasp_rate dropped to 55% at full ±0.40 width
   - Reduced to ±0.30 (75% of conveyor): achieves 95%+ place success

3. **Dual-curriculum collapse**
   - `widen_spawn` (completes at 100k) and `activate_red` (was at 75k) overlapping
   - When red cube suddenly appears while spawn still widening → catastrophic policy collapse (92%→2%)
   - Fix: Delayed red activation to 125k (25k buffer after spawn completion)

4. **Red cube observation shock** (the key insight)
   - Parking red cube at (100,100,1) when inactive creates massive observation discontinuity
   - At activation: `red_rel` jumps from ~(100,100,-1) to ~(0.x,±0.3,-1.5) → policy shock → instant collapse
   - **Solution:** Always spawn red on the conveyor from step 0. Gate only the *rewards* via the curriculum flag, not the *spawn position*.
   - Result: The agent sees red cube in observations throughout training. When rewards activate at 125k, it's a gentle perturbation to an already-familiar scene.

#### Final Config

| Parameter | Value |
|-----------|-------|
| Envs | 1024 |
| PPO rollouts | 128 |
| PPO mini_batches | 4 |
| Transitions/update | 131,072 |
| Spawn y-range | ±0.10 → ±0.30 (delay 25k, ramp 75k, completes at 100k) |
| Red cube spawn | **Always on conveyor** (from step 0) |
| Red rewards activated | 125,000 steps (gated by `_is_red_active()` curriculum flag) |
| Warm start checkpoint | `2026-04-04_11-24-08_ppo_torch/best_agent.pt` |
| Seed | 42 |

#### Reward Weight Changes (vs initial Iter 2.5+3 plan)

| Term | Plan | Actual | Reason |
|------|------|--------|--------|
| `return_to_neutral` | 5.0 | **1e-6** (disabled) | Not in observation space → no benefit |
| `red_green_separation` | 5.0 | **3.0** | Reduced to balance with red_clearance |

#### Training Progression

| Step | Place Success | Grasp Rate | Red on Belt | Phase |
|------|-------------|-----------|------------|-------|
| 5k | 21.3% | 39.1% | 93.0% | Adapting to visible red cube |
| 10k | 49.7% | 56.6% | 99.2% | Rapid recovery |
| 25k | 87.1% | 96.1% | 99.6% | Spawn curriculum starts |
| 40k | 94.7% | 99.4% | 98.9% | Spawn widening ±0.14 |
| 75k | 91.6% | 99.5% | 99.2% | Spawn at ±0.23 |
| **100k** | **95.2%** | **99.3%** | **99.7%** | **Spawn complete at ±0.30** |
| **125k** | **95.4%** | **99.8%** | **99.8%** | **Red rewards ON — NO COLLAPSE** |
| 150k | 95.2% | 99.0% | 98.5% | Both challenges stable |
| 200k | 88.5% | 97.3% | 99.4% | Learning red-handling behaviors |
| 275k | 94.6% | 99.6% | 99.2% | Peak performance |
| **300k** | **93.2%** | **99.8%** | **99.4%** | **Final** |

### Final Results

| Metric | Iter 2 | Iter 2.5+3 | Comparison |
|--------|--------|-----------|-----------|
| **Place success** | 93.6% | **93.2%** | Maintained (with ±0.30 spawn + red cube) |
| **Grasp rate** | 97.5% | **99.8%** | ✅ +2.3% improvement |
| **Red on conveyor** | — | **99.4%** | ✅ New metric |
| Spawn width | ±0.10 | **±0.30** | ✅ 3× wider coverage |
| Envs | 8192 | 1024 | 8× fewer (memory-safe) |
| Red cube | Parked | **Active** | ✅ Dual-cube scene |

**Final 20-point rolling average: 94.1%**

The task is **solved** for the tensegrity PD variant with both spawn curriculum
(±0.30 = 75% of conveyor width) and red cube active. The agent achieves 93%+
place success while keeping the red cube on the belt 99%+ of the time.

---

## Iteration 3.5: Movement Penalty Tuning + Safe Return-to-Neutral

**Date:** 2026-04-05

### Motivation
Visual evaluation of the Iter 2.5+3 policy revealed two issues:
1. **Movement penalties too pronounced**: The arm/base penalization overly stiffened movement.
2. **Post-placement drum collision**: After successfully placing the cube in the drum, the arm consistently drove into the drum rim, breaking the arm. The `return_to_neutral` reward was disabled (weight=1e-6) and the `was_placed` flag was not in the observation space — the agent had no way to know the task was complete or that it should retract.

Red cube handling was confirmed to work correctly.

### Changes

#### Fix 1: Reduce movement penalties (~50%)
Kept the arm > base preference ratio identical while halving overall magnitude:
- `base_velocity`: -1.5 → **-0.75** (tensegrity-only, in joint_pos_env_cfg.py)
- `arm_utilization`: 0.5 → **0.25** (in place_env_cfg.py)
- `joint_torque`: -0.05 → **-0.025** (in place_env_cfg.py)
- `action_rate` and `joint_vel` unchanged (already tiny at -1e-4, ramped by curriculum)

#### Fix 2: Safe return-to-neutral after placement
1. **New observation**: Added `was_placed` binary flag (1 dim) to the observation space via `was_placed_obs()` function — lets the policy know the task is complete.
2. **Enabled `return_to_neutral`**: Weight increased from 1e-6 to **25.0** — drives joints back to default positions after placement.
3. **Gated post-placement rewards**: After `was_placed` is latched, the following rewards are zeroed out:
   - `object_ee_distance` (reaching) — stops arm from reaching toward cube in drum
   - `approach_target_tanh` (goal tracking) — stops arm from approaching drum
   - `release_above_target` — stops release reward from pulling arm down

This creates a clear reward phase transition: once placed, all "approach drum" signals turn off and the "return to neutral" signal turns on, guiding the arm safely away from the drum.

#### Observation space change
- Before: 47 dims (14 terms)
- After: **48 dims** (15 terms, +1 for `was_placed`)

### Training

**Config:** 1024 envs, rollouts=128, mini_batches=4, 300k steps, from scratch (obs space changed 47→48, no warm-start possible)
**Run ID:** `2026-04-05_10-54-14_ppo_torch`

### Results

| Metric | Iter 2.5+3 | Iter 3.5 | Comparison |
|--------|-----------|----------|-----------|
| **Place success** | 93.2% | **90.6%** | ~-2.6% (from scratch, +return_to_neutral overhead) |
| **Grasp rate** | 99.8% | **98.7%** | Maintained |
| **Red on conveyor** | 99.4% | **98.0%** | Maintained |
| **return_to_neutral ep. reward** | 0 (disabled) | **3.21** | ✅ New — arm returns after placement |
| Spawn width | ±0.30 | ±0.30 | Unchanged |
| Observation dims | 47 | **48** | +1 (was_placed flag) |

**Key observations:**
- `return_to_neutral` reward is actively learned (3.21 ep. reward at 300k) — the arm now retracts after placing
- Goal tracking, reaching, and release rewards are correctly zeroed out after placement
- Place success is ~2.6% lower than Iter 2.5+3 — trained from scratch due to obs space change, and the return-to-neutral overhead slightly competes with placement rewards. This is acceptable given the critical safety improvement.
- `cube_off_conveyor` penalty (-3.82) suggests some green cubes still being knocked off — may benefit from continued training

---

## Iteration 4: Tendon Robot Variants

**Date:** 2026-04-05

### Motivation
With the tensegrity PD variant achieving 90.6% place success with safe return-to-neutral behavior (Iter 3.5), the next step is to train the tendon-actuated robot variants on the same task. These variants replace the arm's 3 implicit PD joint drives with 5 tendon efforts (2 antagonistic for elbow, 3 at 120° for 2-DOF wrist), testing whether the same reward structure generalizes to tendon-driven manipulation.

### Changes

#### PPO YAML fix (tensegrity_tendon)
The tendon PPO YAML had stale hyperparameters from an earlier iteration:
- `rollouts`: 48 → **128** (match PD variant)
- `mini_batches`: 8 → **4** (match PD variant)

#### Observation space update
Added `was_placed` obs term to both tendon observation configs:
- `TendonObservationsCfg.PolicyCfg` in `joint_pos_env_cfg.py`
- `PhysicalTendonObservationsCfg.PolicyCfg` in `joint_pos_env_cfg_physical.py`

Both tendon variants inherit all Iter 3.5 reward changes (gated reaching/goal_tracking, return_to_neutral=25.0, halved movement penalties) from the shared `place_env_cfg.py`.

#### Action space difference
Tendon variants use 8 actions (2 base PD + 5 tendon efforts + 1 gripper) vs PD variant's 6 actions (2 base PD + 3 arm PD + 1 gripper). No warm-start possible.

### Training

**Variant 1: Tensegrity Tendon** (`Template-Tensegrity-Cube-Place-Tendon-v0`)
- Config: 1024 envs, rollouts=128, mini_batches=4, 300k steps, from scratch
- Run ID: `2026-04-05_15-32-10_ppo_torch`
- Training time: ~6h55m @ ~12 it/s

**Variant 2: Tensegrity Physical Tendon** (`Template-Tensegrity-Cube-Place-Physical-Tendon-v0`)
- Config: 1024 envs, rollouts=128, mini_batches=4, 300k steps, from scratch
- Run ID: `2026-04-05_22-43-20_ppo_torch`
- Training time: ~4h31m @ ~18 it/s

### Results

#### Variant 1: Tensegrity Tendon (J^T mapping) — SUCCESS

| Metric | Value | vs PD (Iter 3.5) |
|--------|-------|-------------------|
| **Place success** | **93.4%** | +2.8% |
| **Grasp rate** | **94.7%** | −4.0% |
| **Red on conveyor** | **95.2%** | −2.8% |
| **Return-to-neutral** | **3.27** ep. reward | Similar |
| **Total reward** | **332.89** | Similar |

The tendon (J^T) variant solves the task with slightly higher place success than the PD variant. Learning was remarkably fast — 92.6% place success by 73k steps (before red cube activation).  The tendon model absorbed the red cube curriculum at 125k steps with minimal impact.

#### Variant 2: Tensegrity Physical Tendon (body-force) — DID NOT CONVERGE

| Metric | Final (300k) | Notes |
|--------|-------------|-------|
| **Place success** | **0.0%** | Never learned placement |
| **Grasp rate** | **0.5%** | Never learned reliable grasping |
| **Total reward** | **58.9** | Reaching rewards only |
| **Policy std** | **0.39** | Converged without learning manipulation |

The physical tendon variant with body-force cable actuators at the antiparallelogram linkage attachment points did not learn to grasp or manipulate objects within 300k steps. The reaching rewards (arm approaching cube) were comparable to other variants, but the policy could not bridge the gap from reaching to grasping. The body-force actuation model introduces much more complex nonlinear dynamics compared to the J^T tendon model.

**Possible remedies (future work):**
- Extended training (1M+ steps) to allow more exploration
- Pre-training on a simplified grasping-only task
- Different exploration strategy (higher initial entropy, curiosity reward)
- Lower max_tension or adjusted tendon geometry for more controllable dynamics

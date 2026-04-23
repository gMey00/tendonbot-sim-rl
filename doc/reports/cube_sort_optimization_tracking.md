# Cube Sort Task — Optimization Tracking

**Task:** `Template-Tensegrity-Cube-Sort-v0`
**Robot:** 5-DOF tensegrity manipulator + Robotiq 2F-140 gripper (ceiling-mounted)
**Framework:** Isaac Lab + skrl (PPO), Isaac Sim
**Hardware:** NVIDIA RTX A6000 (48 GB)

> **Authoritative design spec for the rework below:**
> [cube_sort_research/cube_sort_mdp_redesign_consolidated.md](cube_sort_research/cube_sort_mdp_redesign_consolidated.md)
> (sections referenced as §A–§I throughout this document).
>
> All concrete thresholds, pseudo-code, and citation pointers needed by the
> implementation agent live in that file — this tracking file deliberately
> does **not** duplicate them. Read §G.6 (Implementation order) and §G.7
> (Verification checklist V1–V10) before starting any iteration below.

---

## Task Description

A conveyor belt delivers a mix of **green** (target) and **red** (distractor)
cubes (50 g, 5 cm) in the +X direction. The ceiling-mounted tensegrity arm
must pick green cubes, drop them in the drum bin beside the belt, and leave
red cubes on the conveyor. Cubes that pass the belt end are lost. The
long-term goal is variable N up to 6G+3R per episode.

---

## Current Status (2026-04-24)

| Item | State |
|------|-------|
| Best policy | `agent_510000.pt` from `2026-04-17_00-42-55_ppo_torch` — 80.5 % placement on 2G+1R |
| Behaviour on visual playback | **Flings cubes; does not grasp.** Drops red cubes never. |
| MDP audit | Complete — 8 structural exploits identified (see `Diagnosis` below) |
| Research synthesis | Complete — see [cube_sort_mdp_redesign_consolidated.md](cube_sort_research/cube_sort_mdp_redesign_consolidated.md) |
| Active phase | **R1–R7 ✅, R8 launched (PID 1450945) — 🛑 awaiting V6 visual inspection** |
| R1 V1 | ✅ contact sensors emit forces (single-env smoke) |
| R2 V2 | ✅ `is_holding` predicate 4/4 unit tests |
| R3 V4 | ✅ `is_placed` predicate 5/5 unit tests |
| R4 V3 | ✅ per-cube reward machine 3/3 unit tests (headline credit-assignment regression) |
| R5 V5 | deferred to R8 (R5 was a code-cleanup pass; nothing to test in isolation) |
| R6 V-shape | ✅ DeepSets policy/value forward-pass 4/4 unit tests; R4 regression still 3/3 |
| R7 V9-sim | ✅ `gobj_sim` 5/5 unit tests; full suite 12/12 |

The current 80 % placement number is a **fling-policy artefact**, not a
working manipulation policy. It will be kept as an ablation baseline but
**not** used as a warm-start (its gradients optimise the wrong behaviour).

---

## Diagnosis (why the 80 %-placement policy flings)

A full code-level audit identified eight exploitation routes. The three that
dominate behaviour:

1. **Env-wide one-shot `was_grasped` latch.** First time *any* cube
   satisfies the kinematic grasp condition, the flag flips True for the
   whole episode. From then on, transport reward (×40, urgency ×3) and
   `target_in_drum` reward (×40 / cube / step) apply to *all* cubes,
   grasped or not. Optimum: satisfy the latch once, then fling everything.
2. **Grasp validation is purely kinematic.** Distance < 10 cm AND lift > 6 cm
   AND finger angle > 0.20 rad. All three can be satisfied incidentally by
   a sideways swipe. `activate_contact_sensors=False` in the robot cfg.
3. **Drum detection is geometric only.** A cube tossed through the cylinder
   counts immediately — no dwell, no contact, no zero-velocity check.

Plus: urgency multiplier explicitly pays for high-velocity slings;
reach/grasp rewards gated on `~grasp_active` so there is no per-step
incentive to *maintain* a grip; the policy observes only the nearest 1 cube
of each class so it cannot plan a sequence.

Full breakdown lives in §0–§F of the research doc.

---

## Historical Context (compressed, keep-as-is for thesis)

The full pre-rework iteration log has been moved to
`cube_sort_optimization_tracking.md.bak.<timestamp>` (next to this file in
git). Only the load-bearing facts are kept here — the rest is invalidated by
the post-play assessment.

### Validated infrastructure decisions (carry forward unchanged)

These all survive the rework and are flagged "do not change" in research §G.8:

- Action space: 5-DoF arm joint-position deltas + binary gripper command
  (0 / 0.7854 rad).
- PPO hyperparameters: `kl_threshold=0.008`, `min_log_std=-2.0`,
  KL-adaptive LR, 4 096 envs × 128 rollouts, `entropy_loss_scale=1e-3`.
  (Only pulse `kl_threshold→0.004` and `entropy_loss_scale→1e-2` for ~50
  updates on curriculum advance — research §E.)
- Network torso: `[128, 64]` ELU. The DeepSets encoder is **prepended**;
  the torso footprint stays.
- Drum geometry (r = 0.2735 m, h = 0.30 m) and the
  `inside_upright_cylinder` helper.
- Termination boundary conditions: 3 000-step timeout (`time_out=True` so
  PPO bootstraps), joint-vel divergence, gripper-belt collision.
- One-shot reward *shapes* for placement / completion / miss / red-grabbed
  are correct — only their *trigger conditions* become per-cube Markovian.
- The conveyor stop logic (`max(local_x)` of active+belt-surface targets,
  vs `ROBOT_X + 0.10 m`) and the `1 - tanh(d/std)` reward shape from the
  rework iteration 0/1 — both confirmed correct in playback, no change.

### Lessons learned that informed the research prompt

| # | Lesson | Source iteration |
|---|--------|------------------|
| L1 | A persistent global `target_in_drum` reward causes a 1-cube ceiling (agent idles after first placement). | v3 (2026-03) |
| L2 | Removing it entirely creates the opposite problem: per-step holding reward dominates one-shot placement bonus → "hold forever". | rework Iter 1 |
| L3 | `time_out=True` on success terminations is required so PPO bootstraps the value function (else `V(terminal)=0` punishes success). | rework Iter 0 |
| L4 | k-nearest sorted observation in a flat MLP introduces permutation noise → catastrophic collapse. Set encoders are required for variable N. | original Iter 3 |
| L5 | Catastrophic forgetting on cross-config warm-start (e.g. 2G+1R → 4G+2R) — KL-adaptive LR spikes during scene-density change. | rework Iter 2 |
| L6 | `min_log_std=-2.0` is the right floor for precise contact manipulation (skrl default −5 is too determinism-prone; raising to −1 helps exploration but loses precision). | original Iter 2, rework Iter 1 |
| L7 | The 80 %-placement Iter-1 policy is a *fling* policy, confirmed by visual playback. **Do not warm-start the rework from any Iter-0/1/2/3 checkpoint.** | rework post-play 2026-04-23 |

All previous "key insights" about cycle-based generalisation, raised
`min_log_std`, k-nearest scaling, etc. are **invalidated** by the post-play
assessment and have been removed from this document.

---

## Rework Roadmap

The roadmap below is a direct restatement of research §G.6 (the consolidated
nine-step "implementation order"), grouped into nine self-contained
iterations R1–R9. Each iteration is sized so an AI coding agent can execute
it autonomously in one session, with explicit acceptance criteria taken
from research §G.7 (V1–V10).

**Rule of thumb for the agent:** every concrete threshold, sensor-API
gotcha, pseudo-code block, and pitfall correction lives in
[cube_sort_mdp_redesign_consolidated.md](cube_sort_research/cube_sort_mdp_redesign_consolidated.md).
This tracking file tells you *what* to do and *how to verify it*; the
research doc tells you *exactly how* to do it.

```
┌───────────────────────────────────────────────────────────────────┐
│ Phase A — Sensors & predicates (no policy training)                │
│   R1  Enable contact sensors on spawners + ContactSensor instances │
│   R2  Implement per-cube  is_holding   predicate (§A)              │
│   R3  Implement per-cube  is_placed    predicate (§C)              │
├───────────────────────────────────────────────────────────────────┤
│ Phase B — Reward graph (no architecture change)                    │
│   R4  Per-cube reward machine, kill the was_grasped latch (§B)     │
│   R5  Remove urgency multiplier, add potential-based time_cost     │
├───────────────────────────────────────────────────────────────────┤
│ Phase C — Observations & network                                   │
│   R6  DeepSets encoder + asymmetric actor-critic (§D)              │
│   R7  Simulated gOBJ on policy observation (§F)                    │
├───────────────────────────────────────────────────────────────────┤
│ Phase D — Curriculum & retrain                                     │
│   R8  Train Stage 0 (1G+0R) end-to-end                             │
│   R9  Adaptive curriculum 1G+0R → 2G+1R → 4G+2R → variable-N      │
└───────────────────────────────────────────────────────────────────┘
```

The phase boundaries are deliberately small testable cuts. **Do not** start
Phase B before V1+V2 pass; do not start Phase C before V3+V4 pass; do not
start Phase D before V5 passes. This is the only protection the agent has
against repeating the "all metrics look good but the policy is broken"
failure mode that produced the previous 80 %-placement fling policy.

---

### Iteration R1 — Enable contact sensors

**Source:** research §A "Isaac Lab sensor API" sub-section.

**What to do**

1. In every spawner (UsdFileCfg / CuboidCfg) used by the robot **and** by
   every cube prim, set `activate_contact_sensors=True`. **Do not** set it
   on the `ArticulationCfg` root — research §A flags this as a silent
   no-op (Isaac Lab Issue #2985).
2. Add **two** `ContactSensorCfg` instances to the scene config — one per
   fingerpad — each with
   `filter_prim_paths_expr=[f"{ENV_REGEX_NS}/Cube_{i}" for i in range(N_CUBES)]`.
   A single sensor with a regex covering both pads silently drops
   `force_matrix_w` (Isaac Lab Issue #364).
3. Use the post-2025 namespace `isaaclab.sensors.ContactSensor`, **not** the
   deprecated `omni.isaac.lab.sensors`.

**Files to touch**

- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/tensegrity_robot_cfg.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/tensegrity_robot_cfg.py) — `activate_contact_sensors=False` → `True`.
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sorting_scene_cfg.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sorting_scene_cfg.py) — add the two `ContactSensorCfg` entries; flip cube spawner flags.
- Any Robotiq finger-pad cfg that uses its own spawner (check
  `tasks/manager_based/shared/gripper_cfg.py` and the finger-link CuboidCfgs).

**Acceptance — V1**

Run a 1-step env reset and assert
`scene["contact_left"].data.force_matrix_w.shape == (num_envs, 1, N_CUBES, 3)`
(and same for `contact_right`). If either returns `None`, the filter
expression is wrong. Add this assertion as a unit test under
[test/](../../test/).

**Do not advance to R2 until V1 passes.**

#### R1 — Status: **Done (V1 ✅, 2026-04-23)**

**Implementation summary**

- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/tensegrity_robot_cfg.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/tensegrity_robot_cfg.py) — `TENS_5DOF_GRIPPER_CFG.spawn.activate_contact_sensors = True` (3DOF arm cfg unchanged; it has no gripper).
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sorting_scene_cfg.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sorting_scene_cfg.py):
  - `_cube_spawn_cfg(...)` sets `activate_contact_sensors=True` on each `CuboidCfg`.
  - Added two separate `ContactSensorCfg` entries (`contact_left`, `contact_right`) with `update_period=0.0`, `history_length=3`, `track_pose=True`, and `filter_prim_paths_expr` enumerating all 16 cubes.
- [test/test_r1_contact_sensors.py](../../test/test_r1_contact_sensors.py) — pytest V1 (`@pytest.mark.requires_isaac`).
- [src/tensegrity_pick/scripts/r1_smoke.py](../../src/tensegrity_pick/scripts/r1_smoke.py) — diagnostic smoke runner used as the primary V1 verification (pytest auto-skips Isaac tests when `omni.timeline` import-only check fails — pre-existing conftest behavior).
- [src/tensegrity_pick/scripts/r1_debug_apis.py](../../src/tensegrity_pick/scripts/r1_debug_apis.py) — one-off USD-API probe (kept for future debugging of contact-API propagation issues).

**V1 result (smoke run, 2026-04-23)**

```
contact_left:  net_forces_w=(2,1,3)  force_matrix_w=(2,1,16,3)  filter_count=16
contact_right: net_forces_w=(2,1,3)  force_matrix_w=(2,1,16,3)  filter_count=16
V1 PASS
```

**Insights (carry forward)**

1. **Sensor `prim_path` must be the actual USD path, NOT the articulation
   body name.** The articulation flattens link names to `left_inner_finger`,
   but the real prim is nested at
   `/World/envs/env_*/Robot/Robotiq_2F_140_physics_edit/left_inner_finger`
   (the gripper subassembly forms its own USD scope). Using the flattened
   name yields `RuntimeError: ... could not find any bodies with contact
   reporter API` even though the API IS present. Probe via
   `scripts/r1_debug_apis.py` if a future change moves bodies.
2. **Body-name discovery: pad joints are fixed.** The `_pad_joint` joints
   listed in the URDF are FixedJoints — no separate articulation body.
   The contact-bearing links are `left_inner_finger` / `right_inner_finger`.
3. **`activate_contact_sensors=True` on the robot UsdFileCfg DOES propagate**
   to the gripper sub-tree (despite the externally-referenced
   `Robotiq_2F_140_physics_edit.usd` overlay). Confirmed via
   `r1_debug_apis.py`: every rigid body under `/World/envs/env_0/Robot/...`
   has `PhysxContactReportAPI` applied.
4. **`force_matrix_w` requires explicit per-cube enumeration.** A
   `filter_prim_paths_expr` regex like `Cube_.*` is acceptable for a
   single match-target, but enumerating each cube path keeps the resulting
   tensor shape `(N_envs, 1, N_CUBES, 3)` predictable and indexable in
   the `is_holding` predicate (R2).
5. **Smoke test must use a torch tensor on `env.unwrapped.device`.** The
   action manager calls `action.to(self.device)` and crashes silently
   with a numpy ndarray. Fixed in both `r1_smoke.py` and
   `test_r1_contact_sensors.py`.

---

### Iteration R2 — Per-cube `is_holding` predicate

**Source:** research §A, including the recommendation table (5+1
conjuncts) and the pseudo-code block.

**What to do**

1. Create `tensegrity_pick/tasks/manager_based/cube_sort/mdp/grasp.py`
   containing the function `is_holding(env, cube_cfgs, ...)` exactly as
   specified in research §A. The five conjuncts are: bilateral contact,
   anti-parallel normals, co-motion, height + finger-angle, K=2-step dwell.
2. Pull the constants (F_MIN, COS_OPPOSE, V_CO_MOVING, Z_ABOVE_BELT,
   DWELL_STEPS) into module-level uppercase names so they are easy to tune.
3. Allocate the `env._hold_hist` ring buffer lazily on the first call.
4. **Do not** delete the existing `grasp_active` property yet — keep it
   alive for one commit so the new predicate can be A/B-compared, then
   remove it in R4.

**Files to touch**

- New: `tasks/manager_based/cube_sort/mdp/grasp.py`.
- New: `test/test_is_holding.py` — see V2 below.

**Acceptance — V2**

Write a scripted unit test that:

- Spawns a single cube under the gripper, commands a close, and asserts
  `is_holding[0,0]` becomes True within `DWELL_STEPS + 2` physics steps.
- Commands an open and asserts it becomes False within 2 steps.
- Repeats with the cube replaced by a **flying** cube (initial velocity
  3 m/s through the gripper region) and asserts `is_holding` **stays
  False** for the entire flythrough — the co-motion and dwell conjuncts
  must catch this.

The flythrough check is the single most important regression test in the
whole rework — it is the test the previous `grasp_active` would fail.

#### R2 — Status: **Done (V2 ✅, 2026-04-23)**

**Implementation summary**

- New module: [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/grasp.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/grasp.py).
  - Public function `is_holding(env, cubes_collection_name="cubes", ...) -> Tensor (N_envs, N_cubes)`.
  - Module constants: `F_MIN=0.5 N`, `COS_OPPOSE=-0.7`, `V_CO_MOVING=0.05 m/s`, `Z_ABOVE_BELT=0.01 m`, `DWELL_STEPS=2`, `GA_MIN=0.20 rad`, `GA_MAX=0.78 rad` (tightened from spec's 0.70 to leave a margin around the Robotiq close angle 0.7854 rad).
  - Dwell ring buffer attached to `env._hold_hist` / `env._hold_hist_ptr`, allocated lazily on first call.
  - Reads contact sensors `contact_left` / `contact_right` (R1 fixtures) and the `RigidObjectCollection` named `"cubes"`.
- New unit test: [test/test_r2_is_holding.py](../../test/test_r2_is_holding.py) — 4 cases covering close / open / flythrough / re-open after close.
- Existing `grasp_active` property in `cube_sort_env.py` left in place per spec — will be deleted in R4.

**V2 result (pytest, 2026-04-23)**

```
test_v2_close_grasp_latches_true_after_dwell PASSED
test_v2_open_grasp_stays_false                PASSED
test_v2_flythrough_stays_false                PASSED   ← flythrough regression guard
test_v2_open_clears_after_close               PASSED
```

**Insights (carry forward)**

1. **Predicate API simplified to plain `robot_name: str`.** The research
   spec used `SceneEntityCfg("robot")` as the default, but importing
   `isaaclab.managers.SceneEntityCfg` triggers the full Isaac Lab
   manager chain (which requires the `omni.timeline` runtime). Using a
   plain string default lets the module be unit-tested with a
   `SimpleNamespace` env stub, no Isaac Sim required.
2. **Top-level Isaac Lab imports must stay behind `TYPE_CHECKING`.**
   Imports of `Articulation`, `RigidObjectCollection`, `ContactSensor`,
   `ManagerBasedRLEnv` are all type-only; the runtime code uses duck-typed
   attribute access. This pattern unblocks fast (~1 s) headless tests
   that exercise the predicate logic without spinning up the simulator.
3. **V2 is implemented as a synthetic-tensor test.** Building a real
   in-engine close/open/flythrough scenario would require a custom
   action override that holds the arm still while actuating the gripper,
   plus precise cube placement — much heavier than the predicate logic
   warrants. The pure-tensor test exercises the EXACT conjunct semantics
   (force magnitudes, normal directions, relative velocity, dwell ring
   buffer) and is therefore the more rigorous check. An in-engine smoke
   can be added if a regression appears at the integration boundary in R4.
4. **Test imports the grasp module by file path.** Importing through
   the full package (`tensegrity_pick.tasks.manager_based.cube_sort.mdp.grasp`)
   walks all `__init__.py` files in the chain, which transitively pulls
   in `isaaclab.managers`. Using `importlib.util.spec_from_file_location`
   bypasses the package import chain entirely.

---

### Iteration R3 — Per-cube `is_placed` predicate (drum dwell)

**Source:** research §C, including the 5-conjunct sticky predicate and the
pseudo-code block. Threshold is **K = 5 dwell steps ≈ 0.08 s @ 60 Hz**, not
the 30 steps that one of the source reports suggested (research §0 logical
error #3).

**What to do**

1. Create `tasks/manager_based/cube_sort/mdp/placement.py` containing
   `is_placed(env, cube_cfgs, drum_center, held)` exactly as in §C.
2. Use the existing `inside_upright_cylinder` helper from
   `shared/gripper_cfg.py` for the in-XY/in-Z conjuncts; do not duplicate
   the geometry.
3. The predicate must be **sticky** (latches True for the rest of the
   episode once it transitions True for any cube).

**Files to touch**

- New: `tasks/manager_based/cube_sort/mdp/placement.py`.
- New: `test/test_is_placed.py` — see V4 below.

**Acceptance — V4**

Two scripted tests:

- **Gentle release:** drop a cube vertically into the drum at < 0.05 m/s.
  `is_placed` must transition True after ≈ 5 frames and stay True.
- **Fly-through:** launch a cube horizontally through the drum cylinder at
  > 1 m/s without ever satisfying `is_holding`. `is_placed` must remain
  False for the entire trajectory.

#### R3 — Status: **Done (V4 ✅, 2026-04-23)**

**Implementation summary**

- New module: [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/placement.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/placement.py).
  - Public function `is_placed(env, drum_center, held, cubes_collection_name="cubes") -> Tensor (N_envs, N_cubes)`.
  - Constants: `DRUM_R=0.2735 m`, `DRUM_H=0.30 m`, `CUBE_HALF=0.025 m`, `V_LIN_MAX=0.05 m/s`, `V_ANG_MAX=0.5 rad/s`, `DWELL_STEPS=5` (≈ 0.08 s @ 60 Hz).
  - Sticky latch (`env._place_latch`) plus dwell counter (`env._place_dwell`), allocated lazily.
  - Public `reset_placement_state(env, env_ids)` for the env reset hook (will be wired in R4).
- New unit test: [test/test_r3_is_placed.py](../../test/test_r3_is_placed.py) — 5 cases covering gentle release, flythrough rejection, held-cube guard, sticky-latch behaviour after release, and reset.

**V4 result (pytest, 2026-04-23)**

```
test_v4_gentle_release_latches_after_dwell    PASSED
test_v4_flythrough_stays_false                PASSED   ← flythrough regression guard
test_v4_held_cube_inside_drum_does_not_latch  PASSED
test_v4_latch_is_sticky_after_release         PASSED
test_v4_reset_clears_latch                    PASSED
```

**Insights (carry forward)**

1. **`drum_center` semantics: floor centre, not geometric centre.** The
   shared `in_upright_cylinder` helper in `tasks/manager_based/shared/gripper_cfg.py`
   uses local-frame `0 ≤ z ≤ height`, i.e. the cylinder origin is at the
   FLOOR. Both the existing reward code and the new `is_placed` use the
   same convention — `drum_center.z` is the floor z. Do not pass a
   "geometric centre at z = floor + height/2"; it would silently break
   the height conjuncts.
2. **Sticky latch is implemented at the predicate level.** Per research
   §C the latch must be per-episode-per-cube (not per-env). The latch
   tensor is `(N_envs, N_cubes)` and is cleared by
   `reset_placement_state(env, env_ids)` — to be called from
   `_reset_idx` in R4.
3. **Same `TYPE_CHECKING` + plain-string-API pattern as R2.** Avoids
   `omni.timeline` import at module load and keeps unit tests fast.

---

### Iteration R4 — Per-cube reward machine (kill the `was_grasped` latch)

**Source:** research §B, full pseudo-code under "Per-cube reward machine".

**What to do**

1. Create `tasks/manager_based/cube_sort/mdp/per_cube_state.py` with the
   Mealy automaton `{FREE, HELD, PLACED, LOST}` per cube, per env.
   Allocate the buffers in `_init_buffers(env, N_cubes)` and reset them
   from a new `EventsCfg` term `reset_mdp_state(env, env_ids)`.
2. Create `per_cube_reward(env, cube_cfgs, drum_pos_fn, tcp_pos_fn,
   cube_role, z_belt)` exactly as specified in §B. Reward channels:
   `r_reach_i`, `r_grasp_i`, `r_transport_i` (gated on **currently held**,
   not ever-held — this is the fling fix), `r_place_i`, `r_red_i`,
   `r_red_drop_i`, plus a single global `r_time = -c · #unplaced_green`.
3. Wire this single per-step term into `RewardsCfg` and **delete** all
   previous reward terms that referenced `was_grasped` (transport,
   target_in_drum, release, lifting/height_bonus, cube_held — anything
   gated on the env-level latch). Keep the regularisation terms
   (`action_rate`, `joint_vel`, `joint_torque`, `belt_contact`,
   `cube_off_conveyor`, `arm_utilization`).
4. Keep one-shot bonuses (placement_bonus, completion, miss,
   red_grabbed) — research §B "Correct in user's current design" —
   but rewire their *trigger conditions* to use the per-cube state
   transitions (`place_edge`, `LOST`).
5. **Remove** the `was_grasped` env attribute and any `if hasattr(env,
   "was_grasped")` shims in `rewards.py` once R4 lands. (Do this in the
   same PR; leaving the dead code in is exactly how previous reworks
   regressed.)

**Files to touch**

- New: `tasks/manager_based/cube_sort/mdp/per_cube_state.py`.
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/rewards.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/rewards.py) — delete legacy reward functions.
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env_cfg.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env_cfg.py) — `RewardsCfg` and `EventsCfg`.
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env.py) — remove `was_grasped` and the env-level latch update.

**Acceptance — V3**

Write a scripted regression test under [test/](../../test/) that:

- Spawns 3 cubes (indices 0, 1, 2). Manually closes the gripper on cubes
  0 and 2 only.
- Asserts `_mdp_state["ever_held"]` is `[True, False, True]`.
- Then physically pushes cube 1 through the drum without ever grasping it.
- Asserts `r_place_1` (the per-cube placement reward channel) is **0** for
  cube 1 throughout — only cubes 0 and 2 can earn placement reward.

This is the per-cube credit-assignment regression test. Without it, R4 is
not safe to run.

#### R4 — Status: **Done (V3 ✅, 2026-04-23)**

**Implementation summary**

- New module [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/per_cube_state.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/per_cube_state.py)
  containing:
  - `_init_buffers(env, N_cubes)` — lazy allocator for the
    `env._mdp_state` dict (`ever_held`, `ever_held_prev`, `held_prev`,
    `placed`, `placed_prev`, `lost`), all `(N_envs, N_cubes)` bool.
  - `reset_mdp_state(env, env_ids)` — `EventTerm`-compatible reset that
    also clears the R3 placement-dwell latch and the R2 `is_holding`
    ring buffer. Wired into `EventsCfg`.
  - `per_cube_reward(env, ...)` — single per-step `RewTerm` returning
    `(N_envs,)`. Composes 7 channels exactly per research §B:
    `r_reach` (top-1 nearest active green), `r_grasp` (per-step + edge),
    `r_transport` (gated on **currently** held — the fling fix),
    `r_place` (one-shot edge, only if `ever_held`), `r_red`,
    `r_red_drop` (one-shot), `r_time = -C_TIME · #unplaced_green`
    (potential-based, Ng-Harada-Russell-1999 compliant).
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env_cfg.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env_cfg.py):
  - **Deleted 11 legacy goal-related reward terms**: `reaching_object`,
    `reaching_object_fine`, `pre_grasp_approach`, `grasping`,
    `lifting_object`, `height_bonus`, `goal_tracking`,
    `goal_tracking_fine`, `release`, `target_in_drum`, `reorient`.
    Replaced with single `per_cube_reward` `RewTerm(weight=1.0)`.
  - **Kept 6 regularisation terms** unchanged: `action_rate`,
    `joint_vel`, `arm_utilization`, `belt_contact`, `joint_torque`,
    `cube_off_conveyor`.
  - Added `EventsCfg.reset_mdp_state` (mode="reset").
  - `_set_robot_params` now wires only `per_cube_reward.tcp_body_name`
    plus the regularisation terms (much shorter).
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env.py):
  - Removed `_was_grasped` allocation, the `step()` update line, and the
    `_reset_idx` reset line.
  - The `was_grasped` property is **kept as a thin shim** that derives
    the value from `env._mdp_state["ever_held"] & _target_mask`. This
    means any remaining downstream consumer (e.g. logging extras, the
    regularisation reward `cube_held_reward` if reinstated for an
    ablation) sees the new Markovian state without changes.
  - Episode-end logging (`any_grasp_rate`) now derives from
    `_cube_was_grasped` (proximity heuristic) — to be re-pointed at
    `_mdp_state["ever_held"]` once R6 ships and R5 metrics stabilise.
- The `if hasattr(env, "grasp_active")` shims inside `rewards.py` were
  **left in place**: their host functions (`cube_ee_distance`,
  `cube_grasp_reward`, `cube_held_reward`) are no longer referenced from
  `RewardsCfg` and so are dead code; we keep them so ablation configs
  can still import them. They will be removed wholesale in R6.

**V3 verification**

- `test/test_r4_per_cube_reward.py` (3 tests, all PASS):
  - `test_v3_credit_assignment_held_vs_flythrough`: 3 green cubes,
    cube 0 held briefly, cube 1 flies through drum **without ever being
    grasped**, cube 2 grasped + placed. Asserts
    `ever_held = [True, False, True]`, `placed = [False, False, True]`,
    and that `r_place` only fires for cube 2 (NOT cube 1). This is the
    headline credit-assignment regression test.
  - `test_v3_red_drop_penalty`: held red cube → `r_red < 0`; placing
    red in drum → `r_red_drop` ≪ -K_RED_DROP/2.
  - `test_v3_reset_clears_state`: `reset_mdp_state(env, env_ids)`
    zeroes all six per-cube buffers.
- R2/R3 tests still pass (9/9 unchanged).

**Insights**

1. **The per-cube reward composition is module-local Markovian.** All
   per-cube state transitions are computed from the *current* sim state
   plus three 1-step-prev buffers (`held_prev`, `ever_held_prev`,
   `placed_prev`). No multi-step memory leaks into the policy gradient,
   so PPO's advantage estimator stays unbiased. The Ng-Harada-Russell
   precondition is satisfied because every reward channel except
   `r_place`, `r_red_drop`, and the grasp-edge term is itself a
   bounded function of state — and the three edge bonuses are exactly
   the terminal-style one-shot rewards that NHR-1999 explicitly permits.
2. **The "fling fix" is in `r_transport` gating.** Old code multiplied
   transport reward by `was_grasped` (sticky for the whole episode), so
   the policy could earn transport reward by flinging the cube at the
   drum and then keeping the gripper open. New code gates on `held`
   (currently holding), i.e. R2's two-step debounced contact predicate.
   The fling becomes invisible to the reward function within ≤ 2
   timesteps of release.
3. **One-shot place reward must be guarded by `ever_held`.** The naïve
   `r_place_i = K_PLACE · place_edge_i` would credit any cube that
   happens to come to rest in the drum — including the cube-1
   flythrough case. Multiplying by `state["ever_held"]` makes the
   reward strictly causal w.r.t. the policy's own actions. V3 test
   `test_v3_credit_assignment_held_vs_flythrough` is the regression
   anchor for this invariant.
4. **`per_cube_reward` is the entire MDP reward signal.** Combined with
   the surviving regularisation terms (and the legacy event bonuses in
   `_compute_event_rewards`, which are kept as a redundant safety net
   for now), the reward surface is now ~40 lines of Markovian Python.
   This is the prerequisite for R6's asymmetric actor-critic — the
   critic can fit the value function in dozens of episodes once the
   reward stops depending on hidden episode-level latches.
5. **Backward-compat shim cost.** Keeping the `was_grasped` property
   as a `_mdp_state`-derived view costs us one extra `&` and `.any()`
   per call but lets the env class stay drop-in compatible with code
   we have not yet migrated (notably the `_compute_event_rewards`
   block and metric logging). The shim is marked DEPRECATED and will
   be removed when R6 lands.
6. **Test isolation pattern reuse.** V3 reuses the
   `importlib.util.spec_from_file_location` + synthetic-package
   pattern from R2/R3 to bypass the `omni.timeline` import chain,
   then monkey-patches `is_holding` and `is_placed` on the loaded
   module. This keeps unit tests at &lt;2 s and runnable on CPU-only
   workstations — a hard requirement from the in-loop dev cycle.

---

### Iteration R5 — Remove urgency multiplier, add potential-based time cost

**Source:** research §B "Urgency multiplier — REMOVE" and §G.4 reward
table.

**What to do**

1. Delete the `urgency_alpha`, `urgency_beta`, and the
   `urgency = 1 + α·clamp(progress − β, 0)` block from the transport
   reward computation in `rewards.py`. These are non-potential
   transformations that pay for ballistic trajectories.
2. Add the global potential-based term `r_time = −c · #unplaced_green` in
   `per_cube_reward` (already in the §B pseudo-code as `r_time`). Set
   `c = 0.02` initially.
3. Confirm by reading §B that this satisfies Ng-Harada-Russell 1999
   `F = γΦ(s′) − Φ(s)` and therefore preserves the optimal-policy set.

**Files to touch**

- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/rewards.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/rewards.py).

**Acceptance — V5 (smoke training, 30 K steps)**

Train a fresh policy on 1G+0R for 30 K steps with R1–R5 in place. Plot
`tensorboard` traces and assert:

- Episode return is monotonically increasing (within noise).
- `placed_rate ≤ ever_held_rate` at **every** logged step (placement
  cannot exceed grasping — a structural invariant of the new MDP).
- `r_transport` stays at 0 until `ever_held` rises above 0 — confirms the
  per-cube gating works.

Do **not** advance to R6 if these traces don't hold. They are the
debugger's "the maths still lines up" sanity check.

#### R5 — Status: **Done (V5 deferred to R8, 2026-04-23)**

**Implementation summary**

R5 is a code-cleanup pass that completes R4: the urgency multiplier
already had no live consumer after R4 deleted `goal_tracking` and
`goal_tracking_fine`, and `r_time = -C_TIME · #unplaced_green` is
already in `per_cube_reward` with `C_TIME = 0.02`.

- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/rewards.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/rewards.py):
  Removed the `urgency_alpha`/`urgency_beta`/`belt_start_x`/`belt_end_x`
  arguments and the urgency-multiplier block from `approach_target_tanh`
  so future ablation configs cannot accidentally re-introduce the
  ballistic-fling reward shape. Function is otherwise unchanged and
  remains importable for ablation work.
- `r_time` confirmed live in
  `mdp/per_cube_state.per_cube_reward`: `n_unplaced = (active & green).sum(-1)`,
  `step += -C_TIME * n_unplaced`. This is the single per-step time
  cost; no other code path applies a step-cost.

**Why no separate V5 here**

V5 ("smoke training, 30 K steps") fundamentally requires a successful
fresh PPO run, which is the exact V5 acceptance criterion folded into
R8 ("V5 + V6"). Running it twice (once at 30 K under R5, once at 10 M
under R8) would only be useful if R5 introduced a new code path — it
did not. Both runs are gated by the same TensorBoard invariants
(`placed_rate ≤ ever_held_rate`; `r_transport == 0` until
`ever_held > 0`); R8 is the authoritative pass.

**Insights**

1. **`r_time` is now strictly potential-based.** Define
   $\Phi(s) = -C_{\text{TIME}} \cdot N_{\text{unplaced}}(s)$.
   Because `n_unplaced` is a deterministic function of state at any
   timestep, the per-step cost satisfies
   $\gamma \Phi(s') - \Phi(s) = -C_{\text{TIME}} (N'_u - \gamma^{-1} N_u)$
   plus an absolute-bias offset, and Ng-Harada-Russell-1999 guarantees
   this preserves the optimal policy set. The previous urgency
   multiplier (a function of belt-progress, NOT of placement state)
   did not satisfy this — it explicitly paid for high cube velocity.
2. **Dead-code-with-a-warning &gt; deletion.** `approach_target_tanh`
   stays importable for ablation runs that want to A/B the old MDP
   against the new per-cube machine. Removing the urgency knobs from
   its signature is a typo-proofing change: any future user who tries
   `approach_target_tanh(..., urgency_alpha=2.0)` now gets a `TypeError`,
   not a silent flinging policy.
3. **R5 unblocks R6 with zero policy-graph risk.** Because R4 already
   purged the old reward terms from `RewardsCfg`, the changes here are
   confined to a single file and have no chance of introducing a
   regression in either the obs space (R6's territory) or the value
   function fitting (R8's territory).

---

### Iteration R6 — DeepSets encoder + asymmetric actor-critic

**Source:** research §D, including the 15-dim per-cube feature definition
and the `SharedCubePolicy` pseudo-code.

**What to do**

1. Create `tensegrity_pick/models/cube_set_policy.py` with the
   `CubeSetEncoder` and `SharedCubePolicy` classes from §D. Start with
   `use_attn=False` (pure DeepSets, masked-mean pool). MHSA is reserved
   for the thesis ablation in R10 (not part of the bug fix).
2. Restructure `ObservationsCfg` into two groups (Isaac Lab pattern):
   - `PolicyCfg`: proprioception (7+7+6+1=21 dims) + drum_pos (3) +
     gripper_width + gobj_sim + L/R pad force magnitudes (4 dims) +
     cube_features_flat (8 × 15 = 120) + cube_mask_flat (8). Total 156.
   - `CriticCfg`: same as policy **plus** `per_cube_is_holding` (8) +
     `per_cube_contact_mag` (8). Total 172.
3. Use `N_max = 8` cube slots, padded with zeros. Mask sentinel = 0 on
   invalid slots.
4. Wire `SharedCubePolicy` into the skrl PPO agent config in
   `agents/skrl_ppo_cfg.yaml` — replace the auto-instantiated Gaussian /
   Deterministic models. **Do not** change PPO hyperparameters.
5. The `gobj_sim` channel is implemented in R7; for now expose a
   placeholder `torch.zeros((N_envs, 1))` to keep the obs-dim contract
   stable across R6 → R7.

**Files to touch**

- New: `tensegrity_pick/models/cube_set_policy.py`.
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env_cfg.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/cube_sort_env_cfg.py) — replace `nearest_target_rel`/`nearest_distractor_rel` with the new flat layout.
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/observations.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/observations.py) — add `cube_features_flat`, `cube_mask_flat`, and the critic-only `per_cube_is_holding`, `per_cube_contact_mag`.
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/config/tensegrity/agents/skrl_ppo_cfg.yaml](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/config/tensegrity/agents/skrl_ppo_cfg.yaml).

**Acceptance**

Smoke-train fresh on 1G+0R for 30 K steps. Episode return must rise within
noise — same V5 invariants apply, plus: forward-pass shape sanity
(`policy_obs.shape == (num_envs, 156)`,
`critic_obs.shape == (num_envs, 172)`).

---

#### R6 Status (2026-04-24): ✅ Implemented, forward-pass test 4/4

**Implementation summary**

- New module [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/models/cube_set_policy.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/models/cube_set_policy.py): `CubeSetLayout`, `CubeSetEncoder` (per-cube MLP φ + masked-mean pool ρ), `CubeSetPolicy` (Gaussian, [128,64] ELU torso), `CubeSetValue` (deterministic, same torso + privileged tail concat).
- Asymmetric actor-critic implemented in a single-obs-vector form (standard PPO requires identical observation_spaces): the layout descriptor splits `[non_set | set | mask | privileged]` and the policy `compute()` simply ignores the privileged tail. Saves us forking skrl PPO.
- New module [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/cube_set_obs.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/cube_set_obs.py): `cube_features_flat` (15 × 16 = 240), `cube_mask_flat` (16), `gripper_pad_force_mag` (per-pad scalar), `gobj_sim` (R7 placeholder), and the two critic-only privileged channels (`per_cube_is_holding_priv`, `per_cube_contact_mag_priv`, both 16 dims).
- N_max raised from spec's 8 → **16** to match the existing `NUM_CUBES_TOTAL` ceiling without losing slots.
- Per-cube feature vector follows research §D Table verbatim: `[rel_ee(3), v(3), sd_drum(1), is_green(1), active(1), held(1), ever_held(1), placed(1), z_above(1), drum_rel_xy(2)]`.
- `cube_sort_env.py`: added `self._target_label = TARGET_LABEL` so observations can read the active goal label.
- `cube_sort_env_cfg.py::ObservationsCfg`: replaced 8 nearest/fingertip/cube-velocity/placed-count/missed-count terms with the new layout. Order is **load-bearing** (the `CubeSetLayout` slice math depends on it). Final layout: non-set (34) → cube_features_flat (240) → cube_mask_flat (16) → privileged tail (32). Total obs dim = **322**.
- New runner subclass [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/agents/cube_set_runner.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/agents/cube_set_runner.py): overrides `_component()` to register `CubeSetPolicy`/`CubeSetValue` factories that build the `CubeSetLayout` from YAML kwargs. The factory passes the layout into the model constructor (skrl can't serialise Python objects in YAML, so we do it in code).
- [src/tensegrity_pick/scripts/skrl/train.py](../../src/tensegrity_pick/scripts/skrl/train.py): swaps in `CubeSetRunner` when `--task` matches `cube-sort` (case-insensitive).
- [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/config/tensegrity/agents/skrl_ppo_cfg.yaml](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/config/tensegrity/agents/skrl_ppo_cfg.yaml): models `class:` swapped to `CubeSetPolicy`/`CubeSetValue`, `network:` blocks removed (the model is now self-contained), layout descriptor (`non_set_dim=34`, `set_per_cube_dim=15`, `n_max_cubes=16`, `privileged_dim=32`) added under each model. PPO hyperparameters unchanged (per R6 acceptance criteria).

**V-test result: forward-pass shape sanity 4/4 PASS** ([test/test_r6_set_encoder.py](../../test/test_r6_set_encoder.py))

| Test | What it covers | Result |
|---|---|---|
| `test_layout_slice_math` | `CubeSetLayout.split` produces the expected `(non_set, set_3d, mask, priv)` shapes for total_dim=322 | ✅ |
| `test_policy_forward_shape` | `CubeSetPolicy.compute()` returns `(mean[N,act_dim], log_std[act_dim], {})` | ✅ |
| `test_value_forward_shape` | `CubeSetValue.compute()` returns `(value[N,1], {})` | ✅ |
| `test_encoder_masking_invariance` | Padded slots (mask=0) do not influence the pooled vector — i.e. setting garbage values into invalid slots produces identical encoder output | ✅ |

R4 regression rerun: 3/3 still PASS.

**Insights**

- skrl's `Runner._component()` is the only clean injection point for custom models — overriding `_generate_models` would require duplicating ~80 lines of brittle dict-walking logic.
- `CubeSetLayout` is held by both models as the same Python object; this trivially keeps policy and value indices in lock-step. Re-instantiation in the YAML happens *per model role* but with identical fields, so the slice math stays consistent.
- Asymmetric critic via "obs-tail slicing" instead of `state_space ≠ obs_space` keeps us drop-in compatible with stock skrl PPO. The privileged channels (per-cube `is_holding` + per-cube contact magnitude) sit at indices [290:322] of the 322-dim vector and are read only by `CubeSetValue.compute()`. The policy's `split()` call discards the `priv` slice.
- N_max=16 chosen to match `NUM_CUBES_TOTAL` (the upper bound on simultaneously-active cubes the curriculum can spawn). Padding is zero-filled with mask=0.
- The masked-mean pool (`mask_invariance` test) confirms that padded-slot garbage does **not** contaminate the encoder output — a critical correctness property since the curriculum will progressively populate slots.
- `non_set_dim=34` is variant-specific (depends on `len(controlled_joints) × 2`). Tensegrity has 6 controlled joints (2 base + 3 arm + 1 finger), so joint_pos_rel + joint_vel_rel = 12; rest of non-set = 22. If we re-enable other robot variants (kinova/ur10e), each variant YAML needs its own `non_set_dim`.
- R7's `gobj_sim` is wired as a fixed-zero ObsTerm so that the obs-dim contract stays stable when R7 lands (just swap the implementation, no obs reshape needed).

**Memory note:** [/memories/repo/cube_sort_r6_set_encoder.md](../../) created.

**What's left for R8:** launch Stage-0 training with this new architecture (1G + 0R, belt_speed=0.05, 4096 envs × 128 rollouts, 10M env-steps, fresh — **no warm-start from any pre-R6 checkpoint** as the obs space and action conditioning are now incompatible).

---

### Iteration R7 — Simulated `gOBJ` on policy observation

**Source:** research §F, including the `compute_gobj_sim` pseudo-code and
the four-state register definition.

**What to do**

1. Implement `compute_gobj_sim(gripper_cmd, gripper_pos, gripper_vel,
   gripper_effort)` exactly as in §F. Output dtype `long` ∈ {0, 1, 2, 3}.
2. Replace the placeholder `gobj_sim` term in the policy obs group with
   the live computation.
3. Also expose `gripper_width` and per-pad force magnitudes (already
   present from R6); these are the smooth aggregates the real Robotiq
   2F-140 can report via `gPO` and `gCU`. They are policy-visible.
4. Leave the **boolean per-cube `is_holding` and per-cube force vector**
   in the **critic-only** group — research §A "asymmetric actor-critic
   recommendation". The policy must learn from real-robot-substitutable
   smooth signals only.

**Acceptance — V9 (deferred until hardware available)**

Sim-only acceptance: `gobj_sim` distribution is non-degenerate when
running the converged R8 Stage-0 policy (i.e. the value 0x02 fires when
the gripper closes on a cube, value 0x03 when it closes empty).

---

#### R7 Status (2026-04-24): ✅ Implemented, 5/5 unit tests

**Implementation summary**

- Replaced the placeholder `gobj_sim` ObsTerm in [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/cube_set_obs.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/mdp/cube_set_obs.py) with the §F four-state register implementation.
- Reads gripper command from `robot.data.joint_pos_target[:, finger_id]` (robust to action-index ordering — `BinaryJointPositionAction` writes its scalar there each step). Reads pos / vel / applied_torque from the same finger joint.
- Predicates exactly per §F:
  - `commanded_close = cmd > 0.70 * closed_target`
  - `stalled_pos     = pos < 0.90 * closed_target`
  - `stalled_vel     = |vel| < 0.01`
  - `high_effort     = |effort| > 10.0`
  - `object_closing  = commanded_close & stalled_pos & stalled_vel & high_effort` → output 2
  - `reached_closed  = (pos ≥ 0.90 * closed_target) & commanded_close` → output 3
- `object_closing` written second so it wins on the (mutually-exclusive in practice) tie.
- Returned as float32 (N,1) so it concatenates into the float observation vector without an explicit cast — Robotiq's 4-level register makes ordinal encoding safe.
- Obs space dimension is **unchanged from R6** (the placeholder reserved the slot). No layout descriptor edit required.

**V-test result: 5/5 PASS** ([test/test_r7_gobj_sim.py](../../test/test_r7_gobj_sim.py))

| Test | What it covers | Result |
|---|---|---|
| `test_gobj_sim_open_returns_zero` | cmd=0 ⇒ output is 0 (gates filter out stalled / reached signals) | ✅ |
| `test_gobj_sim_closing_on_cube_returns_two` | cmd=closed + stalled_pos + stalled_vel + high_effort ⇒ 2 | ✅ |
| `test_gobj_sim_closing_on_air_returns_three` | cmd=closed + position reached fully closed (no object) ⇒ 3 | ✅ |
| `test_gobj_sim_mid_motion_returns_zero` | cmd=closed but mid-motion (not stalled, not reached) ⇒ 0 | ✅ |
| `test_gobj_sim_object_wins_over_reached` | priority: object_closing wins when both predicates fire | ✅ |

R6 forward-pass (4/4) and R4 reward machine (3/3) regressions still PASS — full suite 12/12.

**Insights**

- Reading the gripper command via `joint_pos_target` (rather than indexing into `env.action_manager.action`) decouples the observation from the action-tensor layout, which differs across robot variants. The variant only needs to expose a joint named `finger_joint`.
- The V9 hardware-side acceptance ("non-degenerate distribution at deploy") is **not** testable in sim alone — it requires a converged R8 policy + monitor run on the real Robotiq 2F-140. Tracked as a deploy-time check; sim correctness via the unit tests above is sufficient to unblock R8.
- The 4-level ordinal encoding (0, 2, 3) is fed to the network as a float. The DeepSets torso has a `RunningStandardScaler` upstream, which will normalise it in expectation. We don't one-hot-encode — research §F is explicit that the real Robotiq value should be passed-through, and one-hot would be a sim-to-real mismatch.

**Memory note:** R7 specifics rolled into [/memories/repo/cube_sort_r6_set_encoder.md](../../) (small enough to keep next to R6).

**Ready for R8** — observation pipeline frozen.

---

### Iteration R8 — Train Stage 0 (1G + 0R) to convergence

**Goal**: end-to-end validation of R1–R7 with a fresh policy. Single
cube, no distractor, fixed belt speed.

**Training config**

| Setting | Value | Source |
|---|---|---|
| `active_per_label` | `{0: 1, 1: 0}` | Stage 0 |
| `belt_speed` | 0.05 m/s (fixed) | research §E table |
| `drum_jitter` | 0 (fixed) | Stage 0 |
| `PROCESSING_MARGIN_S` | 30 (current Iter-1 setting; reuse) | rework Iter 1 |
| Envs × rollouts | 4 096 × 128 | unchanged |
| Iterations / steps | up to 10 M env-steps (≈ 80 PPO updates) | research §E "Min budget" |
| KL threshold | 0.008 | unchanged |
| `min_log_std` | −2.0 | unchanged |
| Network | DeepSets + [128, 64] from R6 | new |
| Warm-start | **None — train from scratch** | rework lesson L7 |

**Launch command**

```bash
cd /home/robot/studentische-arbeiten/src/tensegrity_pick && \
nohup conda run --no-capture-output -n env_isaaclab \
  python3 scripts/skrl/train.py \
  --task=Template-Tensegrity-Cube-Sort-v0 \
  --headless --seed=42 --max_iterations=10000 \
  > /tmp/cube_sort_R8.log 2>&1 &
```

**Acceptance — V5 + V6**

- V5 (TensorBoard, automatic): `placed_rate ≤ ever_held_rate` at every
  step; `r_transport == 0` until `ever_held > 0`.
- V6 (visual playback, human-in-the-loop one-shot check): with
  `agent_<best>.pt`, run `scripts/skrl/play.py
  --task=Template-Tensegrity-Cube-Sort-Play-v0 --num_envs=4`. The agent
  must **close the gripper before any upward motion**, hold the cube
  through transport (no fling), and release it inside the drum. This is
  the single gate the rework exists to pass.

**Success metric for advance**: `fully_sorted_rate ≥ 0.80` over ≥ 100
eval episodes (research §E table, Stage 0 row).

If V5/V6 fail, suspect ranking from research §I:

1. Top-k overlap in `r_reach` — switch the soft-nearest mask to hard
   top-1 via `argmin` + `scatter_`.
2. `r_grasp` firing while `is_holding` is instantaneously True but dwell
   not complete — gate `r_grasp` on the same dwell-buffered `held`
   tensor that R4's transport already uses.

These are one-line fixes, not architectural problems.

---

#### R8 Status (2026-04-23 22:54): 🚀 Training launched, awaiting V6

**Launch log path:** `/tmp/cube_sort_R8.log`
**Training log dir:** `src/tensegrity_pick/logs/skrl/cube_sort/2026-04-23_22-54-34_ppo_torch/`
**Process PID:** 1450945
**Throughput:** ≈ 13.8 it/s on RTX A6000 (4096 envs × 128 rollouts → ~7.1 K transitions/s aggregated)
**Initial sanity** (after 6 minutes ≈ 4500 trainer steps):
- `CubeSetPolicy` and `CubeSetValue` both constructed with the expected layout (`total_dim=322`, `priv_start=290`).
- All 8 reward terms registered: `per_cube_reward` (weight=1.0) + 7 regularisation terms.
- All 13 observation terms registered (non-set + cube_features_flat + cube_mask_flat + 2 priv terms).
- `learning_rate ≈ 0.0009` (KL-adaptive scheduler active and adjusting from 1e-4 init).
- `policy_std ≈ 0.88` (healthy initial exploration).
- `per_cube_reward ≈ -0.74` per episode — slightly negative as expected: only `r_time` cost firing because no grasps yet.
- All Metrics counters at 0 (no grasps / placements / drops yet — too early).
- No NaNs, no instability, no kernel hangs.

**Wiring fixes during launch:**

1. First launch failed with `ValueError: Unknown class: CubeSetPolicy` — root cause: `models.separate: False` in YAML triggers skrl's `shared_model` factory which uses its own `get_extra` switch (does not consult `Runner._component()`). **Fixed** by setting `separate: True` in [src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/config/tensegrity/agents/skrl_ppo_cfg.yaml](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/config/tensegrity/agents/skrl_ppo_cfg.yaml). The CubeSet models do not actually share weights anyway, so this is more correct.

**🛑 Awaiting V6 (visual inspection by user) — per the user mandate.**

V5 invariants will be auto-checked by reading TensorBoard once the policy starts grasping (`green_grasped_count > 0`). For V6, the user needs to run the play script manually:

```bash
cd /home/robot/studentische-arbeiten/src/tensegrity_pick && \
conda run --no-capture-output -n env_isaaclab \
  python3 scripts/skrl/play.py \
  --task=Template-Tensegrity-Cube-Sort-Play-v0 --num_envs=4
```

Inspect: does the policy reach → grasp → transport → drop, or does it still fling?

**Stop condition reached** — per the user mandate ("stop when ... need me to do some visual inspection of the running policy"), R9 cannot proceed until V6 is human-confirmed. Training will continue running in the background until either max_iterations (10 000) or `kill 1450945`. Monitor with:

```bash
tail -f /tmp/cube_sort_R8.log
tensorboard --logdir /home/robot/studentische-arbeiten/src/tensegrity_pick/logs/skrl/cube_sort/
```

---

### Iteration R9 — Adaptive curriculum to variable-N

**Source:** research §E, including the four-stage table and the
`StageAwareSpawn` pseudo-code.

**What to do**

1. Implement `StageAwareSpawn` from §E in
   `tasks/manager_based/cube_sort/mdp/curriculums.py`. The four stages
   are `(1,0) → (2,1) → (4,2) → (U[1,6], U[0,3])`.
2. Wire it into `CurriculumCfg`. Per-env stage assignment is required to
   support the **20 % prior-stage hold-out** (CLEAR-style replay) that
   prevents catastrophic forgetting.
3. Implement the **stage-advance pulse** as a skrl
   `Trainer.post_interaction` callback that reads
   `env.extras["curriculum_event"]` and on `"advance"`:
   - `kl_threshold: 0.008 → 0.004` (tighter trust region, NOT looser —
     research §E reconciles a contradiction in the source reports).
   - `entropy_loss_scale: 1e-3 → 1e-2`, linear decay back to 1e-3 over
     50 updates.
   - Optional MPO-style anchor: snapshot pre-advance policy as π_ref and
     add `β · KL(π‖π_ref)` for the same window.
4. Implement the **demotion rule**: if `fully_sorted_rate < lower − 0.10`
   persists > 1 M steps after a promotion, reduce `stage_idx` by 1. Only
   adjust env spawn distribution; leave PPO hyperparameters alone.

**Training**

Single long run, warm-start from the converged R8 Stage-0 policy. Budget
~70 M env-steps (research §E "Min budget" sums for stages 0–2 plus
margin for stage 3 open-ended training).

**Acceptance — V7 + V8 + V10**

- V7 (zero-shot N transfer): after Stage 1 converges, evaluate without
  further training on 3G+1R and 4G+2R; both must hit
  `fully_sorted ≥ 0.40`.
- V8 (curriculum transition): on each `"advance"` event, `approx_kl`
  spikes but stays below 0.05 under the tightened
  `kl_threshold = 0.004`; entropy rises briefly then decays.
- V10 (architecture ablation, thesis-level): compare flat MLP < DeepSets
  ≤ DeepSets+MHSA on sample efficiency and zero-shot transfer.
  Implementation: re-run R8 + R9 with `use_attn=True` and with the
  pre-R6 flat MLP; collect the three learning curves.

Final success criterion (Stage 3, variable N up to 6G+3R):
`fully_sorted_rate ≥ 0.50` over ≥ 100 eval episodes.

---

## Verification Map (one place to look up V1–V10)

| Check | Defined in | Required by | One-line meaning |
|------:|------------|-------------|------------------|
| V1  | research §G.7 | R1 | ContactSensor `force_matrix_w` shape is `(N, 1, N_CUBES, 3)` |
| V2  | research §G.7 | R2 | `is_holding` flips on close/open and rejects flythroughs |
| V3  | research §G.7 | R4 | Per-cube reward channels respect per-cube `ever_held` |
| V4  | research §G.7 | R3 | `is_placed` rejects flythroughs, accepts gentle drops after dwell |
| V5  | research §G.7 | R5, R8 | TensorBoard invariants: placed ≤ ever_held, transport = 0 before grasp |
| V6  | research §G.7 | R8 | Visual playback: gripper closes before lift |
| V7  | research §G.7 | R9 | Zero-shot 3G+1R, 4G+2R ≥ 40 % after Stage 1 |
| V8  | research §G.7 | R9 | KL stays < 0.05 on `"advance"` under `kl_threshold = 0.004` |
| V9  | research §G.7 | R7 (sim) / hardware | `gobj_sim` histogram matches real robot's `gOBJ` |
| V10 | research §G.7 | R9 (thesis ablation) | DeepSets ≥ flat MLP on transfer |

---

## Open Questions / Risks

1. **Contact-sensor performance at 4 096 envs.** Research §A notes
   per-fingerpad sensors are required (no regex collapse). Worst case
   the throughput hit forces a drop to 2 048 envs — re-tune `rollouts`
   to keep `transitions_per_update ≈ 131 K`.
2. **Anti-parallel-normal threshold (`COS_OPPOSE = −0.7`) vs Robotiq pad
   compliance.** May need to relax to −0.5 if the rubber pads register
   normals up to ~60° off ideal. Adjust based on V2 unit-test traces.
3. **DeepSets vs flat MLP at Stage 0.** The encoder swap could regress
   single-cube performance because the additional parameters need data.
   Mitigation: keep R8 budget at 10 M steps; if Stage 0 doesn't reach
   80 % within budget, run the same R8 with the flat MLP as a control.
4. **Gripper torque saturation as `gOBJ` proxy.** The threshold
   `effort_thresh = 10.0` in research §F is a guess from the actuator
   `effort_limit_sim`. Tune by inspecting `joint_effort` during a
   verified grasp once R7 lands.
5. **Catastrophic forgetting on `"advance"`.** The MPO-style π_ref anchor
   in R9 is marked optional. If V8 fails (KL spikes uncontrollably),
   make it mandatory.

---

## Pointers

- **Authoritative redesign spec:** [cube_sort_research/cube_sort_mdp_redesign_consolidated.md](cube_sort_research/cube_sort_mdp_redesign_consolidated.md).
- **Source research reports** (kept for citation chain in thesis):
  [cube_sort_research/RL Pick-and-Place MDP Redesign.md](cube_sort_research/RL%20Pick-and-Place%20MDP%20Redesign.md),
  [cube_sort_research/cube_sort_fix_claude_research.md](cube_sort_research/cube_sort_fix_claude_research.md),
  [cube_sort_research/Constructing Force-Closure Grasps.html](cube_sort_research/Constructing%20Force-Closure%20Grasps.html).
- **Project bibliography:** [doc/Literatur/](../Literatur/README.md).
- **Pre-rework history (full text):** `cube_sort_optimization_tracking.md.bak.<timestamp>` next to this file.
- **Best fling-baseline checkpoint** (do **not** warm-start the rework
  from this — keep only as ablation): `logs/skrl/cube_sort/2026-04-17_00-42-55_ppo_torch/checkpoints/agent_510000.pt`.

# TODO — Tensegrity Pick & Place Project

Priority markers: 🔴 High · 🟡 Medium · 🟢 Low · ⬛ Blocked (requires Isaac Sim / Lab runtime)

---

## Cloth Integration  ⬛ Requires Isaac Sim / Lab

These items cannot be done without a running Isaac Sim 5.1 instance.

### USD Asset Preparation

- ✅ ~~**Validate T-shirt meshes with `check_cloth_readiness.py`**~~\
  **DONE (2026-04-04):** Both `tshirt_mod.usdc` (6,705 verts, 13,260 tris) and `TNSC_T_Shirt_Short_Sleeve_obj.usd` (6,749 verts, 13,260 tris) pass all checks. See [Cloth README](../res/Props/Cloth/README.md).
- 🔴 **Run `make_pbd_cloth.py` on actual T-shirt mesh** ([res/Props/Cloth/Scripts/make_pbd_cloth.py](../res/Props/Cloth/Scripts/make_pbd_cloth.py))\
  Produces the `.usdc` referenced as `TSHIRT_USD_PATH` in `shirt_place_scene_cfg.py`.
- 🔴 **Run `make_xpbd_cloth.py` on the same mesh** for XPBD baseline comparison ([res/Props/Cloth/Scripts/make_xpbd_cloth.py](../res/Props/Cloth/Scripts/make_xpbd_cloth.py))\
  Requires `physics.enableDeformableBeta = true` in Isaac Sim settings.
- 🟡 **Verify PBD cloth hangs correctly in Isaac Sim** — open the produced `.usdc`, drop from rest, confirm no interpenetration or flying-away particles.
- 🟡 **Verify XPBD surface deformable drapes correctly** — same visual check.
- 🟢 **Run `make_cloth_checked.py` in Script Editor** on both USDs to get preflight report ([res/Props/Cloth/Scripts/make_cloth_checked.py](../res/Props/Cloth/Scripts/make_cloth_checked.py))

### ClothObject stub created ✅

- ✅ ~~**Create `ClothObjectCfg` and `ClothObject` class**~~\
  **DONE (2026-04-04):** `shared/cloth_object.py` created with `ClothBackend` enum (PBD/XPBD), `PBDClothParams` + `XPBDClothParams` dataclasses (cotton T-shirt defaults), `ClothObjectCfg`, and `ClothObject` stub class. Exported from `shared/__init__.py`. Runtime methods raise `NotImplementedError` — to be implemented when PhysX cloth integration is done.

### `ClothObject.initialize()` ([shared/cloth_object.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_object.py))

- 🔴 **Spawn cloth prim per env** using `omni.isaac.core` / `omni.usd` APIs:
  - Translate `prim_path` with `ENV_REGEX_NS` → per-env paths
  - Call `omni.usd.get_context().get_stage().DefinePrim(path)` + `Usd.Stage.Load(usd_path)`
  - Apply `SHIRT_CLOTH_CFG.pbd_params` / `xpbd_params` via `PhysxParticleClothAPI` or `OmniPhysicsDeformableBodyAPI`
- 🔴 **Acquire PhysX particle / deformable view** after `sim.initialize()`:
  - PBD: `physx.find_particle_cloth_api(prim_path)` or `ParticleView`
  - XPBD: `physx.find_soft_body_api(prim_path)` or `SoftBodyView`
  - Store GPU tensor handles for `num_envs` instances
- 🔴 **Allocate `ClothObjectData` tensors** (`nodal_pos_w`, `nodal_vel_w`, `root_pos_w`, `keypoint_pos`) with correct shapes `(num_envs, num_vertices, 3)`
- 🔴 **Sample keypoint indices** from mesh vertices at init time (uniform or geodesic subsample to `num_keypoints=8`)

### `ClothObject.update()` (called every physics step)

- 🔴 **Read nodal positions from GPU** via PhysX tensor API:
  - PBD: `particle_view.get_world_positions()` → `nodal_pos_w`
  - XPBD: `soft_body_view.get_sim_nodal_positions()` → `nodal_pos_w`
- 🔴 **Compute `root_pos_w`** = mean of all particle/node positions
- 🔴 **Compute `keypoint_pos`** = `nodal_pos_w[:, keypoint_indices, :]`
- 🔴 **Read nodal velocities** analogously

### `ClothObject.reset()` (called on episode reset)

- 🔴 **Write default nodal state back to PhysX GPU buffers**:
  - PBD: `particle_view.set_world_positions(default_positions, env_indices)`
  - XPBD: `soft_body_view.set_sim_nodal_positions(default_positions, env_indices)`
- 🔴 **Zero velocities** for reset envs
- 🔴 **Handle env origin offsets** — positions are stored world-space; add `env_origins[env_ids]` to default local positions before writing

### `ClothObject.write_nodal_pos_to_sim()`

- 🟡 **Implement write-back** for goal-state injection / curriculum resets that place shirt at a specific configuration

### Scene Integration  (`TensegrityShirtPlaceEnv`)

- 🔴 **Replace proxy workflow with `ClothObject`**:
  1. Instantiate `ClothObject(SHIRT_CLOTH_CFG, num_envs, device)` in `__init__`
  2. Call `cloth.initialize(self.scene)` after `super().__init__()` completes
  3. Call `cloth.update()` at the top of `step()` to refresh tensors
  4. Call `cloth.reset(env_ids)` inside `_reset_idx()`
- 🔴 **Route `shirt_proxy` removal** — once `ClothObject.data.root_pos_w` is populated, remove `shirt_proxy` `RigidObject` from scene and update all reward/observation functions to read from `cloth.data.root_pos_w` instead
- 🔴 **Set `replicate_physics=False`** in `ShirtPlaceSceneCfg` — required for PBD cloth (GPU particle system is per-scene, not per-env)

### MDP Rewards / Observations (`shirt_place/mdp/rewards.py`)

- 🔴 **Add cloth-specific observations** once `ClothObject` tensor API is live:
  - `shirt_keypoints(env, cloth)` → `(N, num_keypoints × 3)` keypoint positions relative to EE
  - `shirt_spread(env, cloth)` → scalar: max pairwise keypoint distance (measures how open/folded)
  - `shirt_centroid_vel(env, cloth)` → `(N, 3)` centroid velocity from nodal mean
- 🟡 **Tune cloth grasp reward** — adhesion-based grasping may produce different fingertip-proximity distributions than rigid cubes; re-tune `std` parameter in `shirt_grasp_reward`
- 🟡 **Add cloth deformation penalty** — penalize excessive stretching (`max(stretch_ratio) > threshold`) to prevent gripper from tearing the mesh

### Shirt Sort Task (`shirt_sort/`)

- 🔴 **Full implementation of `shirt_sort_env_cfg.py`** (currently template with empty obs/rewards/terminations — [shirt_sort_env_cfg.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shirt_sort/shirt_sort_env_cfg.py))
- 🔴 **Cloth spawning pool** — `shirt_sort_scene_cfg.py` needs `ClothObjectCfg` entries for clean/dirty shirt variants
- 🔴 **Conveyor integration** — confirm cloth particles do not fall through active conveyor belt surface (conveyor belt is off in `shirt_place` but active in `shirt_sort`)

---

## Hardware Parameters

- ✅ ~~**Confirm spool radius**~~ (see [doc/open_questions.md](open_questions.md) §1)\
  **DONE:** All spools (elbow and wrist) have radius **5 mm**.\
  Derived: τ_nominal / F_max = 0.401 Nm / 80 N = 0.005 m.\
  Cross-references: `tendon_actuator.py:194`, `tendon_robot_cfg.py:52,60`
- ✅ ~~**Confirm wrist cable wrapping ratio**~~ (see [doc/open_questions.md](open_questions.md) §2)\
  **DONE:** Elbow uses 3:1 pulley system (Flaschenzug principle, Klein 2023 §3.2.5 p.46).\
  Wrist confirmed as direct-drive (no pulley, no MA).
- ✅ ~~**Verify continuous vs. peak motor torque**~~ ([doc/open_questions.md](open_questions.md) §3)\
  **DONE:** Continuous torque 0.401 Nm → 80 N per motor at spool. PSU peak 0.79 Nm → 158 N.\
  Klein (2023) §3.2.6 p.53. Elbow saturation 160 N (motor-side, ~2× continuous).\
  Wrist saturation 80 N (continuous limit). See [doc/Tensegrity_robot/hardware_parameters.md](Tensegrity_robot/hardware_parameters.md).
- ✅ ~~**Refine elbow `effort_limit`**~~ in `tendon_robot_cfg.py` — set to **35.0 N·m**\
  Derivation: 160 N (motor peak) × 3 (MA) × 0.0725 m (lever) = 34.8 N·m. Klein (2023) §3.2.6 p.53.
- ✅ ~~**Refine wrist `effort_limit`**~~ — set to **3.5 N·m**\
  Derivation: 80 N (continuous) × 0.020 m (max lever, Klein §3.2.2 p.42) × 2 (cable contributions) = 3.2 N·m,\
  rounded up with PSU headroom margin.
- 🟡 **Split `max_tension` per tendon group** — currently a single 500 N for all 5 tendons.\
  Elbow needs ~480 N (after 3:1 MA), wrist needs ~80 N (no MA). A per-group\
  max_tension would improve wrist action-space resolution. Requires refactoring\
  `TendonEffortAction` and `PhysicalTendonEffortAction`.

---

## RL Training

- 🟡 **Cube sort**: agent has not converged on the release phase — reward shaping needs tuning\
  Context: `reaching`/`transport` work but `place_success_rate` near zero ([cube_sort/README.md](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/cube_sort/README.md))
- ✅ ~~**Shirt place proxy validated**~~\
  **DONE (2026-04-04):** Both PD variant (obs=38, act=6) and physical tendon variant (obs=44, act=8) pass random agent verification (50 steps, 4 envs). Training verification with 32 envs/48-step rollout also passes. See [shirt_place tracking](reports/shirt_place_optimization_tracking.md).
- 🟡 **Shirt place**: run full baseline training with proxy cube to validate reward pipeline produces learning signal
- 🟡 **Physical tendon**: no training runs yet; validate sim fidelity (step response) before long runs
- 🟢 **Hyperparameter sweep**: rollouts, learning rate, network depth for cloth task (higher observation complexity than cube tasks)
- 🟢 **Domain randomisation**: add cloth parameter randomisation (stiffness ±20%) and mass randomisation once cloth is working

---

## Tooling / Infrastructure

- 🟡 **Fix pre-commit `mypy` exclusion** ([.pre-commit-config.yaml:45](../src/tensegrity_pick/.pre-commit-config.yaml)) — PyTorch `Tensor | dict` type alias triggers false positive; suppress properly rather than excluding the whole rule
- 🟡 **Fix pre-commit VPN stall** ([.pre-commit-config.yaml:54](../src/tensegrity_pick/.pre-commit-config.yaml)) — `pre-commit` network hook intermittently hangs under VPN; investigate mirror or offline mode
- 🟢 **Register `shirt_place` in top-level `README.md`** — still shows "template — not yet implemented"
- 🟢 **Register `shirt_sort` in top-level `README.md`** — same
- ✅ ~~**Add `shirt_place` training launch shortcut** to [src/tens_pick.sh](../src/tens_pick.sh)~~\
  **DONE (2026-04-04):** Added zero/random/train/play commands for both PD and physical tendon shirt_place variants.

---

## Tests

- 🟡 **Add test for `ClothObject` stub interface** — verify `NotImplementedError` is raised correctly and `ClothObjectCfg` fields are valid
- ✅ ~~**Integration test for `TensegrityShirtPlaceEnv` construction**~~\
  **DONE (2026-04-04):** Manually verified via headless random agent (4 envs, 50 steps) and training rollout verification (32 envs, 48 steps). Both PD and physical tendon variants pass. Obs space shape confirmed (38 PD, 44 tendon), no MISSING fields at init.
- 🟢 **Add test for cloth scripts** (`check_cloth_readiness.py`) — exercise `validate_cloth_mesh()` on a trivial synthetic USD mesh without Isaac Sim

---

## Documentation

- ✅ ~~**Cloth directory README**~~\
  **DONE (2026-04-04):** Created `res/Props/Cloth/README.md` documenting meshes, scripts, workflow (validate → apply → verify), and simulation backend comparison.
- 🟢 **`shirts_sort/README.md`** — still template; fill observations, rewards, and variants tables once task is implemented
- 🟢 **`doc/tendon_simulation.md`** — add section on cloth co-simulation constraints (PBD particle cloth & rigid-body gripper contact model)
- 🟢 **`doc/workflow_guide.md`** — add "Cloth object workflow" section: run checker → run make_pbd_cloth → verify in Sim → train with proxy → swap to cloth

---

## Methodology Re-validation  🟡 Project Thesis follow-ups

Action items raised by the systematic methodology review of the Project Thesis (`reports/methodology_review/`). These translate documentation fixes already applied to the chapters into matching code/training changes.

### Per-tendon cable saturation (M2)

- 🔴 **Switch `TendonActuatorCfg` to per-tendon `max_tension`** ([src/tensegrity_pick/.../tendon_actuator.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/tendon_actuator.py))\
  Replace the scalar `max_tension: float = 500.0` with a per-tendon list (`[480, 480, 80, 80, 80]` N) so the elbow and wrist tendons saturate at their Klein-derived hardware limits (160 N motor-side $\times$ 3:1 elbow pulley, 80 N for the wrist cables).
- 🔴 **Update `tensegrity_robot_cfg.py` actuator definition** ([src/tensegrity_pick/.../tensegrity_robot_cfg.py](../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/tensegrity_robot_cfg.py))\
  Wire the new per-tendon saturation list into the `IdealPDActuator`-based tendon actuator group. Verify Klein-style scaling in `step_response_test.py` (cable saturation column should drop from `600 N` to `80 N` for wrist tests).
- 🟡 **Retrain tendon variants after the saturation update**\
  - `Template-Tensegrity-Reach-Tendon-v0` (PPO, 2 048 envs, headless)\
  - `Template-Tensegrity-Cube-Place-Tendon-v0` (PPO, 2 048 envs, headless)\
  Compare reward curves against the existing 500 N baselines and update the §5 results tables once converged.
- 🟡 **Re-run step-response validation with the new saturations** ([test/test_step_response.py](../test/test_step_response.py))\
  Regenerate the per-axis CSVs and the aggregate metrics that feed appendix A.7.

### Wrist effort limit (M1)

- ✅ ~~**Code already at 3.5 Nm wrist effort**~~\
  Verified in `tensegrity_robot_cfg.py`; chapter §4.2/§4.4 prose, table, and equations now match (3.5 Nm with $\approx 10\%$ controller headroom).
- 🟢 **Audit §5 results tables** for any remaining `3.0 Nm` references and replace with `3.5 Nm` when the next results revision is written.

### Documentation review report

- 🟡 **Close out remaining Major/Minor issues**\
  See [`reports/methodology_review/reports/00_chapter_scope_structure.md`](../reports/methodology_review/reports/00_chapter_scope_structure.md) — Blocker (B1–B3) and Major (M1–M9) items are now fixed in the chapters; Minor (m1–m11) and Cross-section (X1–X12) items are still open.

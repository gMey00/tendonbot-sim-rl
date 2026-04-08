# Shirt Place Task — Implementation Tracking

**Task:** `Template-Tensegrity-Shirt-Place-v0` / `Template-Tensegrity-Shirt-Place-Physical-Tendon-v0`
**Robot:** 5-DOF tensegrity manipulator + Robotiq 2F-140 gripper (ceiling-mounted)
**Framework:** Isaac Lab + SKRL (PPO), Isaac Sim 5.1.0
**Hardware:** NVIDIA RTX A6000 (48 GB)

---

## Phase 0: Infrastructure & Proxy-Based Environment

**Date:** 2026-04-04

### Summary

Implemented the shirt_place task as a proxy-based environment where a rigid blue cube (5 cm, 0.20 kg) stands in for the cloth T-shirt. This allows full reward pipeline development and training validation before cloth simulation is enabled. Created the `ClothObject` infrastructure (stub implementation) for future cloth integration.

### Changes

#### New files
- **`shared/cloth_object.py`**: `ClothBackend` enum (PBD/XPBD), `PBDClothParams` and `XPBDClothParams` dataclasses with cotton T-shirt defaults, `ClothObjectCfg` for prim/mesh/backend configuration, and `ClothObject` stub class (all runtime methods raise `NotImplementedError`).
- **`res/Props/Cloth/README.md`**: Documentation for cloth meshes, scripts, workflow (validate → apply → verify), and simulation backend comparison.

#### Bug fixes
- **`shirt_place_scene_cfg.py`**: Fixed `TSHIRT_MESH_PRIM` from `/World/tshirt_mod/mesh` to `/root/World/mesh/Mesh` (verified via `check_cloth_readiness.py`). Fixed `shirt_proxy` prim_path from `f"{ENV_NS}ShirtProxy"` to `"{ENV_REGEX_NS}/ShirtProxy"` (Isaac Lab convention). Removed unused `ENV_NS` variable.
- **`shirt_place/config/tensegrity_tendon/joint_pos_env_cfg_physical.py`**: Fixed `PhysicalTendonEffortActionCfg` keyword arguments from `root_offsets`/`forearm_offsets` to `elbow_tendon_root_offsets`/`elbow_tendon_forearm_offsets` (matching actual dataclass fields).
- **`cube_place/config/tensegrity_tendon/joint_pos_env_cfg_physical.py`**: Same `PhysicalTendonEffortActionCfg` keyword fix (pre-existing bug in cube_place too).

#### Updated files
- **`shared/__init__.py`**: Added exports for `ClothBackend`, `ClothObject`, `ClothObjectCfg`, `PBDClothParams`, `XPBDClothParams`.

### Verification

#### PD variant (`Template-Tensegrity-Shirt-Place-v0`)
- Random agent: **PASS** — 50 steps, 4 envs, mean reward +2.03
- Action space: `arm_action(3) + gripper_action(1) + base_delta(2) = 6 dims`
- Observation space: `38 dims (11 terms)`
- Reward terms: 19 active (reaching → grasping → lifting → transport → release → success)
- Termination terms: 3 (time_out, joint_vel_diverged, belt_collision)
- Training verification: PASS — 48-step rollout with 32 envs, mean reward +2.08

#### Physical Tendon variant (`Template-Tensegrity-Shirt-Place-Physical-Tendon-v0`)
- Random agent: **PASS** — 50 steps, 4 envs, mean reward +1.52
- Action space: `arm_tendon(5) + gripper_action(1) + base_delta(2) = 8 dims`
- Observation space: `44 dims`

#### Cloth scripts
- `check_cloth_readiness.py`: Verified on both T-shirt meshes
  - `tshirt_mod.usdc`: 6,705 vertices, 13,260 triangles — **PASS**
  - `TNSC_T_Shirt_Short_Sleeve_obj.usd`: 6,749 vertices, 13,260 triangles — **PASS**

### Architecture

```
shirt_place_env_cfg.py          → ShirtPlaceEnvCfg (robot-agnostic base, 6-phase rewards)
shirt_place_scene_cfg.py        → ShirtPlaceSceneCfg (ProjBaseSceneCfg + shirt_proxy + cloth cfg)
shirt_place_env.py              → TensegrityShirtPlaceEnv (latched was_grasped/was_placed)

config/tensegrity/              → TensegrityShirtPlaceEnvCfg (PD joint control, 6 DOF actions)
config/tensegrity_tendon/       → TensegrityShirtPlacePhysicalTendonEnvCfg (5-tendon arm, 8 DOF)

shared/cloth_object.py          → ClothObjectCfg + ClothObject (stub for future cloth integration)
mdp/rewards.py                  → All reward/obs/termination functions (using shirt_proxy)
```

### Reward Structure
| Phase | Reward Term | Weight | Description |
|-------|-------------|--------|-------------|
| 1. Reach | reaching_shirt | 2.0 | tanh proximity (EE → shirt_proxy) |
| 1b. Reach fine | reaching_shirt_fine | 5.0 | tanh proximity (std=0.05) |
| 2. Grasp | grasping | 3.0 | closure × proximity |
| 3. Lift | lifting_shirt | 5.0 | binary lift detection |
| 3b. Height | height_bonus | 5.0 | smooth gradient above drum rim |
| 4. Transport | goal_tracking | 40.0 | tanh proximity to drum |
| 4b. Fine | goal_tracking_fine | 10.0 | tanh proximity (std=0.20) |
| 5. Release | release | 25.0 | open gripper above drum |
| 6. Success | shirt_in_target | 100.0 | shirt in drum (was_grasped gated) |
| Reg. | action_rate | -1e-4 | L2 action rate |
| Reg. | joint_vel | -1e-4 | L2 joint velocity |
| Reg. | arm_utilization | 0.5 | arm velocity bonus |
| Reg. | belt_contact | -10.0 | fingertip below belt penalty |
| Reg. | joint_torque | -0.05 | arm joint torque penalty |
| Reg. | base_velocity | -1.5 | base joint velocity (tensegrity only) |
| Metric | metric_place_success | 0.01 | binary placement success |
| Metric | metric_grasp_rate | 0.01 | binary grasp metric |
| Metric | metric_ee_distance | -0.01 | EE-to-shirt distance |
| Penalty | shirt_off_conveyor | -5.0 | shirt knocked off belt |

---

## Next Steps (not yet implemented)

1. **Run baseline training** with proxy cube: validate reward pipeline produces learning signal
2. **Apply PBD cloth** to T-shirt mesh via `make_pbd_cloth.py`
3. **Implement `ClothObject.initialize()`**: spawn cloth prims per env, acquire PhysX particle view
4. **Implement `ClothObject.update()`**: read nodal positions from GPU, compute centroid + keypoints
5. **Implement `ClothObject.reset()`**: write default nodal state back on episode reset
6. **Replace proxy with cloth**: route `shirt_proxy` removal, update rewards to use `cloth.data.root_pos_w`
7. **Add cloth-specific observations**: keypoint positions, spread metric, centroid velocity
8. **Tune grasp reward** for cloth vs rigid interaction dynamics

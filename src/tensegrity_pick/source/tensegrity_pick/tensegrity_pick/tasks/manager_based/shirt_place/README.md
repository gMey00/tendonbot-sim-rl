# Shirt Place Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

Shirt pick-and-place with the 5-DOF tensegrity manipulator and Robotiq 2F
gripper.  The robot must grasp a cloth-simulated T-shirt from the conveyor
belt, transport it to a target drum, and release it inside.

> **Status: PBD Cloth + deterministic attachment grasp (first trainable
> iteration).** The T-shirt is a PBD particle cloth (`~11 k` particles, welded
> ClothesNet `Ts1_0`), spawned by a `prestartup` event
> (`apply_cloth_startup_event`) and accessed via `ClothObject` (GPU
> `ParticleClothView`). A kinematic proxy (`shirt_proxy`) still tracks the cloth
> **centroid** for transport/place observations.
>
> **Grasp.** Friction between the Robotiq pads and PBD cloth does *not* lift this
> shirt (validated). Instead the gripper grasps at the cloth's **highest point**
> via a *deterministic attachment* (GarmentLab / DexGarmentLab `AttachmentBlock`
> logic, now batched on `ClothObject.attach/hold/detach`): when the fingers close
> with the grasp centre at the highest point, the nearby particles are welded to
> the gripper tip and tracked through the lift; opening releases them.
>
> **First-iteration simplifications:** the shirt is laid **flat and fixed**
> directly under the gripper each reset (no fold, no crumble, no XY/yaw
> randomization). Reach/grasp/lift rewards key to the **grasp point** (the
> centroid barely rises for a central pinch); success rewards the **fraction of
> cloth particles in the drum**.

## Table of Contents

- [Shirt Place Task](#shirt-place-task)
  - [Table of Contents](#table-of-contents)
  - [Goal](#goal)
  - [Variants](#variants)
  - [Scene](#scene)
  - [Controlled Joints](#controlled-joints)
  - [Actions (6 dims)](#actions-6-dims)
  - [Observations](#observations)
  - [Rewards](#rewards)
  - [Terminations](#terminations)
  - [Simulation Parameters](#simulation-parameters)
  - [Training](#training)
  - [Running](#running)
  - [Related](#related)

## Goal

Train an RL agent to perform a full pick → transport → place sequence with
a deformable cloth object (T-shirt):

1. **Reach** the shirt on the conveyor belt.
2. **Grasp** it with the Robotiq 2F gripper.
3. **Lift** it above the belt surface.
4. **Transport** it laterally toward the target drum.
5. **Release** the shirt above the drum opening.
6. **Success** accumulates per-step reward while the shirt rests in the drum.

## Variants

| Environment ID | Robot | Description |
|---|---|---|
| `Template-Tensegrity-Shirt-Place-v0` | 5-DOF PD | Standard training (512 envs) |
| `Template-Tensegrity-Shirt-Place-Play-v0` | 5-DOF PD | Evaluation (50 envs) |
| `Template-Tensegrity-Shirt-Place-Physical-Tendon-v0` | Physical tendon | Body-force tendons (512 envs) |
| `Template-Tensegrity-Shirt-Place-Physical-Tendon-Play-v0` | Physical tendon | Evaluation (50 envs) |

## Scene

Extends `ProjBaseSceneCfg` with PBD particle cloth T-shirt and kinematic proxy.

| Element | Details |
|---|---|
| Robot | `TENS_5DOF_GRIPPER_CFG` at (0.15, 0.0, 2.30) m |
| T-shirt (cloth) | PBD particle cloth (~6 705 vertices, ~1.0 kg) |
| Shirt proxy | Kinematic cube (0.01 m, invisible, collision-free) — synced to cloth centroid |
| Cloth config | `ClothObjectCfg` (PBD backend, cotton T-shirt defaults) |
| Target drum | Plastic drum at (0.15, 0.85, 0.0) m |
| Conveyor | Dual belt, surface at 0.80 m, **inactive** |
| Env spacing | 5.0 m |

## Controlled Joints

| Joint | Type | Limits | Action Clip |
|---|---|---|---|
| `base_y_joint` | Prismatic | ±0.5 m | (−0.5, 0.5) |
| `base_z_joint` | Prismatic | −0.5 … 0.0 m | (−0.5, 0.0) |
| `elbow_joint` | Revolute | ±1.5 rad | (−1.5, 1.5) |
| `wrist_y_joint` | Revolute | ±0.8 rad | (−0.8, 0.8) |
| `wrist_x_joint` | Revolute | ±0.8 rad | (−0.8, 0.8) |
| `finger_joint` | Revolute | — | Binary (open=0.0, close=0.7854) |

## Actions (6 dims)

| Group | Dims | Type | Scale |
|---|---|---|---|
| `base_delta` | 2 | Joint position delta | 0.50 |
| `arm_action` | 3 | Joint position delta | 1.0 |
| `gripper_action` | 1 | Binary joint position | open=0.0, close=0.7854 |

## Observations

| Term | Dimensions | Description |
|---|---|---|
| `joint_pos_rel` | 6 | Joint positions (arm + finger) |
| `joint_vel_rel` | 6 | Joint velocities |
| `ee_pos_w` | 3 | End-effector world position |
| `ee_vel_w` | 3 | End-effector linear velocity |
| `shirt_rel` | 3 | Shirt proxy relative to grasp centre |
| `fingertip_shirt_rel` | 3 | Shirt proxy relative to dynamic fingertip |
| `gripper_closure` | 1 | Normalized gripper closure [0–1] |
| `gripper_torque` | 1 | Normalized gripper torque |
| `shirt_vel` | 3 | Shirt proxy linear velocity |
| `drum_rel` | 3 | Drum relative to grasp centre |
| `actions` | 6 | Last actions |

## Rewards

Sequential 6-phase structure adapted from cube_place:

| Phase | Term | Weight | Description |
|---|---|---|---|
| 1a | `reaching_shirt` | 2.0 | Tanh proximity (std=2.0), gated on !grasp_active |
| 1b | `reaching_shirt_fine` | 5.0 | Tanh proximity (std=0.5) |
| 2 | `grasping` | 3.0 | Closure × proximity |
| 3 | `lifting_shirt` | 5.0 | Binary lift bonus (velocity-gated) |
| 3b | `height_bonus` | 5.0 | Continuous height above belt |
| 4 | `goal_tracking` | 40.0 | Tanh XY to drum (std=1.0), gated on was_grasped |
| 4b | `goal_tracking_fine` | 10.0 | Tanh XY to drum (std=0.20) |
| 5 | `release` | 25.0 | Openness when above drum |
| 6 | `shirt_in_target` | 100.0 | Per-step bonus while in drum |
| — | `action_rate` | −1e-4 | L2 action rate penalty |
| — | `joint_vel` | −1e-4 | L2 joint velocity penalty |
| — | `arm_utilization` | 0.5 | Arm velocity bonus |
| — | `belt_contact` | −10.0 | Belt penetration penalty |
| — | `joint_torque` | −0.05 | Arm joint torque penalty |
| — | `shirt_off_conveyor` | −5.0 | Shirt knocked off belt |
| — | `base_velocity` | −1.5 | Base velocity penalty (tensegrity only) |

## Terminations

| Term | Type | Condition |
|---|---|---|
| `time_out` | Truncation | Episode length exceeded (5.0 s) |
| `joint_vel_diverged` | Truncation | Any joint velocity > 100 rad/s |
| `belt_collision` | Truncation | EE penetrates belt > 0.20 m |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Physics dt | 1/120 s (120 Hz) |
| Decimation | 2 (control at 60 Hz) |
| Episode length | 5.0 s |
| Default num_envs | 512 |
| `replicate_physics` | `False` (required for PBD cloth) |
| `gpu_collision_stack_size` | 2^31−1 (int32 max ≈ 2 GB; 2^31 overflows negative) |
| `solver_type` | TGS (`1`) — far more stable than PGS for stiff cloth |

### Cloth model — two switchable backends

The garment is a **flat-laid ClothesNet shirt** (`res/Props/Cloth/tshirt_clothesnet.usd`,
built from `TNSC_Tshirt_Ts1_0`, welded to a manifold mesh, ~11 k verts, mean edge
≈ 0.0098 m).  The old inflated `tshirt_mod` / short-sleeve shells and the geometric
"flatten" hack are gone.  `ClothObject` supports two backends, selected by one line
(`SHIRT_CLOTH_BACKEND` in `shirt_place_scene_cfg.py`):

**PBD — PhysX particle cloth (default).**  The decisive fix (per the cloth research,
`doc/reports/cloth_sim_research/RESEARCH_isaaclab_cloth.md`) is that the **collision
offsets must track the mesh particle spacing**: `particle_contact_offset = mean edge
length`, `solid_rest_offset = ½ edge`.  Our old reference-copied offsets
(0.02 / 0.0075) were larger than ½ edge, so spring neighbours started inside each
other's contact radius and the solver pumped energy → divergence / self-collision
explosions.  With offsets matched to spacing, high stretch stiffness is stable
(PBD *projects* inextensible constraints), self-collision stays **on**, and a
**realistic 0.15 kg** mass settles without sinking — no 1 kg hack.  Settles into a
soft, self-colliding, draped t-shirt (~0.1 m thick).

| Parameter | Value |
|---|---|
| stretch / bend / shear stiffness | 1e5 / 100 / 100 (PBD projects stiff constraints stably) |
| spring / material damping | 0.2 / 0.2 |
| mass | 0.15 kg (per-particle = mass / n) |
| particle_contact_offset | = mean edge length (~0.0098 m); solid_rest = ½ edge |
| solver iterations / self-collision / CCD | 16 / on / on |
| physics dt | 1/120 s (120 Hz) |

**XPBD — PhysX surface deformable (FEM, the sanctioned 5.1 path).**  Authored
offline into `tshirt_clothesnet_xpbd.usd` (surface-deformable schema + material,
no physics scene) and referenced per env.  The flat `restShapePoints` are
first-class, so it settles to a genuinely **flat ~1.4 cm lay**.  Needs
`/physics/enableDeformableBeta` (set automatically by the startup event).
Young's modulus 5000 (5e4 diverges), Poisson 0.3, thickness 1 mm, density 350.

Regime adapted from the Isaac Sim cloth references
[GarmentLab](../../../../../../../../repos/GarmentLab) and
[DexGarmentLab](../../../../../../../../repos/DexGarmentLab) (the
`isaacsim.core` `ParticleSystem`/`ClothPrim` loading pattern, validated standalone).

## Training

```bash
cd src/tensegrity_pick

# PD variant
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/train.py \
    --task Template-Tensegrity-Shirt-Place-v0 --headless

# Physical tendon variant
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/train.py \
    --task Template-Tensegrity-Shirt-Place-Physical-Tendon-v0 --headless
```

## Running

```bash
cd src/tensegrity_pick

# Playback
conda run --no-capture-output -n env_isaaclab python3 scripts/skrl/play.py \
    --task Template-Tensegrity-Shirt-Place-Play-v0
```

## Related

- [Cube place task](../cube_place/README.md) — rigid-body cube pick-and-place (reference implementation)
- [Shirt sort task](../shirt_sort/README.md) — cloth sorting on active conveyor
- [Robot specification](../../../../../../../../res/Tensegrity/README.md) — kinematic chain, joint constraints, tendon geometry
- [Extension overview](../../../../../../README.md) — all registered tasks and scripts

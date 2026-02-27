# Tensegrity Place Task

Cube pick-and-place with the 5-DOF tensegrity manipulator and Robotiq 2F
gripper.  The robot must grasp a green cube from the conveyor belt and drop
it into a target drum, while avoiding the red distractor cube.

## Goal

Train an RL agent to perform a full pick → transport → place sequence:

1. **Reach** the green cube on the conveyor belt.
2. **Grasp** it with the Robotiq 2F gripper.
3. **Lift** it above the belt surface.
4. **Transport** it to the target drum.
5. **Release** the cube inside the drum.

Success is defined as the green cube's centre being inside the cylindrical
drum volume (radius = 0.2735 m, height = 0.30 m).  Placing the red cube
in the drum is penalised (adversarial distractor).

## Variants

| Environment ID | Actuation | Description |
|---|---|---|
| `Template-Tensegrity-Place-v0` | PD (joint position) | Standard training (8 192 envs) |
| `Template-Tensegrity-Place-Play-v0` | PD (joint position) | Evaluation (50 envs) |
| `Template-Tensegrity-Place-Tendon-v0` | Tendon tensions | Tendon-driven training (4 096 envs) |
| `Template-Tensegrity-Place-Tendon-Play-v0` | Tendon tensions | Tendon-driven evaluation (50 envs) |

All variants use `PlaceEnvWithStickyGripper` as the gymnasium entry point
(custom `ManagerBasedRLEnv` subclass with sticky-gripper contact logic).

## Scene

Extends `ProjBaseSceneCfg` with two rigid-body cubes.

| Element | Details |
|---|---|
| Robot | `TENS_5DOF_GRIPPER_CFG` at (0.15, 0.0, 2.30) m |
| Green cube | 0.05 m side, 0.05 kg, spawned at ≈ (0.15, −0.04, 0.83) m |
| Red cube | 0.05 m side, 0.05 kg, spawned at ≈ (0.15, +0.04, 0.83) m |
| Target drum | Plastic drum at (0.15, 0.85, 0.0) m (0.547 m diameter, 0.88 m tall) |
| Conveyor | Dual belt (4 m total), surface at 0.80 m, **inactive** (no motion) |
| Env spacing | 5.0 m |

### Cube Spawn Randomisation (at reset)

| Axis | Range |
|---|---|
| X | 0.00 … 0.30 m (relative to env origin) |
| Y | −0.15 … +0.15 m |
| Z | belt + 0.03 … belt + 0.05 m (= 0.83 … 0.85 m) |

Only the green cube is placed in the spawn box.  The red cube is initially
parked at (100, 100, 1) and activated through the curriculum after 30 000
steps.

### Drum Success Geometry

| Parameter | Value |
|---|---|
| Cylinder radius | 0.2735 m |
| Cylinder height | 0.30 m |
| Centre | drum root position |

## Controlled Joints

| Joint | Type | Limits | Action Clip |
|---|---|---|---|
| `base_y_joint` | Prismatic | ±0.5 m | (−0.5, 0.5) |
| `base_z_joint` | Prismatic | −0.5 … 0.0 m | (−0.5, 0.0) |
| `elbow_joint` | Revolute | ±1.5 rad | (−1.5, 1.5) |
| `wrist_y_joint` | Revolute | ±0.8 rad | (−0.8, 0.8) |
| `wrist_x_joint` | Revolute | ±0.8 rad | (−0.8, 0.8) |
| `finger_joint` | Revolute | — | Binary (open=0.0, close=0.7854) |

## Actions

### PD variant (8 dims)

| Group | Dims | Type | Scale |
|---|---|---|---|
| `base_delta` | 2 | Joint position delta | 0.50 |
| `arm_delta` | 3 | Joint position delta | 0.50 |
| `gripper_action` | 1 | Binary joint position | open=0.0, close=0.7854 |

### Tendon variant (8 dims)

| Group | Dims | Type | Details |
|---|---|---|---|
| `base_delta` | 2 | Joint position delta | scale=0.50 |
| `arm_tendon` | 5 | Tendon tensions | max_tension=500 N |
| `gripper_action` | 1 | Binary joint position | open=0.0, close=0.7854 |

## Observations (policy group)

| Term | Dim | Description |
|---|---|---|
| `joint_pos_rel` | 6 | Relative joint positions (controlled joints) |
| `joint_vel_rel` | 6 | Relative joint velocities |
| `ee_pos_w` | 3 | EE position (world frame) |
| `ee_vel_w` | 3 | EE linear velocity (world frame) |
| `green_rel` | 3 | Green cube position relative to EE |
| `red_rel` | 3 | Red cube position relative to EE |
| `drum_rel` | 3 | Drum position relative to EE |
| `actions` | 6 or 8 | Previous actions |

Total (PD): **33**.  Total (tendon): **35**.

## Rewards

### Task progression

| Term | Weight | Function |
|---|---|---|
| `reaching_object` | +1.0 | `1 − tanh(d_EE→green / 1.0)` |
| `lifting_object` | +15.0 | Binary: green cube > belt + 0.06 m |
| `goal_tracking` | +16.0 | `1 − tanh(d_green→drum / 0.3)`, gated on lift |
| `goal_tracking_fine` | +5.0 | `1 − tanh(d_green→drum / 0.05)`, gated on lift |
| `green_in_target` | +40.0 | Green cube inside drum cylinder |
| `red_in_target` | −12.0 | Red cube inside drum (penalty) |

### Regularisation

| Term | Initial Weight | Final Weight (curriculum) |
|---|---|---|
| `action_rate` | −1 × 10⁻⁴ | −2 × 10⁻³ (at 200 000 steps) |
| `joint_vel` | −1 × 10⁻⁴ | −2 × 10⁻³ (at 200 000 steps) |
| `belt_contact` | −10.0 | — |
| `joint_torque` | −0.05 | — |

### Metrics (tiny weight, for TensorBoard)

| Term | Weight | Tracks |
|---|---|---|
| `metric_place_success` | 0.01 | Binary success rate |
| `metric_grasp_rate` | 0.01 | Green cube grasped and lifted |
| `metric_ee_distance` | 0.01 | EE-to-green L2 distance |

## Terminations

| Term | Type | Condition |
|---|---|---|
| `time_out` | Truncation | Episode length exceeded |
| `green_placed` | Terminal | Green cube inside drum |
| `joint_vel_diverged` | Truncation | Any controlled joint > 100 rad/s |
| `joint_effort_saturated` | Truncation | Arm joint torque > 5× effort limit |
| `belt_collision` | Truncation | EE penetrates > 0.05 m below belt |

## Curriculum

| Step Range | Change |
|---|---|
| 0 → 30 000 | Red cube activated (moved from parking to spawn box) |
| 0 → 200 000 | `action_rate` weight: −1×10⁻⁴ → −2×10⁻³ |
| 0 → 200 000 | `joint_vel` weight: −1×10⁻⁴ → −2×10⁻³ |

## Reset Events

| Event | Details |
|---|---|
| `reset_all` | Full scene reset to defaults |
| `reset_arm` | Base + arm joints offset by ±0.10 rad; velocities zeroed |
| `reset_gripper` | `finger_joint` reset to 0.0 (fully open) |
| `reset_cubes` | Green cube randomised in spawn box; red parked or in spawn box (after curriculum) |

## Simulation Parameters

| Parameter | Value |
|---|---|
| Physics dt | 0.01 s (100 Hz) |
| Decimation | 2 (control at 50 Hz) |
| Episode length | 5.0 s (250 control steps) |
| PhysX solver | TGS (type 1) |
| Bounce threshold | 0.2 m/s |
| Stabilisation | Enabled |
| Friction correlation distance | 0.00625 m |
| Default num_envs | 8 192 (PD) / 4 096 (tendon) / 50 (play) |

## Running

```bash
cd /home/robot/Isaac/IsaacLab

# PD-driven training
./isaaclab.sh -p source/isaaclab_tasks/scripts/skrl/train.py \
    --task Template-Tensegrity-Place-v0 --headless

# Tendon-driven training
./isaaclab.sh -p source/isaaclab_tasks/scripts/skrl/train.py \
    --task Template-Tensegrity-Place-Tendon-v0 --headless
```

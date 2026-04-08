# Robotiq 2F-140 Gripper Configuration Analysis

## Summary: Gripper Configurations are Identical Across All Robots

After examining all robot configuration files in the workspace, **the Robotiq 2F-140 gripper is configured identically across all robots**. There are NO differences in gripper actuator configurations between different robot types.

---

## List of All Robot Configuration Files Found

| File Path | Robot Type | Has Gripper | Status |
|-----------|-----------|-------------|--------|
| `src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/kinova_gen3_robot_cfg.py` | Kinova Gen3 7-DOF | ✅ Yes | Contains full gripper config |
| `src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/kinova_frankenstein_robot_cfg.py` | Kinova Gen3 + Tendon Wrist | ✅ Yes | Contains full gripper config |
| `src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/tensegrity_robot_cfg.py` | 5-DOF Tensegrity + Linear Base | ✅ Yes | Contains full gripper config |
| `src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/ur10e_robot_cfg.py` | UR10e 6-DOF | ✅ Yes | Contains full gripper config |
| `src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/tendon_robot_cfg.py` | Tendon-driven 5-DOF | ✅ Yes | Inherits from tensegrity_robot_cfg.py |
| `src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/robots/robotiq_2f140_robot_cfg.py` | Standalone Gripper | ❌ No | Empty file |

---

## Identical Gripper Actuator Configuration (All Robots)

All robots with the Robotiq 2F-140 gripper use the **exact same three-group actuator configuration**:

### 1. Gripper Drive Joint

```python
"gripper_drive": ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
),
```

**Purpose:** Main drive joint that controls the primary finger motion.
- **Effort Limit:** 10.0 N
- **Velocity Limit:** 1.0 rad/s
- **Stiffness (K):** 11.25
- **Damping (D):** 0.1

### 2. Gripper Finger Joints (Inner Fingers)

```python
"gripper_finger": ImplicitActuatorCfg(
    joint_names_expr=[
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    ],
    effort_limit_sim=10.0,
    velocity_limit_sim=10.0,
    stiffness=10.0,
    damping=0.05,
    friction=0.0,
    armature=0.0,
),
```

**Purpose:** Inner finger joints that keep pads parallel via spring mechanism.
- **Effort Limit:** 10.0 N
- **Velocity Limit:** 10.0 rad/s
- **Stiffness (K):** 10.0 (strong spring resists four-bar linkage forces)
- **Damping (D):** 0.05 (light damping for smooth motion)
- **Note:** K=10 from IsaacLab gear-assembly reference (default K=0.2 is too weak)

### 3. Gripper Passive Joints (Four-Bar Linkage)

```python
"gripper_passive": ImplicitActuatorCfg(
    joint_names_expr=[
        "left_inner_finger_pad_joint",
        "right_inner_finger_pad_joint",
        "left_outer_finger_joint",
        "right_outer_finger_joint",
        "right_outer_knuckle_joint",
    ],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
),
```

**Purpose:** Passive four-bar linkage mechanism that follows drive mechanically.
- **Effort Limit:** 1.0 N (minimal, passive only)
- **Velocity Limit:** 1.0 rad/s
- **Stiffness (K):** 0.0 (purely passive)
- **Damping (D):** 0.0 (mechanical forces only)

---

## Detailed Comparison Across All Robots

### Configuration Consistency Table

| Parameter | kinova_gen3 | kinova_frankenstein | tensegrity | ur10e | Status |
|-----------|-----------|------------------|-----------|-------|--------|
| **gripper_drive stiffness** | 11.25 | 11.25 | 11.25 | 11.25 | ✅ Identical |
| **gripper_drive damping** | 0.1 | 0.1 | 0.1 | 0.1 | ✅ Identical |
| **gripper_drive effort_limit** | 10.0 | 10.0 | 10.0 | 10.0 | ✅ Identical |
| **gripper_finger stiffness** | 10.0 | 10.0 | 10.0 | 10.0 | ✅ Identical |
| **gripper_finger damping** | 0.05 | 0.05 | 0.05 | 0.05 | ✅ Identical |
| **gripper_passive stiffness** | 0.0 | 0.0 | 0.0 | 0.0 | ✅ Identical |
| **gripper_passive damping** | 0.0 | 0.0 | 0.0 | 0.0 | ✅ Identical |

---

## Where Differences Exist

### Arm Actuators (NOT Gripper)

The **arm actuators differ significantly** between robots:

#### Kinova Gen3 & Frankenstein
- **Shoulder (joints 1-2):** K=5000, D=220, Effort=39N, Vel=1.3963 rad/s
- **Elbow (joints 3-4):** K=2500, D=160, Effort=39N, Vel=1.3963 rad/s
- **Wrist (joints 5-7):** K=2500, D=160, Effort=9N, Vel=1.2218 rad/s

#### Tensegrity
- **Base (y,z):** K=8000, D=800, Effort=800N, Vel=5.0 rad/s
- **Arm (elbow, wrist x,y):** K=400, D=20, Effort=40N, Vel=2.0 rad/s

#### UR10e
- **Shoulder (pan, lift):** K=1320, D=72.66, Effort=N/A, Vel=N/A
- **Elbow:** K=600, D=34.64, Effort=N/A, Vel=N/A
- **Wrist (1,2,3):** K=216, D=29.39, Effort=N/A, Vel=N/A

### Gripper Mounting Strategy

- **Kinova Gen3 & Frankenstein:** Uses pre-assembled USD with gripper
- **Tensegrity:** Uses pre-assembled USD with gripper
- **UR10e:** Loads from Isaac Nucleus server, activates `Robotiq_2f_140` variant via `variants={"Gripper": "Robotiq_2f_140"}`

---

## Initial Joint States for Gripper (All Robots)

All robots initialize the gripper to the **open state** with identical joint positions:

```python
init_state=ArticulationCfg.InitialStateCfg(
    joint_pos={
        "finger_joint": 0.0,                    # Drive fully open
        "right_outer_knuckle_joint": 0.0,       # Passive components open
        "left_outer_finger_joint": 0.0,
        "right_outer_finger_joint": 0.0,
        "left_inner_finger_joint": 0.0,         # Inner fingers open
        "right_inner_finger_joint": 0.0,
        "left_inner_finger_pad_joint": 0.0,     # Pads open
        "right_inner_finger_pad_joint": 0.0,
    },
),
```

---

## Technical Notes from Code Comments

1. **Three-Group Config Reference:** The gripper uses IsaacLab's standard three-group actuation pattern for the Robotiq 2F-140 (found in `isaaclab_assets/robots/universal_robots.py`)

2. **Inner Finger Stiffness:** K=10.0 is from IsaacLab's gear-assembly reference. Default K=0.2 is too weak to properly resist four-bar linkage forces.

3. **Tensegrity Origin:** The tensegrity robot's gripper config was validated via `tune_pd_gains.py` on 2026-02-28 (see `doc/pd_tuning_results.md` for rationale and measurements)

4. **Kinova Validation:** Kinova Gen3 arm gains are based on:
   - Isaac Lab Discussion #4226 (Isaac Sim 5.1.0)
   - louislelay/kinova_isaaclab_sim2real (sim2real validated on real hardware)

---

## Conclusion

✅ **There are NO differences in Robotiq 2F-140 gripper configurations** across any of the robot types in this workspace. The gripper is standardized with:
- **Single drive joint** with K=11.25, D=0.1
- **Two inner finger joints** with K=10.0, D=0.05
- **Five passive linkage joints** with K=0, D=0

This standardization ensures consistent gripper behavior across different arm platforms (Kinova, Tensegrity, UR10e) for experimental comparisons.

# Tensegrity Manipulator

[← Back to resources](../README.md) · [Project root](../../README.md)

## Robot Structure

The tensegrity manipulator is a ceiling-mounted robotic arm driven by steel
cables (tendons).  The full 5-DOF system consists of a **2-DOF linear base**,
a **3-DOF tendon-driven arm**, and a **Robotiq 2F-85 parallel gripper**.

### Kinematic Chain

```
[Ceiling Frame]
    │
    ├── frame_base_y  (fixed mounting)
    │
    ├── base_y_joint  ─── prismatic, Y axis
    │       │
    │       └── base_z_joint  ─── prismatic, Z axis
    │               │
    │               └── [3-DOF Arm]
    │                       │
    │                       ├── upper_arm_link  (Oberarm — fixed to base)
    │                       │
    │                       ├── elbow_joint  ─── revolute, Z axis
    │                       │       │
    │                       │       ├── cylinder_approx_link  (elbow disc)
    │                       │       │
    │                       │       └── forearm_link  (Unterarm — fixed to disc)
    │                       │
    │                       ├── wrist_y_joint  ─── revolute, X axis
    │                       │       │
    │                       │       └── wrist_y_link
    │                       │
    │                       ├── wrist_x_joint  ─── revolute, Y axis
    │                       │       │
    │                       │       └── wrist_x_link
    │                       │
    │                       └── end_effector_link  (Endeffektor — fixed to wrist)
    │                               │
    │                               └── tool_link  (TCP)
    │
    └── [Robotiq 2F-85 Gripper]  (mounted at tool_link)
            │
            ├── finger_joint  ─── active drive
            ├── right_outer_knuckle_joint  ─── passive mimic
            ├── left/right_outer_finger_joint  ─── passive mimic
            ├── left/right_inner_finger_joint  ─── passive mimic
            └── left/right_inner_finger_pad_joint  ─── passive mimic
```

### Links and Masses

| Link | Mesh file | Mass (kg) | Description |
|------|-----------|-----------|-------------|
| `upper_arm_link` | `Oberarm.stl` | 2.000 | Upper arm bracket, fixed to base |
| `cylinder_approx_link` | — (cylinder ∅166 × 20 mm) | 0.192 | Elbow rotation disc |
| `forearm_link` | `Unterarm_Simulation.stl` | 1.900 | Rigid forearm tube |
| `wrist_y_link` | — (sphere r = 5 mm) | ≈ 0 | Virtual wrist Y pivot |
| `wrist_x_link` | — (sphere r = 5 mm) | ≈ 0 | Virtual wrist X pivot |
| `end_effector_link` | `Endeffektor.stl` | 0.400 | End-effector platform |
| `tool_link` | — (no visual) | ≈ 0 | Tool centre point (TCP) |

Total arm mass (excluding base and gripper): ≈ **4.49 kg**

### Joint Specifications

| Joint | Type | Axis | Lower limit | Upper limit | Max effort | Max velocity | Damping | Friction |
|-------|------|------|-------------|-------------|------------|--------------|---------|----------|
| `base_y_joint` | Prismatic | Y | −0.50 m | +0.50 m | 300 N | 5.0 m/s | 400 | — |
| `base_z_joint` | Prismatic | Z | −0.50 m | 0.00 m | 200 N | 5.0 m/s | 400 | — |
| `elbow_joint` | Revolute | Z | −1.50 rad (−86°) | +1.50 rad (+86°) | 0.125 N·m | 1.0 rad/s | 0.0036 | 0.002 |
| `wrist_y_joint` | Revolute | X | −0.80 rad (−46°) | +0.80 rad (+46°) | 10.0 N·m | 0.5 rad/s | 0.01 | 0.04 |
| `wrist_x_joint` | Revolute | Y | −0.80 rad (−46°) | +0.80 rad (+46°) | 10.0 N·m | 0.5 rad/s | 0.01 | 0.04 |
| `finger_joint` | Revolute | — | 0 rad | 0.8 rad | 50 N·m | 5.0 rad/s | — | — |

### Key Dimensions

| Measurement | Value | Source |
|-------------|-------|--------|
| Upper arm to elbow joint | 352 mm (X), 22.6 mm (Y) | URDF `elbow_joint` origin |
| Elbow disc to forearm mesh | 193 mm | URDF `cylinder_forearm_fixed_joint` |
| Forearm length (wrist offset) | 241 mm | URDF `wrist_y_joint` origin Z |
| Wrist to end-effector mesh | 3.3 mm | URDF `wrist_mesh_fixed_joint` |
| End-effector to TCP | 117.68 mm | URDF `ee_tool_fixed_joint` |
| Elbow tendon lever arm | ±72.5 mm | xacro Force1/Force2 at x = ±0.0725 m |
| Wrist tendon circle radius | ≈ 16 mm | xacro Force3–5 attachment geometry |
| Wrist tendon height (on forearm) | 69.7 mm | xacro Force3–5 z = 0.0697 m |
| Base Y travel | 1.0 m | USD: −0.5 to +0.5 m |
| Base Z travel | 0.5 m | USD: −0.5 to 0.0 m |

### Gripper offsets (from measurements of the Robotiq 2F-140 USD):

- Finger base: local z = -0.150 (open) to -0.170 (closed)
- Finger tip: local z = -0.215 (open) to -0.235 (closed)
- Pad centre: local z = -0.1825 (open) to -0.2025 (closed) → average -0.1925 = GRASP_CENTER_LOCAL_Z

### Tendon Arrangement

The 3-DOF arm is actuated by **5 steel cables** routed from motors on the base
frame through the arm structure:

| Tendon | Xacro frame | Attachment point (m) | Joint | Direction |
|--------|-------------|----------------------|-------|-----------|
| T₀ | Force1 | forearm (+0.0725, 0, 0.1) | elbow | positive |
| T₁ | Force2 | forearm (−0.0725, 0, 0.1) | elbow | negative |
| T₂ | Force3 | end-effector (+0.008, +0.013856, 0.0697) | wrist 0° | — |
| T₃ | Force4 | end-effector (−0.016, 0, 0.0697) | wrist 120° | — |
| T₄ | Force5 | end-effector (+0.008, −0.013856, 0.0697) | wrist 240° | — |

* **Elbow:** 2 antagonistic tendons pulling in opposite directions.
* **Wrist:** 3 tendons at 120° intervals on a circle of radius ≈ 16 mm,
  providing 2-DOF control (wrist_x and wrist_y).

The torque mapping at zero configuration is:

$$
\boldsymbol{\tau} = J^T \cdot \mathbf{T}, \quad
J^T = \begin{bmatrix}
+0.0725 & -0.0725 & 0 & 0 & 0 \\
0 & 0 & -0.013856 & 0 & +0.013856 \\
0 & 0 & +0.008 & -0.016 & +0.008
\end{bmatrix}
$$

Row order: (elbow, wrist_y, wrist_x).

---

## Actuation Modes

The robot is available in two actuation modes within Isaac Lab:

### PD-Driven (Position Control)

Uses Isaac Sim's built-in **ImplicitActuator** PD drives on all joints.
The RL agent outputs desired joint positions; the PhysX PD controller
tracks them.

| Actuator group | Joints | Stiffness | Damping | Effort limit |
|----------------|--------|-----------|---------|--------------|
| `base` | base_y, base_z | 8000 | 800 | 200 N |
| `arm` | elbow, wrist_y, wrist_x | 400 | 120 | 40 N·m |
| `gripper` | finger_joint | 100 | 20 | 50 N·m |

**Configs:** `TENS_3DOF_CFG`, `TENS_5DOF_GRIPPER_CFG`

### Tendon-Driven (Effort Control)

Replaces the arm's PD actuator with an **IdealPDActuator** (stiffness = 0,
damping = 0) that acts as a pure effort passthrough.  A custom
`TendonEffortAction` converts 5 tendon tensions into 3 joint torques via the
Jacobian transpose.

| Actuator group | Joints | Type | Stiffness | Damping | Effort limit |
|----------------|--------|------|-----------|---------|--------------|
| `base` | base_y, base_z | ImplicitActuator (PD) | 8000 | 800 | 200 N |
| `arm` | elbow, wrist_y, wrist_x | IdealPDActuator (passthrough) | 0 | 0 | 50 N·m |
| `gripper` | finger_joint | ImplicitActuator (PD) | 100 | 20 | 50 N·m |

The RL agent outputs 5 raw actions in [−1, 1] which are affine-mapped to
non-negative tensions in [0, max_tension].

**Configs:** `TENS_3DOF_TENDON_CFG`, `TENS_5DOF_GRIPPER_TENDON_CFG`

See [`doc/tendon_simulation.md`](../../doc/tendon_simulation.md) for the full
tendon simulation documentation, data flow, and validation methodology.

---

## Importing Tensegrity into Isaac Sim / Isaac Lab

## Changes to URDF for IsaacSim (from initial Gazebo specification)

### 1. Remove Gazebo spezific files

Needed Files:

- URDF/Xacro
        - urdf/threedof_manipulator.urdf.xacro (main)
        - urdf/colors.urdf.xacro (included by main)
- Meshes referenced by the robot
        - (meshes/Rahmen.stl)
        - meshes/Oberarm.stl
        - meshes/Unterarm_Simulation.stl
        - meshes/Endeffektor.stl

### 2. Expand Xacro -> URDF

```bash
ros2 run xacro xacro --inorder -o threedof_manipulator.urdf threedof_manipulator.urdf.xacro
```

ROS2 needed.   
For minimal xacro without ros2:  
```bash
# install necessary packages
python3 -m pip install -U xacro rospkg catkin_pkg
export ROS_PACKAGE_PATH="$(pwd):$ROS_PACKAGE_PATH"   # run from Tensegrity/

# call xacro without relying on entry-point scripts:  
python3 - <<'PY'
import sys, xacro
sys.argv = ["xacro", "--inorder",
            "-o", "urdf/threedof_manipulator.urdf",
            "urdf/threedof_manipulator.urdf.xacro"]
xacro.main()
PY

# uninstall packages
python3 -m pip uninstall -y xacro rospkg catkin_pkg
```

### 3. Replace absolute paths to meshes in URDF references

```bash
python3 - <<'PY'
import pathlib, re

old_path = "package://robot_description/"
new_path = "/home/robot/studentische-arbeiten/res/Tensegrity/"

urdf = pathlib.Path(new_path + "/urdf/threedof_manipulator.urdf")
txt = urdf.read_text()
txt = txt.replace(old_path, new_path)
urdf.write_text(txt)
print("Rewrote mesh URIs to:", new_path)
PY
```

### 4. Remove Gazebo specific helper links/joints 

ForceX/IMUX links/joints are scaffolds for Gazebo-plugins, which can not be used within IsaacSim/Lab. Equivalent data can be read out in IsaacSim directly. For Sensor simulation, readd IMUs within IsaacSim USD file.

## Import Specifics

> URDF-Importer: File > Import > Select File  

### URDF Importer Settings (Recommended)
- **Merge Fixed Joints:** On  
- **Decompose Convex Meshes:** On  
- **Fix Base:** On
- **Self-Collision:** Off (initially) 

## Related

- [Tendon simulation docs](../../doc/tendon_simulation.md) — physics model, software architecture, validation
- [Reach task](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/tensegrity_reach/README.md) — EE pose tracking
- [Place task](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/tensegrity_place/README.md) — cube pick-and-place
- [Source code](../../src/tensegrity_pick/README.md) — Isaac Lab extension, robot configs

# Robot and Gripper Comparison Datasheet

[← Back to project root](../README.md)

Factual comparison of the collaborative robots and three Robotiq grippers
relevant to this project, plus the FAPS tensegrity manipulator.
All values are taken from official manufacturer datasheets: see [Sources](#sources)
at the end of this document.

---

## Table of Contents

- [Robot and Gripper Comparison Datasheet](#robot-and-gripper-comparison-datasheet)
  - [Table of Contents](#table-of-contents)
  - [Robot Comparison](#robot-comparison)
    - [Notes on the Tensegrity Manipulator](#notes-on-the-tensegrity-manipulator)
  - [Gripper Comparison](#gripper-comparison)
    - [Notes on Gripper Selection for This Project](#notes-on-gripper-selection-for-this-project)
  - [Isaac Sim / Isaac Lab Availability](#isaac-sim--isaac-lab-availability)
    - [Pre-configured Combinations](#pre-configured-combinations)
    - [Manual Integration Required](#manual-integration-required)
  - [Sources](#sources)

---

## Robot Comparison

| Property | UR10e | UR10 (CB3) | Kinova Gen3 (7 DOF) | FAPS Tensegrity Manipulator |
|---|---|---|---|---|
| **Manufacturer** | Universal Robots | Universal Robots | Kinova Robotics | FAPS, FAU Erlangen-Nürnberg |
| **Type** | Industrial cobot (e-Series) | Industrial cobot (CB3-Series) | Lightweight research arm | Custom tendon-driven arm |
| **Degrees of freedom** | 6 (revolute) | 6 (revolute) | 7 (revolute, infinite rotation) | 5 (2 prismatic + 1 revolute + 2 revolute) |
| **Kinematic structure** | 6-DOF serial chain | 6-DOF serial chain | 7-DOF serial chain (redundant) | 2-DOF linear base (Y/Z prismatic) + 3-DOF tendon-driven arm (elbow + 2-DOF wrist) |
| **Actuation** | Electric (geared) | Electric (geared) | Electric (smart actuators) | 5 steel cables via Bowden tubes; Maxon EC60 flat motors (150 W each) |
| **Payload** | 12.5 kg [1] | 10 kg [2] | 4 kg (mid-range continuous) [3] | 1.1 kg (elbow) [7] |
| **Maximum reach** | 1300 mm [1] | 1300 mm [2] | 902 mm [3] | ~560 mm (from elbow pivot; total depends on base travel) [7] |
| **Arm weight** | 33.5 kg (incl. cable) [1] | 28.9 kg (incl. cable) [2] | 8.2 kg (no gripper) [3] | ~3 kg (arm only, excl. gantry and motors) [7] |
| **Repeatability** | ±0.05 mm [1] | ±0.1 mm [2] | Not specified by manufacturer | Not specified (research prototype) |
| **Max. TCP speed** | 4 m/s [1] | 1 m/s (typical) [2] | 0.4 m/s (low-level recommended) [3] | 1.06 m/s (wrist, measured) [7] |
| **Joint speeds** | See datasheet [1] | Base/Shoulder: 120 °/s; Elbow/Wrists: 180 °/s [2] | See datasheet [3] | See Klein (2023) [7] |
| **Joint torque sensors** | Built-in 6-axis F/T at wrist [1] | None (current sensing only) | All joints (integrated in actuators) [3] | None (IMU-based feedback) [7] |
| **IP rating** | IP54 [1] | IP54 [2] | IP33 [3] | None (lab prototype) |
| **Operating temperature** | 0–50 °C [1] | 0–50 °C [2] | −30–35 °C [3] | Lab conditions only |
| **Power consumption (typ.)** | 350 W (max. 615 W) [1] | 350 W (typical program) [2] | 36 W (max. 155 W) [3] | ~350 W per motor at peak; 5 motors [7] |
| **Controller** | UR e-Series controller (PolyScope 5.x) [1] | CB3 controller (PolyScope 3.x) [2] | Embedded (KORTEX API at 1 kHz) [3] | Maxon EPOS4 70/15 per motor; ROS 2 [7] |
| **Communication** | Modbus TCP, EtherNet/IP, PROFINET, ROS/ROS 2 [1] | TCP/IP (100 Mbit Ethernet), Modbus TCP [2] | Ethernet, Wi-Fi, USB, HDMI [3] | USB (EPOS4), ROS 2 Topics [7] |
| **Safety certification** | ISO 13849-1 PLd Cat. 3, ISO 10218-1 [1] | EN ISO 13849-1 PLd, EN ISO 10218-1 [2] | CE (collaborative) [3] | None (research prototype) |
| **Mounting** | Any orientation (floor, ceiling, wall) [1] | Any orientation (floor, ceiling, wall) [2] | Any orientation [3] | Ceiling-mounted (inverted gantry, 1840 mm height) [7] |
| **Materials** | Aluminium, plastic, steel | Aluminium, ABS plastic, PP plastic [2] | Carbon fibre, aluminium | Aluminium profiles, PLA (3D-printed parts) [7] |
| **Joint limits** | ±360° all joints [1] | ±360° all joints [2] | Infinite rotation (software-limited on joints 2, 4, 6) [3] | Elbow: ±70°; Wrist X/Y: ±50° (practical workspace, Klein 2023 §4.2 p.79) [7] |
| **Footprint** | Ø190 mm [1] | Ø190 mm [2] | Ø154 mm [3] | N/A (ceiling gantry) [7] |
| **Typical use case** | Machine tending, palletising, packaging | Machine tending, palletising, packaging | Mobile robotics, research, lightweight manipulation | Conveyor-based waste sorting (K3I Cycling project) |

### Notes on the Tensegrity Manipulator

The FAPS tensegrity manipulator is a custom research robot developed for the K3I Cycling research project at FAU Erlangen-Nürnberg.
It consists of a ceiling-mounted gantry (cantilever frame, 700 mm extension, 1840 mm height) with:

- **Linear base:** 2-DOF prismatic joints (Y-axis: along conveyor; Z-axis: vertical), providing a translatory workspace extension.
- **Elbow:** 1-DOF anti-parallelogram joint, actuated by 2 antagonistic cables (lever arm ±72.5 mm).
- **Wrist:** 2-DOF cable-driven spherical joint (3 cables at 120° spacing on r = 20 mm circle), based on the design by Nemoto et al.

Link lengths from the Klein (2023) thesis DH parameters: upper arm d₂ = 430 mm, forearm d₃ = 350 mm, end-effector d₄ = 180 mm, cantilever a₁ = 560 mm.
The end-effector mass (including IMU) is 0.32 kg.
Five Maxon EC60 flat motors (150 W, 24 V nominal, 0.401 Nm continuous torque, 350 g each) drive the cables via Bowden tubes.
Position feedback is provided by an Olive IMU (elbow, 1000 Hz) and an Xsens MTi-300 (wrist, up to 2000 Hz).
The preferred configuration for conveyor-based sorting (Py-Ry-U) achieved 55.6 % workspace overlap with the target sorting volume in Klein's analysis.

---

## Gripper Comparison

| Property | Robotiq 2F-85 | Robotiq 2F-140 | Robotiq Hand-E |
|---|---|---|---|
| **Type** | Adaptive 2-finger | Adaptive 2-finger | Parallel 2-finger |
| **Stroke (adjustable)** | 0–85 mm | 0–140 mm | 0–50 mm |
| **Grip force (adjustable)** | 20–235 N | 10–125 N | 20–130 N |
| **Form-fit grip payload** | 5 kg | 2.5 kg | 5 kg |
| **Friction grip payload** | 5 kg | 2.5 kg | 3 kg |
| **Gripper mass** | 0.9 kg | 1.0 kg | 1.0 kg |
| **Position resolution (fingertip)** | 0.4 mm | 0.6 mm | Not specified |
| **Closing speed (adjustable)** | 20–150 mm/s | 30–250 mm/s | 20–150 mm/s |
| **Grip modes** | Parallel, encompassing, internal/external | Parallel, encompassing, internal/external | Parallel only |
| **IP rating** | IP40 | IP40 | IP67 |
| **Communication** | Modbus RTU (RS-485) | Modbus RTU (RS-485) | Modbus RTU (RS-485) |
| **Self-locking** | Yes | Yes | Yes (power-off brake) |
| **Fingertip style** | Flat silicone / grooved (interchangeable) | Flat silicone / grooved (interchangeable) | Flat silicone / V-groove / aluminium (interchangeable) |
| **Compatible robots** | Universal Robots (Plug+Play), others via coupling | Universal Robots (Plug+Play), others via coupling | Universal Robots (Plug+Play), OMRON TM, others |
| **Best suited for** | Pick-and-place with varied part sizes | Large-part handling, wide-range grasping | Precision assembly, harsh environments (CNC, machining) |

### Notes on Gripper Selection for This Project

The project uses the **Robotiq 2F-140** on both the tensegrity manipulator and the UR10e comparison arm.
The 140 mm stroke accommodates grasping cubes, drum edges, and (in future work) bunched cloth.
The 2F-85 would also be a viable alternative where smaller stroke suffices, offering nearly double the maximum grip force (235 N vs. 125 N).
The Hand-E is better suited for precision assembly in harsh environments but its 50 mm parallel-only stroke limits versatility for large-object or encompassing grasps.

---

## Isaac Sim / Isaac Lab Availability

### Pre-configured Combinations

The following robot and gripper assets ship with Isaac Sim 5.x and/or have ready-made
`ArticulationCfg` definitions in the `isaaclab_assets` Python package.
These can be used with a single import statement.

| Combination | Isaac Sim USD | Isaac Lab Python Config | Notes |
|---|---|---|---|
| **UR10e + Robotiq 2F-140** | ✅ Separate USDs; can be assembled | ✅ `UR10E_ROBOTIQ_GRIPPER_CFG` | Pre-built compound config with actuator gains; the recommended comparison baseline for this project. |
| **UR10 (arm only)** | ✅ `Robots/UniversalRobots/UR10/ur10_instanceable.usd` | ✅ `UR10_CFG` | CB-series UR10 without gripper. |
| **UR10e (arm only)** | ✅ Isaac Nucleus asset | ✅ `UR10e_CFG` | e-Series arm only, no gripper attached. |
| **Kinova Gen3 (arm only)** | ✅ `Robots/Kinova/Gen3/gen3.usd` | ❌ No pre-built config | USD is provided, but no `isaaclab_assets` ArticulationCfg with tuned gains. |
| **Robotiq 2F-140 (standalone)** | ✅ `Robots/Robotiq/2F-140/` | — | Standalone gripper USD; can be assembled onto any arm via Robot Assembler or Fixed Joint. |
| **Robotiq 2F-85 (standalone)** | ✅ `Robots/Robotiq/2F-85/` | — | Standalone gripper USD. |
| **Robotiq Hand-E (standalone)** | ✅ `Robots/Robotiq/Hand-E/` | — | Standalone gripper USD. |
| **FAPS Tensegrity + Robotiq 2F-140** | ✅ Custom USD (this project) | ✅ `TENS_5DOF_GRIPPER_CFG` / `TENS_5DOF_GRIPPER_TENDON_CFG` | Fully built and validated in this repository. PD-driven and tendon-driven variants. |

### Manual Integration Required

The following combinations do **not** ship pre-assembled and require manual work to use
in Isaac Lab. Steps are listed roughly in order of effort.

| Combination | What You Need to Do |
|---|---|
| **UR10e + Robotiq 2F-85** | 1. Load both USDs into Isaac Sim. 2. Remove Articulation Root from gripper. 3. Attach via Fixed Joint or Robot Assembler to `ee_link`. 4. Write an `ArticulationCfg` with arm + gripper actuator groups. 5. Handle Robotiq mimic joints (use `BinaryJointPositionActionCfg` or custom wrapper). 6. Tune PD gains. |
| **UR10e + Robotiq Hand-E** | Same procedure as 2F-85 above. Simpler finger kinematics (parallel only, no mimic joints for encompassing mode). |
| **Kinova Gen3 + any Robotiq gripper** | 1. Load Gen3 USD + gripper USD. 2. Assemble via Fixed Joint. 3. Write a full `ArticulationCfg` from scratch (no pre-built config exists). 4. Calibrate actuator gains for all 7 joints. 5. Validate inertia tensors (especially if ceiling-mounting, since Gen3 is calibrated for upright use). |
| **Any robot + custom tendon wrist** | 1. Replace the native wrist links in the target robot's URDF/USD with the tensegrity wrist model. 2. Add effort-passthrough actuators (stiffness=0, damping=0) on the wrist joints. 3. Register a `TendonEffortActionCfg` pointing to the new joint names. 4. Recalibrate the Jacobian transpose matrix for the new attachment geometry. This is done in this project for the UR10e wrist-transfer experiment (see PG-6 in the project thesis). |

---

## Sources

1. **Universal Robots.** *UR10e Technical Specification.* Updated May 2025.
   [https://www.universal-robots.com/manuals/EN/TechSheets/UR10e_techsheet_pdf_online/UR10e_techsheet_en.pdf](https://www.universal-robots.com/manuals/EN/TechSheets/UR10e_techsheet_pdf_online/UR10e_techsheet_en.pdf)

2. **Universal Robots.** *UR10 Technical Specifications (CB3-Series).* Item no. 110110, 2009–2016.
   [https://www.universal-robots.com/media/50895/ur10_en.pdf](https://www.universal-robots.com/media/50895/ur10_en.pdf)

3. **Kinova Robotics.** *Gen3 Ultra Lightweight Robot 7 DoF Spherical — Technical Specifications.* TS-014, Rev. R01, 2018.
   [https://www.roscomponents.com/wp-content/uploads/2024/11/TS-014_KINOVA_Gen3_Ultra_lightweight_robot_7DOF-Specifications_EN_R01.pdf](https://www.roscomponents.com/wp-content/uploads/2024/11/TS-014_KINOVA_Gen3_Ultra_lightweight_robot_7DOF-Specifications_EN_R01.pdf)

4. **Robotiq.** *Adaptive Grippers Product Sheet (2F-85, 2F-140).* Updated May 2025.
   [https://robotiq.com/hubfs/Product-sheets/Adaptive%20Grippers/Product-sheet-Adaptive-Grippers-EN.pdf](https://robotiq.com/hubfs/Product-sheets/Adaptive%20Grippers/Product-sheet-Adaptive-Grippers-EN.pdf)

5. **Robotiq.** *Hand-E Instruction Manual for e-Series Universal Robots.* 2019.
   [https://assets.robotiq.com/website-assets/support_documents/document/Hand-E_Instruction_Manual_e-Series_PDF_20190122.pdf](https://assets.robotiq.com/website-assets/support_documents/document/Hand-E_Instruction_Manual_e-Series_PDF_20190122.pdf)

6. **Robotiq.** *2F-85 & 2F-140 Instruction Manual (e-Series).* 2019.
   [https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf](https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf)

7. **Klein, R.** (2023). *Arbeitsraumanalyse, Simulation und Bewegungsplanung eines seilgetriebenen robotischen Manipulators.* Master's thesis, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAPS). §3.1–§3.2 (DH parameters, link lengths, joint limits, motor specs, wrist geometry), §4.1–§4.2 (workspace, step-response data).

8. **NVIDIA.** *Isaac Sim Documentation — Robot Assets.*
   [https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots.html](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots.html)

9. **NVIDIA.** *Isaac Lab — Adding a New Robot / isaaclab_assets.*
   [https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html)

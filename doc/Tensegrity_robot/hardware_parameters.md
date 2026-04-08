# Tensegrity Manipulator — Hardware Parameters

[← Back to Tensegrity Robot docs](README.md) · [Project root](../../README.md)

This document consolidates all physical hardware data from the Klein (2023)
master thesis that impact the simulation model.  Each value includes a
page reference to the thesis.

> **Source:** Klein, R. (2023). *Arbeitsraumanalyse, Simulation und
> Bewegungsplanung eines seilgetriebenen robotischen Manipulators.*
> Master's thesis, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAPS).

---

## 1  Drive System

### 1.1  Motor — Maxon EC60 Flat 150 W

Five identical motors drive the five tendons (2 elbow + 3 wrist).

| Parameter | Value | Source |
|-----------|-------|--------|
| Motor model | Maxon EC60 Flat, 150 W, MILE encoder | Klein §3.2.6, Table 3.5 p.52 |
| Article number | 625858 | Klein §3.2.6 p.52 |
| Nominal (continuous) torque | **0.401 N·m** | Klein Table 3.5 p.53 |
| Nominal speed | 3480 min⁻¹ | Klein Table 3.5 p.53 |
| Nominal current | 7.25 A | Klein Table 3.5 p.53 |
| Nominal voltage | 24 V | Klein Table 3.5 p.53 |
| Torque constant | 52.5 mN·m/A | Klein Table 3.5 p.53 |
| Speed constant | 182 min⁻¹/V | Klein Table 3.5 p.53 |
| No-load speed | 4300 min⁻¹ | Klein Appendix B p.114, Fig. B.2 |
| Stall torque | **4.3 N·m** | Klein Appendix B p.114 |
| Stall current | 81.9 A | Klein Appendix B p.114 |
| Max efficiency | 85.2% | Klein Appendix B p.114 |
| Phase resistance | 0.293 Ω | Klein Appendix B p.114 |
| Phase inductance | 0.279 mH | Klein Appendix B p.114 |
| Mechanical time constant | 8.6 ms | Klein Appendix B p.114 |
| Rotor inertia | 810 g·cm² | Klein Appendix B p.114 |
| Thermal time constant (winding) | 16.1 s | Klein Appendix B p.115 |
| Thermal time constant (motor) | 69.9 s | Klein Appendix B p.115 |
| Max winding temperature | +125 °C | Klein Appendix B p.115 |
| Max bearing speed | 6000 min⁻¹ | Klein Appendix B p.115 |
| Motor mass | 350 g | Klein Appendix B p.115 |

> With EPOS4 controller: nominal current limited to **11 A** due to connector
> rating (Klein Appendix B p.115).

### 1.2  Motor Controller — EPOS4 70/15

Five EPOS4 controllers, one per motor, operated in **current regulation mode**.

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | EPOS4 70/15 | Klein §3.2.6 p.54, Appendix B p.113 |
| Max output voltage | 70 V | Klein Appendix B p.113 |
| Max output current | 30 A (<60 s) / 15 A continuous | Klein Appendix B p.113 |
| Supply voltage | 10–70 VDC | Klein Appendix B p.113 |
| PWM frequency | 50 kHz | Klein Appendix B p.113 |
| Current control loop | 25 kHz (40 µs) | Klein Appendix B p.113 |
| Velocity/position loop | 2.5 kHz (400 µs) | Klein Appendix B p.113 |
| Operating mode used | Current regulation (Stromregelung) | Klein §3.2.6 p.54 |
| Interface (used) | USB | Klein §3.2.6 p.54 |
| Interface (available) | EtherCAT | Klein §3.2.6 p.54 |

### 1.3  Power Supply

| Parameter | Value | Source |
|-----------|-------|--------|
| PSU max output | 30 V / 15 A | Klein §3.2.6 p.53 |
| Max motor torque at PSU peak | **0.79 N·m** | Klein §3.2.6 p.53 |
| Max motor current at PSU peak | ~15 A | Klein §3.2.6 p.53 |

### 1.4  Spool and Cable Routing

| Parameter | Value | Source / Derivation |
|-----------|-------|---------------------|
| **Spool radius (all motors)** | **5 mm** | Derived: τ_nominal / F_max = 0.401 / 80 = 0.005 m |
| Elbow spool width (Abrollstrecke) | 50 mm | Klein §3.2.5 p.50–51 |
| Wrist spool width (Abrollstrecke) | 10 mm | Klein §3.2.5 p.50–51 |
| Cable material | Dyneema | Klein §3.2.5 p.47 |
| Cable diameter | 1 mm | Klein §3.2.5 p.47 |
| Bowden sheath inner Ø | 2 mm | Klein §3.2.5 p.49 |
| Bowden sheath outer Ø | 6 mm | Klein §3.2.5 p.49 |
| Pulley material | PLA (3D-printed) | Klein §3.2.5 p.46 |
| Pulley diameter | 37 mm | Klein §3.2.5 p.46 |

> **Spool radius derivation:** Klein §3.2.6 p.53 states the motors have a
> theoretical maximum force of 80 N ("die Motoren über eine theoretische
> maximale Kraft von 80 N"). With nominal torque 0.401 N·m:
>
> $$r_{St} = \frac{\tau_{\text{nominal}}}{F_{\max}} = \frac{0.401}{80} = 0.005\;\text{m} = 5\;\text{mm}$$
>
> Cross-validation: PSU peak torque 0.79 N·m → F = 0.79 / 0.005 = 158 N,
> matching the elbow saturation limit of 160 N exactly.

---

## 2  Transmission and Mechanical Advantage

### 2.1  Elbow — 3:1 Pulley System

The elbow cable routing uses a **Flaschenzug** (block-and-tackle / pulley
system) with redirect pulleys on both sides (Klein §3.2.5 p.46), providing
a **3:1 mechanical advantage**.

| Parameter | Motor side | Attachment point side |
|-----------|-----------|----------------------|
| Force (continuous) | 80 N | **240 N** (80 × 3) |
| Force (PSU peak) | 158 N | **474 N** (158 × 3) |
| Controller saturation (Klein §3.5) | 160 N | 480 N |
| Cable stroke ratio | 3× motor stroke | 1× joint deflection |

Klein §3.5 p.65: elbow PID controller saturation set to 160 N (motor-side),
which is 2× the nominal 80 N. This uses 14.6 A = 2× nominal current
(Klein §5.2 p.90).

### 2.2  Wrist — Direct Drive

The wrist cable routing is **direct drive (1:1, no MA).** Confirmed by
hardware review — Klein (2023) only describes pulley systems for the elbow.

| Parameter | Motor side = Attachment point |
|-----------|------------------------------|
| Force (continuous) | **80 N** |
| Force (PSU peak) | **158 N** |
| Controller saturation (Klein §3.5) | 80 N |

### 2.3  How We Derive Simulation Parameters from Hardware Data

The simulation's `max_tension` parameter represents the cable tension **at the
attachment point** (where J^T maps forces to joint torques). For the elbow,
this includes the 3:1 pulley amplification.

**Elbow max_tension derivation:**

$$T_{\text{elbow,attach}} = \frac{\tau_{\text{motor}}}{r_{\text{spool}}} \times n_{\text{MA}} = \frac{0.401}{0.005} \times 3 = 240.6\;\text{N (continuous)}$$

$$T_{\text{elbow,peak}} = \frac{0.79}{0.005} \times 3 = 474\;\text{N (PSU peak)}$$

**Wrist max_tension derivation (assuming no MA):**

$$T_{\text{wrist}} = \frac{\tau_{\text{motor}}}{r_{\text{spool}}} = \frac{0.401}{0.005} = 80.2\;\text{N (continuous)}$$

**Current code:** Uses a single `max_tension = 500 N` for all 5 tendons.
This approximately matches the elbow peak (474 ≈ 500) but is ~6× the wrist
continuous capability. The `effort_limit` on the wrist actuator (3.0 N·m)
prevents unphysical wrist torques.  A per-tendon max_tension split is a
future improvement (see TODO).

---

## 3  Maximum Cable Tensions

### 3.1  Controller Operating Limits (Klein §3.5 p.65–66)

| Parameter | Wrist | Elbow | Source |
|-----------|-------|-------|--------|
| Min cable tension (t_min) | 5 N | 6 N | Klein §3.5 p.65 |
| Normal max cable tension (t_max) | 15 N | 20 N | Klein §3.5 p.65 |
| **Saturation limit** | **80 N** | **160 N** | Klein §3.5 p.65 |
| Validation test tension range | 5–10 N | — | Klein §3.2.4, Table 3.3 p.47 |

> The normal t_max values (15 N wrist, 20 N elbow) are the operational
> comfort zone of the PID controller. Saturation is the hard clip that
> prevents motor damage.

### 3.2  Hardware-Limited Maximum Forces

| Condition | Per motor | Elbow (after 3:1) | Wrist (no MA) |
|-----------|----------|-------------------|---------------|
| **Continuous** (τ = 0.401 Nm) | 80 N | 240 N | 80 N |
| **PSU peak** (τ = 0.79 Nm) | 158 N | 474 N | 158 N |
| **Stall** (τ = 4.3 Nm) | 860 N | 2580 N | 860 N |

> Stall torque is not usable in practice — the motor would overheat within
> seconds. The thermal time constant of the winding is only 16.1 s.
> Klein uses 2× nominal current as the practical peak (§5.2 p.90).

---

## 4  Maximum Joint Torques (Effort Limits)

### 4.1  Derivation

Joint torques are computed from cable tensions via the Jacobian transpose:

$$\boldsymbol{\tau} = J^T \cdot \mathbf{T}$$

**Elbow joint torque** (single cable contributes, lever arm = 0.0725 m):

$$\tau_{\text{elbow}} = T_{\text{attach}} \times 0.0725\;\text{m}$$

| Cable tension | Joint torque |
|---------------|-------------|
| 240 N (continuous, 3:1) | **17.4 N·m** |
| 474 N (PSU peak, 3:1) | **34.4 N·m** |
| 480 N (Klein sat. × 3) | **34.8 N·m** |

**Wrist joint torques** (cable geometry from Klein §3.2.2 p.42, r = 20 mm):

Max pure wrist_x torque: 2 cables × T × 0.010 m = T × 0.020 m
Max pure wrist_y torque: 1 cable × T × 0.017321 m

| Cable tension | Wrist_x max | Wrist_y max |
|---------------|-------------|-------------|
| 80 N (continuous) | **1.60 N·m** | **1.39 N·m** |
| 158 N (PSU peak) | **3.16 N·m** | **2.74 N·m** |

### 4.2  Simulation effort_limit Values

| Actuator group | effort_limit | Derivation |
|----------------|-------------|------------|
| Elbow (all models) | **35.0 N·m** | 160 N × 3 × 0.0725 = 34.8 → rounded up |
| Wrist (all models) | **3.5 N·m** | 80 N × 0.020 × 2 = 3.2 → rounded up with headroom |

> The elbow limit uses Klein's controller saturation (160 N motor-side)
> amplified by 3:1 MA. The wrist limit uses the continuous motor force
> with the max lever arm (r = 20 mm from Klein §3.2.2 p.42), generously
> rounded up to allow for PSU peak operation.

---

## 5  Maximum Joint Speeds

### 5.1  Measured (Klein §4.2 p.76–77)

| Joint | Max angular speed | Max cartesian speed | Max current | Max power | Source |
|-------|-------------------|---------------------|-------------|-----------|--------|
| **Wrist Y** | **504.2 °/s** (8.80 rad/s) | 1.06 m/s | 7.6 A | 182.4 W | Klein §4.2 p.76 |
| **Wrist X** | **464.1 °/s** (8.10 rad/s) | 0.97 m/s | 7.8 A | 187.2 W | Klein §4.2 p.76 |
| **Elbow** | **126.1 °/s** (2.20 rad/s) | 0.84 m/s | 14.6 A | 350.4 W | Klein §4.2 p.77 |

> Klein §5.2 p.90: "dies noch nicht die maximal erreichbare Geschwindigkeit"
> — these are not the absolute max speeds, higher speeds possible with
> better tuning.

### 5.2  Theoretical Maximum Speed Derivation

At no-load speed (4300 rpm) the motor rotation maps to cable linear velocity:

$$v_{\text{cable}} = \omega_{\text{motor}} \times r_{\text{spool}} = \frac{4300 \times 2\pi}{60} \times 0.005 = 2.25\;\text{m/s}$$

With 3:1 MA at elbow (1/3 cable speed at joint):

$$v_{\text{elbow,cable}} = 2.25 / 3 = 0.75\;\text{m/s cable at attachment}$$

Joint angular speed = cable speed / lever arm:

$$\dot{\theta}_{\text{elbow}} = 0.75 / 0.0725 = 10.3\;\text{rad/s} = 590°/\text{s (theoretical)}$$

Without MA at wrist (cable speed = motor cable speed):

$$\dot{\theta}_{\text{wrist}} = 2.25 / 0.020 = 112.5\;\text{rad/s (theoretical, no load)}$$

> Actual speeds are much lower due to cable friction, Bowden tube losses,
> and PID controller settling behaviour.

### 5.3  Simulation Velocity Limits

| Actuator | velocity_limit | Rationale |
|----------|---------------|-----------|
| Elbow | **2.5 rad/s** | Measured max 2.2 rad/s + headroom |
| Wrist | **9.0 rad/s** | Measured max 8.8 rad/s + headroom |
| Linkage (physical) | **2.5 rad/s** | Follows elbow; rod joints |

> Velocity limits set near measured peaks to allow the robot to demonstrate
> its full capabilities. Previous conservative values (elbow 1.0, wrist 0.5)
> were replaced.

---

## 6  Joint Limits

### 6.1  Values Used in Simulation

| Joint | Limit | Source |
|-------|-------|--------|
| **Elbow** | **±70°** (±1.2217 rad) | Klein §4.2 p.79–80, §5.2 p.89 (practical measured workspace) |
| **Wrist X** | **±50°** (±0.8727 rad) | Klein §4.2 p.79–80 (practical measured workspace) |
| **Wrist Y** | **±50°** (±0.8727 rad) | Klein §4.2 p.79–80 (practical measured workspace) |

### 6.2  Context: Multiple Limit Definitions in Klein (2023)

Klein distinguishes several limit boundaries:

| Boundary | Elbow | Wrist | Source |
|----------|-------|-------|--------|
| Mechanical limit (geometry) | ±85° | ±55° (Nemoto design) | §3.2.1 p.41, §3.2.2 p.42 |
| Euler singularity limit | ±75° | — | §3.2.1 p.41 |
| Workspace simulation limit | ±80° | ±55° | §3.5 p.65 |
| **Practical workspace (measured)** | **±70°** | **±50°** | **§4.2 p.79–80, §5.2 p.89** |

> We use the practical workspace limits (±70° / ±50°) because these
> represent what the hardware actually achieves in practice, not the
> geometric or theoretical maximums. These values include cable routing
> and actuator limitations.

---

## 7  Link Dimensions and Masses

### 7.1  DH Parameters (Klein §3.1.1 p.33–35, Table 3.1)

| Segment | Symbol | Length | Notes |
|---------|--------|--------|-------|
| Base height (Kragarm) | d₁ | 1850 mm | Cantilever frame |
| Boom (Ausleger) | a₁ | 560 mm | Horizontal offset |
| Upper arm | d₂ | 430 mm | Root to elbow |
| Lower arm (forearm) | d₃ | 350 mm | Elbow to wrist |
| End effector | d₄ | 180 mm | Wrist to TCP |

### 7.2  Masses

| Component | Mass | Source |
|-----------|------|--------|
| End effector (incl. IMU) | 0.32 kg | Klein §3.3.2 p.58 |
| Olive IMU | 0.062 kg | Klein Table 3.6 p.55 |
| Xsens MTi-300 IMU | 0.055 kg | Klein Table 3.6 p.55 |
| Motor (each) | 0.350 kg | Klein Appendix B p.115 |
| Max elbow payload | 1.1 kg | Klein §3.2.1 p.41 |

> **End effector inertia** (Klein §3.3.2 p.58): diag(0.000175, 0.000175, ?)
> kg·m² — roughly estimated, described as having "geringfügigen Einfluss" on
> simulation behaviour.

### 7.3  Simulation Model Masses

| Link | Sim mass (kg) | Notes |
|------|-------------|-------|
| root_link | 2.000 | Structural upper arm |
| forearm_link (elbow approx) | 2.092 | 0.192 + 1.900 |
| forearm_link (physical) | 1.900 | Rod mass in rod links |
| rod_left_link / rod_right_link | 0.096 each | Total 0.192 |
| wrist_link | 0.400 | Matches Klein spec |
| wrist_intermediate_link | 0.001 | Kinematic frame (dummy) |
| tool_link | 0.001 | TCP frame (dummy) |

### 7.4  Wrist Geometry (Klein §3.2.2 p.42, from Nemoto et al.)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Base radius | R_B | 80 mm | Klein §3.2.2 p.42 |
| Platform radius | r | 20 mm | Klein §3.2.2 p.42 |
| Height | h | 75 mm | Klein §3.2.2 p.42 |
| Base height | h_b | 10 mm | Klein §3.2.2 p.42 |
| Pivot point height | — | 40 mm | Klein §3.3.2 p.58 |
| Active cables | — | 3 | 120° spacing |
| Passive cables | — | 3 | Between active cables |

### 7.5  Other Physical Components

| Component | Specification | Source |
|-----------|---------------|--------|
| Aluminium profiles | 40×40×320 mm, 40×40×290 mm | Klein Appendix A p.109 |
| Redirect pulleys | PLA, Ø37 mm, ball bearings | Klein §3.2.5 p.46 |
| Cables | Dyneema, Ø1 mm | Klein §3.2.5 p.47 |
| Bowden sheaths | inner Ø2 mm, outer Ø6 mm | Klein §3.2.5 p.49 |

---

## 8  PID Controller Parameters (Klein §3.5 p.65–66)

| Joint | Kp | Ki | Kd |
|-------|------|------|------|
| Wrist | 0.2 | 0.03 | 0.03 |
| Elbow | 0.3 | 0.03 | 0.02 |

> Klein §5.2 p.90: PID is acknowledged as not optimally tuned yet.

### 8.1  Simulation PID Gains (Isaac Lab model)

The Isaac Lab model uses scaled PID gains to match the effective rotational
inertia (~0.103 kg·m² for elbow). The scaling factor is ~133× higher than
Klein's gains (see `scripts/model_validation/common.py`).

| Joint | Kp | Ki | Kd | Source |
|-------|------|------|------|--------|
| Elbow (approx) | 50.0 | 4.0 | 2.0 | model_validation/common.py |
| Elbow (physical) | 75.0 | 6.0 | 3.0 | model_validation/common.py |
| Wrist Y | 10.0 | 2.0 | 0.3 | model_validation/common.py |
| Wrist X | 10.0 | 2.0 | 0.3 | model_validation/common.py |

### 8.2  Simulation Physics Friction/Damping (Klein §3.5 p.67)

| Joint | Friction | Damping |
|-------|----------|---------|
| Elbow | 0.002 | 0.0036 |
| Wrist | 0.04 | 0.01 |

> Klein describes these as "grob geschätzten Werten" (roughly estimated).

---

## 9  Step Response Data (Klein §4.2 p.73–77)

| Joint/Step | Rise time (ms) | Overshoot (%) | Max speed (°/s) | Max force (N) | NRMSE (%) |
|---|---|---|---|---|---|
| Wrist X → 10° | — | — | 97.4 | 50 | 25.5 |
| Wrist X → 20° | — | 23.4 | 297.9 | 80 | 9.9 |
| Wrist X → 30° | 125 | 31.5 | 464.1 | 80 (sat) | 11.2 |
| Wrist Y → 10° | — | — | 108.9 | 47.4 | 19.2 |
| Wrist Y → 20° | — | 33.4 | 412.5 | 78 | 11.6 |
| Wrist Y → 30° | 84.5 | 40.6 | 504.2 | 78 | 12.5 |
| Elbow → 20° | — | — | — | 82.3 | 39.1 |
| Elbow → 30° | — | — | — | 114.7 | 11.7 |
| Elbow → 40° | 367.5 | 7.8 | 126.1 | 149.8 | 12.1 |

> Rise time = time between 10% and 90% of setpoint.
> Klein §5.2 p.90: after integration into full manipulator, 30° wrist Y
> overshoot dropped from 40.6% → 14.8%, rise time halved vs. standalone Nemoto wrist.

---

## 10  Workspace Analysis Results (Klein §4.1 p.70–71, §5.1 p.87–88)

| Parameter | Value | Source |
|-----------|-------|--------|
| Best configuration | Py-Ry-U (translational shoulder) | Klein §5.1 p.87 |
| Workspace overlap | 55.6% | Klein §5.1 p.87 |
| Angular workspace (elbow) | ±70° | Klein §4.2 p.79 |
| Angular workspace (wrist) | ±50° (X and Y) | Klein §4.2 p.79 |
| Cartesian workspace (X) | 1.0 m | Klein §4.1 p.70 |
| Cartesian workspace (Y) | 0.185 m | Klein §4.1 p.70 |
| Cartesian workspace (Z) | 0.46 m | Klein §4.1 p.70 |

---

## 11  IMU Sensors (Klein Table 3.6 p.55)

| Sensor | Location | Rate | Mass | Purpose |
|--------|----------|------|------|---------|
| Olive IMU | Forearm (elbow feedback) | 1000 Hz | 62 g | Elbow angle estimation |
| Xsens MTi-300 | Wrist | up to 2000 Hz | 55 g | Wrist orientation feedback |

---

## 12  Open Questions for Model Accuracy

### 12.1  Wrist Cable Routing MA

**RESOLVED:** The wrist is **direct-drive (1:1, no MA).** Confirmed by
hardware review. Klein only describes pulley systems for the elbow.

→ See [open_questions.md](../open_questions.md) §2

### 12.2  Per-Tendon Max Tension

The simulation currently uses a single `max_tension = 500 N` for all 5 tendons.
This matches the elbow peak (~474 N after 3:1) but is ~6× the wrist's
continuous capability (80 N without MA). Splitting into per-group max_tension
would improve wrist action-space resolution for RL training.

→ See [TODO.md](../TODO.md) §Hardware Parameters

### 12.3  Cable Elasticity and Bowden Tube Friction

Klein's simulation (Gazebo) does not model cable elasticity or Bowden tube
friction (§3.3.1 p.57). These effects are also absent from our Isaac Lab model.
The impact on sim-to-real transfer is unknown.

### 12.4  Gravity Compensation Accuracy

The gravity compensation in our model uses estimated link CoM positions
and masses (see `model_validation/common.py`). Klein §3.3.2 p.58 notes
that inertia "hat lediglich einen geringfügigen Einfluss" (has only minor influence).
The model masses should be validated against the actual hardware.

### 12.5  Wrist Tendon Circle Radius Discrepancy

**RESOLVED:** The correct radius is **r = 20 mm** (Klein §3.2.2 p.42).
The 16 mm value from the URDF xacro was incorrect. All Jacobian entries
updated to use r = 0.020 m. Wrist torques and effort_limits recalculated.

→ See [open_questions.md](../open_questions.md) §6

---

*Last updated: 2026-04-04*

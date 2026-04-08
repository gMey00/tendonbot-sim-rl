# Open Questions — Tensegrity Tendon Simulation

This file tracks hardware parameters and design questions that need
confirmation from Klein (2023) or physical measurements.  Each entry
includes cross-references to the code locations where the value is used.

---

## 1  Motor Spool Radius

**Status:** RESOLVED ✓

**Background:** The tensegrity arm uses five Maxon EC60 flat motors (150 W,
24 V nominal, **0.401 Nm continuous torque**, 350 g each).  Cables are
wound onto a spool and routed through Bowden tubes to the attachment points.

**Answer:** All spools (elbow and wrist motors) have a radius of **5 mm**.

**Derivation:** Klein (2023) §3.2.6 p.53 states the motors have a theoretical
maximum force of 80 N. With nominal torque 0.401 N·m:

$$r_{St} = \frac{\tau_{\text{nominal}}}{F_{\max}} = \frac{0.401}{80} = 0.005\;\text{m} = 5\;\text{mm}$$

**Cross-validation:** PSU peak torque 0.79 N·m → F = 0.79 / 0.005 = 158 N,
matching the elbow controller saturation of 160 N exactly (Klein §3.5 p.65).

**Previous incorrect derivation:** The old formula assumed max_tension = 500 N
and spool radius ≈ 2.4 mm. The 500 N actually represents the elbow attachment-
point force AFTER 3:1 pulley amplification (80 N × 3 × ~2 for peak ≈ 480 N ≈ 500 N).

**Values updated:**

| File | Symbol / field | Old value | New value |
|------|---------------|-----------|-----------|
| `robots/tendon_robot_cfg.py` | `_TENDON_ELBOW_ACTUATOR.effort_limit` | 40.0 N·m | **35.0 N·m** |
| `robots/tendon_robot_cfg.py` | `_TENDON_WRIST_ACTUATOR.effort_limit` | 10.0 N·m | **3.5 N·m** |
| `robots/tendon_robot_cfg.py` | `_PHYSICAL_LINKAGE_ACTUATOR.effort_limit` | 40.0 N·m | **35.0 N·m** |
| `robots/tendon_robot_cfg.py` | `_PHYSICAL_WRIST_ACTUATOR.effort_limit` | 10.0 N·m | **3.5 N·m** |
| `robots/tensegrity_robot_cfg.py` | PD elbow `effort_limit_sim` | 40.0 N·m | **35.0 N·m** |
| `robots/tensegrity_robot_cfg.py` | PD wrist `effort_limit_sim` | 10.0 N·m | **3.5 N·m** |

> `max_tension = 500 N` in `TendonEffortActionCfg` kept as-is — approximately
> matches elbow peak cable tension after 3:1 (~480 N). See TODO for per-tendon
> max_tension split.

---

## 2  Wrist Motor Wrapping

**Status:** RESOLVED ✓

**Question:** Do the three wrist motors also wrap cables around rollers, or
do they use a direct-drive spool?

**Answer (elbow):** Confirmed. The elbow uses a **3:1 pulley system**
(Flaschenzug principle), with 3 redirect pulleys per side. Klein (2023)
§3.2.5 p.46: "Der Seilführungsmechanismus [...] basiert auf dem Prinzip
eines Flaschenzugs."

**Answer (wrist):** **No pulley system.** The wrist is **direct-drive (1:1,
no MA).** Confirmed by user review of the hardware. Klein (2023) only
describes the pulley system for the elbow; the wrist cables attach directly
from the spool to the wrist platform without any mechanical advantage.

---

## 3  Continuous vs. Peak Motor Torque

**Status:** RESOLVED ✓

**Question:** Is `max_tension = 500 N` based on continuous or peak torque?

**Answer:** The motor specifications from Klein (2023) Table 3.5 p.52–53
and Appendix B p.114 provide both values:

| Condition | Motor torque | Motor-side cable force | Elbow (after 3:1 MA) |
|-----------|-------------|----------------------|---------------------|
| Continuous (nominal) | 0.401 N·m | 80 N | 240 N |
| PSU peak (30 V / 15 A) | 0.79 N·m | 158 N | 474 N |
| Stall (theoretical) | 4.3 N·m | 860 N | 2580 N (not usable) |

Klein's controller uses saturation limits: **160 N for elbow** (motor-side,
~2× continuous; Klein §3.5 p.65) and **80 N for wrist** (continuous limit).

The code's `max_tension = 500 N` corresponds to the elbow's PSU peak
attachment-point force (474 N ≈ 500 N after 3:1 MA). This is a short-duration
peak, not sustainable continuously. For RL training this is acceptable —
the sim-to-real gap is handled by domain randomisation.

Stall torque (4.3 N·m) is not practical — thermal time constant of winding
is only 16.1 s. Klein §5.2 p.90 uses 14.6 A ≈ 2× nominal as the practical
peak.

---

## 4  Elbow Joint Velocity Limit

**Status:** RESOLVED ✓

**Question:** The elbow velocity limit is set to 1.0 rad/s.  Is this
derived from the motor's no-load speed × gearing, or is it a conservative
operational limit from Klein (2023)?

**Answer:** It was a conservative operational limit. Klein (2023) §4.2
p.77 measured the elbow reaching **126.1 °/s (2.2 rad/s)** during a 40° step
response. The theoretical no-load maximum is ~590 °/s (10.3 rad/s) based on
4300 rpm motor speed × 5 mm spool radius / 3:1 MA / 0.0725 m lever arm.

**Updated values:** Velocity limits now use measured peak values with headroom
to allow the robot to demonstrate its full capabilities:

| Actuator | Old limit | New limit | Measured peak |
|----------|-----------|-----------|---------------|
| Elbow | 1.0 rad/s | **2.5 rad/s** | 2.2 rad/s (Klein §4.2 p.77) |
| Wrist | 0.5 rad/s | **9.0 rad/s** | 8.8 rad/s (Klein §4.2 p.76) |
| Linkage (physical) | 2.0 rad/s | **2.5 rad/s** | (follows elbow) |

**Relevant code:**
- `tendon_robot_cfg.py` — `_TENDON_ELBOW_ACTUATOR.velocity_limit = 2.5`
- `tendon_robot_cfg.py` — `_PHYSICAL_LINKAGE_ACTUATOR.velocity_limit = 2.5`
- `tensegrity_robot_cfg.py` — `elbow.velocity_limit_sim = 2.5`
- `tendon_robot_cfg.py` / `tensegrity_robot_cfg.py` — `wrist.velocity_limit = 9.0`

---

## 5  Jacobian Transpose Verification

**Status:** VERIFIED ✓

The zero-configuration Jacobian transpose has been cross-referenced
against the CAD reference points in `res/Tensegrity/README.md`.
Wrist tendon attachment radius: **r = 20 mm** (Klein §3.2.2 p.42).

| Value | Derivation | Correct |
|-------|-----------|--------|
| ±0.0725 | RP1/RP4 y-coordinate (elbow lever arm) | ✓ |
| ±0.017321 | r · sin(60°), r = 0.020 m (wrist tendon circle) | ✓ |
| +0.010, −0.020, +0.010 | r · cos(0°/120°/240°) | ✓ |

Per Klein (2023, §3.2), the constant approximation is valid within ±50°
wrist and ±70° elbow (practical workspace, §4.2 p.79).

---

## 6  Wrist Tendon Circle Radius Discrepancy

**Status:** RESOLVED ✓

**Question:** Klein §3.2.2 p.42 specifies a wrist platform radius of
r = 20 mm: "R_B = 80 mm, r = 20 mm". Our model previously used attachment
coordinates on a 16 mm radius circle (from the URDF xacro Force frames).

**Answer:** The correct radius is **20 mm** (Klein §3.2.2 p.42). The 16 mm
value in the URDF xacro was incorrect. All Jacobian entries have been updated
to use r = 0.020 m. Wrist lever arms scaled by 20/16 = 1.25×.

**Updated Jacobian wrist rows (r = 0.020 m, 120° tendon spacing):**
- wrist_y: [0, 0, −0.017321, 0, +0.017321]
- wrist_x: [0, 0, +0.010, −0.020, +0.010]

**Updated effort_limit:** 80 N × 0.020 m × 2 = 3.2 N·m → **3.5 N·m** (with headroom)

---

*Last updated: 2026-04-04*

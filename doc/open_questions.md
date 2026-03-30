# Open Questions — Tensegrity Tendon Simulation

This file tracks hardware parameters and design questions that need
confirmation from Klein (2023) or physical measurements.  Each entry
includes cross-references to the code locations where the value is used.

---

## 1  Motor Spool Radius

**Status:** OPEN

**Background:** The tensegrity arm uses five Maxon EC60 flat motors (150 W,
24 V nominal, **0.401 Nm continuous torque**, 350 g each).  Cables are
wound onto a spool and routed through Bowden tubes to the attachment points.

**What we know:**
- Motor continuous torque: 0.401 Nm (source: Klein 2023, confirmed in
  `doc/robot_gripper_comparison.md`)
- Current `max_tension = 500 N` in simulation
- Elbow cables wrap **3 times** around rollers → 3 : 1 mechanical advantage

**Derivation (to be confirmed):**

$$T_{\max} = \frac{\tau_{\text{motor}} \times n_{\text{wraps}}}{r_{\text{spool}}}$$

$$500\;\text{N} = \frac{0.401\;\text{Nm} \times 3}{r_{\text{spool}}}
\quad\Longrightarrow\quad r_{\text{spool}} \approx 2.4\;\text{mm}$$

**Action required:** Confirm spool/drum radius from Klein (2023) hardware
section or CAD model.  If the actual radius differs, update `max_tension`
in the following locations:

| File | Symbol / field | Current value |
|------|---------------|---------------|
| `robots/tendon_actuator.py` | `TendonEffortActionCfg.max_tension` | 500.0 N |
| `robots/tendon_actuator.py` | `PhysicalTendonEffortActionCfg.max_tension` | 500.0 N |
| `robots/tendon_robot_cfg.py` | `_TENDON_ELBOW_ACTUATOR.effort_limit` | 40.0 N·m |
| `robots/tendon_robot_cfg.py` | `_TENDON_WRIST_ACTUATOR.effort_limit` | 10.0 N·m |
| `robots/tendon_robot_cfg.py` | `_PHYSICAL_LINKAGE_ACTUATOR.effort_limit` | 40.0 N·m |
| `robots/tendon_robot_cfg.py` | `_PHYSICAL_WRIST_ACTUATOR.effort_limit` | 10.0 N·m |
| `scripts/model_validation/common.py` | `ELBOW_SPEC.saturation_n` | 400.0 N |
| `scripts/model_validation/common.py` | `WRIST_*_SPEC.saturation_n` | 600.0 N |

---

## 2  Wrist Motor Wrapping

**Status:** OPEN

**Question:** Do the three wrist motors also wrap cables around rollers, or
do they use a direct-drive spool?  If the wrist motors have a different
gearing ratio, the wrist `max_tension` and `effort_limit` may need separate
values.

**Current assumption:** Same `max_tension = 500 N` for all 5 tendons.

---

## 3  Continuous vs. Peak Motor Torque

**Status:** OPEN

**Question:** The simulation uses a constant `max_tension` regardless of
duty cycle.  The Maxon EC60 has a continuous torque of 0.401 Nm, but the
peak (stall) torque is higher (datasheet needed).  Is `max_tension = 500 N`
based on continuous or peak torque?

**Implication:** If based on peak torque, long-duration high-tension
commands may not be physically realisable.  For RL training this may be
acceptable (sim-to-real gap handled by domain randomisation), but
validation scripts should note the distinction.

---

## 4  Elbow Joint Velocity Limit

**Status:** LOW PRIORITY

**Question:** The elbow velocity limit is set to 1.0 rad/s.  Is this
derived from the motor's no-load speed × gearing, or is it a conservative
operational limit from Klein (2023)?

**Relevant code:**
- `tendon_robot_cfg.py` — `_TENDON_ELBOW_ACTUATOR.velocity_limit = 1.0`
- `tendon_robot_cfg.py` — `_PHYSICAL_LINKAGE_ACTUATOR.velocity_limit = 2.0`

Note: The physical model's linkage actuator uses 2.0 rad/s because each rod
joint moves about half the elbow angle.

---

## 5  Jacobian Transpose Verification

**Status:** VERIFIED ✓

The zero-configuration Jacobian transpose has been cross-referenced
against the CAD reference points in `res/Tensegrity/README.md`:

| Value | Derivation | Correct |
|-------|-----------|---------|
| ±0.0725 | RP1/RP4 y-coordinate (elbow lever arm) | ✓ |
| ±0.013856 | r · sin(60°), r = 0.016 m (wrist tendon circle) | ✓ |
| +0.008, −0.016, +0.008 | r · cos(0°/120°/240°) | ✓ |

Per Klein (2023, §3.2), the constant approximation is valid within ±55°
wrist and ±70° elbow.

---

*Last updated: 2026-03-29*

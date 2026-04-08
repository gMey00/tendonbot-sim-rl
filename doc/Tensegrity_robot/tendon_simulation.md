# Tendon Simulation for the Tensegrity Manipulator

This document describes the tendon-driven simulation layer added to the
IsaacLab tensegrity project.  It explains the physical model, the software
architecture, how to use the tendon-driven robot in new tasks, and how to
validate the simulation against the master thesis results.

---

## 1  Physical Background

The tensegrity manipulator uses **cables (tendons)** instead of direct motor
drives on its revolute joints.  Each joint is actuated by one or more
antagonistic tendons whose tensions generate a net torque around the joint
axis.

### 1.1  Tendon Arrangement

| Tendon | Xacro frame | Attachment point | Joint |
|--------|------------|-----------------|-------|
| T₀ | Force1 | forearm (+0.0725, 0, 0.1) m | elbow (+) |
| T₁ | Force2 | forearm (−0.0725, 0, 0.1) m | elbow (−) |
| T₂ | Force3 | end-effector (+0.010, +0.017321, 0.0697) m | wrist 0° |
| T₃ | Force4 | end-effector (−0.020, 0, 0.0697) m | wrist 120° |
| T₄ | Force5 | end-effector (+0.010, −0.017321, 0.0697) m | wrist 240° |

* **Elbow**: 2 antagonistic tendons (pull in opposite directions).
* **Wrist**: 3 tendons at 120° intervals around a circle of radius r = 20 mm
  (Klein §3.2.2 p.42), providing 2-DOF control (wrist_x and wrist_y).

### 1.2  Jacobian-Transpose Torque Mapping

At the zero configuration the relationship between tendon tensions **T** ∈ ℝ⁵
and joint torques **τ** ∈ ℝ³ is:

$$
\boldsymbol{\tau} = J^T \cdot \mathbf{T}
$$

where the constant Jacobian transpose is:

$$
J^T = \begin{bmatrix}
+0.0725 & -0.0725 & 0 & 0 & 0 \\
0 & 0 & -0.017321 & 0 & +0.017321 \\
0 & 0 & +0.010 & -0.020 & +0.010
\end{bmatrix}
$$

Row order: (elbow, wrist_y, wrist_x).  This is a **zero-configuration
approximation** — the actual moment arms change slightly with joint angle due
to the geometric cable routing, but the error is small for the operating range
(±50° wrist, ±70° elbow) per Klein (2023, §3.2 / §4.2 p.79).

### 1.3  Physical Model (Antiparallelogram Linkage)

An alternative *physical* model replaces the single `elbow_joint` with a
four-bar **antiparallelogram linkage** — the mechanism actually present on the
hardware.  The linkage introduces four revolute joints (`rod_left_joint`,
`rod_right_joint`, `coupler_left_joint`, `coupler_right_joint`) whose 
geometry constrains elbow motion to a single degree of freedom.

**Elbow actuation — body forces:**  Instead of applying elbow torque via J^T,
the two elbow tendons are modelled as cable segments pulling between physical
attachment points on the `root_link` and `forearm_link`.  At each physics step
the force direction is computed from the current body poses, making the
torque mapping configuration-dependent (no zero-config approximation).

**Wrist actuation — constant J^T:**  The 3 wrist tendons continue to use the
constant Jacobian-transpose mapping because the wrist geometry does not change
appreciably within the operating range (±50°, Klein 2023 §4.2 p.79).

| Elbow tendon | Root offset (body-local m) | Forearm offset (body-local m) |
|-------------|---------------------------|------------------------------|
| T₀ (positive) | (0, +0.0725, −0.34) | (0, +0.0725, −0.02) |
| T₁ (negative) | (0, −0.0725, −0.34) | (0, −0.0725, −0.02) |

The physical model is implemented in `PhysicalTendonEffortAction` and
configured via `PhysicalTendonEffortActionCfg`.

### 1.4  Motor and Cable Specifications

The manipulator uses **5 × Maxon EC60** BLDC motors (one per tendon):

| Parameter | Value | Source |
|-----------|-------|--------|
| Continuous torque | 0.401 N·m | Motor datasheet / Klein (2023) Table 3.5 p.52 |
| Stall torque | 4.3 N·m | Motor datasheet / Klein (2023) Appendix B p.114 |
| Power | 150 W | Motor datasheet |
| Voltage | 24 V nominal | Motor datasheet |
| Mass per motor | 350 g | Motor datasheet |
| Spool radius (all motors) | 5 mm | Derived: τ_nominal / F_max = 0.401 / 80 = 0.005 m |
| Cable wrapping (elbow) | 3:1 pulley system | Klein (2023) §3.2.5 p.46 — Flaschenzug principle |
| Wrist motor wrapping | Direct-drive (no MA) | Not stated in thesis; open question |
| Motor-side max force (continuous) | 80 N per motor | Klein (2023) §3.2.6 p.53: τ / r_spool = 0.401 / 0.005 |
| Motor-side max force (PSU peak) | 158 N per motor | Klein (2023) §3.2.6 p.53: 0.79 Nm / 0.005 m |
| Elbow cable tension at attachment (continuous) | 240 N | 80 N × 3:1 MA |
| Elbow cable tension at attachment (peak) | 474 N | 158 N × 3:1 MA |
| Klein elbow saturation (controller) | 160 N (motor-side) | Klein (2023) §3.5 p.65 |
| Klein wrist saturation (controller) | 80 N | Klein (2023) §3.5 p.65 |
| max_tension (code) | 500 N | `TendonEffortActionCfg` — matches elbow peak ~480 N |

> **Note:** The code uses a single `max_tension = 500 N` for all 5 tendons.
> This approximately matches the elbow's peak attachment-point force (474 N after 3:1 MA)
> but is ~6× higher than the wrist motor's continuous capability (80 N, no MA).
> The wrist `effort_limit = 3.0 N·m` prevents unphysical wrist torques from
> high cable tensions.  A per-tendon max_tension would improve action-space
> resolution for the wrist — see TODO.

### 1.5  Why Not PhysX Spatial Tendons?

Isaac Sim / PhysX 5.x provides spatial and fixed tendon primitives.  We chose
**not** to use them for two reasons:

1. **Numerical instabilities**: The Isaac Sim documentation warns that spatial
   tendons can cause solver instabilities, especially with stiff systems.
2. **Force reporting**: PhysX excludes tendon forces from the
   `joint_force_report`, making it impossible to observe or reward tendon
   behaviour in RL tasks.

Instead, we implement the tendon model as a custom `ActionTerm` in IsaacLab.
This gives full control over the tension-to-torque mapping and allows the
forces to be applied as standard joint effort targets.

---

## 2  Software Architecture

### 2.1  Package Layout

```
tensegrity_pick/
├── robots/                              ← NEW: centralised robot definitions
│   ├── __init__.py                      #   exports all configs + TendonEffortAction
│   ├── tensegrity_robot_cfg.py          #   TENS_3DOF_CFG, TENS_5DOF_GRIPPER_CFG
│   ├── tendon_actuator.py               #   TendonEffortAction + TendonEffortActionCfg
│   └── tendon_robot_cfg.py              #   TENS_3DOF_TENDON_CFG, TENS_5DOF_GRIPPER_TENDON_CFG
├── tasks/
│   └── manager_based/
│       ├── reach/
│       │   ├── reach_env_cfg.py                     # Shared reach task config
│       │   └── config/tensegrity_tendon/            # Tendon-driven reach variant
│       ├── shared/                      # Shared configs across tasks
│       └── cube_place/
│           ├── place_env_cfg.py                     # Base MDP + Tensegrity PD env configs
│           └── config/tensegrity_tendon/            # Tendon-driven place variant
```

### 2.2  Key Classes

| Class | Module | Role |
|-------|--------|------|
| `TendonEffortAction` | `robots.tendon_actuator` | `ActionTerm` subclass: maps tendon tensions → joint torques via J^T |
| `TendonEffortActionCfg` | `robots.tendon_actuator` | Configclass for the above; specifies joint names, J^T, max tension |
| `PhysicalTendonEffortAction` | `robots.tendon_actuator` | Body-force elbow tendons + J^T wrist; replaces elbow_joint with linkage |
| `PhysicalTendonEffortActionCfg` | `robots.tendon_actuator` | Configclass for the physical model; specifies attachment offsets |
| `TENS_3DOF_TENDON_CFG` | `robots.tendon_robot_cfg` | 3-DOF arm with `IdealPDActuator(stiffness=0, damping=0)` |
| `TENS_5DOF_GRIPPER_TENDON_CFG` | `robots.tendon_robot_cfg` | 5-DOF + gripper; arm is tendon-driven, base + gripper are PD |
| `TENS_3DOF_PHYSICAL_TENDON_CFG` | `robots.tendon_robot_cfg` | 3-DOF arm with antiparallelogram linkage + effort passthrough |
| `TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG` | `robots.tendon_robot_cfg` | 5-DOF physical + gripper; linkage elbow, J^T wrist |

### 2.3  How It Works (Data Flow)

```
RL Agent → raw actions ∈ [-1, 1]^5
    ↓ TendonEffortAction.process_actions()
Affine map → tensions ∈ [0, max_tension]^5
    ↓ torch.bmm(J^T, T)
Joint torques τ ∈ ℝ³
    ↓ TendonEffortAction.apply_actions()
asset.set_joint_effort_target(τ)
    ↓ IsaacLab Articulation._apply_actuator_model()
IdealPDActuator(stiffness=0, damping=0) → τ_out = feedforward effort
    ↓ PhysX
Joint motion
```

The `IdealPDActuator` with zero stiffness and zero damping acts as a pure
effort passthrough: the only torque applied is the feedforward effort from
the tendon mapping.

---

## 3  Using the Tendon Robot in a New Task

Adding tendon-driven control to any manager-based task requires two changes:

### 3.1  Scene: Use the Tendon Robot Config

```python
from tensegrity_pick.robots import TENS_5DOF_GRIPPER_TENDON_CFG

@configclass
class MySceneCfg(ProjBaseSceneCfg):
    robot = TENS_5DOF_GRIPPER_TENDON_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.15, 0.0, 2.3)),
    )
```

### 3.2  Actions: Add TendonEffortActionCfg

```python
from tensegrity_pick.robots import TendonEffortActionCfg
from tensegrity_pick.robots.tendon_actuator import DEFAULT_JACOBIAN_TRANSPOSE

@configclass
class ActionsCfg:
    # Base joints: keep using standard PD position control
    base_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
        scale=0.25,
        use_default_offset=True,
    )
    # Arm joints: tendon-driven
    arm_tendon = TendonEffortActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        num_tendons=5,
        max_tension=500.0,
        jacobian_transpose=DEFAULT_JACOBIAN_TRANSPOSE,
    )
```

That's it.  All other MDP components (rewards, observations, terminations,
curriculum) can remain unchanged.

---

## 4  Registered Environments

| Environment ID | Drive | Task |
|---------------|-------|------|
| `Template-Reach-Tensegrity-v0` | PD position | Reach |
| `Template-Reach-Tensegrity-Tendon-v0` | Tendon effort (J^T) | Reach |
| `Template-Reach-Tensegrity-Physical-Tendon-v0` | Tendon effort (body-force) | Reach |
| `Template-Tensegrity-Cube-Place-v0` | PD position | Cube Place |
| `Template-Tensegrity-Cube-Place-Tendon-v0` | Tendon effort (J^T) | Cube Place |
| `Template-Tensegrity-Cube-Place-Physical-Tendon-v0` | Tendon effort (body-force) | Cube Place |
| `Template-Tensegrity-Shirt-Place-v0` | PD position | Shirt Place |
| `Template-Tensegrity-Shirt-Place-Physical-Tendon-v0` | Tendon effort (body-force) | Shirt Place |

Play variants (`*-Play-v0`) use 50 envs for evaluation.

---

## 5  Step Response Validation

The `scripts/model_validation/` directory contains a comprehensive validation
pipeline for **both** tendon models.  See
[`scripts/model_validation/README.md`](../src/tensegrity_pick/scripts/model_validation/README.md)
for full usage details.

### 5.1  Variants

| Variant | CLI flag | Elbow actuation | Wrist actuation | Data dir |
|---------|----------|----------------|----------------|----------|
| `elbow_approx` | `--variant elbow_approx` | J^T (constant) | J^T (constant) | `tendon/data/` |
| `physical` | `--variant physical` | Body forces | J^T (constant) | `tendon_physical/data/` |

Both variants share the same PID controller gains and wrist J^T mapping;
they differ only in how elbow torque is applied.  The pipeline script
`run_validation.sh` runs both by default.

```bash
# Run full pipeline (PD + both tendon variants + base + plots)
bash scripts/model_validation/run_validation.sh

# Physical tendon only
bash scripts/model_validation/run_validation.sh --tendon-physical-only
```

### 5.2  Test Protocol

1. **PID controller** converts angle set-points to desired torques
   (thesis-identical gains: wrist Kp=0.2/Ki=0.03/Kd=0.03,
   elbow Kp=0.3/Ki=0.03/Kd=0.02).
2. **Gravity compensation** per Mukherjee et al.: τ_a = m·g·l_c·sin(θ).
3. **Tension distribution** maps desired torques to cable tensions via
   the pseudoinverse of J^T, clamped to physical limits.
4. **Step inputs**: wrist 0→10°, 0→20°, 0→30°; elbow 0→20°, 0→30°, 0→40°.
5. After each step, the set-point returns to 0° and the system settles.

### 5.2  Metrics

| Metric | Definition | Thesis reference |
|--------|-----------|-----------------|
| Rise time | Time from 10% to 90% of target | §3.5 |
| Overshoot | (peak − target) / target × 100% | §3.5 |
| RMSE | √(mean(error²)) | Eq. 3.11 |
| NRMSE | RMSE / (max − min) × 100% | Eq. 3.12 |

### 5.3  Running the Test

```bash
cd /home/robot/Isaac/IsaacLab
./isaaclab.sh -p /home/robot/studentische-arbeiten/src/tensegrity_pick/scripts/step_response_test.py \
    --headless --num-envs 1 --output-dir ./step_response_results
```

Outputs:
- `step_response_{joint}.png` — angle and tension time-series plots
- `metrics_{joint}.png` — summary table
- `nrmse_comparison.png` — bar chart comparing Isaac Sim vs. Klein (2023) Gazebo results
- `step_response_results.csv` — numerical data

### 5.4  Reference Data from Klein (2023)

Thesis Table 4.2 — simulation NRMSE (Gazebo vs. reality):

| Joint | 10° | 20° | 30° | 40° |
|-------|-----|-----|-----|-----|
| Wrist X | 25.5% | 9.9% | 11.2% | — |
| Wrist Y | 19.2% | 11.6% | 12.5% | — |
| Elbow | — | 39.1% | 11.7% | 12.1% |

---

## 6  References

1. **Klein, M.** (2023). *Arbeitsraumanalyse, Simulation und Bewegungsplanung
   eines seilgetriebenen robotischen Manipulators*. Master's thesis,
   Friedrich-Alexander-Universität Erlangen-Nürnberg.

2. **Nemoto, T., Niiyama, R., Iwata, H.** (2019). *Design and Analysis of 
   a Compact Tendon-Driven Rotary Joint with Application to a Robotic Arm*.
   — Cited as [84] in Klein (2023); foundational control principle for
   tendon-driven orientation regulation.

3. **Mukherjee, S., Kamal, A., Hirai, S.** — Gravity compensation formula
   τ_a = m·g·l_c·sin(θ), cited as [85] in Klein (2023), Eq. 3.2.

4. **NVIDIA** (2024). *Isaac Sim Documentation — PhysX Tendons*.
   https://docs.isaacsim.omniverse.nvidia.com/ — Documents spatial/fixed
   tendon API and known limitations (force reporting, stability).

5. **Borgstrom, P.H., Jordan, B.L., Sukhatme, G.S., Batalin, M.A.,
   Kaiser, W.J.** (2009). *Rapid Computation of Optimally Safe Tension
   Distributions for Parallel Cable-Driven Robots*. IEEE Transactions on
   Robotics, 25(6). — General framework for cable tension distribution.

6. **Oh, S.-R., Agrawal, S.K.** (2005). *Cable Suspended Planar Robots
   with Redundant Cables: Controllers with Positive Tensions*. IEEE
   Transactions on Robotics, 21(3). — Theoretical basis for redundant
   cable systems with tension constraints.

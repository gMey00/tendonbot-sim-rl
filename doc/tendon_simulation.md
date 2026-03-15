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
| T₂ | Force3 | end-effector (+0.008, +0.013856, 0.0697) m | wrist 0° |
| T₃ | Force4 | end-effector (−0.016, 0, 0.0697) m | wrist 120° |
| T₄ | Force5 | end-effector (+0.008, −0.013856, 0.0697) m | wrist 240° |

* **Elbow**: 2 antagonistic tendons (pull in opposite directions).
* **Wrist**: 3 tendons at 120° intervals around a circle of radius ≈ 16 mm,
  providing 2-DOF control (wrist_x and wrist_y).

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
0 & 0 & -0.013856 & 0 & +0.013856 \\
0 & 0 & +0.008 & -0.016 & +0.008
\end{bmatrix}
$$

Row order: (elbow, wrist_y, wrist_x).  This is a **zero-configuration
approximation** — the actual moment arms change slightly with joint angle due
to the geometric cable routing, but the error is small for the operating range
(±55° wrist, ±70° elbow) per Klein (2023, §3.2).

### 1.3  Why Not PhysX Spatial Tendons?

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
| `TENS_3DOF_TENDON_CFG` | `robots.tendon_robot_cfg` | 3-DOF arm with `IdealPDActuator(stiffness=0, damping=0)` |
| `TENS_5DOF_GRIPPER_TENDON_CFG` | `robots.tendon_robot_cfg` | 5-DOF + gripper; arm is tendon-driven, base + gripper are PD |

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
| `Template-Tensegrity-Reach-v0` | PD position | Reach |
| `Template-Tensegrity-Reach-Tendon-v0` | Tendon effort | Reach |
| `Template-Tensegrity-Cube-Place-v0` | PD position | Cube Place |
| `Template-Tensegrity-Cube-Place-Tendon-v0` | Tendon effort | Cube Place |

Play variants (`*-Play-v0`) use 50 envs for evaluation.

---

## 5  Step Response Validation

The script `scripts/step_response_test.py` replicates the testing methodology
from Klein (2023, §3.5 and §4.2–4.3):

### 5.1  Test Protocol

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

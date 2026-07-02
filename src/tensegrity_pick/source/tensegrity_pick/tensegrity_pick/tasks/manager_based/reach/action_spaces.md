# Reach — Action Spaces (F140 comparison grid)

[← Back to reach task overview](README.md)

The 6 comparison arms each train in **four action spaces**. Everything else —
reward, target distribution, observation structure, terminations — is identical
across the grid so the action-space controllers can be compared fairly (see
[Fairness constraints](#fairness-constraints)).

| Suffix | Action space | Action dim | Controller config funnel | PPO config |
|---|---|---|---|---|
| *(none)* | Joint (EMA) | N | [`config/f140_reach_common.py`](config/f140_reach_common.py) | `agents/skrl_ppo_cfg.yaml` |
| `-IK-Rel` | Differential IK, relative | 6 | [`config/ik_reach_common.py`](config/ik_reach_common.py) | `agents/skrl_ppo_ik_cfg.yaml` |
| `-IK-Abs` | Differential IK, absolute | 7 | [`config/ik_reach_common.py`](config/ik_reach_common.py) | `agents/skrl_ppo_ikabs_cfg.yaml` |
| `-OSC` | Operational-Space Control | 13 | [`config/osc_reach_common.py`](config/osc_reach_common.py) | `agents/skrl_ppo_osc_cfg.yaml` |

N = number of controlled joints (6 for UR-F140, 7 for Kinova-F140; +2 tensegrity
wrist joints for the Frankenstein arms).

## Joint (EMA joint position)

`EMAJointPositionToLimitsActionCfg` with `alpha = 0.2`: the policy outputs one
value per controlled joint in [−1, 1], mapped to the joint limits and smoothed
with an exponential moving average so the implicit PD can track the targets.
The baseline action space — the policy must resolve all redundancy itself.

## Differential IK (relative and absolute)

Isaac Lab's `DifferentialInverseKinematicsAction` with a damped-least-squares
(DLS) solver over the controlled joints, tracking `robotiq_base_link`:

- **IK-Rel** (`use_relative_mode=True`, `scale=0.5`): the policy commands a
  per-step EE pose *delta* (3 position + 3 rotation).
- **IK-Abs**: the policy commands the full target pose (3 position +
  4 quaternion) in the robot root frame.

Both stiffen every non-gripper actuator to `IK_ARM_STIFFNESS/DAMPING = 800/160`
(cf. Isaac Lab's `FRANKA_PANDA_HIGH_PD_CFG`) so the arm tracks the IK-solved
joint targets crisply; the robots' native USD gains are far too underdamped.
For the Frankenstein arms the tensegrity wrist joints are part of the IK joint
set — the DLS solver handles the extra redundancy.

**PPO difference (IK-Abs only):** `initial_log_std = −1.0` (all other spaces
use 0.0). IK-Abs actions are an absolute EE pose in metres; std = 1 Gaussian
exploration swamps the 0.2–0.4 m target box and slows convergence
(ur10_frankenstein_ikabs: 8.9 cm/73 % → 5.1 cm/83 % with −1.0).

## Operational-Space Control (OSC)

`OperationalSpaceControllerActionCfg` with `target_types=["pose_abs"]` and
`impedance_mode="variable_kp"`: the policy commands an absolute EE pose
(7 dims) **and** the task-space stiffness (6 dims, scaled ×100, clamped to
50–200); damping follows critically (`ζ = 1`). The controller solves for joint
torques, so the arm actuators run pure-effort (implicit PD zeroed) — except:

- **Gripper groups** keep their reach-stabilised PD (as in every variant).
- **The compliant tensegrity wrist** (`wrist_x/y_joint`, Frankenstein arms)
  keeps its PD and is **excluded from the OSC joint set / Jacobian**
  (`TENSEGRITY_WRIST_JOINTS` in `osc_reach_common.py`). Torque-controlling
  these soft, near-massless joints (effort limit 3.5 N·m) removes all damping
  from them; they oscillate and diverge the whole arm — this was the root
  cause of the 41–65 cm / 0 %-success OSC+Frankenstein failures. With the
  wrist position-held, OSC corrects wrist deflection with the rigid arm
  joints instead.

Further settings (empirically validated 2026-07-02, see
[`doc/reports/reach_optimization_tracking.md`](../../../../../../../../doc/reports/reach_optimization_tracking.md)
iterations 14–15):

- **Partial inertial dynamics decoupling**
  (`inertial_dynamics_decoupling=True` + `partial_…=True`): scales the
  task-space gains by the arm's translational/rotational inertia blocks so the
  same kp/ζ behave consistently across the light UR5e and the heavy UR10
  (ur5e_f140_osc: 11.9 cm/67 % → 2.1 cm/93 %). *Full* decoupling
  (Λ = (J M⁻¹ Jᵀ)⁻¹, franka sample) is ill-conditioned on these arm+gripper
  articulations; a damping ratio of 2 instead of decoupling over-damps
  (44 cm). Do not change these without re-running the OSC grid.
- **Null-space posture control** toward the default configuration, enabled
  only for redundant arms (> 6 OSC joints, i.e. the 7-DOF Kinovas — the URs
  are non-redundant once the wrist is excluded).

## Fairness constraints

When touching any of these configs, keep the comparison fair:

1. Reward = the Isaac Lab reference reach reward, identical everywhere.
2. Target distribution (the uniform position box) identical everywhere.
3. Observation structure identical everywhere (sin/cos joint positions,
   joint velocities, pose command, previous actions).
4. A controller/PPO setting may differ **between action spaces** but must be
   identical **across arms within one action space**.

## Results by action space

Converged means over the 24-variant grid (seed 42, 100k timesteps,
2026-07-02; position error / % of steps within 5 cm):

| Space | Mean pos. error | Mean success |
|---|---|---|
| Joint (EMA) | 3.1 cm | 91 % |
| IK-Rel | 3.4 cm | 88 % |
| IK-Abs | 4.0 cm | 84 % |
| OSC | 7.0 cm (3.3 cm excl. the ur10_frankenstein seed outlier) | 87 % |

Per-variant numbers and history: [README — Training Results](README.md#training-results)
and the [optimization tracking log](../../../../../../../../doc/reports/reach_optimization_tracking.md).

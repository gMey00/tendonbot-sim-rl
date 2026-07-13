# Reach Task

[← Back to extension overview](../../../../../../README.md) · [Project root](../../../../../../../../README.md)

End-effector pose tracking, based on the IsaacLab reference reach task. The
robot must move its EE body to randomly sampled targets. Two variant families
share the same MDP (rewards, terminations, curriculum, sim parameters):

- **F140 comparison grid** — 6 industrial arms × 4 action spaces
  (**24 variants**), built for the action-space study
  ([research brief](../../../../../../../../doc/reports/RESEARCH_BRIEF_action_spaces.md)).
  Targets are a uniform position box; orientation is intentionally loose.
- **Tensegrity family** — the original 5-DOF tensegrity manipulator (PD,
  Tendon, Physical Tendon, Physical Hierarchical). Targets are FK-sampled
  full poses, reachable by construction.

![Task Scene](figures/scene_setup.png)

## Purpose of this task (read this first)

Reach is **not** primarily a robot-vs-robot benchmark. Its role in the project is:

1. **Controller hardening** — get each *(robot × action space)* controller working
   robustly on *reachable* targets, so it can be reused in the harder downstream
   cloth tasks without hitting breaking points (divergence, NaNs, collapsed
   episodes). Reach is the cheap place to find and fix those failure modes.
2. **Per-robot action-space selection** — for *each* robot, determine which action
   space / control mechanism (joint · IK-Rel · IK-Abs · OSC) works best, to pick
   the controller to carry forward per robot.
3. **Tensegrity-wrist evaluation** — determine whether the compliant 2-DOF
   tensegrity wrist ("Frankenstein" arms) helps, and under which action spaces
   (it participates under joint/IK but is currently *excluded* from the OSC
   Jacobian — see [`action_spaces.md`](action_spaces.md)).

Because the comparison is **within** each robot (not across robots), "fairness"
here means giving each controller **reachable** targets — not a shared cross-robot
distribution. This is why **per-robot FK-sampled targets** (each robot samples its
own reachable poses via forward kinematics) are the right design for the hardening
runs: a shared box or a shared reference-robot distribution inevitably lands in
some robot's unreachable / singular region, which *manufactures* the very breaking
points we want to eliminate. The uniform-box grid below is retained as a settled
position-only comparison; the FK-target work (tracking log iterations 17–20) is
the controller-hardening direction. See the
[tracking log](../../../../../../../../doc/reports/reach_optimization_tracking.md)
for the multi-seed evidence that the box's fixed (unreachable) orientation was the
root cause of both the OSC-on-UR instability and the "0 % orientation on UR arms".

**THESIS results (iteration 22, 2026-07-13) — the final consistent dataset.**
One locked config everywhere: `REACH_FK_TARGETS=1` (fixed sampler:
self-collision filter + floor clearance), Kinova sector
`REACH_FK_HALF_RANGE=1.0`, **EMA α=0.2 on every action space**
(`REACH_TS_EMA=0.2` for task-space; the joint action has it built in);
5 seeds {0,1,2,42,123}. Cell = pos cm / pos % / pose % (pose = position < 5 cm
AND orientation < 0.3 rad):

| Arm | Joint | IK-Rel | IK-Abs | OSC |
|---|---|---|---|---|
| Kinova-F140 | 3.4 / 90 / 43 | 3.7 / 86 / 45 | 3.1 / 90 / 35 | **2.9 / 91 / 32** |
| Kinova-Frankenstein | **4.6 / 85 / 37** | 6.0 / 69 / 29 | 5.1 / 80 / 26 | 3.9 / 85 / 11 |
| UR5e-F140 | **3.4 / 91 / 86** | 6.1 / 71 / 37 | 5.8 / 77 / 45 | 9.1 / 43 / 16 |
| UR5e-Frankenstein | **5.3 / 79 / 60** | 7.9 / 59 / 31 | 6.1 / 75 / 44 | 12.4 / 20 / 7 |
| UR10-F140 | **5.2 / 87 / 84** | 10.0 / 60 / 35 | 11.3 / 47 / 22 | 13.2 / 31 / 16 |
| UR10-Frankenstein | **6.2 / 80 / 64** | 10.8 / 49 / 27 | 10.7 / 51 / 28 | 17.1 / 10 / 3 |

Per-robot action-space answer: **joint control wins overall** — decisively on
the 6-DOF UR arms, marginally on the 7-DOF Kinova, where the action-space
choice barely matters and `kinova_f140_osc` is the grid's tightest task-space
cell (redundancy properly exploited once targets are well-scoped). Task-space
fallback of record: **IK-Abs + EMA**. Hardening lessons baked into this table:
task-space actions must be EMA-smoothed (iteration 21 — doubles pose%; a 6-D
rotation action and stiffer OSC are *worse*); **no OSC on non-redundant (UR)
arms** (bimodal divergence on box targets, consistently worst on FK; EMA
mixed-to-harmful there); the tensegrity wrist is a **capability extension**
(wrist A/B on the fixed sampler: locking it on wrist-requiring targets costs
22 pose-points, 57 → 35 %) and is only usable under joint/IK control —
wrist-under-OSC is closed (pure-effort diverges, damped-compliant strictly
worse). Per-variant figures and reports (`figures/`, `reports/`) are
regenerated from the seed-42 thesis runs; the full 132-run dataset (tfevents +
final checkpoints) is preserved in
[`logs/skrl/theses_logs/reach/`](../../../../../../logs/skrl/theses_logs/).

## Table of Contents

- [Variants](#variants)
  - [F140 comparison grid (24 variants)](#f140-comparison-grid-24-variants)
  - [Tensegrity family (4 variants)](#tensegrity-family-4-variants)
- [Documentation Map](#documentation-map)
- [Directory Structure](#directory-structure)
- [Shared MDP](#shared-mdp)
  - [Targets (commands)](#targets-commands)
  - [Observations (policy group)](#observations-policy-group)
  - [Rewards](#rewards)
  - [Terminations, Curriculum, Events](#terminations-curriculum-events)
  - [Simulation Parameters](#simulation-parameters)
- [Training](#training)
  - [On the Alex cluster (recommended)](#on-the-alex-cluster-recommended)
  - [Locally](#locally)
  - [Evaluation and Play](#evaluation-and-play)
- [Training Results](#training-results)
  - [F140 grid](#f140-grid)
  - [Tensegrity family](#tensegrity-family)
  - [Regenerating Plots and Reports](#regenerating-plots-and-reports)
- [Related](#related)

## Variants

### F140 comparison grid (24 variants)

Task IDs follow `Template-Reach-<Arm>[-IK-Rel|-IK-Abs|-OSC]-v0`; append
`-Play` before `-v0` for the evaluation config (e.g.
`Template-Reach-UR5e-F140-OSC-Play-v0`). No suffix = joint (EMA) action space.

| Arm | DOF (controlled) | Joint | IK-Rel | IK-Abs | OSC | Config |
|---|---|---|---|---|---|---|
| `UR5e-F140` | 6 | ✓ | ✓ | ✓ | ✓ | [`config/ur5e_f140/`](config/ur5e_f140/) |
| `UR5e-Frankenstein` | 8 (6+2 wrist) | ✓ | ✓ | ✓ | ✓ | [`config/ur5e_frankenstein/`](config/ur5e_frankenstein/) |
| `UR10-F140` | 6 | ✓ | ✓ | ✓ | ✓ | [`config/ur10_f140/`](config/ur10_f140/) |
| `UR10-Frankenstein` | 8 (6+2 wrist) | ✓ | ✓ | ✓ | ✓ | [`config/ur10_frankenstein/`](config/ur10_frankenstein/) |
| `Kinova-F140` | 7 | ✓ | ✓ | ✓ | ✓ | [`config/kinova_f140/`](config/kinova_f140/) |
| `Kinova-Frankenstein` | 9 (7+2 wrist) | ✓ | ✓ | ✓ | ✓ | [`config/kinova_frankenstein/`](config/kinova_frankenstein/) |

"F140" = rigid Robotiq 2F-140 gripper; "Frankenstein" = the same arm with the
compliant 2-DOF **tensegrity wrist** spliced between flange and gripper
(+2 controlled DOF, redundant). All six arms are floor-mounted upright at
`(0.75, 1.0, 0.75)` and track the EE body `robotiq_base_link`. Robot
definitions: [`robots/`](../../../robots/) (`ur5e`, `ur10`, `kinova_gen3`,
`*_frankenstein` configs).

The four action spaces (controllers, gains, PPO differences, fairness rules)
are documented in **[action_spaces.md](action_spaces.md)**.

### Tensegrity family (4 variants)

| Environment ID | Robot | DOF | Actuation | Action dim | Config |
|---|---|---|---|---|---|
| `Template-Reach-Tensegrity-v0` | Tensegrity 5-DOF | 5 | PD (joint position) | 5 | [`config/tensegrity/`](config/tensegrity/) |
| `Template-Reach-Tensegrity-Tendon-v0` | Tensegrity 5-DOF | 5 | Tendon (Jacobian-transpose) | 5 | [`config/tensegrity_tendon/`](config/tensegrity_tendon/) |
| `Template-Reach-Tensegrity-Physical-Tendon-v0` | Tensegrity 5-DOF (physical 4-bar elbow) | 7 | Body-force tendons (direct tension control) | 7 (2 base + 5 tensions) | [`config/tensegrity_tendon/`](config/tensegrity_tendon/) |
| `Template-Reach-Tensegrity-Physical-Hierarchical-v0` | Tensegrity 5-DOF (physical 4-bar elbow) | 7 | Body-force tendons via inner PID→tension loop | 5 (2 base + 3 set-points) | [`config/tensegrity_tendon/`](config/tensegrity_tendon/) |

Each also has a `…-Play-v0` twin. Details: controlled joints and tendon
actuation are documented in the
[physical elbow spec](config/tensegrity_tendon/physical_elbow_spec.md); both
physical variants use a hidden PD **FK reference robot** for target
sampling (see tracking log, iteration 13).

**Physical-variant specifics (2026-07-09 rework** — full change log in
[physical_variant_fix_report.md](../../../../../../../../doc/reports/physical_variant_fix_report.md)**):**

- Cable min/max length limits between the attachment points
  ([0.0913, 0.2577] m over the ±70° elbow workspace): wind-up stop +
  spring–damper stretch stop (cap 500 N); per-tendon hardware tension
  saturation `[480, 480, 80, 80, 80]` N; 50 ms motor/spool tension lag.
- `linkage_integrity_penalty` reward term, −1/step while the closure-anchor
  gap exceeds 3 cm OR the chain is parallelogram-branch-flipped — closes the
  "break the linkage" reward hack. (A *termination* here is itself exploitable:
  the net-negative reach reward makes early termination a suicide exit —
  measured episode-length collapse 180 → 5 steps.)
- Observations measure the lower-arm rotation from the **forearm body twist**
  (`lower_arm_angle`/`lower_arm_ang_vel`, not raw linkage joints) and expose
  per-cable length/rate plus the applied (lag-filtered) tensions to the
  policy (obs 30 direct / 28 hierarchical).
- 120 Hz physics / decimation 4 (policy still 30 Hz); `*_awake.usd` bakes
  (per-body sleep thresholds zeroed — a sleeping articulation ignores body
  forces). Known open issue: a GPU-solver quasi-static breakaway
  (~10–20 N·m) documented in the fix-report addendum.
- Targets sample the *achievable* workspace (`joint_range_margin = 0.10`):
  boundary poses cannot be held by the cable drives (wrist under-actuated
  near ±50° with the gripper mass; elbow stop-grinding at ±70°).
- The hierarchical variant tracks 3 joint set-points (elbow ±60°, wrist ±40°)
  with the retuned step-response PID (slew 1.2/5/5 rad/s, per-joint integral
  clamps, anti-windup zones) → block-wise tension distribution → the same
  body-force channel. A scripted **IK+PID heuristic baseline**
  (`scripts/skrl/heuristic_physical_ik.py`, `--agent heuristic` in
  `evaluate_reach.py`) validates the controller separately from RL.

> The former ceiling-mounted `Template-Reach-UR10e-v0` / `Template-Reach-Kinova-v0`
> variants were superseded by the F140 comparison grid and are no longer
> registered (their old training artifacts remain under `figures/ur10e`,
> `reports/ur10e`, `figures/kinova`, `reports/kinova`).

## Documentation Map

| Document | Content |
|---|---|
| this README | variant matrix, shared MDP, training workflow, results |
| [action_spaces.md](action_spaces.md) | the 4 controllers of the grid: joint EMA, IK-Rel/Abs, OSC (wrist handling, inertial decoupling), per-space PPO settings, fairness rules |
| [physical_elbow_spec.md](config/tensegrity_tendon/physical_elbow_spec.md) | tensegrity physical-elbow kinematics and actuation |
| [optimization tracking log](../../../../../../../../doc/reports/reach_optimization_tracking.md) | full change history (iterations 0–15) with root-cause analyses |
| [reach task handoff](../../../../../../../../doc/reach_task_handoff.md) | mission/state snapshot of the grid study (2026-07-01) |
| [Alex quickstart](../../../../../../../../doc/Alex_cluster/alex_quickstart.md) | cluster workflow (Slurm, rendering caveats) |

## Directory Structure

```
reach/
├── README.md                     # ← you are here
├── action_spaces.md              # the 4 action spaces of the F140 grid
├── reach_env_cfg.py              # Base MDP config (shared by ALL variants)
├── mdp/
│   ├── fk_sampled_pose_command.py  # command term: FK sampling + uniform-box mode + metrics
│   ├── rewards.py                  # tracking terms + *_reached success terms
│   ├── observations.py             # joint_pos_sin_cos
│   └── events.py                   # clamp_infinite_joint_limits
├── config/
│   ├── f140_reach_common.py      # shared grid setup: mount, box command, joint action, reward
│   ├── ik_reach_common.py        # IK-Rel/Abs action + arm stiffening
│   ├── osc_reach_common.py       # OSC action (wrist exclusion, partial decoupling)
│   ├── <arm>/                    # ur5e_f140, ur5e_frankenstein, ur10_f140,
│   │   │                         # ur10_frankenstein, kinova_f140, kinova_frankenstein
│   │   ├── joint_pos_env_cfg.py  # arm setup + joint action (+ _PLAY subclass)
│   │   ├── ik_rel_env_cfg.py     # + IK-Rel action
│   │   ├── ik_abs_env_cfg.py     # + IK-Abs action
│   │   ├── osc_env_cfg.py        # + OSC action
│   │   └── agents/skrl_ppo{,_ik,_ikabs,_osc}_cfg.yaml
│   ├── tensegrity/               # tensegrity PD variant
│   └── tensegrity_tendon/        # tendon + physical-tendon variants
├── figures/<variant>/            # training plots (per variant)
└── reports/<variant>/            # training reports (per variant)
```

## Shared MDP

All variants inherit [`reach_env_cfg.py`](reach_env_cfg.py), which mirrors the
IsaacLab reference reach task.

### Targets (commands)

One target per episode (`resampling_time_range = (1e9, 1e9)`), generated by
the project's `FKSampledPoseCommand` in one of two modes:

| | F140 grid | Tensegrity family |
|---|---|---|
| Mode | `uniform_ranges` position box | FK-sampled full pose |
| Position | x (0.30, 0.50), y (−0.20, 0.20), z (0.25, 0.50) m in the base frame | forward kinematics of uniformly sampled joint configurations |
| Orientation | gripper-down (`pitch = π`), free yaw — **intentionally loose** | reachable by construction |

The box fits inside every comparison arm's envelope (farthest corner ≈ 86 % of
UR5e reach). A *fixed* orientation target is geometrically unreachable across
the box for the 6-DOF UR arms (verified by
[`scripts/diagnose_box_orientation.py`](../../../../../../../../src/tensegrity_pick/scripts/diagnose_box_orientation.py));
reach is therefore a **position task** — do not re-add orientation reward
weight (see tracking log, iterations 14–15).

### Observations (policy group)

| Term | Dim | Description |
|---|---|---|
| `joint_pos` | 2N | Controlled joint positions, **sin/cos encoded** (removes the ±π wrap for continuous Kinova joints; ±0.01 uniform noise) |
| `joint_vel` | N | Controlled joint velocities (±0.01 noise) |
| `pose_command` | 7 | Target pose (x, y, z, qw, qx, qy, qz) in root frame |
| `actions` | M | Previous actions (M = action dim of the variant) |

`enable_corruption = True` during training, disabled in the `-Play` configs.

### Rewards

The **IsaacLab reference reach reward, unchanged** — restoring it exactly was
the key fix of the grid study (tracking log, iteration 14). Do not rebalance.

| Term | Weight | Function |
|---|---|---|
| `end_effector_position_tracking` | −0.2 | L2 position error |
| `end_effector_position_tracking_fine_grained` | +0.1 | `1 − tanh(d / 0.1)` |
| `end_effector_orientation_tracking` | −0.1 | Quaternion error magnitude |
| `action_rate` | −0.0001 → −0.005 | curriculum ramp at 4 500 steps |
| `joint_vel` | −0.0001 → −0.001 | curriculum ramp at 4 500 steps |

Success metrics (logging only, weight 1×10⁻⁶ — divide the TensorBoard value by
1e-6 to recover the success fraction):

| Term | Threshold |
|---|---|
| `position_reached` | position error < **0.05 m** |
| `orientation_reached` | orientation error < **0.3 rad** |
| `pose_reached` | both |

> **Reading the metrics:** `Episode_Reward/<term>` is a *per-step mean*, not an
> episodic sum. Mean position error [m] = −`end_effector_position_tracking` / 0.2.
> A run with tiny tracking error but 0 % success has collapsed episodes — check
> `Episode / Total timesteps (mean)` (= 180 when healthy).

### Terminations, Curriculum, Events

| Term | Condition |
|---|---|
| `time_out` | episode length exceeded |
| `joint_vel_diverged` | any *controlled* joint velocity > 100 rad/s (physics divergence guard; 500 rad/s on the physical variants — the near-massless wrist frame link spikes harmlessly on reset transients) |


Reset: controlled joints to default ± 0.125 rad offset, zero velocity;
`clamp_infinite_joint_limits` (reset-mode) replaces infinite/oversized joint
limits (Kinova continuous joints, UR ±2π joints) with finite ranges.

> **Kinova before-bake clamp.** The Kinova continuous joints are authored beyond
> PhysX's `[-2π, 2π]`, so PhysX threw `setLimitParams()` errors when it *baked*
> the articulation at `sim.reset()` — before the reset-mode clamp runs (fixed
> for stepping, but noisy at load). The Kinova robot spawner is therefore wrapped
> (`spawn_usd_with_clamped_joint_limits`, `reach/mdp/events.py`) to clamp the
> source (`env_0`) joint-limit USD attributes *before* the clone/bake, using the
> same `default ± range/2` rule so the baked limits are unchanged. See
> [tracking log iteration 16](../../../../../../../../doc/reports/reach_optimization_tracking.md).

### Simulation Parameters

| Parameter | Value |
|---|---|
| Physics dt | 1/60 s (physical variants: **1/120 s**) |
| Decimation | 2 (physical variants: 4) — control always at 30 Hz |
| Episode length | **6.0 s (180 control steps)** |
| num_envs | 4 096 (train) / 50 (play) |
| Gravity | disabled on the robot (kinematic task) |
| Gripper joints | PD-stabilised (stiffness 100, damping 100, armature 10) |

## Training

### On the Alex cluster (recommended)

See the [Alex quickstart](../../../../../../../../doc/Alex_cluster/alex_quickstart.md). From
the repo root on a login node:

```bash
source .config/env_vars.sh

# Single variant (100k timesteps ≈ 30–45 min on one GPU)
bash tools/train_alex.sh -t 02:00:00 Template-Reach-UR5e-F140-OSC-v0 --headless

# Smoke test a config change first (cheap)
bash tools/train_alex.sh -i 60 -t 00:20:00 -j smoke Template-Reach-UR5e-F140-OSC-v0 --headless --num_envs 1024

# The whole 24-variant grid, or a subset
bash tools/train_reach_alex.sh
bash tools/train_reach_alex.sh --arms "UR5e-F140 Kinova-F140" --spaces "osc ikabs"

# Multi-seed study of one variant
bash tools/train_alex.sh -s "7 42 123" -t 02:00:00 Template-Reach-UR10-Frankenstein-OSC-v0 --headless
```

### Locally

```bash
cd src/tensegrity_pick

conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/train.py \
    --task Template-Reach-UR5e-F140-v0 --headless

# Tensegrity family pipeline (train + plots + reports)
./scripts/train_reach.sh tensegrity tensegrity_tendon
```

### Evaluation and Play

```bash
# Rigorous evaluation → JSON (success_rate / success_held / position_error / reach_time)
conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/evaluate_reach.py \
    --task Template-Reach-UR5e-F140-Play-v0

# Interactive play of the latest checkpoint (workstation only — rendering
# does NOT work on Alex, see doc/Alex_cluster/alex_quickstart.md §7a)
conda run --no-capture-output -n env_isaaclab \
    python3 scripts/skrl/play.py \
    --task Template-Reach-UR5e-F140-Play-v0 --num_envs 10
```

## Training Results

### F140 grid

Converged values (last 10 % of training), **5 seeds {0,1,2,42,123}, 100k
timesteps**, runs of 2026-07-08. Cell = **mean position error (cm) / mean % of
steps within 5 cm** over the 5 seeds. Orientation is loose by design (position
task); see the pose-tracking read-out below.

| Arm | Joint | IK-Rel | IK-Abs | OSC |
|---|---|---|---|---|
| UR5e-F140 | 3.2 / 91 % | 2.3 / 93 % | **1.9 / 95 %** | 15.8 / 65 % ‡ |
| UR5e-Frankenstein | 3.6 / 88 % | 3.7 / 89 % | 4.2 / 85 % | 28.0 / 51 % ‡ |
| UR10-F140 | 3.4 / 91 % | 3.3 / 90 % | 3.7 / 90 % | 7.3 / 83 % ‡ |
| UR10-Frankenstein | 5.7 / 83 % | 4.7 / 86 % | 5.0 / 86 % | 23.0 / 55 % ‡ |
| Kinova-F140 | 2.7 / 90 % | 3.0 / 90 % | 4.2 / 80 % | 2.7 / 90 % |
| Kinova-Frankenstein | 4.1 / 83 % | 2.9 / 92 % | 2.3 / 92 % | 2.6 / 91 % |

‡ **OSC is only seed-robust on the redundant Kinova arms.** On the 6-DOF UR arms
it is **bimodal** — converges on some seeds, diverges (20–60 cm, collapsed
episodes) on others; the mean hides that. The single-seed(42) grid reported these
as passing because seed 42 happened to be a converging seed. The Kinova-OSC cells
are tight (std ≤ 0.8 cm). Full per-seed breakdown and std: see the
[tracking log](../../../../../../../../doc/reports/reach_optimization_tracking.md),
**iteration 17**. Open problem: stabilise OSC on the non-redundant UR arms.

**Orientation (pose-tracking).** Reading `pose_reached` (position AND orientation
< 0.3 rad, the honest signal — `orientation_reached` alone is inflated by
collapsed OSC episodes): real orientation tracking is a **Kinova-only,
redundancy-scaled** phenomenon — best `Kinova-Frankenstein IK-Abs ≈ 29 %`, then
Kinova-Frankenstein joint/ik ≈ 21 %, Kinova-F140 4–12 %. **Every UR arm (rigid and
Frankenstein) is 0–3 %** — the tensegrity wrist does not rescue orientation there
(P2: geometrically unreachable for 6-DOF-effective arms). OSC suppresses it even
on the redundant arm (Kinova-Frankenstein-OSC ≈ 1 %, wrist excluded from the
Jacobian). Iteration 16 shows a gated reward lifts this to 75 % where the
redundancy is controller-accessible (`kinova_f140_osc`).

Per-variant figures and reports are generated into
[`figures/<variant>/`](figures/) and [`reports/<variant>/`](reports/) — e.g.
[figures/ur5e_f140_osc/](figures/ur5e_f140_osc/),
[reports/ur5e_f140_osc/](reports/ur5e_f140_osc/). Log-dir names use the
variant naming `<arm>[_ik|_ikabs|_osc]` (no suffix = joint).

### Tensegrity family

Historical results (older configs — 12 s episodes, FK full-pose targets; see
the tracking log for context):

| Variant | Steps | Total Reward | Report | Date |
|---|---|---|---|---|
| Tensegrity PD | 48k | +0.77 | [reports/tensegrity/](reports/tensegrity/) | 2026-04-06 |
| Tensegrity Tendon | 48k | +0.75 | [reports/tensegrity_tendon/](reports/tensegrity_tendon/) | 2026-04-06 |
| Tensegrity Physical Tendon | 150k | −0.35 (best) | [reports/tensegrity_physical_tendon/](reports/tensegrity_physical_tendon/) | 2026-04-08 |

### Regenerating Plots and Reports

```bash
cd src/tensegrity_pick

# One variant (uses the latest run automatically; --variant name = log-dir name)
conda run --no-capture-output -n env_isaaclab \
    python3 scripts/plot_reach_training_results.py --variant ur5e_f140_osc

# All 24 grid variants
for arm in ur10_f140 ur10_frankenstein ur5e_f140 ur5e_frankenstein kinova_f140 kinova_frankenstein; do
  for sfx in "" _ik _ikabs _osc; do
    conda run --no-capture-output -n env_isaaclab \
        python3 scripts/plot_reach_training_results.py --variant "$arm$sfx"
  done
done

# Multi-seed aggregate plot
conda run --no-capture-output -n env_isaaclab \
    python3 scripts/plot_reach_training_results.py --seeds \
    logs/skrl/reach/ur10_frankenstein_osc/*_ppo_torch_seed*
```

## Related

- [Action spaces of the grid](action_spaces.md) — controllers, gains, fairness rules
- [Robot specification](../../../../../../../../res/Tensegrity/README.md) — tensegrity kinematic chain, tendon geometry
- [Tendon simulation](../../../../../../../../doc/Tensegrity_robot/tendon_simulation.md) — physics model and validation
- [Cube place task](../cube_place/README.md) · [Cube sort task](../cube_sort/README.md)
- [Extension overview](../../../../../../README.md) — all registered tasks and scripts

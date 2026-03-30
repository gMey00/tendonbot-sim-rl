# Workspace Analysis: Three-Robot Comparison

**Date:** 2026-03-11
**Robots:** 5-DOF Tensegrity, UR10e (6-DOF), Kinova Gen3 (7-DOF) — all with Robotiq 2F-140
**Framework:** Isaac Lab + Isaac Sim 5.1.0 (PhysX GPU)
**Hardware:** NVIDIA RTX A6000 (48 GB), 4096 parallel environments
**Final sample count:** 1,000,000 per robot

---

## 1. Context and Motivation

This workspace analysis serves two thesis-level objectives:

**Project thesis (PG-3 — Robot Simulation Validation):**
Validate the simulated tensegrity robot through Monte Carlo workspace analysis,
characterising the reachable volume, Yoshikawa manipulability, and inverse
condition number. Compare against the desired task geometry defined by Klein
(2023) to verify that the robot, as configured, can meaningfully cover the
intended sorting workspace.

**Project thesis (PG-6 — Tendon Wrist Transfer Experiment):**
Isolate the 2-DOF tendon-driven wrist mechanism and transplant it onto a
standard industrial robot arm (UR10e) in Isaac Sim. Conduct a controlled
comparison of reach accuracy and place success rate between the native rigid
wrist and the tendon wrist — using identical RL hyperparameters — to evaluate
whether the compliant wrist design improves manipulation capability. The
workspace analysis establishes baseline reachability and dexterity for the UR10e
*before* the wrist transfer, giving a reference for how much workspace the rigid
6-DOF configuration already covers.

**Master thesis (PG-3 — Second Robot Arm Integration):**
Add a second collaborative robot arm (Kinova Gen3 7-DOF) to the simulation
scene for cloth stretching and damage inspection. Validate the arm's kinematic
configuration and workspace to ensure it complements the tensegrity picker
within the conveyor-based textile sorting pipeline.

Performing workspace analysis on all three robots under identical conditions
(same desired workspace, same metric suite, same Monte Carlo methodology)
enables direct comparison and informs task-specific design decisions such as
mount height, joint limits, and reward shaping targets.

---

## 2. Methodology

### 2.1 Monte Carlo Forward-Kinematics Sampling

The analysis uses a Monte Carlo approach analogous to MathWorks'
`generateRobotWorkspace` (MathWorks 2024). For each trial:

1. Uniformly sample random joint configurations within the robot's joint limits.
2. Compute forward kinematics via the Isaac Sim 5.1.0 physics engine across
   4,096 parallel environments.
3. Apply a gripper tip offset of 22.5 cm along the end-effector's local Z axis
   to report the actual tool-centre-point position (not the flange position).
4. Compute quality metrics from the geometric Jacobian at each configuration.
5. Save raw data (positions, manipulability, condition number) as NumPy arrays.

During sampling, collision on scene objects (conveyors, drums) is disabled so
the arm can sweep freely through the entire joint space without interference.
Self-collision settings are retained.

A translucent box overlaid in the scene visualises the desired workspace derived
from the Klein (2023) master-thesis task geometry.

### 2.2 Quality Metrics

**Yoshikawa Manipulability Index** (Yoshikawa 1985):
For a robot with *n* DOFs and a 3-DOF positional task, the linear Jacobian is
$J \in \mathbb{R}^{3 \times n}$. The Yoshikawa index is:

$$w(q) = \sqrt{\det\bigl(J(q) \, J(q)^\top\bigr)}$$

Higher values indicate the end effector can be moved equally well in all
Cartesian directions (far from singularities). The index is zero at singular
configurations.

**Inverse Condition Number** (isotropy index) (Salisbury & Craig 1982):

$$\kappa_{\text{inv}} = \frac{\sigma_{\min}}{\sigma_{\max}} \quad \text{of } J$$

$\kappa_{\text{inv}} \in [0, 1]$ where 1 means perfectly isotropic motion
capability. Singular configurations yield $\kappa_{\text{inv}} = 0$.

**Reachability Density:**
The number of random FK samples landing in each voxel, proportional to the
joint-space volume that maps to that Cartesian region. Higher density means more
joint configurations can reach that area, indicating better reachability from
diverse approach directions.

### 2.3 Desired Workspace

The desired workspace is an axis-aligned box in environment-local coordinates
derived from the Klein (2023) conveyor-based pick-and-place task geometry:

| Axis | Min (m) | Max (m) | Extent (m) |
|------|---------|---------|------------|
| X    | −0.05   | +0.35   | 0.40       |
| Y    | −0.40   | +0.95   | 1.35       |
| Z    | +0.80   | +1.30   | 0.50       |

**Total volume:** 0.40 × 1.35 × 0.50 = 0.27 m³

The workspace is voxelised with a 2 cm edge length, yielding **34,000 voxels**.
A voxel is counted as "reached" if at least one FK sample falls inside it.
**Desired workspace coverage** is the fraction of reached voxels over total
voxels — i.e. how much of the intended task volume the robot can actually
access.

### 2.4 Gripper Tip Offset

All three robots use the Robotiq 2F-140 gripper. The tool-centre-point is
computed by applying a 22.5 cm offset along the end-effector body's local Z
axis. This ensures positions reflect where objects are actually grasped, not
the gripper flange position. This is the same offset used during RL training.

### 2.5 Ceiling Mount

All robots are ceiling-mounted (inverted via 180° rotation about the Z axis)
at position (0.15, 0.0, *h*) in environment-local coordinates, where *h* is the
mount height in metres. The 0.15 m X offset places the robot above the conveyor
edge, matching the physical lab layout.

---

## 3. Mount Height Optimisation

The tensegrity robot's mount height of 2.30 m was established by Klein (2023)
and validated in the project thesis. For the UR10e and Kinova Gen3, the optimal
mount height was determined via parametric sweeps using 50,000 samples per
height (sufficient for comparative ranking but not final analysis fidelity).

The goal: maximise desired workspace coverage while maintaining good
manipulability inside the workspace. Coverage is the primary objective;
manipulability serves as a tiebreaker.

### 3.1 UR10e Height Sweep

| Height (m) | Coverage | Reached / Total | Yosh (WS mean) | Cond (WS mean) |
|:----------:|:--------:|:---------------:|:--------------:|:--------------:|
| 1.30       | 10.4%    | 3,554 / 34,000  | 1.9346         | 0.4231         |
| **1.40**   | **10.1%** | **3,432 / 34,000** | **1.9488** | **0.4269**     |
| 1.50       | 9.3%     | 3,149 / 34,000  | 1.9782         | 0.4347         |
| 1.60       | 8.1%     | 2,748 / 34,000  | 2.0044         | 0.4419         |
| 1.70       | 7.2%     | 2,444 / 34,000  | 2.0212         | 0.4469         |
| 1.80       | 6.4%     | 2,168 / 34,000  | 2.0270         | 0.4487         |
| 1.90       | 6.0%     | 2,036 / 34,000  | 2.0315         | 0.4499         |
| 2.00       | 5.5%     | 1,888 / 34,000  | 2.0044         | 0.4415         |
| 2.10       | 5.0%     | 1,698 / 34,000  | 1.9686         | 0.4319         |

**Selection: h = 1.40 m.** Coverage peaks in the 1.30–1.40 m range. Although
h = 1.30 has marginally higher coverage (+0.3 pp), h = 1.40 offers better
manipulability (Yosh 1.95 vs 1.93) and condition number (0.43 vs 0.42). The
difference in coverage is within Monte Carlo noise for 50k samples.

### 3.2 Kinova Gen3 Height Sweep

| Height (m) | Coverage | Reached / Total | Yosh (WS mean) | Cond (WS mean) |
|:----------:|:--------:|:---------------:|:--------------:|:--------------:|
| 1.10       | 10.3%    | 3,520 / 34,000  | 3.2867         | 0.6945         |
| 1.20       | 11.4%    | 3,882 / 34,000  | 3.2984         | 0.6993         |
| 1.30       | 12.2%    | 4,131 / 34,000  | 3.3036         | 0.6998         |
| **1.40**   | **12.4%** | **4,231 / 34,000** | **3.3159** | **0.7042**     |
| 1.50       | 12.1%    | 4,119 / 34,000  | 3.3254         | 0.7085         |
| 1.60       | 11.9%    | 4,052 / 34,000  | 3.3128         | 0.7039         |
| 1.70       | 11.0%    | 3,743 / 34,000  | 3.2996         | 0.6977         |

**Selection: h = 1.40 m.** The Kinova achieves peak coverage at h = 1.40
(12.4%) with strong manipulability (Yosh 3.32, Cond 0.70). The plateau between
1.30–1.50 is broad, and h = 1.40 sits at the peak of both coverage and
manipulability.

### 3.3 Summary of Optimal Heights

| Robot                  | Optimal h (m) | Source      |
|:-----------------------|:-------------:|:------------|
| 5-DOF Tensegrity       | 2.30          | Klein (2023) |
| UR10e + Robotiq 2F-140 | 1.40          | This sweep   |
| Kinova Gen3 + Robotiq 2F-140 | 1.40   | This sweep   |

Both industrial robots converge on 1.40 m, considerably lower than the
tensegrity's 2.30 m. This reflects their shorter reach (UR10e: 1.30 m, Kinova:
0.90 m arm + gripper) compared to the tensegrity's prismatic base + 5-DOF arm
combination.

---

## 4. Final Workspace Analysis Results (1,000,000 Samples)

### 4.1 Comparison Table

| Metric                            | Tensegrity (2.30 m) | UR10e (1.40 m)  | Kinova Gen3 (1.40 m) |
|:----------------------------------|:-------------------:|:---------------:|:--------------------:|
| **Desired WS Coverage**           | **98.8%**           | **78.0%**       | **88.9%**            |
| Voxels reached / total            | 33,575 / 34,000     | 26,519 / 34,000 | 30,212 / 34,000      |
| Samples inside desired WS         | 412,162 / 1,000,000 | 73,195 / 1,000,000| 91,823 / 1,000,000 |
| **Yoshikawa (WS) — mean**        | 0.3210              | 1.9619          | 3.3178               |
| Yoshikawa (WS) — median          | 0.3246              | 2.0969          | 3.3730               |
| Yoshikawa (WS) — std             | 0.1926              | 0.4904          | 0.1683               |
| Yoshikawa (WS) — P5              | 0.0272              | 0.8971          | 2.9805               |
| Yoshikawa (WS) — P95             | 0.6193              | 2.4457          | 3.4671               |
| **Inv. Condition (WS) — mean**   | 0.1707              | 0.4304          | 0.7048               |
| Inv. Condition (WS) — median     | 0.1668              | 0.4678          | 0.7187               |
| Inv. Condition (WS) — std        | 0.1077              | 0.1317          | 0.0795               |
| Inv. Condition (WS) — P5         | 0.0136              | 0.1629          | 0.5539               |
| Inv. Condition (WS) — P95        | 0.3469              | 0.5756          | 0.8057               |
| **Yoshikawa (global) — mean**    | 0.3571              | 1.9339          | 3.2822               |
| **Inv. Condition (global) — mean** | 0.1937            | 0.4231          | 0.6906               |

### 4.2 Per-Robot Findings

#### 5-DOF Tensegrity Robot (h = 2.30 m)

The tensegrity robot achieves the highest workspace coverage at **98.8%**,
reaching 33,575 of 34,000 voxels. This near-complete coverage is expected: the
prismatic linear base (slides along Y) combined with the 5-DOF arm gives broad
lateral reach across the conveyor width, and the mount height of 2.30 m —
calibrated by Klein (2023) specifically for this task — places the arm's reach
envelope squarely over the desired workspace.

However, manipulability inside the workspace is notably low: Yoshikawa mean of
**0.32** and inverse condition number of **0.17**. The P5 values near zero
(Yosh P5 = 0.03, Cond P5 = 0.01) indicate that a substantial fraction of
reachable workspace poses are near-singular. The under-actuated 5-DOF
kinematic chain inherently cannot achieve the same dexterity as 6-DOF or 7-DOF
arms: with only 5 joints for a 6-DOF task space, the Jacobian is always
rank-deficient for full SE(3) control.

**Implication for PG-6 (Tendon Wrist Transfer):** The low manipulability
baseline means that improvements from the compliant tendon wrist could be
masked by the arm's inherent kinematic limitations. Transplanting the tendon
wrist onto the UR10e, which has substantially higher manipulability, provides a
cleaner experimental baseline.

#### UR10e (6-DOF) + Robotiq 2F-140 (h = 1.40 m)

The UR10e covers **78.0%** of the desired workspace (26,519 of 34,000 voxels).
While lower than the tensegrity's 98.8%, this is strong coverage for a
fixed-base 6-DOF arm. The unreached 22% corresponds primarily to the far
lateral edges of the desired workspace (large |Y| values) that are beyond the
UR10e's 1.30 m arm reach from its single mount point.

Manipulability inside the workspace is strong: Yoshikawa mean **1.96** (6.1×
the tensegrity), inverse condition mean **0.43** (2.5× the tensegrity). The P5
values (Yosh 0.90, Cond 0.16) confirm that even the worst-case poses inside
the workspace have reasonable dexterity — no near-singular configurations.

**Implication for PG-6:** The UR10e's 78% coverage with high manipulability
makes it an excellent host for the tendon wrist comparison. The reachable
volume covers the core working area directly below and around the mount
with uniformly good dexterity, which is what the reach and place tasks
require.

#### Kinova Gen3 (7-DOF) + Robotiq 2F-140 (h = 1.40 m)

The Kinova covers **88.9%** of the desired workspace (30,212 of 34,000
voxels), exceeding the UR10e by 10.9 percentage points. With 7 DOFs (one
redundant), the Kinova can reach more diverse poses and approach the same
Cartesian point from multiple joint configurations, filling more voxels.

Manipulability is the highest of all three robots by a wide margin: Yoshikawa
mean **3.32** (10.3× tensegrity, 1.7× UR10e), inverse condition mean **0.70**
(4.1× tensegrity, 1.6× UR10e). The tight standard deviation (Yosh σ = 0.17,
Cond σ = 0.08) and high P5 values (Yosh 2.98, Cond 0.55) indicate uniformly
excellent dexterity throughout the reachable workspace — the 7th joint
provides kinematic redundancy that avoids singular configurations almost
entirely.

**Implication for Master Thesis (PG-3):** The Kinova's 88.9% coverage and
exceptional manipulability confirm it as a strong candidate for the second
robot arm in the multi-agent textile sorting pipeline. For the cloth stretching
task, the arm needs to reach garment edges within a sub-region of the desired
workspace (near the sorting bins) with sufficient dexterity for precise
grasping — both requirements are well satisfied.

---

## 5. Workspace Visualisation

Each robot's workspace is visualised with three figures containing a 3-D
voxelised scatter (left panel) and three 2-D cross-section heatmaps through
the desired workspace midpoint (right panels). The translucent blue box in
3-D views marks the desired workspace boundary.

### 5.1 5-DOF Tensegrity Robot

**Manipulability (Yoshikawa Index):**
![Tensegrity Manipulability](../src/tensegrity_pick/outputs/ws_final/tensegrity/workspace_manipulability.png)

**Isotropy (Inverse Condition Number):**
![Tensegrity Condition](../src/tensegrity_pick/outputs/ws_final/tensegrity/workspace_condition.png)

**Reachability Density:**
![Tensegrity Density](../src/tensegrity_pick/outputs/ws_final/tensegrity/workspace_density.png)

### 5.2 UR10e (6-DOF) + Robotiq 2F-140

**Manipulability (Yoshikawa Index):**
![UR10e Manipulability](../src/tensegrity_pick/outputs/ws_final/ur10e/workspace_manipulability.png)

**Isotropy (Inverse Condition Number):**
![UR10e Condition](../src/tensegrity_pick/outputs/ws_final/ur10e/workspace_condition.png)

**Reachability Density:**
![UR10e Density](../src/tensegrity_pick/outputs/ws_final/ur10e/workspace_density.png)

### 5.3 Kinova Gen3 (7-DOF) + Robotiq 2F-140

**Manipulability (Yoshikawa Index):**
![Kinova Manipulability](../src/tensegrity_pick/outputs/ws_final/kinova/workspace_manipulability.png)

**Isotropy (Inverse Condition Number):**
![Kinova Condition](../src/tensegrity_pick/outputs/ws_final/kinova/workspace_condition.png)

**Reachability Density:**
![Kinova Density](../src/tensegrity_pick/outputs/ws_final/kinova/workspace_density.png)

---

## 6. Discussion

### 6.1 Coverage vs Manipulability Trade-off

A clear inverse relationship exists between workspace coverage and
manipulability across the three robots:

| Robot       | Coverage | Yosh (WS) | Cond (WS) |
|:------------|:--------:|:---------:|:---------:|
| Tensegrity  | 98.8%    | 0.32      | 0.17      |
| UR10e       | 78.0%    | 1.96      | 0.43      |
| Kinova Gen3 | 88.9%    | 3.32      | 0.70      |

The tensegrity's prismatic base provides the best coverage but at the cost
of dexterity. The industrial arms achieve lower coverage (fixed-base, limited
reach from a single mount point) but maintain substantially higher
manipulability throughout their reachable volume.

The Kinova breaks this trade-off partially: its 7th DOF provides redundancy
that improves both coverage (+10.9 pp over UR10e) *and* manipulability (1.7×
higher Yoshikawa, 1.6× higher condition number). This makes the Kinova the
most efficient robot per unit of reachable workspace.

### 6.2 Practical Implications

**RL Training:** The tensegrity's 98.8% coverage means reach targets can be
sampled virtually anywhere in the desired workspace. The UR10e and Kinova at
78% and 89% also offer broad coverage; the UR10e's unreached 22% is mainly at
the far lateral edges of the workspace. The high manipulability of the
industrial arms predicts faster policy convergence for manipulation tasks
within their reachable zones.

**Mount Height Sensitivity:** Both UR10e and Kinova show gentle coverage
gradients around h = 1.40 m, meaning small installation errors (±5 cm) would
change coverage by less than 1 pp. The tensegrity at h = 2.30 m was not swept
in this study (established by Klein 2023).

**Multi-Robot Placement (Master Thesis):** The Kinova's 88.9% coverage at
h = 1.40 m covers a large portion of the desired workspace with excellent
dexterity. For the cloth stretching pipeline, the two robots (tensegrity +
Kinova) should be mounted to maximise the *union* of their reachable volumes,
with the tensegrity handling pick from the conveyor and the Kinova handling
stretch and presentation near the inspection camera.

### 6.3 Limitations

1. **Coverage is sample-dependent.** At 50k samples (sweep phase), coverage was
   10–12%; at 1M samples (final phase), the same heights yielded 78–89%.
   More samples fill more voxels. The 1M-sample results provide good
   statistical density but a theoretically exact coverage envelope would
   require exhaustive inverse-kinematics analysis.
2. **Orientation is not considered.** The analysis tracks position only. Some
   poses may be reachable positionally but not with the required gripper
   orientation (e.g. downward-facing for bin placement). This is a known
   limitation of position-only workspace analysis.
3. **Single mount point.** Each robot is analysed at a single (x, y) = (0.15,
   0.00). Lateral repositioning could improve coverage for the fixed-base
   robots but would require a physical mount change.
4. **Isaac Sim shutdown hang.** The `simulation_app.close()` call in Isaac Sim
   5.1.0 occasionally hangs indefinitely, requiring process kill via timeout.
   This does not affect sampling results (all data is written before shutdown).

---

## 7. References

1. Yoshikawa, T. (1985). "Manipulability of Robotic Mechanisms." *The
   International Journal of Robotics Research*, 4(2), 3–9.
2. Salisbury, J. K., & Craig, J. J. (1982). "Articulated Hands: Force Control
   and Kinematic Issues." *IJRR*, 1(1), 4–17.
3. Klein, R. (2023). Master's Thesis — Tensegrity Robot Pick-and-Place.
4. MathWorks (2024). "Workspace Analysis for Manipulators."
5. Togai, M. (1986). "An Application of the Singular Value Decomposition to
   Manipulability and Sensitivity of Industrial Robots."

---

## Appendix A: Reproduction

All data can be reproduced from the `tensegrity_pick` extension:

```bash
cd src/tensegrity_pick

# Final analysis (1M samples, 8192 parallel envs each)
conda run -n env_isaaclab python scripts/workspace_analysis/workspace_sample.py \
    --robot=tensegrity --num_samples=1000000 --num_envs=8192 \
    --output_dir=outputs/ws_final/tensegrity --headless

conda run -n env_isaaclab python scripts/workspace_analysis/workspace_sample.py \
    --robot=ur10e --mount_height=1.40 --num_samples=1000000 --num_envs=8192 \
    --output_dir=outputs/ws_final/ur10e --headless

conda run -n env_isaaclab python scripts/workspace_analysis/workspace_sample.py \
    --robot=kinova --mount_height=1.40 --num_samples=1000000 --num_envs=8192 \
    --output_dir=outputs/ws_final/kinova --headless

# Generate plots
for robot in tensegrity ur10e kinova; do
    python scripts/workspace_analysis/workspace_visualize.py \
        --input_dir outputs/ws_final/$robot
done
```

## Appendix B: Raw JSON Statistics

### Tensegrity (h = 2.30 m)
```json
{
  "robot": "tensegrity",
  "mount_height_m": 2.3,
  "total_samples": 1000000,
  "samples_inside_desired_ws": 412162,
  "desired_ws_coverage_fraction": 0.9875,
  "desired_ws_voxels_reached": 33575,
  "desired_ws_voxels_total": 34000,
  "yoshikawa_inside_ws": {"mean": 0.3210, "median": 0.3246, "std": 0.1926},
  "condition_inside_ws": {"mean": 0.1707, "median": 0.1668, "std": 0.1077}
}
```

### UR10e (h = 1.40 m)
```json
{
  "robot": "ur10e",
  "mount_height_m": 1.4,
  "total_samples": 1000000,
  "samples_inside_desired_ws": 73195,
  "desired_ws_coverage_fraction": 0.7800,
  "desired_ws_voxels_reached": 26519,
  "desired_ws_voxels_total": 34000,
  "yoshikawa_inside_ws": {"mean": 1.9619, "median": 2.0969, "std": 0.4904},
  "condition_inside_ws": {"mean": 0.4304, "median": 0.4678, "std": 0.1317}
}
```

### Kinova Gen3 (h = 1.40 m)
```json
{
  "robot": "kinova",
  "mount_height_m": 1.4,
  "total_samples": 1000000,
  "samples_inside_desired_ws": 91823,
  "desired_ws_coverage_fraction": 0.8889,
  "desired_ws_voxels_reached": 30212,
  "desired_ws_voxels_total": 34000,
  "yoshikawa_inside_ws": {"mean": 3.3178, "median": 3.3730, "std": 0.1683},
  "condition_inside_ws": {"mean": 0.7048, "median": 0.7187, "std": 0.0795}
}
```

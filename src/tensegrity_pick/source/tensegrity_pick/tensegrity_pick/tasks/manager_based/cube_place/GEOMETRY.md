# Tensegrity Place – Empirical Geometry Reference

> **Measured with `scripts/measure_positions.py`** (50 zero-action steps,
> `base_z_joint = −0.25`, 1 env, headless).  Re-run the script and
> update this file whenever the robot, gripper, or mount changes.
>
> **Model reference:** `res/Tensegrity/README.md` — Key Dimensions, CAD
> Reference Points, and Gripper Offsets sections.

## Ceiling-mount orientation

The robot is mounted on the ceiling at z = 2.30 m.  The arm hangs
downward.  At the home position **all link frames have approximately
identity orientation** (local axes ≈ world axes):

- **Local +Z = World +Z** (upward, toward ceiling)
- **Local −Z = World −Z** (downward, toward floor/cube)

All finger-pad offsets in `rewards.py` must therefore be **negative**
so that `quat_apply(ee_quat, [0, 0, -offset])` yields a position
*below* `tool_link_0` in world coordinates.

## Key positions (local frame, env origin subtracted)

| Body / Point                         |  x     |  y      |  z      | Ref Z*  | Notes                                |
|--------------------------------------|--------|---------|---------|---------|--------------------------------------|
| base_y_link (mount)                  |  0.150 |  0.000  |  2.300  |  —      | ceiling attachment                   |
| base_connector_link                  |  0.190 |  0.000  |  2.300  |  —      | x-offset to arm axis                 |
| base_z_link                          |  0.190 |  0.000  |  2.542  |  —      | prismatic rail carriage              |
| tool_link (base output)              |  0.190 |  0.000  |  2.042  |  —      | linear-base output frame             |
| root_link                            |  0.190 |  0.000  |  2.042  |  2.042  | upper arm, arm origin                |
| forearm_link                         |  0.190 |  0.000  |  1.612  |  1.612  | elbow disc + forearm tube            |
| wrist_intermediate_link              |  0.190 |  0.000  |  1.206  |  1.206  | kinematic frame (dummy)              |
| wrist_link                           |  0.190 |  0.000  |  1.206  |  1.206  | lower wrist + IMU housing            |
| **tool_link_0**                      |  0.190 |  0.000  |  1.070  |  1.070  | gripper base (arm TCP)               |
| robotiq_base_link                    |  0.190 |  0.000  |  1.070  |  —      | body origin ≈ tool_link_0            |
| left_inner_finger                    |  0.190 |  0.000  |  1.070  |  —      | body origin ≈ tool_link_0            |
| right_inner_finger                   |  0.190 |  0.000  |  1.070  |  —      | body origin ≈ tool_link_0            |
| grasp_center (offset −0.1925)        |  0.190 |  0.000  | **0.877** |  —    | 5.2 cm above cube centre             |
| finger_tip_open  (−0.215)            |  0.190 |  0.000  | **0.855** |  —    | 3.0 cm above cube centre             |
| finger_tip_closed (−0.235)           |  0.190 |  0.000  | **0.835** |  —    | 1.0 cm above cube centre             |
| **Green cube centre**                |  0.152 | −0.067  | **0.825** |  —    | spawned on belt                      |
| Belt surface                         |  —     |  —      |  0.800  |  —      | CONVEYOR_SURFACE_HEIGHT_M            |

*\* Ref Z = expected world Z from README link origins: `mount_z + base_z_value + arm_offset`.*

### Reference validation (README § Key Dimensions)

| Link                    | Arm offset (m) | Expected Z | Measured Z | Δ        |
|-------------------------|-----------------|------------|------------|----------|
| root_link               |  0.000          |  2.042     |  2.042     |  0.000   |
| forearm_link            | −0.430          |  1.612     |  1.612     |  0.000   |
| wrist_intermediate_link | −0.836          |  1.206     |  1.206     |  0.000   |
| wrist_link              | −0.836          |  1.206     |  1.206     |  0.000   |
| tool_link_0             | −0.972          |  1.070     |  1.070     |  0.000   |

All reference positions match within < 0.001 m (physics settling tolerance).

## Key distances at default pose

| From → To                       | Distance |
|----------------------------------|----------|
| tool_link_0 → cube               |  0.257 m |
| grasp_center → cube              |  0.093 m |
| finger_tip_open → cube           |  0.082 m |
| grasp_center vertical gap        |  +0.052 m (above cube) |
| finger_tip_open vertical gap     |  +0.030 m (above cube) |

## tool_link_0 quaternion (w, x, y, z)

```
(1.0000, -0.0001, 0.0000, -0.0000)
```

This is approximately **identity**: local axes ≈ world axes.
Local +Z = world +Z (upward).

## Gripper offset constants (from README § Gripper Offsets)

| Constant              | Value (m) | Direction in world |
|-----------------------|-----------|--------------------|
| `GRASP_CENTER_LOCAL_Z` | −0.1925   | downward (−Z)      |
| `FINGER_TIP_OPEN_Z`   | −0.215    | downward (−Z)      |
| `FINGER_TIP_CLOSED_Z`  | −0.235    | downward (−Z)      |

## Common mistakes to avoid

1. **Sign of offsets**: The new model has identity quaternion at
   `tool_link_0` (local +Z = world +Z = upward).  Finger pad offsets
   must be **negative** so that `quat_apply(ee_quat, [0, 0, -offset])`
   projects **downward** toward the cube.  This is the opposite of the
   old URDF-based model which had a 180° rotation at the EE.

2. **`tool_link` vs `tool_link_0`**: In the 5-DOF model, `tool_link`
   is the **linear-base output frame** (at root level, z ≈ 2.04), NOT
   the arm TCP.  The arm TCP is `tool_link_0` (gripper base, z ≈ 1.07).

3. **`forearm_link` body frame vs mesh origin**: The body frame of
   `forearm_link` sits at the **elbow joint** (−0.430 m from root),
   not at the forearm mesh centroid (−0.4975 m).  Always use measured
   body positions, not mesh origins, for distance calculations.

4. **FK chain estimates**: Do not manually compute world-Z positions
   from joint offsets.  Run `scripts/measure_positions.py` to get
   ground truth.

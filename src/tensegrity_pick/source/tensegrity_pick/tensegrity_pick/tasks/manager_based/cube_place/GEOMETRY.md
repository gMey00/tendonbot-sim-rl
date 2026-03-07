# Tensegrity Place – Empirical Geometry Reference

> **Measured with `scripts/measure_positions.py`** (50 zero-action steps,
> `base_z_joint = −0.25`, 1 env, headless).  Re-run the script and
> update this file whenever the robot, gripper, or mount changes.

## Ceiling-mount orientation

The robot is mounted **upside-down** on the ceiling at z = 2.30 m.
In the `tool_link_0` local frame:

- **Local +Z → World −Z** (downward, toward floor/cube)
- **Local −Z → World +Z** (upward, toward ceiling)

All finger-pad offsets in `rewards.py` must therefore be **positive**
so that `quat_apply(ee_quat, [0, 0, +offset])` yields a position
*below* `tool_link_0` in world coordinates.

## Key positions (local frame, env origin subtracted)

| Body / Point               |  x     |  y     |  z     | Notes                                |
|-----------------------------|--------|--------|--------|--------------------------------------|
| Mount                       |  0.150 |  0.000 |  2.300 | ceiling attachment                   |
| tool_link (prismatic rail)  |  0.190 |  0.000 |  2.042 | base_z_joint = −0.258                |
| upper_arm_link              |  0.168 |  0.007 |  1.939 |                                      |
| forearm_link                |  0.191 |  0.007 |  1.394 |                                      |
| wrist_y_link                |  0.191 |  0.007 |  1.153 |                                      |
| end_effector_link           |  0.191 |  0.007 |  1.150 |                                      |
| **tool_link_0**             |  0.191 |  0.006 |  1.032 | gripper base                         |
| left_inner_finger           |  0.191 |  0.006 |  1.032 | body origin ≈ tool_link_0            |
| right_inner_finger          |  0.191 |  0.006 |  1.032 | body origin ≈ tool_link_0            |
| grasp_center (offset +0.1925) | 0.192 | 0.006 | **0.840** | 1.5 cm above cube centre         |
| finger_tip_open  (+0.215)  |  0.192 |  0.006 | **0.817** | 0.8 cm below cube centre           |
| finger_tip_closed (+0.235) |  0.192 |  0.006 | **0.797** | 2.8 cm below cube centre           |
| **Green cube centre**       |  0.169 | −0.008 | **0.825** | spawned on belt                    |
| Belt surface                |  —     |  —     |  0.800 | CONVEYOR_SURFACE_HEIGHT_M            |

## Key distances at default pose

| From → To                       | Distance |
|----------------------------------|----------|
| tool_link_0 → cube               |  0.209 m |
| **grasp_center → cube**          |  **0.030 m** |
| finger_tip_open → cube           |  0.028 m |
| grasp_center vertical gap        |  +0.015 m (above cube) |
| finger_tip_open vertical gap     |  −0.008 m (below cube) |

## tool_link_0 quaternion (w, x, y, z)

```
(-0.0006, 0.4998, -0.8661, 0.0020)
```

This rotation maps local +Z approximately to world −Z (downward).

## Common mistakes to avoid

1. **Sign of offsets**: The Robotiq 2F-140 CAD/URDF defines finger pads
   along local −Z (pointing away from base).  For a **ceiling-mounted**
   robot, local −Z points **upward** in world, so the offsets must be
   **positive** to project downward toward the cube.

2. **Body origins vs. mesh extents**: All Robotiq finger body origins
   (`left_inner_finger`, etc.) coincide with `tool_link_0` in the USD.
   The actual pad surfaces are 0.14–0.24 m further along the local +Z
   axis (world-downward).  Always use `_project_local_z_offset` with the
   constants from `rewards.py`.

3. **FK chain estimates**: Do not manually compute world-Z positions from
   URDF joint offsets — the FK is complex and easy to get wrong.
   Instead, run `scripts/measure_positions.py` to get ground truth.

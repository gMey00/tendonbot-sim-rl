# Tensegrity Arm Meshes

Source file for USD generation:

- **`tensegrity_arm_cad.usdc`** — Blender-exported USDC containing all visual
  and collision meshes with materials.  This is the single source of truth read
  by `tools/build_tensegrity_arm_usd.py`.

## Mesh Objects in the USDC

### Visual Meshes (full-resolution)

| Object name | Target link | Description |
|-------------|-------------|-------------|
| `upper_arm_visual` | `root_link` | Upper arm bracket |
| `elbow_approx_visual` | `forearm_link` | Elbow cylinder disc |
| `forearm_visual` | `forearm_link` | Rigid forearm tube |
| `lower_wrist_imu_visual` | `wrist_link` | Lower wrist + IMU housing |

### Visual Meshes (low-resolution)

| Object name | Target link | Description |
|-------------|-------------|-------------|
| `upper_arm_visual_low_res` | `root_link` | Decimated upper arm |
| `forearm_visual_low_res` | `forearm_link` | Decimated forearm |
| `lower_wrist_imu_visual_low_res` | `wrist_link` | Decimated lower wrist |

### Collision Meshes (custom convex hulls)

| Object name | Target link | Description |
|-------------|-------------|-------------|
| `upper_arm_collision` | `root_link` | Upper arm collision hull |
| `elbow_approx_collision` | `forearm_link` | Elbow disc collision hull |
| `forearm_collision` | `forearm_link` | Forearm collision hull |
| `lower_wrist_imu_collision` | `wrist_link` | Wrist collision hull |

## Other Files

- `mini_wrist_lower.stl` — lower wrist component (used by `twodof_wrist_mini`)
- `mini_wrist_upper.stl` — upper wrist component (used by `twodof_wrist_mini`)

## Deprecated

Legacy STL meshes (`Oberarm.stl`, `Unterarm_Simulation.stl`, `Endeffektor.stl`,
etc.) have been moved to `deprecated/meshes/`.  They were part of the original
URDF-based workflow and are no longer used.
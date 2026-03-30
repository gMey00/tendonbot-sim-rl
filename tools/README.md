# Tools

[← Back to project root](../README.md)

Setup and installation scripts for the development environment.

## Contents

| Script | Description |
|--------|-------------|
| `install_IsaacLab.sh` | Automated installer for NVIDIA Isaac Sim and Isaac Lab. Creates the conda environment (`env_isaaclab`), installs Isaac Sim, clones and builds Isaac Lab, and installs the selected RL framework (default: skrl). |
| `build_tensegrity_arm_usd.py` | Builds a physics-ready Isaac Lab articulation USD for the tensegrity arm from a Blender-exported mesh file (`tensegrity_arm_cad.usdc`). |

## Usage

### install_IsaacLab.sh

```bash
# From the repository root
cd tools
bash install_IsaacLab.sh          # default: installs skrl
bash install_IsaacLab.sh sb3      # alternative: installs Stable-Baselines3
```

> The script reads environment variables from `.config/env_vars.sh`.
> Make sure that file exists and defines `CONDA_PATH`, `INSTALL_PATH`, and `PROJECT_PATH`.

### build_tensegrity_arm_usd.py

Generates `res/Tensegrity/threedof_arm/tensegrity_threedof_arm_<variant>.usd` from the
Blender-exported source mesh `res/Tensegrity/meshes/tensegrity_arm_cad.usdc`.

**Requirements:** Isaac Sim's `pxr` library.

The script is self-bootstrapping: it detects Isaac Sim via the `ISAAC_PATH` environment
variable (set automatically by `python.sh`) and adds the required extscache paths at
startup — no manual `PYTHONPATH` or `LD_LIBRARY_PATH` setup needed.

**Option A — Isaac Sim's bundled Python (recommended):**
```bash
# Conda must NOT be active — deactivate first if needed
conda deactivate
~/Isaac/IsaacSim/python.sh tools/build_tensegrity_arm_usd.py
```

**Option B — your own conda env (one-time setup):**
```bash
source ~/Isaac/IsaacSim/setup_conda_env.sh   # run once per env after IsaacSim updates
python tools/build_tensegrity_arm_usd.py
```

> **Important:** Do NOT call `python.sh` while a conda environment is active.
> The script detects this and prints a clear error with the exact commands to fix it.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` / `-v` | `elbow_approx` | Robot variant: `elbow_approx`, `elbow_approx_lo`, `physical`, or `physical_lo` |
| `--src` | `res/Tensegrity/meshes/tensegrity_arm_cad.usdc` | Source mesh USDC from Blender |
| `--out-dir` | `res/Tensegrity/threedof_arm/` | Output directory |

**Example — build all variants:**
```bash
conda deactivate
~/Isaac/IsaacSim/python.sh tools/build_tensegrity_arm_usd.py --variant elbow_approx
~/Isaac/IsaacSim/python.sh tools/build_tensegrity_arm_usd.py --variant elbow_approx_lo
~/Isaac/IsaacSim/python.sh tools/build_tensegrity_arm_usd.py --variant physical
~/Isaac/IsaacSim/python.sh tools/build_tensegrity_arm_usd.py --variant physical_lo
```

**After building:**
1. Open the output USD in Isaac Sim and press Play to verify joints and collision.
2. Use Robot Assembler to attach the Robotiq 2F-140 gripper at `tool_link`.
   Save the assembled result as `res/Tensegrity/fivedof_manipulator/fivedof_manipulator_gripper.usd`.
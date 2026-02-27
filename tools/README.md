# Tools

[← Back to project root](../README.md)

Setup and installation scripts for the development environment.

## Contents

| Script | Description |
|--------|-------------|
| `install_IsaacLab.sh` | Automated installer for NVIDIA Isaac Sim and Isaac Lab. Creates the conda environment (`env_isaaclab`), installs Isaac Sim, clones and builds Isaac Lab, and installs the selected RL framework (default: skrl). |

## Usage

```bash
# From the repository root
cd tools
bash install_IsaacLab.sh          # default: installs skrl
bash install_IsaacLab.sh sb3      # alternative: installs Stable-Baselines3
```

> The script reads environment variables from `.config/env_vars.sh`.
> Make sure that file exists and defines `CONDA_PATH`, `INSTALL_PATH`, and `PROJECT_PATH`.
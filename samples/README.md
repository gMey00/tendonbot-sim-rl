# Samples

[← Back to project root](../README.md)

Reference Isaac Lab projects used during initial development to learn the framework.
Each sample is a self-contained Isaac Lab extension with its own training scripts and
environment configurations.

## Contents

| Directory | Description |
|-----------|-------------|
| `Cartpole_example/` | Classic cart-pole balancing task — the default Isaac Lab template project. Includes training logs and a shell helper (`cartpole.sh`). |
| `UR10_pose_example/` | End-effector pose reaching task for a UR10 robot with gripper. Includes training logs and a USD asset (`UR-with-gripper.usd`). |

## Usage

These samples are **not** required for the main project.  They serve as reference
implementations showing how Isaac Lab extensions, gym registrations, and skrl training
scripts are structured.

To run a sample independently:

```bash
cd samples/Cartpole_example/Cartpole
python -m pip install -e source/Cartpole
python scripts/skrl/train.py --task=Template-Cartpole-v0 --headless
```
#!/bin/bash
# Sets up a *plain-Python* OpenUSD + PhysxSchema environment from the Isaac Sim
# package shipped with Isaac Lab, WITHOUT booting the Kit runtime.
#
# This is enough to: open/author/save USD, apply UsdPhysics + PhysxSchema +
# OmniPhysics deformable APIs, and inspect prims.  It is NOT enough to run the
# omni.physx solver (that needs the full Kit app) — use python.sh for that.
#
# Usage:  source _usdenv.sh   &&   "$KPY" your_script.py
ISAAC="/home/robot/Isaac/IsaacLab/_isaac_sim"
USDLIB="$ISAAC/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
PHYSXLIB="$ISAAC/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353"
PIPBUNDLE="$ISAAC/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle"
export KPY="$ISAAC/kit/python/bin/python3"
export KIT_PLUGINS="$ISAAC/kit/plugins/bindings-python"
export PYTHONPATH="$USDLIB:$PHYSXLIB:$PIPBUNDLE:$PYTHONPATH"
export LD_LIBRARY_PATH="$USDLIB/bin:$PHYSXLIB/bin:$ISAAC/kit/python/lib:$LD_LIBRARY_PATH"
# Register the PhysxSchema + OmniPhysics deformable USD schema plugins so that
# UsdPhysics/PhysxSchema/OmniPhysics prim types resolve in plain Python.
export PXR_PLUGINPATH_NAME="$PHYSXLIB/plugins/PhysxSchema/resources:$PHYSXLIB/plugins/PhysxSchemaAddition/resources:$PHYSXLIB/plugins/OmniUsdPhysicsDeformableSchema/resources:$PXR_PLUGINPATH_NAME"

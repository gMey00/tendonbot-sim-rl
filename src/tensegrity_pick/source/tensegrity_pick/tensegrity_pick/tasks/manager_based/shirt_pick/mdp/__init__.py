# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for the shirt pick task."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Import the task-local rewards submodule AFTER the star import.  NOTE:
# ``from . import rewards`` would silently no-op here — the star import of
# isaaclab.envs.mdp already bound a ``rewards`` attribute (its own submodule)
# on this package, and ``from pkg import item`` prefers an existing attribute.
# ``from .rewards import *`` forces the real submodule import, which also
# re-binds the package attribute to the task-local module (shirt_place
# pattern).
from .rewards import *  # noqa: F401, F403, E402

from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_mdp import *  # noqa: F401, F403
from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import (  # noqa: F401
    ee_lin_vel_w,
    ee_pos_w,
    gripper_closure,
    gripper_torque_residual,
    joint_vel_out_of_limit,
)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for the shirt distribute task."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_mdp import *  # noqa: F401, F403
from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import (  # noqa: F401
    belt_collision_termination,
    belt_contact_penalty,
    ee_lin_vel_w,
    ee_pos_w,
    gripper_closure,
    gripper_torque_residual,
    joint_vel_l2_controlled,
    joint_vel_out_of_limit,
)

# Local task functions LAST and star-imported: a plain ``from . import rewards``
# would silently keep isaaclab's own ``rewards`` module bound (hard-won lesson).
from .rewards import *  # noqa: F401, F403

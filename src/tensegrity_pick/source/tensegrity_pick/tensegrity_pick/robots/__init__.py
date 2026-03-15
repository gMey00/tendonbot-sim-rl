# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralised robot configurations and actuator modules for the tensegrity project.

This package separates hardware-specific definitions (URDF-derived configs, tendon
geometry, actuator mappings) from RL task logic so that any task can import a
ready-to-use robot with a single import line.
"""

from .tendon_actuator import TendonEffortAction, TendonEffortActionCfg  # noqa: F401
from .tendon_robot_cfg import (  # noqa: F401
    TENS_3DOF_TENDON_CFG,
    TENS_5DOF_GRIPPER_TENDON_CFG,
)
from .tensegrity_robot_cfg import (  # noqa: F401
    TENS_3DOF_CFG,
    TENS_5DOF_GRIPPER_CFG,
)
from .ur10e_robot_cfg import UR10E_GRIPPER_CFG  # noqa: F401
from .kinova_gen3_robot_cfg import KINOVA_GEN3_GRIPPER_CFG  # noqa: F401

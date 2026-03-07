"""Backward-compatibility shim — canonical definitions live in ``tensegrity_pick.robots``.

All existing imports of ``TENS_3DOF_CFG`` / ``TENS_5DOF_GRIPPER_CFG`` from this
module continue to work unchanged.
"""

from tensegrity_pick.robots.tensegrity_robot_cfg import (  # noqa: F401
    PROJ_ASSETS_PATH,
    TENS_3DOF_CFG,
    TENS_5DOF_GRIPPER_CFG,
)
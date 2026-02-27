"""Backward-compatibility shim — canonical definitions live in ``tensegrity_pick.robots.tendon_actuator``.

All existing imports from this module continue to work unchanged.
"""

from tensegrity_pick.robots.tendon_actuator import (  # noqa: F401
    DEFAULT_JACOBIAN_TRANSPOSE,
    NUM_TENDONS,
    TendonEffortAction,
    TendonEffortActionCfg,
)

# Keep old private name for code that referenced it
_DEFAULT_JACOBIAN_TRANSPOSE = DEFAULT_JACOBIAN_TRANSPOSE

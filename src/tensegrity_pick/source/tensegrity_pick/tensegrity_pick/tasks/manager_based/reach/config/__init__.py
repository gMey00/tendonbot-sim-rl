# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for manager-based reach-tracking environments."""

from . import tensegrity  # noqa: F401
from . import tensegrity_tendon  # noqa: F401

# Floor-standing F140 comparison arms (mounted upright at (0.75, 1.0, 0.75),
# FK-sampled reachable EE-pose targets).
from . import ur10_f140  # noqa: F401
from . import ur10_frankenstein  # noqa: F401
from . import ur5e_f140  # noqa: F401
from . import ur5e_frankenstein  # noqa: F401
from . import kinova_f140  # noqa: F401
from . import kinova_frankenstein  # noqa: F401

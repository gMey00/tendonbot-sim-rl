# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shirt pick task (cloth-sorting pipeline, task 1).

The retrieving robot picks a shirt from the conveyor at its highest point and
holds it in front of the inspection camera.  Robot-agnostic formulation —
robot variants live in ``config/``.
"""

from . import config  # noqa: F401

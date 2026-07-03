# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shirt distribute task (cloth-sorting pipeline, task 3).

After classification, the second robot throws/places the shirt into the bin
matching its condition label (reusable / recyclable / trash).  Goal-
conditioned formulation: the label only selects which bin position enters the
observation.  Robot-agnostic — robot variants live in ``config/``.
"""

from . import config  # noqa: F401

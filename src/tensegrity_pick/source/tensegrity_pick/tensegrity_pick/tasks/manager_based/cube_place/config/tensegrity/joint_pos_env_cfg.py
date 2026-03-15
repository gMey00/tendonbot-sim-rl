# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensegrity PD-driven cube-placement environment configurations.

Re-exports the full PD environment configs from the task base module so
that gym registration entry points resolve to this package path.
"""

from tensegrity_pick.tasks.manager_based.cube_place.place_env_cfg import (  # noqa: F401
    TensegrityPlaceEnvCfg,
    TensegrityPlaceEnvCfg_PLAY,
)

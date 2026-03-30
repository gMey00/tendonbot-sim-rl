# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Tensegrity-Cube-Place-Tendon-v0",
    entry_point="tensegrity_pick.tasks.manager_based.cube_place.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:TensegrityPlaceTendonEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Tensegrity-Cube-Place-Tendon-Play-v0",
    entry_point="tensegrity_pick.tasks.manager_based.cube_place.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:TensegrityPlaceTendonEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# ── Physical tendon model (body-force elbow) ──────────────────────────────────

gym.register(
    id="Template-Tensegrity-Cube-Place-Physical-Tendon-v0",
    entry_point="tensegrity_pick.tasks.manager_based.cube_place.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg_physical:TensegrityPlacePhysicalTendonEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Tensegrity-Cube-Place-Physical-Tendon-Play-v0",
    entry_point="tensegrity_pick.tasks.manager_based.cube_place.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg_physical:TensegrityPlacePhysicalTendonEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

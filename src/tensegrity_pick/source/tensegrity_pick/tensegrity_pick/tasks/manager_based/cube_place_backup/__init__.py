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
    id="Template-Tensegrity-Cube-Place-v0",
    entry_point=f"{__name__}.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_place_env_cfg:TensegrityPlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Tensegrity-Cube-Place-Play-v0",
    entry_point=f"{__name__}.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_place_env_cfg:TensegrityPlaceEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Tensegrity-Cube-Place-Tendon-v0",
    entry_point=f"{__name__}.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_place_env_cfg:TensegrityPlaceTendonEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Tensegrity-Cube-Place-Tendon-Play-v0",
    entry_point=f"{__name__}.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_place_env_cfg:TensegrityPlaceTendonEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# UR10e variants
##

gym.register(
    id="Template-UR10e-Cube-Place-v0",
    entry_point=f"{__name__}.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_place_env_cfg:UR10ePlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-UR10e-Cube-Place-Play-v0",
    entry_point=f"{__name__}.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_place_env_cfg:UR10ePlaceEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

##
# Kinova Gen3 variants
##

gym.register(
    id="Template-Kinova-Cube-Place-v0",
    entry_point=f"{__name__}.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_place_env_cfg:KinovaPlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Kinova-Cube-Place-Play-v0",
    entry_point=f"{__name__}.place_env:TensegrityPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kinova_place_env_cfg:KinovaPlaceEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

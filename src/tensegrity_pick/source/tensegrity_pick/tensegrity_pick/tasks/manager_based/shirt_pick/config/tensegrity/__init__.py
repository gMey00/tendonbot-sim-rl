# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Shirt-Pick-Tensegrity-v0",
    entry_point="tensegrity_pick.tasks.manager_based.shirt_pick.shirt_pick_env:ShirtPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:TensegrityShirtPickEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Shirt-Pick-Tensegrity-Play-v0",
    entry_point="tensegrity_pick.tasks.manager_based.shirt_pick.shirt_pick_env:ShirtPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:TensegrityShirtPickEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Stage-2 S2: learned grasp-point refinement head (mdp/grasp_head.py).
gym.register(
    id="Template-Shirt-Pick-Head-Tensegrity-v0",
    entry_point="tensegrity_pick.tasks.manager_based.shirt_pick.shirt_pick_env:ShirtPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:TensegrityShirtPickHeadEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_head_cfg.yaml",
    },
)

gym.register(
    id="Template-Shirt-Pick-Head-Tensegrity-Play-v0",
    entry_point="tensegrity_pick.tasks.manager_based.shirt_pick.shirt_pick_env:ShirtPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:TensegrityShirtPickHeadEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_head_cfg.yaml",
    },
)

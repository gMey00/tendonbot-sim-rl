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
    id="Template-Shirt-Distribute-UR5e-F140-v0",
    entry_point="tensegrity_pick.tasks.manager_based.shirt_distribute.shirt_distribute_env:ShirtDistributeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5eF140ShirtDistributeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Shirt-Distribute-UR5e-F140-Play-v0",
    entry_point="tensegrity_pick.tasks.manager_based.shirt_distribute.shirt_distribute_env:ShirtDistributeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5eF140ShirtDistributeEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

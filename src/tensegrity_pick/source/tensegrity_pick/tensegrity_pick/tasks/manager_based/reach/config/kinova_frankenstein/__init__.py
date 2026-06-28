# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Reach-Kinova-Frankenstein-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:KinovaFrankensteinReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Reach-Kinova-Frankenstein-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:KinovaFrankensteinReachEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# ── Differential-IK (relative pose) variant ───────────────────────────────
gym.register(
    id="Template-Reach-Kinova-Frankenstein-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:KinovaFrankensteinIKRelReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_ik_cfg.yaml",
    },
)

gym.register(
    id="Template-Reach-Kinova-Frankenstein-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:KinovaFrankensteinIKRelReachEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_ik_cfg.yaml",
    },
)


# ── Differential-IK (absolute pose) variant ───────────────────────────────
gym.register(
    id="Template-Reach-Kinova-Frankenstein-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:KinovaFrankensteinIKAbsReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_ikabs_cfg.yaml",
    },
)

gym.register(
    id="Template-Reach-Kinova-Frankenstein-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:KinovaFrankensteinIKAbsReachEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_ikabs_cfg.yaml",
    },
)


# ── Operational-Space-Control (OSC) variant ───────────────────────────────
gym.register(
    id="Template-Reach-Kinova-Frankenstein-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:KinovaFrankensteinOSCReachEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_osc_cfg.yaml",
    },
)

gym.register(
    id="Template-Reach-Kinova-Frankenstein-OSC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:KinovaFrankensteinOSCReachEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_osc_cfg.yaml",
    },
)

"""skrl ``Runner`` for the shirt_distribute per-goal PPO (mode-collapse fix).

Registers two extra component names on top of the stock Runner:

  - ``GoalMultiHeadValue`` → the per-bin multi-head value factory.
  - ``PerGoalPPO``         → PPO with per-goal value + advantage normalization.

A stock PPO yaml routed through this Runner behaves identically to the stock
Runner (the overrides only add names), so the legacy baseline config still runs
unchanged for A/B comparison.  ``_generate_agent`` gains a ``PerGoalPPO`` branch
mirroring skrl's single-agent PPO path (the stock method's hardcoded class list
does not include custom agents); every other agent class delegates to ``super``.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Type

from skrl.agents.torch.ppo import PPO_CFG
from skrl.utils.runner.torch import Runner as _StockRunner

from .goal_film import _film_policy_factory, _film_value_factory
from .goal_value import goal_multi_head_value_factory
from .per_goal_ppo import PerGoalPPO


class PerGoalRunner(_StockRunner):
    """Stock skrl Runner + per-goal PPO / multi-head value components."""

    def _component(self, name: str) -> Type:  # type: ignore[override]
        lname = name.lower()
        if lname == "goalmultiheadvalue":
            return goal_multi_head_value_factory  # type: ignore[return-value]
        if lname == "filmgoalpolicy":
            return _film_policy_factory  # type: ignore[return-value]
        if lname == "filmgoalvalue":
            return _film_value_factory  # type: ignore[return-value]
        if lname == "pergoalppo":
            return PerGoalPPO  # type: ignore[return-value]
        if lname == "pergoalppo_cfg":
            return PPO_CFG  # type: ignore[return-value]
        return super()._component(name)

    def _generate_agent(self, env, cfg: dict[str, Any], models):  # type: ignore[override]
        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if agent_class != "pergoalppo":
            return super()._generate_agent(env, cfg, models)

        # --- PerGoalPPO single-agent path (mirrors skrl Runner PPO branch) ---
        cfg = copy.deepcopy(cfg)
        device = env.device
        num_envs = env.num_envs
        observation_space = env.observation_space
        state_space = env.state_space
        action_space = env.action_space

        # memory
        memory_class = self._component(cfg["memory"]["class"])
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]
        memory = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # agent cfg (identical plumbing to the stock PPO branch)
        agent_cfg = dataclasses.asdict(PPO_CFG(**self._process_cfg(cfg["agent"])))
        agent_cfg.get("observation_preprocessor_kwargs", {}).update(
            {"size": observation_space, "device": device}
        )
        agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": state_space, "device": device})
        agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})

        return PerGoalPPO(
            cfg=agent_cfg,
            models=models["agent"],
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )


__all__ = ["PerGoalRunner"]

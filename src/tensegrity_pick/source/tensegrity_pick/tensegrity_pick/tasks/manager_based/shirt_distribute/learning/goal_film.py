"""FiLM-conditioned, per-goal multi-head actor + critic (mode-collapse round 2).

After per-goal value/advantage normalization + a per-goal *critic* broke the
collapse (Phase 2), the policy still could not hold all three bins ≥ 0.85
simultaneously: it can drive one bin to 0.96 while another sits at 0.79 — a
*balance* failure, i.e. residual interference in the shared **actor** (the
critic is already per-goal). Research report §C decision tree: "if the actor
can't express per-goal behavior → strengthen goal conditioning (FiLM, #2)"; the
per-goal actor heads are the direct analog of the multi-head critic that worked.

Two capacity levers, single rollout (no Distral-style 3× cost):

  1. **FiLM trunk** (Perez et al. 2018): the goal one-hot generates per-layer
     (γ, β) that modulate every hidden activation ``h ← (1+Δγ)·h + β`` — the goal
     *gates* computation instead of being one concatenated input among 45. The
     FiLM generators are zero-initialized so training starts at identity (stable).
  2. **Per-goal heads**: 3 actor mean+log_std heads and 3 critic value heads,
     selected by the commanded bin (goal one-hot argmax over the last dims).

Both models recover the goal from the trailing ``num_bins`` observation dims
(argmax survives RunningStandardScaler; see ``per_goal_ppo``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


_TRUNK_HIDDEN: tuple[int, ...] = (256, 128, 64)


class FiLMTrunk(nn.Module):
    """MLP trunk whose every hidden layer is FiLM-modulated by the goal.

    ``forward(x, goal)`` → (B, last_hidden). ``goal`` is the (B, num_bins)
    one-hot; the FiLM generators start at identity (γ=1, β=0)."""

    def __init__(self, in_dim: int, goal_dim: int, hidden: Sequence[int] = _TRUNK_HIDDEN) -> None:
        super().__init__()
        self.linears = nn.ModuleList()
        self.films = nn.ModuleList()
        last = in_dim
        for h in hidden:
            self.linears.append(nn.Linear(last, h))
            film = nn.Linear(goal_dim, 2 * h)
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)  # start at Δγ=0, β=0 → identity modulation
            self.films.append(film)
            last = h
        self.out_dim = last
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        h = x
        for lin, film in zip(self.linears, self.films):
            h = self.act(lin(h))
            gamma, beta = film(goal).chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
        return h


class FiLMGoalPolicy(GaussianMixin, Model):
    """FiLM-trunk Gaussian actor with per-goal mean + log_std heads."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Any = None,
        num_bins: int = 3,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -3.0,
        max_log_std: float = 0.5,
        initial_log_std: float = -0.7,
        **_: Any,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            role="policy",
        )
        self.num_bins = int(num_bins)
        self.trunk = FiLMTrunk(self.num_observations, self.num_bins)
        self.mean_heads = nn.ModuleList(
            [nn.Linear(self.trunk.out_dim, self.num_actions) for _ in range(self.num_bins)]
        )
        # per-goal state-independent log_std (each goal its own exploration scale)
        self.log_std = nn.Parameter(initial_log_std * torch.ones(self.num_bins, self.num_actions))

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict]:
        x = inputs.get("states")
        if x is None:
            x = inputs["observations"]
        goal = x[:, -self.num_bins:]
        bins = goal.argmax(dim=-1)  # (B,)
        h = self.trunk(x, goal)
        means = torch.stack([head(h) for head in self.mean_heads], dim=1)  # (B, num_bins, A)
        idx = bins.view(-1, 1, 1).expand(-1, 1, self.num_actions)
        mean = means.gather(1, idx).squeeze(1)  # (B, A)
        log_std = self.log_std[bins]  # (B, A)
        return mean, {"log_std": log_std}


class FiLMGoalValue(DeterministicMixin, Model):
    """FiLM-trunk deterministic critic with one value head per goal."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Any = None,
        num_bins: int = 3,
        clip_actions: bool = False,
        **_: Any,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role="value")
        self.num_bins = int(num_bins)
        self.trunk = FiLMTrunk(self.num_observations, self.num_bins)
        self.value_heads = nn.ModuleList([nn.Linear(self.trunk.out_dim, 1) for _ in range(self.num_bins)])

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict]:
        x = inputs.get("states")
        if x is None:
            x = inputs["observations"]
        goal = x[:, -self.num_bins:]
        bins = goal.argmax(dim=-1, keepdim=True)
        h = self.trunk(x, goal)
        all_values = torch.cat([head(h) for head in self.value_heads], dim=-1)  # (B, num_bins)
        return all_values.gather(1, bins), {}


def _film_policy_factory(observation_space, action_space, device, return_source: bool = False, **kwargs):
    num_bins = int(kwargs.get("num_bins", 3))
    if return_source:
        return f"FiLMGoalPolicy(num_bins={num_bins})"
    return FiLMGoalPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        num_bins=num_bins,
        clip_actions=bool(kwargs.get("clip_actions", False)),
        clip_log_std=bool(kwargs.get("clip_log_std", True)),
        min_log_std=float(kwargs.get("min_log_std", -3.0)),
        max_log_std=float(kwargs.get("max_log_std", 0.5)),
        initial_log_std=float(kwargs.get("initial_log_std", -0.7)),
    )


def _film_value_factory(observation_space, action_space, device, return_source: bool = False, **kwargs):
    num_bins = int(kwargs.get("num_bins", 3))
    if return_source:
        return f"FiLMGoalValue(num_bins={num_bins})"
    return FiLMGoalValue(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        num_bins=num_bins,
    )


__all__ = ["FiLMGoalPolicy", "FiLMGoalValue", "_film_policy_factory", "_film_value_factory"]

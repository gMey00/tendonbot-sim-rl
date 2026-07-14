"""Per-goal multi-head value network for shirt_distribute (mode-collapse fix B1).

Research report ``doc/reports/tmp/RESEARCH_REPORT_goal_conditioned_mode_collapse``
§C #1 ("MultiCriticAL single-actor / multi-critic", Mysore et al. 2022): keep a
single shared Gaussian actor but give the *critic* one output head per commanded
bin so cross-goal value estimates stop interfering.  The shared trunk keeps the
representation compact (only 3 discrete goals) while each head owns its goal's
value scale.

Head selection is driven by the goal one-hot that ``target_bin_onehot`` appends
as the LAST ``num_bins`` observation dims (see ``mdp.rewards.target_bin_onehot``
and the env-cfg obs group).  The one-hot survives ``RunningStandardScaler``
argmax-intact — every one-hot column shares identical running stats under uniform
goal sampling, so per-column standardisation is monotonic and ``argmax`` still
recovers the commanded bin.  That lets the model self-select the head from its
own (preprocessed) input, with no side channel.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, Model


_TRUNK_HIDDEN: tuple[int, ...] = (256, 128, 64)


def _make_trunk(in_dim: int, hidden: Sequence[int] = _TRUNK_HIDDEN) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ELU())
        last = h
    return nn.Sequential(*layers), last


class GoalMultiHeadValue(DeterministicMixin, Model):
    """Deterministic value model with one linear head per commanded bin.

    The trunk mirrors the baseline value MLP ``[256, 128, 64]`` ELU; the single
    output linear is replaced by ``num_bins`` linear heads.  The active head is
    the argmax of the goal one-hot occupying the last ``num_bins`` observation
    dims, gathered per row.
    """

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
        trunk, last = _make_trunk(self.num_observations)
        self.trunk = trunk
        self.heads = nn.ModuleList([nn.Linear(last, 1) for _ in range(self.num_bins)])

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict]:
        x = inputs.get("states")
        if x is None:
            x = inputs["observations"]
        h = self.trunk(x)
        # (B, num_bins): every head's value; select the commanded one per row.
        all_values = torch.cat([head(h) for head in self.heads], dim=-1)
        bins = x[:, -self.num_bins:].argmax(dim=-1, keepdim=True)  # (B, 1)
        value = all_values.gather(1, bins)  # (B, 1)
        return value, {}


def goal_multi_head_value_factory(
    observation_space, action_space, device, return_source: bool = False, **kwargs
):
    """skrl model-instantiator factory (mirrors the cube-set factory contract).

    The Runner passes ``state_space`` too; it is absorbed by ``**kwargs`` and
    ignored (the value model uses the observation space, as the actor does)."""
    num_bins = int(kwargs.get("num_bins", 3))
    if return_source:
        return f"GoalMultiHeadValue(num_bins={num_bins})"
    return GoalMultiHeadValue(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        num_bins=num_bins,
    )


__all__ = ["GoalMultiHeadValue", "goal_multi_head_value_factory"]

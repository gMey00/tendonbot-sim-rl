"""DeepSets-encoder policy + value models for the cube-sorting task (R6).

Implements the architecture from
``doc/reports/cube_sort_research/cube_sort_mdp_redesign_consolidated.md`` §D.

Observation layout assumed (concatenated, in this exact order):

  [0 : F]      : non-set features (proprio + EE + drum + gripper smooth signals)
  [F : F+S*N]  : per-cube features, flattened — N slots × S features
  [F+S*N : F+S*N+N] : per-cube validity mask (1.0 for active, 0.0 for padded)
  [F+S*N+N : end]   : OPTIONAL privileged channels (e.g. per-cube is_holding,
                       per-cube contact magnitude). Read by the value model
                       only — the policy ignores this tail.

Both ``CubeSetPolicy`` and ``CubeSetValue`` share the same observation
vector (standard PPO requires identical obs spaces for both heads); the
asymmetry is achieved purely by which slice each model reads. This is a
strict subset of research §D — it keeps us drop-in compatible with skrl's
stock PPO ``Runner`` while preserving the privileged-information advantage.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


# ---------------------------------------------------------------------------
# Layout descriptor
# ---------------------------------------------------------------------------

class CubeSetLayout:
    """Indices into the concatenated observation tensor.

    All indices are end-exclusive, matching ``tensor[:, start:end]``
    semantics. The class is a thin dataclass-like container so the
    same descriptor can be threaded through both policy + value model
    without re-deriving slice math.
    """

    def __init__(
        self,
        non_set_dim: int,
        set_per_cube_dim: int,
        n_max_cubes: int,
        privileged_dim: int = 0,
    ) -> None:
        self.non_set_dim = int(non_set_dim)
        self.set_per_cube_dim = int(set_per_cube_dim)
        self.n_max_cubes = int(n_max_cubes)
        self.privileged_dim = int(privileged_dim)

        f, s, n = self.non_set_dim, self.set_per_cube_dim, self.n_max_cubes
        self.set_start = f
        self.set_end = f + s * n
        self.mask_start = self.set_end
        self.mask_end = self.mask_start + n
        self.priv_start = self.mask_end
        self.priv_end = self.priv_start + self.privileged_dim
        self.total_dim = self.priv_end

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(non_set, set_3d, mask, privileged)``.

        ``set_3d`` has shape ``(B, n_max_cubes, set_per_cube_dim)`` and
        ``mask`` has shape ``(B, n_max_cubes)``.
        """
        non_set = x[:, : self.non_set_dim]
        set_flat = x[:, self.set_start : self.set_end]
        set_3d = set_flat.view(x.shape[0], self.n_max_cubes, self.set_per_cube_dim)
        mask = x[:, self.mask_start : self.mask_end]
        priv = x[:, self.priv_start : self.priv_end]
        return non_set, set_3d, mask, priv


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CubeSetEncoder(nn.Module):
    """Per-cube MLP φ + masked-mean pool ρ (DeepSets, Zaheer et al. 2017)."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (64, 64), out_dim: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ELU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.phi = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, set_3d: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``set_3d``: (B, N, S); ``mask``: (B, N) ∈ {0,1}.  Returns (B, out_dim)."""
        b, n, s = set_3d.shape
        feats = self.phi(set_3d.reshape(b * n, s)).view(b, n, -1)  # (B, N, out)
        m = mask.unsqueeze(-1).clamp(0.0, 1.0)                      # (B, N, 1)
        denom = m.sum(dim=1).clamp(min=1.0)                         # (B, 1) but here (B,1)? compute below
        pooled = (feats * m).sum(dim=1) / denom                     # (B, out)
        return pooled


# ---------------------------------------------------------------------------
# Models — one Gaussian policy + one deterministic value
# ---------------------------------------------------------------------------

_TORSO_HIDDEN = (128, 64)
_ENCODER_HIDDEN = (64, 64)
_ENCODER_OUT = 64


def _make_torso(in_dim: int, hidden: Sequence[int] = _TORSO_HIDDEN) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ELU())
        last = h
    return nn.Sequential(*layers), last  # type: ignore[return-value]


class CubeSetPolicy(GaussianMixin, Model):
    """DeepSets-encoded Gaussian policy for variable-N cube observation."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Any = None,
        layout: CubeSetLayout | None = None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -2.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        **_: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            role="policy",
        )
        if layout is None:
            raise ValueError("CubeSetPolicy requires an explicit `layout` (CubeSetLayout).")
        self.layout = layout
        self.encoder = CubeSetEncoder(layout.set_per_cube_dim, _ENCODER_HIDDEN, _ENCODER_OUT)
        torso_in = layout.non_set_dim + _ENCODER_OUT
        torso, last = _make_torso(torso_in)
        self.torso = torso
        self.mean = nn.Linear(last, self.num_actions)
        self.log_std = nn.Parameter(initial_log_std * torch.ones(self.num_actions))

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = inputs["states"]
        non_set, set_3d, mask, _priv = self.layout.split(x)
        pooled = self.encoder(set_3d, mask)
        h = self.torso(torch.cat([non_set, pooled], dim=-1))
        return self.mean(h), self.log_std, {}


class CubeSetValue(DeterministicMixin, Model):
    """DeepSets-encoded deterministic value head.

    The value function additionally consumes the privileged tail (per-cube
    ``is_holding`` + per-cube contact magnitude) when present — research §D
    "asymmetric actor-critic". Implemented as an extra linear projection
    that is concatenated into the torso input, so the policy and value
    use the same overall observation tensor.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Any = None,
        layout: CubeSetLayout | None = None,
        clip_actions: bool = False,
        **_: Any,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions, role="value")
        if layout is None:
            raise ValueError("CubeSetValue requires an explicit `layout` (CubeSetLayout).")
        self.layout = layout
        self.encoder = CubeSetEncoder(layout.set_per_cube_dim, _ENCODER_HIDDEN, _ENCODER_OUT)
        torso_in = layout.non_set_dim + _ENCODER_OUT + layout.privileged_dim
        torso, last = _make_torso(torso_in)
        self.torso = torso
        self.head = nn.Linear(last, 1)

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict]:
        x = inputs["states"]
        non_set, set_3d, mask, priv = self.layout.split(x)
        pooled = self.encoder(set_3d, mask)
        h = self.torso(torch.cat([non_set, pooled, priv], dim=-1))
        return self.head(h), {}


__all__ = [
    "CubeSetLayout",
    "CubeSetEncoder",
    "CubeSetPolicy",
    "CubeSetValue",
]

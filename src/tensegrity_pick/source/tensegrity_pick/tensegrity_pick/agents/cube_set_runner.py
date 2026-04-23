"""Custom skrl ``Runner`` for the cube-sort DeepSets architecture (R6).

Overrides ``_component`` to resolve two extra component names from the
agent YAML:

  - ``CubeSetPolicy`` → factory that builds a
    :class:`tensegrity_pick.models.cube_set_policy.CubeSetPolicy`.
  - ``CubeSetValue``  → factory that builds a
    :class:`tensegrity_pick.models.cube_set_policy.CubeSetValue`.

Both factories accept the raw layout descriptor fields
(``non_set_dim``, ``set_per_cube_dim``, ``n_max_cubes``, ``privileged_dim``)
from the YAML, plus the standard ``observation_space``/``action_space``/
``device`` triple that skrl injects.

Use this Runner exactly like the stock one — the train script just needs
to import this class and pass it instead of ``skrl.utils.runner.torch.Runner``.
"""

from __future__ import annotations

from typing import Any, Type

from skrl.utils.runner.torch import Runner as _StockRunner

from tensegrity_pick.models.cube_set_policy import (
    CubeSetLayout,
    CubeSetPolicy,
    CubeSetValue,
)


def _build_layout(kwargs: dict) -> CubeSetLayout:
    return CubeSetLayout(
        non_set_dim=int(kwargs.pop("non_set_dim")),
        set_per_cube_dim=int(kwargs.pop("set_per_cube_dim")),
        n_max_cubes=int(kwargs.pop("n_max_cubes")),
        privileged_dim=int(kwargs.pop("privileged_dim", 0)),
    )


def _cube_set_policy_factory(
    observation_space, action_space, device, return_source: bool = False, **kwargs
):
    layout = _build_layout(kwargs)
    if return_source:
        return f"CubeSetPolicy(layout={layout.__dict__})"
    return CubeSetPolicy(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        layout=layout,
        **kwargs,
    )


def _cube_set_value_factory(
    observation_space, action_space, device, return_source: bool = False, **kwargs
):
    layout = _build_layout(kwargs)
    if return_source:
        return f"CubeSetValue(layout={layout.__dict__})"
    return CubeSetValue(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        layout=layout,
        **kwargs,
    )


class CubeSetRunner(_StockRunner):
    """skrl ``Runner`` extended with the cube-set model factories."""

    def _component(self, name: str) -> Type:  # type: ignore[override]
        lname = name.lower()
        if lname == "cubesetpolicy":
            return _cube_set_policy_factory  # type: ignore[return-value]
        if lname == "cubesetvalue":
            return _cube_set_value_factory  # type: ignore[return-value]
        return super()._component(name)


__all__ = ["CubeSetRunner"]

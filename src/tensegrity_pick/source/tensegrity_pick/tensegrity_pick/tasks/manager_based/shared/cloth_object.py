# cloth_object.py
#
# Minimal wrapper for cloth simulation objects in Isaac Lab.
# Provides a uniform interface for both PBD particle cloth and XPBD
# surface deformable backends.
#
# Isaac Lab's DeformableObjectCfg wraps FEM volumetric soft bodies
# (physx.SoftBodyView) — NOT particle cloth or surface deformables.
# This module bridges that gap with a thin wrapper that:
#   1. Stores cloth configuration (backend type, mesh path, parameters)
#   2. Provides placeholder methods for state access (nodal_pos, reset)
#   3. Will be filled in once PhysX tensor API access is verified
#
# Reference: cloth_simulation_isaacsim_research.md §4 (Isaac Lab integration)
#
# NOTE: This is a stub.  The actual PhysX tensor API calls
#       (SimulationView.create_particle_cloth_view) require Isaac Sim
#       runtime and must be implemented inside the running environment.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import torch


class ClothBackend(Enum):
    """Supported cloth simulation backends in Isaac Sim 5.1."""

    PBD = "pbd"
    XPBD = "xpbd"


@dataclass(frozen=True)
class PBDClothParams:
    """PBD particle cloth material parameters (cotton T-shirt defaults)."""

    stretch_stiffness: float = 2000.0
    bend_stiffness: float = 200.0
    shear_stiffness: float = 100.0
    spring_damping: float = 0.3
    drag: float = 0.02
    lift: float = 0.02
    friction: float = 0.8
    adhesion: float = 0.2
    mass_kg: float = 0.2
    solver_position_iterations: int = 16
    self_collision: bool = True


@dataclass(frozen=True)
class XPBDClothParams:
    """XPBD surface deformable material parameters (cotton T-shirt defaults)."""

    youngs_modulus: float = 5000.0
    poissons_ratio: float = 0.3
    elasticity_damping: float = 0.01
    dynamic_friction: float = 0.8
    density: float = 350.0
    solver_position_iterations: int = 16
    self_collision: bool = True


@dataclass
class ClothObjectCfg:
    """Configuration for a cloth simulation object.

    Used by scene configs to declare cloth objects.  The actual PhysX
    setup happens at environment startup, not at config time.
    """

    prim_path: str = "{ENV_REGEX_NS}/Shirt"
    usd_path: str = ""
    mesh_prim_path: str = ""  # relative path within USD to the Mesh prim
    backend: ClothBackend = ClothBackend.PBD
    pbd_params: PBDClothParams = field(default_factory=PBDClothParams)
    xpbd_params: XPBDClothParams = field(default_factory=XPBDClothParams)
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    num_keypoints: int = 8


@dataclass
class ClothObjectData:
    """Runtime state tensors for a cloth object across parallel envs.

    Shapes:
        nodal_pos_w:  (num_envs, num_vertices, 3)
        nodal_vel_w:  (num_envs, num_vertices, 3)
        root_pos_w:   (num_envs, 3) — mean of all nodal positions
        keypoint_pos: (num_envs, num_keypoints, 3) — subsampled positions
    """

    nodal_pos_w: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    nodal_vel_w: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    root_pos_w: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    keypoint_pos: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    default_nodal_state_w: torch.Tensor = field(default_factory=lambda: torch.empty(0))


class ClothObject:
    """Minimal wrapper for cloth simulation objects in Isaac Lab envs.

    This class will manage the PhysX particle cloth / surface deformable
    lifecycle across parallel environments.

    Current status: STUB — provides the interface contract.  Internal
    PhysX tensor API calls will be implemented once the simulation
    pipeline is verified end-to-end.
    """

    def __init__(self, cfg: ClothObjectCfg, num_envs: int, device: str) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._data = ClothObjectData()
        self._keypoint_indices: list[int] = []

    @property
    def data(self) -> ClothObjectData:
        return self._data

    @property
    def num_vertices(self) -> int:
        if self._data.nodal_pos_w.numel() == 0:
            return 0
        return self._data.nodal_pos_w.shape[1]

    def initialize(self) -> None:
        """Called after scene construction to set up PhysX views.

        TODO: Implement:
          - For PBD: SimulationView.create_particle_cloth_view()
          - For XPBD: SimulationView.create_deformable_body_view()
          - Compute keypoint indices via farthest point sampling
          - Store default nodal state for reset
        """
        raise NotImplementedError(
            "ClothObject.initialize() not yet implemented. "
            "Requires Isaac Sim runtime with PhysX tensor API."
        )

    def update(self) -> None:
        """Read current cloth state from simulation into data tensors.

        TODO: Implement:
          - Read nodal positions and velocities from PhysX view
          - Compute root_pos_w as mean of nodal positions
          - Extract keypoint positions
        """
        raise NotImplementedError(
            "ClothObject.update() not yet implemented."
        )

    def reset(self, env_ids: Sequence[int]) -> None:
        """Reset cloth to initial state for specified environments.

        TODO: Implement:
          - Write default_nodal_state_w back to simulation for env_ids
          - For PBD: set particle positions and velocities
          - For XPBD: write nodal state to sim
        """
        raise NotImplementedError(
            "ClothObject.reset() not yet implemented."
        )

    def write_nodal_pos_to_sim(
        self, nodal_pos: torch.Tensor, env_ids: Sequence[int]
    ) -> None:
        """Write nodal positions directly to the simulation.

        Used for randomized reset (e.g., shifted spawn position).
        """
        raise NotImplementedError(
            "ClothObject.write_nodal_pos_to_sim() not yet implemented."
        )

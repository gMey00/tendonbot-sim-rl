"""Per-cube grasp predicate for the cube-sorting task (Iteration R2).

Implements the multi-conjunct ``is_holding`` predicate.

The predicate is intentionally **per cube** (not per env) so the reward
machine in R4 can differentiate which cube is currently held in the
gripper.  The output shape is ``(N_envs, NUM_CUBES_TOTAL)``.

Conjuncts (all must hold for ``DWELL_STEPS`` consecutive frames):

1. **Bilateral contact** — both fingerpads register a normal force above
   ``F_MIN`` against the same cube (filtered ``force_matrix_w`` from the
   two ``ContactSensor`` instances created in R1).
2. **Anti-parallel normals** — the unit force vectors on the two pads
   point at each other (cosine ≤ ``COS_OPPOSE``); rules out two pads
   pushing the same face from one side.
3. **Co-motion** — the cube's linear velocity is within ``V_CO_MOVING``
   of the TCP velocity; rules out a cube flying through the open gripper.
4. **Above belt + finger angle in grip range** — geometric sanity check.
5. **Dwell** — instant predicate must be True for ``DWELL_STEPS``
   consecutive physics steps (ring buffer stored on ``env._hold_hist``).

This file is consumed by the R4 reward refactor.  Until R4 lands, the
existing ``grasp_active`` property in ``cube_sort_env.py`` is kept alive
for A/B comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - type-only imports (avoid omni.timeline at import time)
    from isaaclab.assets import Articulation, RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

# ---------------------------------------------------------------------------
# Tunable constants — see research §A "Recommendation" table.
# ---------------------------------------------------------------------------
F_MIN: float = 0.5
"""Per-finger normal force threshold (N).

Cube weight ``m·g = 0.05·9.81 ≈ 0.49 N``.  With friction ``μ ≈ 0.5``
(Isaac Lab ``RigidBodyMaterialCfg`` default), the minimum per-finger
normal force to resist gravity is ``W/(2μ) ≈ 0.49 N``.  ``F_MIN = 0.5``
gives ~1× safety factor.
"""

COS_OPPOSE: float = -0.7
"""Maximum allowed cosine of the angle between left/right pad normals.

``-0.7`` corresponds to ≈135°; rules out the degenerate case where both
pads contact the cube from the same side.
"""

V_CO_MOVING: float = 0.05
"""Maximum allowed cube–TCP relative speed (m/s).

A flying cube passing through the gripper region cannot satisfy this
because its velocity differs from the slow-moving TCP by far more than
5 cm/s.
"""

Z_ABOVE_BELT: float = 0.01
"""Minimum height above the belt surface (m) for a cube to count as held."""

DWELL_STEPS: int = 2
"""Number of consecutive physics steps the instant predicate must hold."""

GA_MIN: float = 0.20
"""Minimum gripper joint angle (rad) — below this the gripper is too open."""

GA_MAX: float = 0.78
"""Maximum gripper joint angle (rad) — above this the gripper is closed empty.

Robotiq 2F-140 ``finger_joint`` close pos is 0.7854 rad; ``GA_MAX = 0.78``
keeps a small margin so a fully-closed empty gripper is excluded.
"""

# Buffer attribute names on ``env`` (avoid collisions with other predicates)
_HIST_ATTR = "_hold_hist"
_PTR_ATTR = "_hold_hist_ptr"


def _ensure_buffers(
    env: ManagerBasedRLEnv,
    num_cubes: int,
    device: torch.device,
) -> None:
    """Allocate the dwell ring buffer lazily on first call."""
    if not hasattr(env, _HIST_ATTR):
        setattr(
            env,
            _HIST_ATTR,
            torch.zeros(
                (env.num_envs, num_cubes, DWELL_STEPS),
                dtype=torch.bool,
                device=device,
            ),
        )
        setattr(env, _PTR_ATTR, 0)


def is_holding(
    env: "ManagerBasedRLEnv",
    cubes_collection_name: str = "cubes",
    robot_name: str = "robot",
    tcp_body_name: str = "tool_link_0",
    gripper_joint: str = "finger_joint",
    left_sensor: str = "contact_left",
    right_sensor: str = "contact_right",
    z_belt: float = 0.0,
) -> torch.Tensor:
    """Compute the per-cube ``is_holding`` predicate.

    Args:
        env: The ``ManagerBasedRLEnv`` instance (used to access ``scene``
            and to attach the persistent dwell ring buffer).
        cubes_collection_name: Scene key of the ``RigidObjectCollection``
            holding all cubes.  Default ``"cubes"``.
        robot_cfg: ``SceneEntityCfg`` for the robot articulation.
        tcp_body_name: Name of the body used as the TCP for the co-motion
            check.  Default matches ``EE_BODY_NAME`` in the env.
        gripper_joint: Name of the actuated gripper joint.
        left_sensor: Scene key of the left fingerpad ``ContactSensor``.
        right_sensor: Scene key of the right fingerpad ``ContactSensor``.
        z_belt: World-frame Z of the belt surface (m).

    Returns:
        Tensor of shape ``(N_envs, N_cubes)`` and dtype ``torch.bool``.
        ``True`` for cubes that satisfy all conjuncts for ``DWELL_STEPS``
        consecutive physics steps.
    """
    # --- 1. Bilateral contact + anti-parallel normals --------------------
    sensor_l: "ContactSensor" = env.scene[left_sensor]
    sensor_r: "ContactSensor" = env.scene[right_sensor]
    # force_matrix_w shape: (N_envs, 1, N_cubes, 3) — one source body per env.
    fm_l = sensor_l.data.force_matrix_w
    fm_r = sensor_r.data.force_matrix_w
    assert fm_l is not None and fm_r is not None, (
        "is_holding: contact sensor force_matrix_w is None — check filter "
        "prim regex (R1)."
    )
    FL = fm_l.squeeze(1)  # (N, M, 3)
    FR = fm_r.squeeze(1)  # (N, M, 3)
    mag_L = FL.norm(dim=-1)
    mag_R = FR.norm(dim=-1)
    bilateral = (mag_L > F_MIN) & (mag_R > F_MIN)

    n_L = FL / mag_L.clamp_min(1e-6).unsqueeze(-1)
    n_R = FR / mag_R.clamp_min(1e-6).unsqueeze(-1)
    opposing = (n_L * n_R).sum(dim=-1) < COS_OPPOSE  # (N, M)

    # --- 2. Co-motion + above-belt --------------------------------------
    robot: "Articulation" = env.scene[robot_name]
    cubes: "RigidObjectCollection" = env.scene[cubes_collection_name]

    tcp_idx = robot.data.body_names.index(tcp_body_name)
    v_tcp = robot.data.body_lin_vel_w[:, tcp_idx]                # (N, 3)
    v_cube = cubes.data.object_lin_vel_w                         # (N, M, 3)
    p_cube = cubes.data.object_pos_w                             # (N, M, 3)

    co_moving = (v_cube - v_tcp.unsqueeze(1)).norm(dim=-1) < V_CO_MOVING
    above_belt = p_cube[..., 2] > (z_belt + Z_ABOVE_BELT)

    # --- 3. Gripper angle in grip range ---------------------------------
    g_idx = robot.data.joint_names.index(gripper_joint)
    g_ang = robot.data.joint_pos[:, g_idx].unsqueeze(-1)         # (N, 1)
    gripper_ok = (g_ang > GA_MIN) & (g_ang < GA_MAX)             # broadcast to (N, M)

    instant = bilateral & opposing & co_moving & above_belt & gripper_ok

    # --- 4. Dwell via ring buffer ---------------------------------------
    num_cubes = instant.shape[1]
    _ensure_buffers(env, num_cubes=num_cubes, device=instant.device)
    hist: torch.Tensor = getattr(env, _HIST_ATTR)
    ptr: int = getattr(env, _PTR_ATTR)
    hist[:, :, ptr] = instant
    setattr(env, _PTR_ATTR, (ptr + 1) % DWELL_STEPS)
    return hist.all(dim=-1)

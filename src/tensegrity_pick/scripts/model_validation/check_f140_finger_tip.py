"""Calibrate the Robotiq 2F-140 finger-tip offsets per EE reference frame.

The cloth-grasp machinery (``ClothSortingEnvBase._finger_tip_pos`` and the
belt-penalty helpers in ``shared/gripper_cfg.py``) models the dynamic finger
tip as a local-Z offset from the EE body.  Those offsets were calibrated on
the tensegrity's ``tool_link_0`` frame; the UR/Kinova F140 variants use the
``robotiq_base_link`` frame, whose origin and axis convention differ.

This probe measures, for a given task variant, the finger-PAD midpoint
(mean of ``left/right_inner_finger_pad`` body origins) expressed in the local
frame of each candidate EE body, at gripper OPEN and CLOSED.  On the
tensegrity (where the true tip offsets −0.215/−0.235 on ``tool_link_0`` are
validated) the same articulation also contains ``robotiq_base_link``, so the
tensegrity run directly yields the transferable ``robotiq_base_link`` tip
offsets used by ``shared/gripper_cfg.py::FRAME_TIP_TABLE``.

Uses the (fast-booting, cloth-free) reach envs — same gripper hardware.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    # reference frame (tensegrity carries the calibrated tool_link_0):
    PYTHONUNBUFFERED=1 python scripts/model_validation/check_f140_finger_tip.py \
        --headless --task Template-Reach-Tensegrity-v0
    # verify the F140 arms read the same robotiq_base_link-local values:
    PYTHONUNBUFFERED=1 python scripts/model_validation/check_f140_finger_tip.py \
        --headless --task Template-Reach-UR5e-F140-v0
    PYTHONUNBUFFERED=1 python scripts/model_validation/check_f140_finger_tip.py \
        --headless --task Template-Reach-Kinova-F140-v0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="F140 finger-tip frame calibration probe.")
parser.add_argument("--task", type=str, default="Template-Reach-UR5e-F140-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_conjugate  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import (  # noqa: E402
    FINGER_JOINT_CLOSE_POS,
)

# EE frames worth reporting (superset across robot families).
CANDIDATE_FRAMES = ("tool_link_0", "robotiq_base_link", "end_effector_link")
# Robotiq four-bar mimic ratios (same table as ClothSortingEnvBase).
GRIPPER_MIMIC = {
    "right_outer_knuckle_joint": 1.0,
    "left_outer_finger_joint": 0.0,
    "right_outer_finger_joint": 0.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": -1.0,
    "left_inner_finger_pad_joint": 1.0,
    "right_inner_finger_pad_joint": 1.0,
}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    with torch.inference_mode():
        env.reset()

    robot = u.scene["robot"]
    names = robot.body_names
    print(f"\nbody names: {names}")
    pads = [i for i, n in enumerate(names) if n.endswith("inner_finger_pad")]
    if not pads:  # this gripper USD has no separate pad bodies
        pads = [i for i, n in enumerate(names) if n.endswith("inner_finger")]
    assert len(pads) == 2, f"expected 2 finger(-pad) bodies, found {pads}"
    frames = [(n, names.index(n)) for n in CANDIDATE_FRAMES if n in names]
    fj = robot.joint_names.index("finger_joint")
    mimic = [(robot.joint_names.index(j), r) for j, r in GRIPPER_MIMIC.items()
             if j in robot.joint_names]

    for label, finger in (("OPEN", 0.0), ("CLOSED", FINGER_JOINT_CLOSE_POS)):
        with torch.inference_mode():
            ids = [fj] + [i for i, _ in mimic]
            pose = torch.tensor(
                [finger] + [finger * r for _, r in mimic], device=u.device
            ).unsqueeze(0).repeat(u.num_envs, 1)
            robot.write_joint_position_to_sim(pose, joint_ids=ids)
            robot.write_joint_velocity_to_sim(torch.zeros_like(pose), joint_ids=ids)
            robot.set_joint_position_target(pose, joint_ids=ids)
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(u.cfg.sim.dt)

        pad_mid = 0.5 * (robot.data.body_pos_w[:, pads[0], :]
                         + robot.data.body_pos_w[:, pads[1], :])
        print(f"\n[{label}] finger_joint={finger:.4f}  pad-mid world={pad_mid[0].tolist()}")
        for fname, fi in frames:
            fp = robot.data.body_pos_w[:, fi, :]
            fq = robot.data.body_quat_w[:, fi, :]
            local = quat_apply(quat_conjugate(fq), pad_mid - fp)
            print(f"  pad-mid in {fname:>20s} local frame: "
                  f"[{local[0, 0]:+.4f}, {local[0, 1]:+.4f}, {local[0, 2]:+.4f}]")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

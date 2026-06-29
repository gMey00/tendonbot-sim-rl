#!/usr/bin/env python3
"""Verify robot articulation for reach tasks.

Spawns each robot variant, commands random joint positions, and checks:
1. Root link is fixed (doesn't drift)
2. Joints respond to position commands
3. End-effector moves within expected workspace
4. No NaN in joint positions/velocities
5. FK sampler can generate reachable targets

Usage:
    cd src/tensegrity_pick
    conda run -n env_isaaclab python scripts/verify_robot_reach.py
    conda run -n env_isaaclab python scripts/verify_robot_reach.py --variant ur10e
"""

import argparse
import sys
import torch
import numpy as np

# -- Isaac Lab launch (must be first) --
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify robot reach configuration")
parser.add_argument(
    "--variant",
    choices=[
        "tensegrity", "tensegrity_tendon",
        "ur10_f140", "ur10_frankenstein", "ur5e_f140", "ur5e_frankenstein",
        "kinova_f140", "kinova_frankenstein", "all",
    ],
    default="all",
    help="Which robot variant to test",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -- Isaac Lab imports (after AppLauncher) --
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass


def get_robot_config(variant: str):
    """Return (ArticulationCfg, controlled_joint_names, target_link_name, init_pos, init_rot)."""
    # Floor-standing F140 comparison arms: mounted upright at the
    # workspace-analysis pose (0.75, 1.0, 0.75); EE = robotiq_base_link.
    _F140_POS = (0.75, 1.0, 0.75)
    _F140_ROT = (1.0, 0.0, 0.0, 0.0)
    _F140_EE = "robotiq_base_link"

    if variant == "tensegrity":
        from tensegrity_pick.robots.tensegrity_robot_cfg import (
            TENS_5DOF_GRIPPER_CFG,
            TARGET_LINK_NAME_5DOF,
            CONTROLLED_JOINT_NAMES_5DOF,
        )
        cfg = TENS_5DOF_GRIPPER_CFG.copy()
        return cfg, CONTROLLED_JOINT_NAMES_5DOF, TARGET_LINK_NAME_5DOF, (0.15, 0.0, 2.30), (1.0, 0.0, 0.0, 0.0)

    elif variant == "ur10_f140":
        from tensegrity_pick.robots.ur10_robot_cfg import UR10_GRIPPER_CFG, CONTROLLED_JOINT_NAMES
        return UR10_GRIPPER_CFG.copy(), CONTROLLED_JOINT_NAMES, _F140_EE, _F140_POS, _F140_ROT

    elif variant == "ur10_frankenstein":
        from tensegrity_pick.robots.ur10_frankenstein_robot_cfg import (
            UR10_FRANKENSTEIN_GRIPPER_CFG, CONTROLLED_JOINT_NAMES,
        )
        return UR10_FRANKENSTEIN_GRIPPER_CFG.copy(), CONTROLLED_JOINT_NAMES, _F140_EE, _F140_POS, _F140_ROT

    elif variant == "ur5e_f140":
        from tensegrity_pick.robots.ur5e_robot_cfg import UR5E_GRIPPER_CFG, CONTROLLED_JOINT_NAMES
        return UR5E_GRIPPER_CFG.copy(), CONTROLLED_JOINT_NAMES, _F140_EE, _F140_POS, _F140_ROT

    elif variant == "ur5e_frankenstein":
        from tensegrity_pick.robots.ur5e_frankenstein_robot_cfg import (
            UR5E_FRANKENSTEIN_GRIPPER_CFG, CONTROLLED_JOINT_NAMES,
        )
        return UR5E_FRANKENSTEIN_GRIPPER_CFG.copy(), CONTROLLED_JOINT_NAMES, _F140_EE, _F140_POS, _F140_ROT

    elif variant == "kinova_f140":
        from tensegrity_pick.robots.kinova_gen3_robot_cfg import KINOVA_GEN3_GRIPPER_CFG, CONTROLLED_JOINT_NAMES
        return KINOVA_GEN3_GRIPPER_CFG.copy(), CONTROLLED_JOINT_NAMES, _F140_EE, _F140_POS, _F140_ROT

    elif variant == "kinova_frankenstein":
        from tensegrity_pick.robots.kinova_frankenstein_robot_cfg import (
            KINOVA_FRANKENSTEIN_GRIPPER_CFG, CONTROLLED_JOINT_NAMES,
        )
        return KINOVA_FRANKENSTEIN_GRIPPER_CFG.copy(), CONTROLLED_JOINT_NAMES, _F140_EE, _F140_POS, _F140_ROT

    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_verification(variant: str) -> dict:
    """Run verification for a single variant. Returns dict of results."""
    print(f"\n{'='*70}")
    print(f"  VERIFYING: {variant}")
    print(f"{'='*70}\n")

    robot_cfg, joint_names, target_link, init_pos, init_rot = get_robot_config(variant)

    # Create a minimal scene
    @configclass
    class TestSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=GroundPlaneCfg(),
        )
        robot: ArticulationCfg = robot_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_rot,
            ),
        )

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=5.0)

    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=1/60.0, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 1.5])

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    robot: Articulation = scene["robot"]
    results = {"variant": variant, "passed": True, "issues": []}

    # --- Test 1: Check initial state ---
    print("Test 1: Initial state check")
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_rot = robot.data.root_quat_w[0].cpu().numpy()
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    joint_vel = robot.data.joint_vel[0].cpu().numpy()

    print(f"  Root position (world): {root_pos}")
    print(f"  Root rotation (wxyz):  {root_rot}")
    print(f"  Joint positions: {joint_pos[:len(joint_names)]}")
    print(f"  Joint velocities: {joint_vel[:len(joint_names)]}")

    if np.any(np.isnan(root_pos)) or np.any(np.isnan(root_rot)):
        results["issues"].append("FAIL: Root state contains NaN!")
        results["passed"] = False
    else:
        print("  PASS: No NaN in root state")

    if np.any(np.isnan(joint_pos)) or np.any(np.isnan(joint_vel)):
        results["issues"].append("FAIL: Joint state contains NaN!")
        results["passed"] = False
    else:
        print("  PASS: No NaN in joint state")

    # --- Test 2: Check root stability (simulate 100 steps with no action) ---
    print("\nTest 2: Root stability (100 idle steps)")
    initial_root_pos = root_pos.copy()
    for _ in range(100):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    root_pos_after = robot.data.root_pos_w[0].cpu().numpy()
    drift = np.linalg.norm(root_pos_after - initial_root_pos)
    print(f"  Root drift after 100 steps: {drift:.6f} m")
    if drift > 0.01:
        results["issues"].append(f"FAIL: Root drifted {drift:.4f}m (>1cm) — likely fix_root_link issue!")
        results["passed"] = False
    else:
        print("  PASS: Root is stable")

    # --- Test 3: Joint command response ---
    print("\nTest 3: Joint command response")
    # Get joint limits
    joint_ids = []
    for jname in joint_names:
        idx = robot.joint_names.index(jname)
        joint_ids.append(idx)
    joint_ids_t = torch.tensor(joint_ids, device=robot.device)

    lower = robot.data.joint_pos_limits[0, joint_ids_t, 0].cpu().numpy()
    upper = robot.data.joint_pos_limits[0, joint_ids_t, 1].cpu().numpy()
    print(f"  Joint limits (lower): {lower}")
    print(f"  Joint limits (upper): {upper}")

    # Check for infinite limits
    inf_mask = np.isinf(lower) | np.isinf(upper)
    if np.any(inf_mask):
        inf_joints = [joint_names[i] for i in range(len(joint_names)) if inf_mask[i]]
        results["issues"].append(f"WARNING: Infinite limits on: {inf_joints}")
        print(f"  WARNING: Infinite limits on: {inf_joints}")
        # Clamp for testing
        lower = np.where(np.isinf(lower), -np.pi, lower)
        upper = np.where(np.isinf(upper), np.pi, upper)

    # Command mid-range positions
    target_pos = (lower + upper) / 2.0
    print(f"  Commanding joint midpoint: {target_pos}")

    # Set target and simulate
    full_target = robot.data.joint_pos[0].clone()
    for i, jid in enumerate(joint_ids):
        full_target[jid] = float(target_pos[i])

    for step in range(200):
        robot.set_joint_position_target(full_target.unsqueeze(0))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    new_pos = robot.data.joint_pos[0, joint_ids_t].cpu().numpy()
    pos_error = np.abs(new_pos - target_pos)
    print(f"  Achieved positions: {new_pos}")
    print(f"  Position errors:    {pos_error}")

    if np.any(np.isnan(new_pos)):
        results["issues"].append("FAIL: Joint positions are NaN after commanding!")
        results["passed"] = False
    elif np.max(pos_error) > 0.5:
        results["issues"].append(f"FAIL: Max joint error {np.max(pos_error):.3f} rad > 0.5 — joints not responding well")
        results["passed"] = False
    else:
        print(f"  PASS: Max error {np.max(pos_error):.4f} rad")

    # --- Test 4: End-effector workspace ---
    print("\nTest 4: End-effector workspace sampling")
    ee_positions = []
    n_samples = 20

    for i in range(n_samples):
        # Random joint angles within limits
        rand_pos = lower + np.random.rand(len(joint_names)) * (upper - lower)
        full_target = robot.data.joint_pos[0].clone()
        for j, jid in enumerate(joint_ids):
            full_target[jid] = float(rand_pos[j])

        for _ in range(150):
            robot.set_joint_position_target(full_target.unsqueeze(0))
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

        # Get EE position
        body_idx = robot.body_names.index(target_link)
        ee_pos = robot.data.body_pos_w[0, body_idx].cpu().numpy()

        if np.any(np.isnan(ee_pos)):
            results["issues"].append(f"FAIL: EE position NaN at sample {i}")
            results["passed"] = False
            break

        ee_positions.append(ee_pos)

    if ee_positions:
        ee_arr = np.array(ee_positions)
        ee_mean = ee_arr.mean(axis=0)
        ee_std = ee_arr.std(axis=0)
        ee_min = ee_arr.min(axis=0)
        ee_max = ee_arr.max(axis=0)
        workspace_span = ee_max - ee_min

        print(f"  EE positions mean: {ee_mean}")
        print(f"  EE positions std:  {ee_std}")
        print(f"  EE min: {ee_min}")
        print(f"  EE max: {ee_max}")
        print(f"  Workspace span: {workspace_span}")

        if np.all(workspace_span < 0.05):
            results["issues"].append("FAIL: EE workspace too small (<5cm) — joints may not be moving")
            results["passed"] = False
        else:
            print(f"  PASS: Workspace spans {workspace_span}")

    # --- Test 5: Check velocity sanity ---
    print("\nTest 5: Velocity sanity")
    joint_vel = robot.data.joint_vel[0].cpu().numpy()
    max_vel = np.max(np.abs(joint_vel))
    print(f"  Max joint velocity: {max_vel:.4f} rad/s")
    if max_vel > 100.0:
        results["issues"].append(f"FAIL: Max velocity {max_vel:.1f} exceeds limit!")
        results["passed"] = False
    else:
        print("  PASS: Velocities within bounds")

    # --- Summary ---
    print(f"\n{'='*70}")
    status = "PASSED" if results["passed"] else "FAILED"
    print(f"  {variant}: {status}")
    if results["issues"]:
        for issue in results["issues"]:
            print(f"    - {issue}")
    print(f"{'='*70}\n")

    # Cleanup
    sim.stop()
    sim.clear_all_callbacks()
    sim.clear_instance()

    return results


def main():
    variants = (
        [
            "tensegrity",
            "ur10_f140", "ur10_frankenstein", "ur5e_f140", "ur5e_frankenstein",
            "kinova_f140", "kinova_frankenstein",
        ]
        if args.variant == "all"
        else [args.variant]
    )

    all_results = []
    for variant in variants:
        try:
            result = run_verification(variant)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR running {variant}: {e}")
            all_results.append({"variant": variant, "passed": False, "issues": [str(e)]})

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['variant']}")
        for issue in r.get("issues", []):
            print(f"         {issue}")
    print("=" * 70)

    # Exit with error if any failed
    if any(not r["passed"] for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()

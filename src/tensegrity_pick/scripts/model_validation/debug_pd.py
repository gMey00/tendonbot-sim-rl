"""Quick debug: does ImplicitActuatorCfg (K=400, D=20) actually move the robot?"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_3DOF_CFG

ARM_JOINTS = ["elbow_joint", "wrist_y_joint", "wrist_x_joint"]

robot_cfg = TENS_3DOF_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0), joint_pos={j: 0.0 for j in ARM_JOINTS}
    ),
    spawn=TENS_3DOF_CFG.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            effort_limit_sim=400.0,
            velocity_limit_sim=10.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
)

@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

sim = SimulationContext(sim_utils.SimulationCfg(dt=1/120.0, render_interval=4, device="cuda:0"))
scene = InteractiveScene(DebugSceneCfg(num_envs=1, env_spacing=3.0))
sim.reset(); scene.reset()

robot: Articulation = scene["robot"]
joint_ids, joint_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
dev = robot.device
print(f"\njoint_ids={joint_ids}  joint_names={joint_names}")
print(f"is_fixed_base={robot.is_fixed_base}")
print(f"num_joints={robot.num_joints}")
print(f"stiffness in sim: {robot.root_physx_view.get_dof_stiffnesses()}")
print(f"damping in sim: {robot.root_physx_view.get_dof_dampings()}")
print(f"effort_limits in sim: {robot.root_physx_view.get_dof_max_forces()}")

print("\n--- 50 warmup steps at target=0 ---")
zero = torch.zeros(1, 3, device=dev)
for i in range(50):
    robot.set_joint_position_target(zero, joint_ids=joint_ids)
    scene.write_data_to_sim(); sim.step(); scene.update(1/120.0)
pos = robot.data.joint_pos[0, joint_ids]
print(f"After warmup: joint_pos = {[f'{math.degrees(float(p)):.3f}' for p in pos]} deg")

print("\n--- 200 steps at target=20 deg elbow ---")
tgt = torch.zeros(1, 3, device=dev)
tgt[0, 0] = math.radians(20.0)
for i in range(200):
    robot.set_joint_position_target(tgt, joint_ids=joint_ids)
    scene.write_data_to_sim(); sim.step(); scene.update(1/120.0)
    if i % 40 == 0:
        pos = robot.data.joint_pos[0, joint_ids]
        print(f"  step {i:3d}: joint_pos = {[f'{math.degrees(float(p)):.3f}' for p in pos]} deg")

pos = robot.data.joint_pos[0, joint_ids]
print(f"\nFinal: joint_pos = {[f'{math.degrees(float(p)):.3f}' for p in pos]} deg  (target=20 deg)")
print("EXPECTED: elbow ~ 20 deg if drives work")

sim.stop()
simulation_app.close()

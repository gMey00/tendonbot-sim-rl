"""Drive the physical hierarchical reach variant with the scripted IK+PID heuristic.

Validation tool for the inner PID→tension controller: a DLS-IK solver
(:class:`~scripts.skrl.heuristic_physical_ik.HeuristicHierarchicalIKAgent`)
computes joint set-points and the hierarchical action term's inner PID tracks
them on the physical 4-bar robot — no learned policy involved.  If this reaches
targets cleanly, the controller is sound and any RL shortfall is a policy/
training issue, not a controller one.

Usage (workstation with a display; rendering does NOT work on Alex):
    cd src/tensegrity_pick
    conda run --no-capture-output -n env_isaaclab python3 \
        scripts/diagnostics/play_physical_heuristic.py --num_envs 4 --real_time
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted IK+PID heuristic for the physical hierarchical reach task.")
parser.add_argument("--task", type=str, default="Template-Reach-Tensegrity-Physical-Hierarchical-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle to wall-clock for viewing.")
parser.add_argument("--ik_iters", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0 = run until the window closes).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402

# Load the heuristic agent by file path — ``scripts/skrl/`` is not a package and
# adding it to sys.path would shadow the installed ``skrl`` library.
import importlib.util as _ilu  # noqa: E402
import os as _os  # noqa: E402

_agent_path = _os.path.join(_os.path.dirname(__file__), "..", "skrl", "heuristic_physical_ik.py")
_spec = _ilu.spec_from_file_location("heuristic_physical_ik", _os.path.abspath(_agent_path))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HeuristicHierarchicalIKAgent = _mod.HeuristicHierarchicalIKAgent


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        dt = env.unwrapped.step_dt
    except AttributeError:
        dt = 1.0 / 30.0

    print(f"[heuristic] task={args_cli.task}  num_envs={args_cli.num_envs}  ik_iters={args_cli.ik_iters}")
    obs, _ = env.reset()
    agent = HeuristicHierarchicalIKAgent(env, ik_iters=args_cli.ik_iters)

    # Running success readout from the command term's live metrics.
    cmd_term = env.unwrapped.command_manager.get_term("ee_pose")
    threshold = float(cmd_term.cfg.success_threshold)

    step = 0
    reach_hits = 0
    reach_windows = 0
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = agent.act(env)
            env.step(actions)

        pos_err = cmd_term.metrics["position_error"]
        reach_hits += int((pos_err < threshold).sum().item())
        reach_windows += pos_err.numel()
        if step % 30 == 0:
            ik_res = getattr(agent, "last_ik_residual", None)
            trk = getattr(agent, "last_track_err", None)
            extra = ""
            if ik_res is not None:
                extra = f"  | IK residual mean={ik_res.mean():.3f} max={ik_res.max():.3f} m"
            if trk is not None:
                extra += (f"  set-point track_err(elbow/wristY/wristX)="
                          f"{trk[:, 2].mean():.3f}/{trk[:, 3].mean():.3f}/{trk[:, 4].mean():.3f} rad")
            print(
                f"[heuristic] step {step:4d}  "
                f"pos_err mean/min/max = {pos_err.mean():.3f} / {pos_err.min():.3f} / {pos_err.max():.3f} m  "
                f"within {threshold:.2f} m: {reach_hits / max(1, reach_windows):.2f}{extra}"
            )

        step += 1
        if args_cli.max_steps and step >= args_cli.max_steps:
            break
        if args_cli.real_time:
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
    import os
    os._exit(0)

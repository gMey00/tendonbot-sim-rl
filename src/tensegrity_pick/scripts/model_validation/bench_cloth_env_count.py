"""Stage-0 de-risk: parallel cloth env-count benchmark (single N per process).

Measures, for ONE env count N (Isaac Sim cannot rebuild the stage in-process,
so the sweep runs one process per N — see ``bench_cloth_env_count.sh``):

  * construct  — gym.make wall time (scene build + cloth authoring + PhysX
                 init + the one-off 320-step pre-settle)
  * steps/s    — steady-state zero-action env.step rate (after warmup)
  * env·steps/s — steps/s × N (the throughput number that matters for PPO)
  * GPU memory — torch CUDA allocated + reserved, and driver-level used
  * sanity     — NaN check + cloth bounded (max |pos − env_origin| < 5 m)
  * reset cost — wall time of a full env.reset() at N envs

Emits one machine-readable line:
  RESULT,N,construct_s,steps_per_s,env_steps_per_s,cuda_alloc_gb,cuda_reserved_gb,driver_used_gb,reset_s,sane

PhysX buffer overflow warnings ("collisionStackSize buffer overflow",
"Contact buffer overflow") print to the console — the driver script greps the
log for them; they mean silently degraded cloth physics at that N.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab
    PYTHONUNBUFFERED=1 python scripts/model_validation/bench_cloth_env_count.py \
        --headless --num_envs 128

    # full sweep + CSV (recommended):
    bash scripts/model_validation/bench_cloth_env_count.sh
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cloth env-count benchmark (one N per run).")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--warmup_steps", type=int, default=30)
parser.add_argument("--bench_steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402


def driver_used_gb() -> float:
    try:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024**3
    except Exception:
        return float("nan")


def main() -> None:
    n = args_cli.num_envs
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)

    t0 = time.perf_counter()
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    t_construct = time.perf_counter() - t0
    u = env.unwrapped
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    print(f"[bench] N={n} constructed in {t_construct:.1f}s "
          f"({u.cloth.num_particles} particles/cloth → {n * u.cloth.num_particles} total)")

    with torch.inference_mode():
        for _ in range(args_cli.warmup_steps):
            env.step(zero)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args_cli.bench_steps):
            env.step(zero)
        torch.cuda.synchronize()
        t_bench = time.perf_counter() - t0

    steps_per_s = args_cli.bench_steps / t_bench
    env_steps_per_s = steps_per_s * n

    # sanity: cloth finite and bounded around its env origin
    u.cloth.update()
    pts = u.cloth.nodal_pos_w
    origins = u.scene.env_origins.unsqueeze(1)
    sane = bool(torch.isfinite(pts).all()) and bool(
        ((pts - origins).norm(dim=-1).max() < 5.0)
    )

    cuda_alloc = torch.cuda.memory_allocated() / 1024**3
    cuda_reserved = torch.cuda.memory_reserved() / 1024**3
    drv = driver_used_gb()

    t0 = time.perf_counter()
    with torch.inference_mode():
        env.reset()
    t_reset = time.perf_counter() - t0

    print(f"[bench] N={n}: {steps_per_s:.2f} steps/s, {env_steps_per_s:.0f} env·steps/s, "
          f"CUDA {cuda_alloc:.2f}/{cuda_reserved:.2f} GB (driver {drv:.2f} GB), "
          f"reset {t_reset:.2f}s, sane={sane}")
    print(f"RESULT,{n},{t_construct:.1f},{steps_per_s:.3f},{env_steps_per_s:.1f},"
          f"{cuda_alloc:.2f},{cuda_reserved:.2f},{drv:.2f},{t_reset:.2f},{sane}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

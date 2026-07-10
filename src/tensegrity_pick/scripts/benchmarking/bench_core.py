"""Atomic benchmark harness — measure ONE configuration point in one process.

Isaac Sim cannot rebuild its stage in-process, so every ``(task, num_envs,
overrides)`` point is a fresh process. This script is that process. It is the
reusable primitive called by:
  * a human, directly, for a one-off measurement, and
  * ``run_matrix.py`` (local loop or SLURM array), which passes a point index into
    a manifest file written by the matrix expander.

It generalises the original ``model_validation/bench_cloth_env_count.py``:
  * arbitrary task (not just the cloth env),
  * config-driven overrides via dotted paths (so optimisation levers — solver
    iterations, dt, decimation, cloth backend, fp32/tf32 — are swept *without code
    changes*, reading the project's own env configs),
  * richer metric set (per-step latency spread, GPU power/util/energy, peak memory,
    host RAM) collected through :mod:`bench_metrics`,
  * full reproducibility provenance through :mod:`bench_env`,
  * a single machine-readable ``RESULT_JSON,{...}`` line for robust aggregation.

Usage
-----
    cd src/tensegrity_pick && conda activate env_isaaclab

    # explicit one-off
    python scripts/benchmarking/bench_core.py --headless \
        --task Template-Tensegrity-Shirt-Place-v0 --num_envs 64

    # one point out of a manifest (how run_matrix.py / SLURM arrays call it)
    python scripts/benchmarking/bench_core.py --headless \
        --point-file /path/to/manifest.jsonl --point-index 7

    # ad-hoc optimisation-lever override
    python scripts/benchmarking/bench_core.py --headless \
        --task ... --num_envs 64 \
        --set sim.physx.solver_position_iteration_count=8 --set decimation=2

Overrides (``--set a.b.c=value`` or the ``overrides`` dict in a manifest point)
are applied to the parsed ``env_cfg`` by dotted path. Values are parsed as JSON
first (so ``8``, ``true``, ``0.0166`` keep their type), falling back to string.
NOTE: exact attribute paths depend on the task config; unknown paths raise so a
typo fails loudly instead of silently benchmarking the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# bench_metrics / bench_env are siblings — make them importable when this file is
# run as a plain script (python scripts/benchmarking/bench_core.py ...).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher  # noqa: E402

# ---------------------------------------------------------------------------
# CLI  (AppLauncher args are appended so --headless/--device work as everywhere)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Atomic single-point benchmark harness.")
parser.add_argument("--task", type=str, default=None, help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=None, help="Parallel env count N.")
parser.add_argument("--warmup_steps", type=int, default=30, help="Un-timed steps before measurement.")
parser.add_argument("--bench_steps", type=int, default=240, help="Timed steady-state steps.")
parser.add_argument("--seed", type=int, default=0, help="Env/torch seed (reproducibility).")
parser.add_argument("--set", dest="overrides", action="append", default=[],
                    help="Dotted-path env_cfg override a.b.c=value (JSON-typed). Repeatable.")
parser.add_argument("--cloth-ref", type=str, default=None,
                    help="module:attr of the shared cloth cfg singleton to mutate "
                         "(e.g. tensegrity_pick.tasks.manager_based.shirt_place."
                         "shirt_place_scene_cfg:SHIRT_CLOTH_CFG). Required when a point "
                         "carries cloth_overrides.")
parser.add_argument("--cloth-set", dest="cloth_overrides", action="append", default=[],
                    help="Dotted-path cloth-cfg override a.b.c=value applied to --cloth-ref. Repeatable.")
parser.add_argument("--point-file", type=str, default=None,
                    help="Manifest JSONL; take one point (task/num_envs/overrides/meta) by index.")
parser.add_argument("--point-index", type=int, default=None, help="Row index into --point-file.")
parser.add_argument("--out", type=str, default=None,
                    help="If given, write the RESULT JSON object to this path (one line).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()


# ---------------------------------------------------------------------------
# Resolve the configuration point (explicit CLI or a manifest row) BEFORE launch.
# ---------------------------------------------------------------------------
def _load_point() -> dict:
    """Merge a manifest row (if any) with explicit CLI flags. CLI wins on conflict
    only where the user actually passed a value."""
    point: dict = {"task": args_cli.task, "num_envs": args_cli.num_envs,
                   "overrides": {}, "cloth_overrides": {},
                   "cloth_cfg_ref": args_cli.cloth_ref, "meta": {}}
    if args_cli.point_file is not None:
        if args_cli.point_index is None:
            raise SystemExit("--point-file requires --point-index")
        with open(args_cli.point_file) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        row = rows[args_cli.point_index]
        point["task"] = row.get("task", point["task"])
        point["num_envs"] = row.get("num_envs", point["num_envs"])
        point["overrides"].update(row.get("overrides", {}))
        point["cloth_overrides"].update(row.get("cloth_overrides", {}))
        point["cloth_cfg_ref"] = row.get("cloth_cfg_ref", point["cloth_cfg_ref"])
        point["meta"] = row.get("meta", {})
        point["meta"].setdefault("point_index", args_cli.point_index)
        if "seed" in row:
            args_cli.seed = row["seed"]
    # CLI --set / --cloth-set layer on top of the manifest overrides.
    for item in args_cli.overrides:
        key, _, raw = item.partition("=")
        point["overrides"][key.strip()] = _parse_scalar(raw)
    for item in args_cli.cloth_overrides:
        key, _, raw = item.partition("=")
        point["cloth_overrides"][key.strip()] = _parse_scalar(raw)
    if point["cloth_overrides"] and not point["cloth_cfg_ref"]:
        raise SystemExit("cloth_overrides given but no --cloth-ref / cloth_cfg_ref to apply them to.")
    if not point["task"]:
        raise SystemExit("No task given (pass --task or a manifest with 'task').")
    if not point["num_envs"]:
        raise SystemExit("No num_envs given (pass --num_envs or a manifest with 'num_envs').")
    return point


def _parse_scalar(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


POINT = _load_point()

# ---------------------------------------------------------------------------
# Launch the simulator, THEN import the heavy / torch-bound modules.
# ---------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402

import bench_env  # noqa: E402
import bench_metrics as bm  # noqa: E402


# ---------------------------------------------------------------------------
# Config-driven overrides
# ---------------------------------------------------------------------------
def apply_overrides(cfg, overrides: dict) -> dict:
    """Set ``cfg.a.b.c = value`` for each ``"a.b.c": value`` pair.

    Returns the *effective* values read back after assignment, so the record shows
    what was actually benchmarked (guards against a lever that a task silently
    re-derives in ``__post_init__``). Raises on an unknown path.
    """
    applied = {}
    for path, value in overrides.items():
        parts = path.split(".")
        obj = cfg
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise AttributeError(f"override path '{path}': no attribute '{p}' on {type(obj).__name__}")
            obj = getattr(obj, p)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise AttributeError(f"override path '{path}': no attribute '{leaf}' on {type(obj).__name__}")
        setattr(obj, leaf, value)
        applied[path] = getattr(obj, leaf)
    return applied


def _apply_cloth_overrides_everywhere(cfg, singleton, overrides: dict) -> dict:
    """Apply cloth overrides so they actually take effect, working around the task's
    ``__post_init__`` (which re-sets e.g. ``solver_position_iterations`` on the shared
    singleton *after* a pre-parse override).

    Targets, post-parse:
      * the module singleton  → what the ``ClothObject`` view + fingerprint read;
      * every cloth cfg found inside ``cfg.events.*.params`` → what the startup event
        actually authors into the PBD/XPBD solver (``cloth_cfg.pbd_params`` at
        ``_apply_pbd_cloth``). This is the one that determines real solver behaviour.
    """
    targets = []
    if singleton is not None:
        targets.append(singleton)
    events = getattr(cfg, "events", None)
    if events is not None:
        for term_name in vars(events):
            term = getattr(events, term_name, None)
            params = getattr(term, "params", None)
            if isinstance(params, dict):
                for v in params.values():
                    if hasattr(v, "pbd_params") or hasattr(v, "xpbd_params"):
                        targets.append(v)
    applied = {}
    for t in targets:
        applied = apply_overrides(t, overrides)  # identical overrides → same readback
    return applied


def _total_obs_dim(obs_manager) -> int:
    """Sum the per-group observation width. ``group_obs_dim`` maps group→shape
    tuple (e.g. (45,)) in Isaac Lab, so take the last shape element, not ``.shape``."""
    try:
        total = 0
        for v in obs_manager.group_obs_dim.values():
            total += v[-1] if isinstance(v, (tuple, list)) else int(v)
        return int(total)
    except Exception:
        return -1


def _g(obj, path, default=None):
    """Safe nested getattr by dotted path ('sim.physx.solver_type')."""
    for p in path.split("."):
        if obj is None or not hasattr(obj, p):
            return default
        obj = getattr(obj, p)
    return obj


def _derive_labels(task: str) -> dict:
    """Parse robot / action_space labels from the registered task id so the
    artifact schema has the columns the report's Section-4 schema expects."""
    t = task.lower()
    if "ik-rel" in t:
        action_space = "ik_rel"
    elif "ik-abs" in t:
        action_space = "ik_abs"
    elif "osc" in t:
        action_space = "osc"
    else:
        action_space = "joint"
    for tok, name in (("ur5e", "ur5e"), ("ur10", "ur10"), ("kinova", "kinova"),
                      ("tensegrity", "tensegrity")):
        if tok in t:
            robot = name
            break
    else:
        robot = "unknown"
    if "frankenstein" in t:
        robot += "+wrist"          # arm + tensegrity wrist
    if "tendon" in t:
        robot += "+tendon"
    if "physical" in t:
        robot += "+physical"
    return {"robot": robot, "action_space": action_space}


def config_fingerprint(cfg, task: str, cloth) -> dict:
    """The physics/robot/config columns the report's artifact schema requires.
    Everything is best-effort (getattr-guarded) so a task missing a field never
    aborts the benchmark."""
    fp = _derive_labels(task)
    fp["dt"] = _g(cfg, "sim.dt")
    fp["decimation"] = _g(cfg, "decimation")
    fp["render_interval"] = _g(cfg, "sim.render_interval")
    fp["solver_type"] = _g(cfg, "sim.physx.solver_type")           # 0=PGS, 1=TGS
    fp["enhanced_determinism"] = _g(cfg, "sim.physx.enable_enhanced_determinism")
    fp["precision"] = "fp32"   # Isaac PhysX default; tf32 flags live in environment blob
    # PhysX GPU buffer sizes — the report's threat #1 (fixed buffers, overflow regime).
    for k in ("gpu_collision_stack_size", "gpu_max_particle_contacts",
              "gpu_found_lost_aggregate_pairs_capacity", "gpu_total_aggregate_pairs_capacity",
              "gpu_max_rigid_contact_count", "gpu_max_rigid_patch_count", "gpu_heap_capacity"):
        fp[f"physx.{k}"] = _g(cfg, f"sim.physx.{k}")
    # Cloth backend + solver profile (only for cloth tasks).
    if cloth is not None:
        fp["cloth_backend"] = getattr(getattr(cloth.cfg, "backend", None), "value", None) \
            if hasattr(cloth, "cfg") else None
        fp["cloth_solver_pos_iters"] = _g(cloth, "cfg.pbd_params.solver_position_iterations")
    return fp


def main() -> None:
    task = POINT["task"]
    n = int(POINT["num_envs"])
    torch.manual_seed(args_cli.seed)

    record: dict = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": task,
        "num_envs": n,
        "seed": args_cli.seed,
        "warmup_steps": args_cli.warmup_steps,
        "bench_steps": args_cli.bench_steps,
        "requested_overrides": POINT["overrides"],
        "requested_cloth_overrides": POINT["cloth_overrides"],
        "cloth_cfg_ref": POINT["cloth_cfg_ref"],
        "meta": POINT["meta"],
        "environment": bench_env.capture_environment(),
        "status": "ok",
    }

    try:
        # Cloth-cfg overrides (usd_path, pbd_params.*) must hit the SHARED module
        # singleton BEFORE parse_env_cfg: the env reads that global directly, and
        # parse_env_cfg deep-copies it into the spawn event — mutating it first
        # keeps both the spawned mesh and the ClothObject view consistent. A
        # post-parse env_cfg override cannot reach it (config holds a copy).
        cloth_singleton = None
        if POINT["cloth_overrides"]:
            import importlib
            mod_path, _, attr = POINT["cloth_cfg_ref"].partition(":")
            cloth_singleton = getattr(importlib.import_module(mod_path), attr)
            apply_overrides(cloth_singleton, POINT["cloth_overrides"])  # pre-parse (usd_path/spawn)

        cfg = parse_env_cfg(task, device=args_cli.device, num_envs=n)
        record["applied_overrides"] = apply_overrides(cfg, POINT["overrides"])

        # Re-apply cloth overrides AFTER parse (post-__post_init__) to the singleton
        # AND the startup-event cloth cfg — the latter is what actually authors the
        # PBD solver, so this is what makes a solver-iteration sweep take effect.
        if POINT["cloth_overrides"]:
            record["applied_cloth_overrides"] = _apply_cloth_overrides_everywhere(
                cfg, cloth_singleton, POINT["cloth_overrides"])

        # -- construction (scene build + cloth authoring + PhysX init + presettle) --
        t0 = time.perf_counter()
        env = gym.make(task, cfg=cfg)
        t_first_reset0 = time.perf_counter()
        env.reset()
        t_construct = t_first_reset0 - t0
        t_first_reset = time.perf_counter() - t_first_reset0
        u = env.unwrapped

        action_dim = u.action_manager.total_action_dim
        zero = torch.zeros(u.num_envs, action_dim, device=u.device)
        record.update({
            "action_dim": int(action_dim),
            "obs_dim": _total_obs_dim(u.observation_manager),
            "construct_s": t_construct,
            "first_reset_s": t_first_reset,
        })
        # Cloth-specific size (only tasks that carry a ClothObject).
        cloth = getattr(u, "cloth", None) or getattr(u, "_cloth", None)
        if cloth is not None and hasattr(cloth, "num_particles"):
            record["cloth_particles_per_env"] = int(cloth.num_particles)
            record["cloth_particles_total"] = int(cloth.num_particles * n)

        # Config fingerprint (robot/action_space/physics/buffers) for the schema.
        record["config"] = config_fingerprint(cfg, task, cloth)

        # -- warmup (excluded from timing; lets clocks/caches reach steady state) --
        with torch.inference_mode():
            for _ in range(args_cli.warmup_steps):
                env.step(zero)
            torch.cuda.synchronize()

            # -- measured window: throughput + latency + GPU telemetry ------------
            bm.reset_torch_peak()
            timer = bm.StepTimer()
            sampler = bm.GpuSampler(device_index=0).start()
            for _ in range(args_cli.bench_steps):
                timer.tic()
                env.step(zero)
                torch.cuda.synchronize()  # per-step sync → honest latency distribution
                timer.toc()
            gpu = sampler.stop()

        record.update(timer.summary(n))
        record.update(bm.torch_mem())
        record["driver_used_gb"] = bm.driver_used_gb()
        record.update(bm.host_stats())
        record["gpu_power_w_mean"] = gpu.power_w_mean
        record["gpu_power_w_max"] = gpu.power_w_max
        record["gpu_util_mean"] = gpu.util_gpu_mean
        record["gpu_mem_util_mean"] = gpu.util_mem_mean
        record["gpu_mem_used_mb_max"] = gpu.mem_used_mb_max
        record["gpu_sm_clock_mhz_mean"] = gpu.sm_clock_mhz_mean
        record["gpu_temp_c_max"] = gpu.temp_c_max
        record["gpu_energy_j"] = gpu.energy_j
        # Energy per million env-steps — the hardware-agnostic efficiency figure.
        esps = record.get("env_steps_per_s", float("nan"))
        record["energy_j_per_Mstep"] = (gpu.energy_j / (record.get("n_steps", 0) * n) * 1e6
                                        if gpu.energy_j == gpu.energy_j and n else float("nan"))

        # -- physics sanity: finite + bounded cloth (skip for non-cloth tasks) ----
        record["sane"] = _sanity(u, cloth)

        # -- reset cost at N envs -------------------------------------------------
        t0 = time.perf_counter()
        with torch.inference_mode():
            env.reset()
        record["reset_s"] = time.perf_counter() - t0

        env.close()
    except Exception as exc:  # a failed point must still emit a row, not vanish
        import traceback
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()

    _emit(record)


def _sanity(u, cloth) -> bool:
    """Finite state + (for cloth tasks) cloth bounded around its env origin."""
    if cloth is None:
        return True
    try:
        cloth.update()
        pts = cloth.nodal_pos_w
        origins = u.scene.env_origins.unsqueeze(1)
        finite = bool(torch.isfinite(pts).all())
        bounded = bool((pts - origins).norm(dim=-1).max() < 5.0)
        return finite and bounded
    except Exception:
        return False


def _emit(record: dict) -> None:
    line = "RESULT_JSON," + json.dumps(record, default=str)
    print(line, flush=True)
    if args_cli.out:
        with open(args_cli.out, "w") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    # human-friendly one-liner
    if record.get("status") == "ok":
        print(f"[bench] {record['task']} N={record['num_envs']}: "
              f"{record.get('steps_per_s', float('nan')):.1f} steps/s, "
              f"{record.get('env_steps_per_s', float('nan')):.0f} env·steps/s, "
              f"peak {record.get('cuda_max_reserved_gb', float('nan')):.1f} GB reserved / "
              f"{record.get('driver_used_gb', float('nan')):.1f} GB driver, "
              f"{record.get('gpu_power_w_mean', float('nan')):.0f} W, "
              f"sane={record.get('sane')}", flush=True)
    else:
        print(f"[bench] {record['task']} N={record['num_envs']}: FAILED — {record.get('error')}",
              flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()

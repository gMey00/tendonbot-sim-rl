"""Expand a benchmark spec into points and dispatch them (local loop or SLURM array).

One *spec* (YAML in ``specs/``) declares an experiment: a base config plus swept
axes (Cartesian product) plus optional explicit points and repeats. This tool
expands it into a flat **manifest** (one JSON point per line) and runs each point
through the atomic ``bench_core.py`` harness — one process per point, because
Isaac Sim cannot rebuild its stage in-process.

Dispatch modes
--------------
* ``expand`` — write the manifest + print the plan; run nothing (inspect first).
* ``local``  — run every point sequentially on this workstation, collecting each
               ``RESULT_JSON`` line into ``<run>/results.jsonl`` + ``results.csv``.
* ``slurm``  — write the manifest and submit a SLURM **array** (index = point);
               each array task writes ``points/point_<i>.json``. Aggregate later
               with ``--dispatch collect``.
* ``collect``— gather ``points/point_*.json`` from a SLURM run into results.jsonl/csv.

Outputs land under ``doc/reports/data/benchmarks/<spec-name>/<machine>_<timestamp>/``
so A6000 and Alex results sit side by side and never overwrite each other.

Usage
-----
    cd src/tensegrity_pick && conda activate env_isaaclab

    # inspect the plan
    python scripts/benchmarking/run_matrix.py specs/env_count_sweep.yaml --dispatch expand

    # run locally on the A6000
    python scripts/benchmarking/run_matrix.py specs/env_count_sweep.yaml --dispatch local

    # submit to Alex, then collect once the array finishes
    python scripts/benchmarking/run_matrix.py specs/env_count_sweep.yaml --dispatch slurm
    python scripts/benchmarking/run_matrix.py --dispatch collect \
        --run-dir doc/reports/data/benchmarks/env_count_sweep/alex_2026....
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_env  # noqa: E402

try:
    import yaml
except Exception:  # pragma: no cover
    print("PyYAML is required (pip install pyyaml / conda install pyyaml).", file=sys.stderr)
    raise

# Axis-key routing:
#   'task'/'num_envs'/'seed'  -> top-level point fields
#   'cloth.<path>'            -> cloth_overrides (mutated on the cloth cfg singleton
#                                named by base.cloth_cfg_ref, BEFORE parse_env_cfg)
#   anything else             -> env_cfg dotted override (e.g. 'sim.dt', 'decimation')
_TOP_LEVEL = {"task", "num_envs", "seed"}

HERE = Path(__file__).resolve().parent
# repo root = .../studentische-arbeiten ; scripts live at src/tensegrity_pick/scripts/benchmarking
REPO_ROOT = HERE.parents[3]
PROJECT_DIR = HERE.parents[1]                 # src/tensegrity_pick
DATA_ROOT = REPO_ROOT / "doc/reports/data/benchmarks"
BENCH_CORE = "scripts/benchmarking/bench_core.py"  # relative to PROJECT_DIR


# ---------------------------------------------------------------------------
# Spec → manifest expansion
# ---------------------------------------------------------------------------
def expand_spec(spec: dict) -> list[dict]:
    """Cartesian product of ``axes`` × ``repeats``, merged onto ``base``, plus any
    explicit ``points``. Each result is a self-contained point dict for bench_core."""
    base = spec.get("base", {})
    base_overrides = dict(base.get("overrides", {}))
    base_cloth = dict(base.get("cloth_overrides", {}))
    cloth_ref = base.get("cloth_cfg_ref")
    axes = spec.get("axes", {}) or {}
    repeats = int(spec.get("repeats", 1))

    axis_keys = list(axes.keys())
    axis_vals = [axes[k] if isinstance(axes[k], list) else [axes[k]] for k in axis_keys]
    combos = list(itertools.product(*axis_vals)) if axis_keys else [()]

    points: list[dict] = []
    for rep in range(repeats):
        for combo in combos:
            point = {
                "task": base.get("task"),
                "num_envs": base.get("num_envs"),
                "seed": base.get("seed", 0) + rep,   # distinct seed per repeat
                "overrides": dict(base_overrides),
                "cloth_overrides": dict(base_cloth),
                "cloth_cfg_ref": cloth_ref,
                "meta": {"spec": spec.get("name"), "repeat": rep},
            }
            for key, val in zip(axis_keys, combo):
                if key in _TOP_LEVEL:
                    point[key] = val
                elif key.startswith("cloth."):
                    point["cloth_overrides"][key[len("cloth."):]] = val
                else:
                    point["overrides"][key] = val
                point["meta"][f"axis_{key}"] = val
            points.append(point)

    for extra in spec.get("points", []) or []:
        point = {
            "task": extra.get("task", base.get("task")),
            "num_envs": extra.get("num_envs", base.get("num_envs")),
            "seed": extra.get("seed", base.get("seed", 0)),
            "overrides": {**base_overrides, **extra.get("overrides", {})},
            "cloth_overrides": {**base_cloth, **extra.get("cloth_overrides", {})},
            "cloth_cfg_ref": extra.get("cloth_cfg_ref", cloth_ref),
            "meta": {"spec": spec.get("name"), "explicit": True, **extra.get("meta", {})},
        }
        points.append(point)

    # exclusion filters: drop a point if it matches ALL key/values of any rule
    for rule in (spec.get("filters", {}) or {}).get("exclude", []) or []:
        points = [p for p in points if not _matches(p, rule)]

    if not points:
        raise SystemExit("Spec expanded to zero points — check axes/base/filters.")
    for p in points:
        if not p["task"] or not p["num_envs"]:
            raise SystemExit(f"Point missing task/num_envs: {p}")
        if p["cloth_overrides"] and not p["cloth_cfg_ref"]:
            raise SystemExit(
                "Spec sets cloth.* overrides but no base.cloth_cfg_ref to apply them to.")
    return points


def _matches(point: dict, rule: dict) -> bool:
    for k, v in rule.items():
        pv = point.get(k, point["overrides"].get(k))
        if pv != v:
            return False
    return True


# ---------------------------------------------------------------------------
# Run-dir + manifest
# ---------------------------------------------------------------------------
def make_run_dir(spec_name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = DATA_ROOT / spec_name / f"{bench_env.detect_machine()}_{stamp}"
    (run_dir / "points").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir: Path, points: list[dict], spec: dict) -> Path:
    manifest = run_dir / "manifest.jsonl"
    with open(manifest, "w") as fh:
        for p in points:
            fh.write(json.dumps(p) + "\n")
    (run_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (run_dir / "environment.json").write_text(
        json.dumps(bench_env.capture_environment(), indent=2, default=str))
    return manifest


# ---------------------------------------------------------------------------
# Dispatch: local
# ---------------------------------------------------------------------------
def dispatch_local(run_dir: Path, manifest: Path, points: list[dict], args) -> None:
    results_path = run_dir / "results.jsonl"
    py = os.environ.get("BENCH_PYTHON", sys.executable)
    for i, p in enumerate(points):
        print(f"\n=== [{i + 1}/{len(points)}] {p['task']} N={p['num_envs']} "
              f"overrides={p['overrides']} cloth={p['cloth_overrides']} ===", flush=True)
        out = run_dir / "points" / f"point_{i}.json"
        log = run_dir / "points" / f"point_{i}.log"
        cmd = [py, BENCH_CORE, "--headless",
               "--point-file", str(manifest), "--point-index", str(i),
               "--warmup_steps", str(args.warmup_steps),
               "--bench_steps", str(args.bench_steps),
               "--out", str(out)]
        # bench_core reads the seed from the manifest row; run in PROJECT_DIR so the
        # relative BENCH_CORE path and conda activation-set PYTHONPATH both resolve.
        # Combined stdout+stderr → per-point log so PhysX overflow warnings (C++
        # layer, not catchable in-process) can be grepped during aggregation.
        with open(log, "w") as lf:
            proc = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=lf,
                                  stderr=subprocess.STDOUT)
        if proc.returncode != 0 and not out.exists():
            # harness crashed before emitting — record a stub so the row isn't lost
            out.write_text(json.dumps({
                "status": "process_crash", "task": p["task"], "num_envs": p["num_envs"],
                "meta": p["meta"], "returncode": proc.returncode}) + "\n")
        rec = _read_point(out)
        tag = rec.get("status", "?")
        print(f"  -> {tag}: {rec.get('env_steps_per_s', float('nan')):.0f} env·steps/s "
              f"(overflow warns: {_overflow_count(log)})", flush=True)
    _aggregate(run_dir, results_path)


# PhysX GPU fixed-buffer overflow signatures (report threat #1: silent physics
# degradation → any throughput measured in this regime is invalid).
_OVERFLOW_RE = re.compile(
    r"buffer overflow|collisionStackSize|foundLostPairsCapacity|"
    r"foundLostAggregatePairsCapacity|contact.*dropped|patch.*overflow|"
    r"PxgDynamicsMemoryConfig", re.IGNORECASE)


def _overflow_count(log: Path) -> int:
    if not log.exists():
        return 0
    try:
        return sum(1 for line in log.read_text(errors="ignore").splitlines()
                   if _OVERFLOW_RE.search(line))
    except Exception:
        return 0


def _read_point(out: Path) -> dict:
    if not out.exists():
        return {"status": "missing"}
    try:
        return json.loads(out.read_text().strip().splitlines()[-1])
    except Exception:
        return {"status": "parse_error"}


# ---------------------------------------------------------------------------
# Dispatch: SLURM array (Alex)
# ---------------------------------------------------------------------------
SBATCH_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --array=0-{last_index}{throttle}
#SBATCH --output={run_dir}/points/slurm_%A_%a.out
#SBATCH --export=NONE

# NHR@FAU reproducible-env boilerplate (mirrors tools/train_alex.sh).
unset SLURM_EXPORT_ENV
source "{project_path}/.config/env_vars.sh"
module load gcc/15.2.0 2>/dev/null || true
if module load python 2>/dev/null && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    export CONDA_PKGS_DIRS="${{INSTALL_PATH}}/.cache/conda_pkgs"
else
    source "${{CONDA_PATH}}/etc/profile.d/conda.sh"
fi
conda activate "${{ISAACLAB_ENV_NAME}}"

export http_proxy=http://proxy.nhr.fau.de:80  https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80  HTTPS_PROXY=http://proxy.nhr.fau.de:80
export no_proxy=localhost,127.0.0.1,.nhr.fau.de,.fau.de
export NO_PROXY=localhost,127.0.0.1,.nhr.fau.de,.fau.de

cd "{project_path}/src/tensegrity_pick"
# Shadow any editable `tensegrity_pick` install with THIS checkout's package, so a
# worktree (or any checkout != the env's pip -e target) benchmarks its own code.
# The pip editable install is a sys.meta_path finder queried AFTER PathFinder, so a
# PYTHONPATH prepend wins without touching the shared conda env.
export PYTHONPATH="{project_path}/src/tensegrity_pick/source/tensegrity_pick:${{PYTHONPATH:-}}"
{stage_block}
echo "point ${{SLURM_ARRAY_TASK_ID}} on $(hostname)"
python -c "import tensegrity_pick as t; print('[shadow] tensegrity_pick ->', t.__file__)" || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun --ntasks=1 python {bench_core} --headless \\
    --point-file "{manifest}" --point-index "${{SLURM_ARRAY_TASK_ID}}" \\
    --warmup_steps {warmup} --bench_steps {bench} \\
    --out "{run_dir}/points/point_${{SLURM_ARRAY_TASK_ID}}.json" \\
    2>&1 | tee "{run_dir}/points/point_${{SLURM_ARRAY_TASK_ID}}.log"
"""


def dispatch_slurm(run_dir: Path, manifest: Path, points: list[dict], args) -> None:
    stage_block = (
        'echo "[stage] copying res/ to $TMPDIR"\n'
        'mkdir -p "$TMPDIR/res" && cp -r "${PROJECT_PATH}/res/." "$TMPDIR/res/"\n'
        'export TENSEGRITY_ASSET_ROOT="$TMPDIR/res"\n' if args.stage else ""
    )
    gres = os.environ.get("ALEX_GRES", "gpu:rtxpro6k:1")
    sbatch = SBATCH_TEMPLATE.format(
        job_name=f"bench_{run_dir.parent.name}",
        partition=gres.split(":")[1],
        gres=gres,
        cpus=os.environ.get("ALEX_CPUS_PER_GPU", "32"),
        time=args.time,
        last_index=len(points) - 1,
        throttle=f"%{args.max_parallel}" if args.max_parallel else "",
        run_dir=str(run_dir),
        project_path=os.environ.get("PROJECT_PATH", str(REPO_ROOT)),
        stage_block=stage_block,
        bench_core=BENCH_CORE,
        manifest=str(manifest),
        warmup=args.warmup_steps,
        bench=args.bench_steps,
    )
    script = run_dir / "submit.sbatch"
    script.write_text(sbatch)
    print(f"Wrote SLURM array script: {script}")
    print(f"  array 0-{len(points) - 1}"
          + (f" (throttled to {args.max_parallel} concurrent)" if args.max_parallel else ""))
    if args.dry_run:
        print("--dry-run: not submitting. Inspect the script above, then re-run without --dry-run.")
        return
    subprocess.run(["sbatch", str(script)], check=True)
    print(f"Submitted. When the array finishes, aggregate with:\n"
          f"  python {Path(__file__).name} --dispatch collect --run-dir {run_dir}")


# ---------------------------------------------------------------------------
# Aggregation (local end-of-run, or SLURM 'collect')
# ---------------------------------------------------------------------------
def _aggregate(run_dir: Path, results_path: Path) -> None:
    rows = []
    for pf in sorted((run_dir / "points").glob("point_*.json"),
                     key=lambda p: int(p.stem.split("_")[1])):
        try:
            r = json.loads(pf.read_text().strip().splitlines()[-1])
        except Exception as exc:
            print(f"  warn: could not parse {pf.name}: {exc}", file=sys.stderr)
            continue
        # Fold the PhysX-overflow grep into the physics gate: a point is only
        # valid if it ran clean AND emitted no fixed-buffer overflow warnings.
        idx = pf.stem.split("_")[1]
        log = run_dir / "points" / f"point_{idx}.log"
        n_ovf = _overflow_count(log)
        r["physx_overflow_warns"] = n_ovf
        r["physics_ok"] = bool(r.get("sane", False)) and n_ovf == 0 and r.get("status") == "ok"
        rows.append(r)
    with open(results_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    _write_csv(run_dir / "results.csv", rows)
    ok = sum(r.get("status") == "ok" for r in rows)
    valid = sum(r.get("physics_ok", False) for r in rows)
    ovf = sum(r.get("physx_overflow_warns", 0) > 0 for r in rows)
    print(f"\nAggregated {len(rows)} points ({ok} ran ok, {valid} physics-valid, "
          f"{ovf} with overflow warnings) → {results_path}")
    print(f"Flat CSV → {run_dir / 'results.csv'}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Flatten scalar top-level fields + environment.* into a wide CSV for quick
    inspection; the JSONL remains the lossless source of truth."""
    flat_rows = []
    _nested = {"environment": "env", "meta": "meta", "config": "cfg",
               "applied_overrides": "ov", "applied_cloth_overrides": "clov"}
    for r in rows:
        flat = {}
        for k, v in r.items():
            if k in _nested and isinstance(v, dict):
                for ek, ev in v.items():
                    flat[f"{_nested[k]}.{ek}"] = ev
            elif isinstance(v, (str, int, float, bool)) or v is None:
                flat[k] = v
        flat_rows.append(flat)
    cols: list[str] = []
    for fr in flat_rows:
        for k in fr:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(flat_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", nargs="?", help="Path to a spec YAML (under specs/).")
    ap.add_argument("--dispatch", choices=("expand", "local", "slurm", "collect"),
                    default="expand")
    ap.add_argument("--run-dir", help="Existing run dir (required for --dispatch collect).")
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--bench_steps", type=int, default=240)
    ap.add_argument("--time", default="04:00:00", help="SLURM wall time per array task.")
    ap.add_argument("--max-parallel", type=int, default=8,
                    help="SLURM array throttle (%%N concurrent). 0 = unlimited.")
    ap.add_argument("--stage", action="store_true", help="SLURM: stage res/ to node-local $TMPDIR.")
    ap.add_argument("--dry-run", action="store_true", help="SLURM: write the script, do not submit.")
    args = ap.parse_args()

    if args.dispatch == "collect":
        if not args.run_dir:
            ap.error("--dispatch collect requires --run-dir")
        run_dir = Path(args.run_dir)
        _aggregate(run_dir, run_dir / "results.jsonl")
        return

    if not args.spec:
        ap.error("a spec YAML is required (except for --dispatch collect)")
    spec = yaml.safe_load(Path(args.spec).read_text())
    spec.setdefault("name", Path(args.spec).stem)
    if "warmup_steps" in spec.get("base", {}):
        args.warmup_steps = spec["base"]["warmup_steps"]
    if "bench_steps" in spec.get("base", {}):
        args.bench_steps = spec["base"]["bench_steps"]

    points = expand_spec(spec)
    print(f"Spec '{spec['name']}' → {len(points)} point(s):")
    for i, p in enumerate(points):
        extra = f"  cloth={p['cloth_overrides']}" if p["cloth_overrides"] else ""
        print(f"  [{i}] {p['task']}  N={p['num_envs']}  seed={p['seed']}  "
              f"overrides={p['overrides'] or '{}'}{extra}")

    if args.dispatch == "expand":
        run_dir = make_run_dir(spec["name"])
        manifest = write_manifest(run_dir, points, spec)
        print(f"\nManifest written: {manifest}\n(no runs dispatched; --dispatch local|slurm to execute)")
        return

    run_dir = make_run_dir(spec["name"])
    manifest = write_manifest(run_dir, points, spec)
    print(f"\nRun dir: {run_dir}")
    if args.dispatch == "local":
        dispatch_local(run_dir, manifest, points, args)
    elif args.dispatch == "slurm":
        dispatch_slurm(run_dir, manifest, points, args)


if __name__ == "__main__":
    main()

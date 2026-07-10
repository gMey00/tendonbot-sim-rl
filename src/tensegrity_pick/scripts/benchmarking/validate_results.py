"""Validity report over benchmark result artifacts (report Section 4/6 gate).

For each run dir it summarises: point count, how many ran ok / passed the physics
gate / hit PhysX overflow, any errored points (with the error), and the seed-to-seed
dispersion (coefficient of variation of env·steps/s per (task,N,config) group). The
report flags CV > 15 % — the threshold above which the study design says to add
repeats before drawing conclusions [Agarwal 2021].

Usage:
    python3 scripts/benchmarking/validate_results.py --auto
    python3 scripts/benchmarking/validate_results.py --runs doc/reports/data/benchmarks/*/*
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "doc/reports/data/benchmarks"
CV_WARN = 0.15


def load_run(run_dir: Path) -> list[dict]:
    jl = run_dir / "results.jsonl"
    if jl.exists():
        return [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    rows = []
    for pf in sorted((run_dir / "points").glob("point_*.json")):
        try:
            rows.append(json.loads(pf.read_text().strip().splitlines()[-1]))
        except Exception:
            pass
    return rows


def group_key(r: dict) -> tuple:
    cfg = r.get("config", {})
    return (r.get("task"), int(r.get("num_envs", -1)),
            cfg.get("robot"), cfg.get("action_space"),
            json.dumps(r.get("applied_overrides", {}), sort_keys=True),
            json.dumps(r.get("applied_cloth_overrides", {}), sort_keys=True))


def report_run(run_dir: Path) -> None:
    rows = load_run(run_dir)
    if not rows:
        print(f"\n### {run_dir.name}: (no data yet)")
        return
    n = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    valid = [r for r in rows if r.get("physics_ok", r.get("sane", False)) and r.get("status") == "ok"]
    ovf = [r for r in rows if r.get("physx_overflow_warns", 0)]
    errs = [r for r in rows if r.get("status") != "ok"]
    print(f"\n### {run_dir.parent.name}/{run_dir.name}")
    print(f"  points={n}  ok={len(ok)}  physics-valid={len(valid)}  "
          f"overflow={len(ovf)}  errored={len(errs)}")
    for r in errs:
        print(f"  ERROR  {r.get('task')} N={r.get('num_envs')}: "
              f"{str(r.get('error'))[:130]}")
    for r in ovf:
        print(f"  OVERFLOW  {r.get('task')} N={r.get('num_envs')}: "
              f"{r.get('physx_overflow_warns')} warnings  (throughput INVALID)")
    # seed dispersion
    groups: dict = defaultdict(list)
    for r in valid:
        v = r.get("env_steps_per_s")
        if isinstance(v, (int, float)) and v == v:
            groups[group_key(r)].append(v)
    hi_cv = []
    for k, vals in groups.items():
        if len(vals) >= 2:
            m = statistics.fmean(vals)
            cv = (statistics.pstdev(vals) / m) if m else 0.0
            if cv > CV_WARN:
                hi_cv.append((k, cv, len(vals)))
    if hi_cv:
        print(f"  HIGH-VARIANCE groups (CV>{CV_WARN:.0%}, add repeats):")
        for k, cv, nn in sorted(hi_cv, key=lambda x: -x[1]):
            print(f"    CV={cv:.0%} (n={nn})  task={k[0]} N={k[1]} robot={k[2]} act={k[3]}")
    else:
        print(f"  seed dispersion OK (all groups with >=2 repeats have CV<={CV_WARN:.0%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--runs", nargs="*", default=[])
    args = ap.parse_args()
    dirs: list[Path] = []
    if args.auto:
        for spec_dir in sorted(DATA_ROOT.iterdir()) if DATA_ROOT.exists() else []:
            if spec_dir.is_dir():
                runs = sorted([d for d in spec_dir.iterdir()
                               if (d / "results.jsonl").exists() or (d / "points").exists()])
                if runs:
                    dirs.append(runs[-1])
    for pat in args.runs:
        dirs += [Path(p) for p in glob.glob(pat)]
    if not dirs:
        print("No run dirs found.")
        return
    print("=" * 66)
    print("BENCHMARK VALIDITY REPORT")
    print("=" * 66)
    for d in dirs:
        report_run(d)


if __name__ == "__main__":
    main()

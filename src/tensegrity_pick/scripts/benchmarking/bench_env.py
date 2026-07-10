"""Reproducibility capture — hardware, software, and repo provenance.

Every benchmark row is tagged with the output of :func:`capture_environment` so
that results collected on the **A6000 workstation** and on the **Alex RTX PRO
6000** nodes are directly comparable and re-runnable months later. This directly
serves the cross-hardware comparison + reproducibility protocol (Questions D & F
of ``doc/reports/RESEARCH_PROMPT_benchmarking_suite.md``).

Kept import-light on purpose: ``torch`` is optional so the capture also works from
an offline aggregation shell. No Isaac imports.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from typing import Optional

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


def _run(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5.0,
                              check=True).stdout.strip()
    except Exception:
        return None


def _git(*args: str) -> Optional[str]:
    return _run(["git", *args])


def detect_machine() -> str:
    """Mirror ``.config/env_vars.sh`` machine detection so bench rows carry the
    same profile label as training runs."""
    forced = os.environ.get("MACHINE")
    if forced:
        return forced
    host = socket.gethostname()
    if os.environ.get("HPCVAULT") or "alex" in host or host.startswith("login"):
        return "alex"
    if host.startswith("faps"):
        return "faps_workstation"
    return host or "unknown"


def gpu_info() -> dict[str, object]:
    info: dict[str, object] = {}
    smi = shutil.which("nvidia-smi")
    if smi:
        q = _run([smi, "--id=0",
                  "--query-gpu=name,memory.total,driver_version,compute_cap,"
                  "power.limit,clocks.max.sm,ecc.mode.current",
                  "--format=csv,noheader,nounits"])
        if q:
            fields = [v.strip() for v in q.split(",")]
            keys = ["gpu_name", "gpu_mem_total_mb", "driver_version", "compute_cap",
                    "power_limit_w", "sm_clock_max_mhz", "ecc_mode"]
            info.update(dict(zip(keys, fields)))
    if _HAVE_TORCH and torch.cuda.is_available():
        info.setdefault("gpu_name", torch.cuda.get_device_name(0))
        info["torch_cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["cuda_device_count"] = torch.cuda.device_count()
        # tf32 state matters for the fp32-vs-tf32 optimisation lever (Question E).
        info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        info["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    return info


def software_versions() -> dict[str, object]:
    v: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if _HAVE_TORCH:
        v["torch"] = torch.__version__
        v["cuda_runtime"] = getattr(torch.version, "cuda", None)
        v["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    # Version-probe the sim/RL stack without importing the heavy modules.
    for mod in ("isaaclab", "isaacsim", "skrl", "gymnasium"):
        v[mod] = _pkg_version(mod)
    return v


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def repo_provenance() -> dict[str, object]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def slurm_context() -> dict[str, object]:
    """SLURM identifiers when running as an Alex array task (empty locally)."""
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURMD_NODENAME")
    return {k.lower(): os.environ[k] for k in keys if k in os.environ}


def capture_environment() -> dict[str, object]:
    """The full provenance blob attached to every benchmark record."""
    env = {"machine": detect_machine(), "hostname": socket.gethostname()}
    env.update(gpu_info())
    env.update(software_versions())
    env.update(repo_provenance())
    slurm = slurm_context()
    if slurm:
        env["slurm"] = slurm
    return env


if __name__ == "__main__":
    import json
    print(json.dumps(capture_environment(), indent=2, default=str))

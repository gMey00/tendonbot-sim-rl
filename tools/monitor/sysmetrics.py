"""Live system-performance metrics: GPU (nvidia-smi), CPU / RAM (psutil).

Designed to run **wherever the monitor runs**:

* On the FAPS workstation it samples the local GPUs and CPU.
* On the Alex cluster the monitor is launched *on the compute node* through
  ``srun --jobid=<id> --overlap`` (see ``tools/watch_alex.sh``), so the same
  ``nvidia-smi`` / ``psutil`` calls report that node's resources — no cluster-
  specific code path is needed here.

Everything degrades gracefully: a missing ``nvidia-smi`` (CPU-only host) or a
missing ``psutil`` simply yields empty sections plus an ``errors`` note, instead
of crashing the dashboard.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field

try:
    import psutil
except Exception:  # pragma: no cover - psutil is an install-time dependency
    psutil = None  # type: ignore


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GpuStat:
    index: int
    name: str
    util_pct: float
    mem_used_mb: float
    mem_total_mb: float
    temp_c: float
    power_w: float
    power_limit_w: float
    procs: int = 0

    @property
    def mem_pct(self) -> float:
        return (self.mem_used_mb / self.mem_total_mb * 100.0) if self.mem_total_mb else 0.0


@dataclass
class ProcStat:
    pid: int
    name: str
    cpu_pct: float
    rss_gb: float
    cmdline: str


@dataclass
class SysSnapshot:
    ts: float
    hostname: str
    cpu_pct: float
    per_cpu: list[float]
    cpu_count: int
    load1: float
    load5: float
    load15: float
    mem_used_gb: float
    mem_total_gb: float
    mem_pct: float
    swap_used_gb: float
    swap_total_gb: float
    gpus: list[GpuStat] = field(default_factory=list)
    train_procs: list[ProcStat] = field(default_factory=list)
    slurm_job: str | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU sampling via nvidia-smi
# ---------------------------------------------------------------------------

_GPU_QUERY = (
    "index,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw,power.limit"
)


def _to_float(token: str) -> float:
    """Parse an nvidia-smi field; '[N/A]' and blanks become 0.0."""
    token = token.strip()
    if not token or token.startswith("[") or token.lower() in ("n/a", "na"):
        return 0.0
    try:
        return float(token)
    except ValueError:
        return 0.0


def sample_gpus(errors: list[str]) -> list[GpuStat]:
    """Query all visible GPUs via nvidia-smi. Returns [] on any failure."""
    if shutil.which("nvidia-smi") is None:
        errors.append("nvidia-smi not found (CPU-only host?)")
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        errors.append(f"nvidia-smi failed: {exc}")
        return []
    if out.returncode != 0:
        msg = out.stderr.strip().splitlines()[0] if out.stderr.strip() else "unknown error"
        errors.append(f"nvidia-smi error: {msg}")
        return []

    gpus: list[GpuStat] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            gpus.append(
                GpuStat(
                    index=int(_to_float(parts[0])),
                    name=parts[1],
                    util_pct=_to_float(parts[2]),
                    mem_used_mb=_to_float(parts[3]),
                    mem_total_mb=_to_float(parts[4]),
                    temp_c=_to_float(parts[5]),
                    power_w=_to_float(parts[6]),
                    power_limit_w=_to_float(parts[7]),
                )
            )
        except (ValueError, IndexError):
            continue

    # Best-effort per-GPU process count (one extra cheap call).
    _annotate_gpu_procs(gpus)
    return gpus


def _annotate_gpu_procs(gpus: list[GpuStat]) -> None:
    if not gpus:
        return
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return
    if out.returncode != 0:
        return
    # gpu_uuid → index mapping is not trivially available here; instead just
    # report the total compute-app count on the first GPU when there is a single
    # GPU (the common Alex single-GPU-per-job case). For multi-GPU hosts we skip.
    n = len([ln for ln in out.stdout.strip().splitlines() if ln.strip()])
    if len(gpus) == 1:
        gpus[0].procs = n


# ---------------------------------------------------------------------------
# CPU / memory / training-process sampling via psutil
# ---------------------------------------------------------------------------

def _is_python(name: str, cmd: str) -> bool:
    """True if the process is a Python interpreter (not a shell mentioning it)."""
    name = (name or "").lower()
    if "python" in name or name in ("pt_main_thread", "isaac-sim", "kit"):
        return True
    # Fallback: argv[0] is a python binary (some launchers rename the process).
    first = cmd.split(None, 1)[0].lower() if cmd else ""
    return first.endswith("python") or "python3" in first


def _find_train_procs(errors: list[str]) -> list[ProcStat]:
    """Find running Isaac Lab / skrl training workers (best effort).

    Matches an actual **Python interpreter** whose command line runs a
    ``train.py`` entry point — this deliberately excludes shell wrappers that
    merely mention "isaac"/"python"/"train" in their arguments (e.g. the conda
    activation command that launches the job), which previously false-matched.
    """
    if psutil is None:
        return []
    procs: list[ProcStat] = []
    self_pid = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            if p.info["pid"] == self_pid:
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if not cmd:
                continue
            low = cmd.lower()
            # Skip our own monitor / editors that merely mention the path.
            if "tools.monitor" in low or "tools/monitor" in low:
                continue
            if "train.py" not in low or not _is_python(p.info["name"], cmd):
                continue
            mem = p.info["memory_info"]
            procs.append(
                ProcStat(
                    pid=p.info["pid"],
                    name=p.info["name"] or "python",
                    cpu_pct=p.cpu_percent(interval=None),
                    rss_gb=(mem.rss / 1e9) if mem else 0.0,
                    cmdline=cmd,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue
    procs.sort(key=lambda pr: pr.rss_gb, reverse=True)
    return procs[:6]


# ---------------------------------------------------------------------------
# Collector (keeps rolling history for sparkline plots)
# ---------------------------------------------------------------------------

class SysMetricsCollector:
    """Samples system metrics and keeps a bounded rolling history.

    A single instance is created per dashboard session and reused across
    refreshes so ``cpu_percent`` deltas and the history plots are meaningful.
    """

    def __init__(self, history: int = 120) -> None:
        self.history = history
        self.cpu_hist: deque[float] = deque(maxlen=history)
        self.mem_hist: deque[float] = deque(maxlen=history)
        # Per-GPU histories keyed by GPU index.
        self.gpu_util_hist: dict[int, deque[float]] = {}
        self.gpu_mem_hist: dict[int, deque[float]] = {}
        self._primed = False
        # Prime psutil's per-process CPU meter so the first real read is sane.
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None, percpu=True)
            except Exception:
                pass

    def sample(self) -> SysSnapshot:
        errors: list[str] = []
        hostname = socket.gethostname()
        slurm_job = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")

        cpu_pct = 0.0
        per_cpu: list[float] = []
        cpu_count = os.cpu_count() or 0
        load1 = load5 = load15 = 0.0
        mem_used = mem_total = mem_pct = 0.0
        swap_used = swap_total = 0.0
        train_procs: list[ProcStat] = []

        if psutil is not None:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                per_cpu = psutil.cpu_percent(interval=None, percpu=True)
                cpu_count = psutil.cpu_count(logical=True) or cpu_count
                vm = psutil.virtual_memory()
                mem_used = vm.used / 1e9
                mem_total = vm.total / 1e9
                mem_pct = vm.percent
                sw = psutil.swap_memory()
                swap_used = sw.used / 1e9
                swap_total = sw.total / 1e9
            except Exception as exc:
                errors.append(f"psutil sample failed: {exc}")
            try:
                if hasattr(os, "getloadavg"):
                    load1, load5, load15 = os.getloadavg()
            except OSError:
                pass
            train_procs = _find_train_procs(errors)
        else:
            errors.append("psutil not installed — CPU/RAM metrics unavailable")

        gpus = sample_gpus(errors)

        # Update rolling histories.
        self.cpu_hist.append(cpu_pct)
        self.mem_hist.append(mem_pct)
        for g in gpus:
            self.gpu_util_hist.setdefault(g.index, deque(maxlen=self.history)).append(g.util_pct)
            self.gpu_mem_hist.setdefault(g.index, deque(maxlen=self.history)).append(g.mem_pct)

        self._primed = True
        return SysSnapshot(
            ts=time.time(),
            hostname=hostname,
            cpu_pct=cpu_pct,
            per_cpu=per_cpu,
            cpu_count=cpu_count,
            load1=load1,
            load5=load5,
            load15=load15,
            mem_used_gb=mem_used,
            mem_total_gb=mem_total,
            mem_pct=mem_pct,
            swap_used_gb=swap_used,
            swap_total_gb=swap_total,
            gpus=gpus,
            train_procs=train_procs,
            slurm_job=slurm_job,
            errors=errors,
        )

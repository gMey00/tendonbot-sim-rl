"""Metric collectors for the benchmarking suite (no Isaac Lab dependency).

This module is deliberately import-light so it can be reused from any process
(the atomic ``bench_core.py`` harness, offline aggregation, unit tests). It has
no ``isaaclab`` / ``omni`` imports and must never acquire them — everything here
is pure Python + optional ``torch`` / ``psutil`` / ``nvidia-smi``.

Collectors
----------
* :class:`GpuSampler`     — background GPU telemetry (power / util / mem / clocks)
                            via ``pynvml`` if importable, else polling ``nvidia-smi``.
* :func:`torch_mem`       — PyTorch CUDA caching-allocator counters (alloc/reserved/peak).
* :func:`driver_used_gb`  — driver-level used VRAM (``torch.cuda.mem_get_info``).
* :func:`host_stats`      — process + system RAM and CPU via ``psutil`` (optional).
* :class:`StepTimer`      — per-step latency accumulator → p50/p95/mean throughput.

Design note (research-driven metrics): the metric *set* below is the baseline the
existing bench already justified (throughput + memory + reset). Metrics flagged
``TODO(research)`` are placeholders to be confirmed/parameterised once
``doc/reports/RESEARCH_PROMPT_benchmarking_suite.md`` comes back — energy-per-Mstep,
SM-occupancy, and any cloth-fidelity-vs-speed metric are expected additions and
already have hooks here.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependencies — degrade gracefully, never hard-fail a benchmark run.
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401  (present in the Isaac env; guard for offline tests)
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover
    _HAVE_PSUTIL = False

try:
    import pynvml  # nvidia-ml-py; not installed by default → nvidia-smi fallback
    pynvml.nvmlInit()
    _HAVE_NVML = True
except Exception:
    _HAVE_NVML = False


# ---------------------------------------------------------------------------
# GPU telemetry sampler
# ---------------------------------------------------------------------------

@dataclass
class GpuSample:
    """One telemetry reading."""

    t: float
    power_w: float = float("nan")
    util_gpu_pct: float = float("nan")
    util_mem_pct: float = float("nan")
    mem_used_mb: float = float("nan")
    sm_clock_mhz: float = float("nan")
    temp_c: float = float("nan")


@dataclass
class GpuSummary:
    """Aggregated telemetry over a sampling window."""

    n_samples: int = 0
    power_w_mean: float = float("nan")
    power_w_max: float = float("nan")
    util_gpu_mean: float = float("nan")
    util_mem_mean: float = float("nan")
    mem_used_mb_max: float = float("nan")
    sm_clock_mhz_mean: float = float("nan")
    temp_c_max: float = float("nan")
    # Energy over the window, integrated from power (mean_power × window_seconds).
    # TODO(research): confirm energy-reporting convention (J vs Wh, per-Mstep
    # normalisation) against the ML energy-reporting sources in the research report.
    window_s: float = float("nan")
    energy_j: float = float("nan")


class GpuSampler:
    """Poll GPU telemetry on a background thread between ``start()`` and ``stop()``.

    Prefers ``pynvml`` (in-process, low overhead); falls back to spawning
    ``nvidia-smi --query-gpu`` each interval. Both work headless on the A6000
    workstation and on Alex compute nodes. The sampler is best-effort: if neither
    backend is available every field stays NaN and the benchmark still runs.

    Parameters
    ----------
    device_index : the CUDA device to watch (default 0; on Alex a job sees one GPU).
    interval_s   : polling period. 0.2 s balances resolution vs. self-overhead.
    """

    _SMI_FIELDS = (
        "power.draw",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "clocks.sm",
        "temperature.gpu",
    )

    def __init__(self, device_index: int = 0, interval_s: float = 0.2):
        self.device_index = device_index
        self.interval_s = interval_s
        self._samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._smi = shutil.which("nvidia-smi")

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "GpuSampler":
        self._samples.clear()
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> GpuSummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._summarise()

    # -- polling ------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            s = self._read()
            if s is not None:
                self._samples.append(s)
            self._stop.wait(self.interval_s)

    def _read(self) -> Optional[GpuSample]:
        now = time.perf_counter()
        if _HAVE_NVML:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                return GpuSample(
                    t=now,
                    power_w=pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    util_gpu_pct=float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
                    util_mem_pct=float(pynvml.nvmlDeviceGetUtilizationRates(h).memory),
                    mem_used_mb=pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2,
                    sm_clock_mhz=float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
                    temp_c=float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
                )
            except Exception:
                return None
        if self._smi is None:
            return None
        try:
            out = subprocess.run(
                [self._smi, f"--id={self.device_index}",
                 f"--query-gpu={','.join(self._SMI_FIELDS)}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0, check=True,
            ).stdout.strip()
            vals = [v.strip() for v in out.split(",")]

            def f(i: int) -> float:
                try:
                    return float(vals[i])
                except (ValueError, IndexError):
                    return float("nan")

            return GpuSample(now, f(0), f(1), f(2), f(3), f(4), f(5))
        except Exception:
            return None

    # -- aggregation --------------------------------------------------------
    def _summarise(self) -> GpuSummary:
        if not self._samples:
            return GpuSummary()
        window = self._samples[-1].t - self._samples[0].t

        def col(attr: str) -> list[float]:
            return [getattr(s, attr) for s in self._samples
                    if getattr(s, attr) == getattr(s, attr)]  # drop NaN

        power = col("power_w")
        energy = (statistics.fmean(power) * window) if power and window > 0 else float("nan")
        return GpuSummary(
            n_samples=len(self._samples),
            power_w_mean=statistics.fmean(power) if power else float("nan"),
            power_w_max=max(power) if power else float("nan"),
            util_gpu_mean=statistics.fmean(col("util_gpu_pct")) if col("util_gpu_pct") else float("nan"),
            util_mem_mean=statistics.fmean(col("util_mem_pct")) if col("util_mem_pct") else float("nan"),
            mem_used_mb_max=max(col("mem_used_mb")) if col("mem_used_mb") else float("nan"),
            sm_clock_mhz_mean=statistics.fmean(col("sm_clock_mhz")) if col("sm_clock_mhz") else float("nan"),
            temp_c_max=max(col("temp_c")) if col("temp_c") else float("nan"),
            window_s=window,
            energy_j=energy,
        )


# ---------------------------------------------------------------------------
# Memory + host collectors
# ---------------------------------------------------------------------------

def torch_mem(device: Optional[str] = None) -> dict[str, float]:
    """PyTorch CUDA caching-allocator counters, in GiB.

    ``max_allocated`` is the honest *peak* since the last
    ``reset_peak_memory_stats`` — call :func:`reset_torch_peak` at the start of the
    measured window. Reserved ≥ allocated because PyTorch caches freed blocks; the
    research report (Question D) asks for peak-vs-steady reporting under exactly
    this caching behaviour.
    """
    if not _HAVE_TORCH or not torch.cuda.is_available():
        return {k: float("nan") for k in
                ("cuda_alloc_gb", "cuda_reserved_gb", "cuda_max_alloc_gb", "cuda_max_reserved_gb")}
    g = 1024**3
    return {
        "cuda_alloc_gb": torch.cuda.memory_allocated(device) / g,
        "cuda_reserved_gb": torch.cuda.memory_reserved(device) / g,
        "cuda_max_alloc_gb": torch.cuda.max_memory_allocated(device) / g,
        "cuda_max_reserved_gb": torch.cuda.max_memory_reserved(device) / g,
    }


def reset_torch_peak(device: Optional[str] = None) -> None:
    if _HAVE_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def driver_used_gb() -> float:
    """Driver-level used VRAM (all processes) in GiB — the number that actually
    bounds how many envs fit. Complements the PyTorch counters, which miss the
    Isaac/PhysX native allocations living outside the torch allocator."""
    if not _HAVE_TORCH:
        return float("nan")
    try:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024**3
    except Exception:
        return float("nan")


def host_stats() -> dict[str, float]:
    """Process RSS + system RAM + CPU utilisation (best-effort via psutil)."""
    if not _HAVE_PSUTIL:
        return {k: float("nan") for k in ("host_rss_gb", "host_ram_used_gb", "host_cpu_pct")}
    g = 1024**3
    vm = psutil.virtual_memory()
    return {
        "host_rss_gb": psutil.Process().memory_info().rss / g,
        "host_ram_used_gb": vm.used / g,
        "host_cpu_pct": psutil.cpu_percent(interval=None),
    }


# ---------------------------------------------------------------------------
# Step-latency timer
# ---------------------------------------------------------------------------

@dataclass
class StepTimer:
    """Accumulate per-step wall times and summarise throughput + latency spread.

    Report both aggregate throughput (steps/s) and the p50/p95 per-step latency:
    a high mean throughput can hide periodic solver-driven stalls, which matter for
    the cloth-scaling questions. ``env_steps_per_s`` (steps/s × N) is the PPO-
    relevant figure.
    """

    _durations: list[float] = field(default_factory=list)
    _t: float = 0.0

    def tic(self) -> None:
        self._t = time.perf_counter()

    def toc(self) -> None:
        self._durations.append(time.perf_counter() - self._t)

    def summary(self, num_envs: int) -> dict[str, float]:
        d = self._durations
        if not d:
            return {}
        total = sum(d)
        steps_per_s = len(d) / total if total > 0 else float("nan")
        return {
            "n_steps": len(d),
            "steps_per_s": steps_per_s,
            "env_steps_per_s": steps_per_s * num_envs,
            "step_ms_mean": 1e3 * statistics.fmean(d),
            "step_ms_p50": 1e3 * statistics.median(d),
            "step_ms_p95": 1e3 * (statistics.quantiles(d, n=20)[18] if len(d) >= 20 else max(d)),
            "step_ms_max": 1e3 * max(d),
        }

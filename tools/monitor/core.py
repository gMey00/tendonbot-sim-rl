"""Data layer: run discovery, TensorBoard parsing, metric categorization, ETA."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tbparse import SummaryReader


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    """Metadata about a single training run."""

    run_dir: str
    task: str
    status: str
    target_timesteps: int
    algorithm: str
    framework: str
    # Derived at construction
    short_name: str = ""
    mtime: float = 0.0

    def __post_init__(self) -> None:
        if not self.short_name:
            self.short_name = os.path.basename(self.run_dir)
        if self.mtime == 0.0:
            status_file = os.path.join(self.run_dir, "run_status.json")
            if os.path.exists(status_file):
                self.mtime = os.path.getmtime(status_file)


@dataclass
class MetricGroup:
    """A category of metrics discovered from TensorBoard tags."""

    name: str
    tags: list[str] = field(default_factory=list)


@dataclass
class ProgressInfo:
    """Current training progress and ETA estimates."""

    current_step: int = 0
    target_steps: int = 0
    pct: float = 0.0
    elapsed_secs: float = 0.0
    eta_secs: float | None = None
    steps_per_sec: float = 0.0
    num_checkpoints: int = 0


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def _scan_single_root(
    log_root: Path,
    include_all: bool,
) -> list[RunInfo]:
    """Scan a single directory tree for run_status.json files."""
    runs: list[RunInfo] = []
    if not log_root.exists():
        return runs

    for status_file in sorted(log_root.rglob("run_status.json")):
        try:
            with open(status_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        info = RunInfo(
            run_dir=str(status_file.parent),
            task=data.get("task", "unknown"),
            status=data.get("status", "unknown"),
            target_timesteps=int(data.get("target_timesteps", 0)),
            algorithm=data.get("algorithm", "ppo"),
            framework=data.get("framework", "torch"),
        )

        if include_all or info.status == "running":
            runs.append(info)

    return runs


def discover_runs(
    log_roots: str | Path | list[str | Path],
    include_all: bool = False,
) -> list[RunInfo]:
    """Recursively scan one or more *log_roots* for run_status.json files.

    Returns a de-duplicated list of RunInfo sorted by modification time
    (most recent first).  If *include_all* is False only "running" runs
    are returned.
    """
    if isinstance(log_roots, (str, Path)):
        log_roots = [log_roots]

    seen_dirs: set[str] = set()
    runs: list[RunInfo] = []

    for root in log_roots:
        for info in _scan_single_root(Path(root), include_all):
            if info.run_dir not in seen_dirs:
                seen_dirs.add(info.run_dir)
                runs.append(info)

    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def latest_event_mtime(run_dir: str | Path) -> float:
    """Return the newest mtime among a run's TensorBoard event files.

    Used as a cheap liveness signal for the run list without paying the cost of
    a full tbparse load: a "running" run whose event file has not been touched
    for a while is very likely a job that died without writing its final status.
    Returns 0.0 if no event file is present.
    """
    newest = 0.0
    try:
        for ev in Path(run_dir).glob("events.out.tfevents*"):
            try:
                newest = max(newest, ev.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return newest


def run_is_stale(run_info: "RunInfo", max_age_secs: float = 180.0) -> bool:
    """Heuristic: a run marked 'running' whose events stopped updating.

    Only ever reports True for runs whose status is still 'running' — a real
    completed/failed run is never "stale". A fresh event file (updated within
    *max_age_secs*) is considered live.
    """
    if run_info.status != "running":
        return False
    mtime = latest_event_mtime(run_info.run_dir)
    if mtime == 0.0:
        return False  # no events yet — treat as just-started, not stale
    return (time.time() - mtime) > max_age_secs


def find_log_roots(workspace: str | Path | None = None) -> list[Path]:
    """Return **all** existing log root directories.

    Checks both ``<workspace>/logs/skrl/`` and
    ``<workspace>/src/tensegrity_pick/logs/skrl/`` (the two locations used
    by different launch methods).
    """
    candidates: list[Path] = []
    bases = []
    if workspace:
        bases.append(Path(workspace))
    bases.append(Path.cwd())

    for base in bases:
        candidates.append(base / "logs" / "skrl")
        candidates.append(base / "src" / "tensegrity_pick" / "logs" / "skrl")

    # De-duplicate resolved paths and keep only existing ones
    seen: set[str] = set()
    roots: list[Path] = []
    for c in candidates:
        resolved = str(c.resolve())
        if resolved not in seen and c.is_dir():
            seen.add(resolved)
            roots.append(c)

    return roots


# ---------------------------------------------------------------------------
# TensorBoard parsing
# ---------------------------------------------------------------------------

# Each category maps to a tuple of acceptable tag prefixes. Multiple prefixes
# per category make the categorizer robust to logging-format changes across
# skrl / Isaac Lab versions: the reward and metric tags used to be written with
# an "Info / " prefix (skrl <= 2.0, e.g. "Info / Episode_Reward/reaching") but
# newer runs drop it ("Episode_Reward/reaching"). We accept both so historical
# and current runs both categorize — the previous single-prefix match silently
# dropped every reward/metric term for new runs.
_TAG_CATEGORIES = {
    "rewards": ("Info / Episode_Reward/", "Episode_Reward/"),
    "custom_metrics": ("Info / Metrics/", "Metrics/"),
    "total_reward": ("Reward / ",),
    "policy": ("Policy / ",),
    "loss": ("Loss / ",),
    "learning": ("Learning / ",),
    "episode": ("Episode / ",),
    "stats": ("Stats / ",),
}

# Flat list of every known prefix, longest first, so tag_display_name strips the
# most specific match (e.g. "Info / Episode_Reward/" before "Episode_Reward/").
_ALL_PREFIXES = sorted(
    {p for prefixes in _TAG_CATEGORIES.values() for p in prefixes},
    key=len,
    reverse=True,
)


def _tag_matches(tag: str, prefixes: tuple[str, ...]) -> bool:
    """True if *tag* starts with any of the category's accepted prefixes."""
    return any(tag.startswith(p) for p in prefixes)


def load_scalars(run_dir: str | Path) -> pd.DataFrame:
    """Read all scalar events from the TensorBoard event file(s) in *run_dir*."""
    reader = SummaryReader(str(run_dir), extra_columns={"wall_time"})
    df = reader.scalars
    return df


def categorize_tags(df: pd.DataFrame) -> dict[str, MetricGroup]:
    """Group all discovered tags into semantic categories."""
    if df.empty:
        return {}

    all_tags = sorted(df["tag"].unique())
    groups: dict[str, MetricGroup] = {}

    for category_key, prefixes in _TAG_CATEGORIES.items():
        matching = [t for t in all_tags if _tag_matches(t, prefixes)]
        if matching:
            groups[category_key] = MetricGroup(name=category_key, tags=matching)

    # Catch any uncategorized tags
    categorized = set()
    for g in groups.values():
        categorized.update(g.tags)
    uncategorized = [t for t in all_tags if t not in categorized]
    if uncategorized:
        groups["other"] = MetricGroup(name="other", tags=uncategorized)

    return groups


def tag_display_name(tag: str) -> str:
    """Extract a human-readable short name from a full TensorBoard tag."""
    for prefix in _ALL_PREFIXES:  # longest-first so the most specific one wins
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return tag


# ---------------------------------------------------------------------------
# Latest values & trends
# ---------------------------------------------------------------------------

def get_latest_values(
    df: pd.DataFrame,
    tags: list[str],
) -> dict[str, tuple[float, float | None]]:
    """Return {tag: (latest_value, prev_value)} for the given tags.

    prev_value is the second-to-last value, used for trend arrows.
    """
    result: dict[str, tuple[float, float | None]] = {}
    for tag in tags:
        sub = df[df["tag"] == tag].sort_values("step")
        if sub.empty:
            continue
        latest = float(sub.iloc[-1]["value"])
        prev = float(sub.iloc[-2]["value"]) if len(sub) >= 2 else None
        result[tag] = (latest, prev)
    return result


def get_series(
    df: pd.DataFrame,
    tag: str,
    ema_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for a tag. Optionally apply EMA smoothing."""
    sub = df[df["tag"] == tag].sort_values("step")
    if sub.empty:
        return np.array([]), np.array([])
    steps = sub["step"].values.astype(float)
    values = sub["value"].values.astype(float)
    if ema_alpha > 0:
        values = _ema(values, ema_alpha)
    return steps, values


def _ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average with weight *alpha* (0..1, higher = more smooth)."""
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * data[i]
    return smoothed


# ---------------------------------------------------------------------------
# Progress & ETA
# ---------------------------------------------------------------------------

def compute_progress(
    df: pd.DataFrame,
    run_info: RunInfo,
) -> ProgressInfo:
    """Compute current training progress, speed, and estimated time of arrival."""
    info = ProgressInfo(target_steps=run_info.target_timesteps)

    # Count checkpoints
    ckpt_dir = os.path.join(run_info.run_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        info.num_checkpoints = len(
            [f for f in os.listdir(ckpt_dir) if f.startswith("agent_") and f.endswith(".pt")]
        )

    # Find current step from total reward tag (most reliable)
    reward_tags = [
        "Reward / Total reward (mean)",
        "Reward / Instantaneous reward (mean)",
    ]
    for tag in reward_tags:
        sub = df[df["tag"] == tag].sort_values("step")
        if not sub.empty:
            info.current_step = int(sub.iloc[-1]["step"])
            break
    else:
        # Fallback: max step across all tags
        if not df.empty:
            info.current_step = int(df["step"].max())

    if info.target_steps > 0:
        info.pct = min(info.current_step / info.target_steps * 100, 100.0)

    # Compute speed and ETA from wall_time
    if not df.empty and "wall_time" in df.columns:
        # Use the reward tag for consistent timing
        for tag in reward_tags:
            sub = df[df["tag"] == tag].sort_values("step")
            if len(sub) >= 2:
                break
        else:
            sub = df.sort_values("step")

        if len(sub) >= 2:
            first_wall = float(sub.iloc[0]["wall_time"])
            last_wall = float(sub.iloc[-1]["wall_time"])
            first_step = int(sub.iloc[0]["step"])
            last_step = int(sub.iloc[-1]["step"])

            info.elapsed_secs = last_wall - first_wall
            step_delta = last_step - first_step
            if info.elapsed_secs > 0 and step_delta > 0:
                info.steps_per_sec = step_delta / info.elapsed_secs
                remaining = max(0, info.target_steps - info.current_step)
                info.eta_secs = remaining / info.steps_per_sec

    return info


def format_duration(seconds: float | None) -> str:
    """Format seconds into a human-readable string like '2h 15m 30s'."""
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    if hours < 24:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    days = hours // 24
    hrs = hours % 24
    return f"{days}d {hrs:02d}h {mins:02d}m"

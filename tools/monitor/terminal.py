"""Rich-based terminal dashboard with plotext inline charts."""

from __future__ import annotations

import datetime
import os
import select
import shutil
import sys
import termios
import threading
import time

import numpy as np
import plotext as plt
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .core import (
    MetricGroup,
    ProgressInfo,
    RunInfo,
    categorize_tags,
    compute_progress,
    discover_runs,
    find_log_roots,
    format_duration,
    get_latest_values,
    get_series,
    latest_event_mtime,
    load_scalars,
    run_is_stale,
    tag_display_name,
)
from .sysmetrics import SysMetricsCollector, SysSnapshot


# ---------------------------------------------------------------------------
# View names and keyboard navigation
# ---------------------------------------------------------------------------

_VIEW_NAMES = ["Overview", "Reward Charts", "Policy", "Performance", "Runs"]

# Unicode blocks for inline sparklines (eight levels).
_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def _sparkline(values, width: int = 40, vmin: float | None = None, vmax: float | None = None) -> str:
    """Render a sequence of numbers as a compact unicode sparkline string."""
    vals = [float(v) for v in values][-width:]
    if not vals:
        return ""
    lo = min(vals) if vmin is None else vmin
    hi = max(vals) if vmax is None else vmax
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    out = []
    for v in vals:
        frac = (v - lo) / (hi - lo)
        idx = int(round(frac * (len(_SPARK_BLOCKS) - 1)))
        idx = max(0, min(len(_SPARK_BLOCKS) - 1, idx))
        out.append(_SPARK_BLOCKS[idx])
    return "".join(out)


def _bar(pct: float, width: int = 20, color: str = "green") -> Text:
    """A horizontal fill bar as a Rich Text object, coloured by load level."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    # Escalate colour with load unless the caller forced one.
    if color == "auto":
        color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="dim")
    return t


def _key_reader(
    stop_event: threading.Event,
    view_holder: list,
    sel_holder: list,
    runs_holder: list,
    active_holder: list,
    wake_event: threading.Event,
    reload_event: threading.Event,
) -> None:
    """Background thread: read raw keypresses and drive navigation / job switch.

    Uses a SURGICAL termios change: only disables ICANON + ECHO on the
    INPUT flags.  Output flags (OPOST / ONLCR) are left untouched so Rich's
    \\n characters still translate to CR+LF — preventing garbled rendering.
    tty.setraw() must NOT be used here because it clears OPOST.

    Keys
    ----
    1-5            switch view (Overview / Reward / Policy / Performance / Runs)
    [ / ]          switch the active job to previous / next running run
    ← / →          same as [ / ]  (quick job cycling from any view)
    ↑ / ↓          move the highlight in the Runs view
    Enter          activate the highlighted run (Runs view)
    q / Ctrl+C     quit

    Sets `wake_event` on every change so the render loop updates immediately;
    sets `reload_event` when the active run changes so the data loader reloads
    that run's scalars right away instead of waiting for the next cycle.
    """
    if not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()

    def _switch_active(delta: int) -> None:
        """Move the active run by *delta* within the current runs list."""
        runs = runs_holder[0] or []
        if not runs:
            return
        cur = active_holder[0]
        try:
            idx = next(i for i, r in enumerate(runs) if r.run_dir == cur.run_dir)
        except StopIteration:
            idx = 0
        idx = (idx + delta) % len(runs)
        active_holder[0] = runs[idx]
        sel_holder[0] = idx
        reload_event.set()
        wake_event.set()

    def _activate_selected() -> None:
        runs = runs_holder[0] or []
        if not runs:
            return
        idx = max(0, min(sel_holder[0], len(runs) - 1))
        active_holder[0] = runs[idx]
        reload_event.set()
        wake_event.set()

    def _move_selection(delta: int) -> None:
        runs = runs_holder[0] or []
        if not runs:
            return
        sel_holder[0] = (sel_holder[0] + delta) % len(runs)
        wake_event.set()

    old_settings = termios.tcgetattr(fd)
    try:
        # Build new attrs: copy everything, then only touch local (input) flags
        new_settings = [list(v) if isinstance(v, list) else v for v in old_settings]
        # c_lflag is index 3: clear ICANON (line buffering) and ECHO
        new_settings[3] = old_settings[3] & ~termios.ICANON & ~termios.ECHO
        # VMIN=1: return after 1 char; VTIME=0: no inter-char timer
        cc = list(old_settings[6])
        cc[termios.VMIN] = 1
        cc[termios.VTIME] = 0
        new_settings[6] = cc
        # OPOST (output post-processing) and ONLCR stay ON — Rich needs them
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)

        while not stop_event.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            ch = os.read(fd, 1).decode("utf-8", errors="ignore")
            if ch == '\x1b':  # escape sequence — could be an arrow key
                seq = ""
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    seq = os.read(fd, 2).decode("utf-8", errors="ignore")
                if seq == '[C':      # → right : next job
                    _switch_active(+1)
                elif seq == '[D':    # ← left : previous job
                    _switch_active(-1)
                elif seq == '[A':    # ↑ up : move selection up (Runs view)
                    _move_selection(-1)
                elif seq == '[B':    # ↓ down : move selection down (Runs view)
                    _move_selection(+1)
            elif ch in ('\x03', 'q', 'Q'):  # Ctrl+C or Q
                stop_event.set()
                wake_event.set()  # unblock main loop immediately
            elif ch in ('\r', '\n'):  # Enter — activate highlighted run
                _activate_selected()
            elif ch in ('1', '2', '3', '4', '5'):
                view_holder[0] = int(ch) - 1
                wake_event.set()  # instant view switch
            elif ch in ('[', ','):   # previous job
                _switch_active(-1)
            elif ch in (']', '.'):   # next job
                _switch_active(+1)
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def _data_loader(
    stop_event: threading.Event,
    active_holder: list,
    data_holder: list,
    runs_holder: list,
    log_roots: list,
    include_all: bool,
    wake_event: threading.Event,
    reload_event: threading.Event,
    interval: float,
) -> None:
    """Background thread: reload scalars for the *active* run + refresh run list.

    Stores ``(run_info, df, groups, progress)`` in ``data_holder[0]`` — the run
    is bundled with its data so the render loop always draws a consistent pair
    even when the user switches jobs mid-load. Also refreshes the discovered
    runs list into ``runs_holder[0]`` so the Runs view and job-switching keys
    see newly started / finished jobs.

    Waits on ``reload_event`` so a job switch reloads immediately instead of
    waiting up to *interval* seconds; ``stop_event`` still wakes it on quit.
    """
    while not stop_event.is_set():
        run_info = active_holder[0]
        try:
            df = load_scalars(run_info.run_dir)
            groups = categorize_tags(df)
            progress = compute_progress(df, run_info)
            data_holder[0] = (run_info, df, groups, progress)
        except Exception:
            pass
        # Refresh the discovered-runs list (cheap: reads run_status.json only).
        try:
            runs = discover_runs(log_roots, include_all=include_all)
            if runs:
                runs_holder[0] = runs
        except Exception:
            pass
        wake_event.set()  # signal render loop: new data ready
        # Sleep, but wake early on quit OR on a job switch.
        reload_event.wait(timeout=interval)
        reload_event.clear()


def _sys_loader(
    stop_event: threading.Event,
    collector: SysMetricsCollector,
    sys_holder: list,
    wake_event: threading.Event,
    interval: float,
) -> None:
    """Background thread: sample GPU/CPU/RAM metrics on a steady ~2s cadence.

    Runs independently of the (possibly slow) data-reload interval so the
    Performance view's live gauges and sparklines stay smooth. Capped at the
    data interval so a very fast --interval doesn't oversample nvidia-smi.
    """
    period = max(1.0, min(2.0, interval))
    while not stop_event.is_set():
        try:
            sys_holder[0] = collector.sample()
        except Exception:
            pass
        wake_event.set()
        stop_event.wait(timeout=period)


def _build_nav(current_view: int, active: RunInfo | None = None, n_runs: int = 0) -> Group:
    """Navigation bar: Rule / view tabs / active-job + hints / Rule."""
    line = Text(justify="center")
    line.append("  Views: ", style="bold dim")
    for i, name in enumerate(_VIEW_NAMES):
        key = str(i + 1)
        if i == current_view:
            line.append(f" {key}:{name} ", style="bold white on blue")
        else:
            line.append(f" {key}:{name} ", style="bright_black")
        if i < len(_VIEW_NAMES) - 1:
            line.append("  │  ", style="dim")

    hint = Text(justify="center")
    if active is not None and n_runs > 0:
        hint.append("  Job: ", style="bold dim")
        hint.append(active.short_name, style="bold cyan")
        if n_runs > 1:
            hint.append(f"  ([ ] or ←/→ to switch, {n_runs} runs)", style="dim")
    hint.append("     Q / Ctrl+C: Quit", style="dim")
    return Group(Rule(style="bright_blue"), line, hint, Rule(style="bright_blue"))


def _build_compact_status(run_info: RunInfo, progress: ProgressInfo) -> Panel:
    """Single-row status banner used in chart views."""
    status_color = {"running": "green", "completed": "blue", "failed": "red"}.get(
        run_info.status, "yellow"
    )
    line = Text()
    line.append(run_info.task, style="bold cyan")
    line.append("  │  ", style="dim")
    line.append(f"{progress.current_step:,} / {progress.target_steps:,}", style="white")
    line.append(f"  ({progress.pct:.1f}%)", style="yellow")
    if progress.steps_per_sec > 0:
        line.append("  │  ", style="dim")
        line.append(f"{progress.steps_per_sec:.0f} steps/s", style="green")
    line.append("  │  ETA: ", style="dim")
    line.append(format_duration(progress.eta_secs), style="cyan")
    line.append("  │  ", style="dim")
    line.append(run_info.status.upper(), style=f"bold {status_color}")
    return Panel(line, height=3, border_style="dim")


# ---------------------------------------------------------------------------
# Per-view builders
# ---------------------------------------------------------------------------

def _view_overview(
    df, groups, progress: ProgressInfo, run_info: RunInfo,
    nav: Group, term_width: int,
) -> Group:
    header = build_header(run_info, progress)
    rewards_panel = build_rewards_table(df, groups)
    learning_panel = build_learning_table(df, groups)
    tables_grid = Table.grid(expand=True)
    tables_grid.add_column(ratio=1)
    tables_grid.add_column(ratio=1)
    if groups.get("custom_metrics"):
        tables_grid.add_row(Group(rewards_panel, build_custom_metrics_table(df, groups)), learning_panel)
    else:
        tables_grid.add_row(rewards_panel, learning_panel)
    ts = Text(
        f"  Last refresh: {datetime.datetime.now().strftime('%H:%M:%S')}  "
        f"│  Auto-refresh every {_INTERVAL_HINT}s",
        style="dim",
    )
    return Group(nav, header, tables_grid, ts)


def _view_reward_charts(
    df, groups, progress: ProgressInfo, run_info: RunInfo,
    nav: Group, term_width: int, term_height: int,
) -> Group:
    status = _build_compact_status(run_info, progress)
    # Each plot gets half the remaining rows (minus borders)
    available = max(16, term_height - 10)  # 4 nav + 3 status + 3 timestamp rows
    plot_h = max(6, (available - 4) // 2)  # 4 for two panel borders
    rplot = build_reward_plot(df, groups, width=term_width - 4, height=plot_h)
    dplot = build_rewards_decomposition_plot(df, groups, width=term_width - 4, height=plot_h)
    ts = Text(
        f"  Last refresh: {datetime.datetime.now().strftime('%H:%M:%S')}  "
        f"│  Auto-refresh every {_INTERVAL_HINT}s",
        style="dim",
    )
    return Group(nav, status, rplot, dplot, ts)


def _view_policy(
    df, groups, progress: ProgressInfo, run_info: RunInfo,
    nav: Group, term_width: int, term_height: int,
) -> Group:
    status = _build_compact_status(run_info, progress)
    available = max(16, term_height - 10)
    plot_h = max(12, available - 4)
    pplot = build_policy_plot(df, groups, width=term_width - 4, height=plot_h)
    ts = Text(
        f"  Last refresh: {datetime.datetime.now().strftime('%H:%M:%S')}  "
        f"│  Auto-refresh every {_INTERVAL_HINT}s",
        style="dim",
    )
    return Group(nav, status, pplot, ts)


def _view_performance(
    snapshot: SysSnapshot | None, collector: SysMetricsCollector,
    run_info: RunInfo, progress: ProgressInfo,
    nav: Group, term_width: int, term_height: int,
) -> Group:
    status = _build_compact_status(run_info, progress)
    body = build_performance_panels(snapshot, collector, term_width, term_height)
    ts = Text(
        f"  Last refresh: {datetime.datetime.now().strftime('%H:%M:%S')}  "
        f"│  Perf sampled every ~2s",
        style="dim",
    )
    return Group(nav, status, body, ts)


def _view_runs(
    runs: list, active: RunInfo, sel_idx: int,
    nav: Group, term_width: int, term_height: int = 40,
) -> Group:
    # Reserve rows for nav (4), panel borders/header (4), help + timestamp (2).
    max_rows = max(3, term_height - 12)
    table = build_runs_table(runs, active, sel_idx, max_rows=max_rows)
    help_line = Text(
        "  ↑/↓ move   Enter jump to highlighted   [ ] or ←/→ cycle active job   "
        "1-5 switch view",
        style="dim",
    )
    ts = Text(
        f"  Last refresh: {datetime.datetime.now().strftime('%H:%M:%S')}  "
        f"│  Auto-refresh every {_INTERVAL_HINT}s",
        style="dim",
    )
    return Group(nav, table, help_line, ts)


_INTERVAL_HINT: float = 10.0  # updated at runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trend_arrow(current: float, prev: float | None) -> str:
    if prev is None:
        return "─"
    diff = current - prev
    if abs(diff) < 1e-6:
        return "[dim]─[/]"
    return "[green]▲[/]" if diff > 0 else "[red]▼[/]"


def _colored_value(val: float) -> str:
    if val > 0:
        return f"[green]{val:>10.4f}[/]"
    elif val < 0:
        return f"[red]{val:>10.4f}[/]"
    return f"{val:>10.4f}"


def _plotext_to_rich(width: int = 80, height: int = 15) -> Text:
    """Render current plotext figure as a Rich Text object (handles ANSI codes)."""
    plt.plotsize(width, height)
    return Text.from_ansi(plt.build())


# ---------------------------------------------------------------------------
# Dashboard panels
# ---------------------------------------------------------------------------

def build_header(run_info: RunInfo, progress: ProgressInfo) -> Panel:
    """Top header panel with run metadata and progress."""
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    left = Table.grid(padding=(0, 2))
    left.add_column(style="bold cyan", justify="right")
    left.add_column()
    left.add_row("Task", run_info.task)
    left.add_row("Algorithm", f"{run_info.algorithm.upper()} / {run_info.framework}")
    left.add_row("Run Dir", run_info.short_name)
    status_color = {"running": "green", "completed": "blue", "failed": "red"}.get(
        run_info.status, "yellow"
    )
    left.add_row("Status", f"[{status_color}]{run_info.status.upper()}[/]")

    right = Table.grid(padding=(0, 2))
    right.add_column(style="bold cyan", justify="right")
    right.add_column()
    right.add_row("Steps", f"{progress.current_step:,} / {progress.target_steps:,}")
    right.add_row("Progress", f"{progress.pct:.1f}%")
    right.add_row("Speed", f"{progress.steps_per_sec:,.0f} steps/s" if progress.steps_per_sec > 0 else "N/A")
    right.add_row("Elapsed", format_duration(progress.elapsed_secs))
    right.add_row("ETA", format_duration(progress.eta_secs))
    right.add_row("Checkpoints", str(progress.num_checkpoints))

    grid.add_row(left, right)

    # Build a simple text progress bar
    bar_width = 60
    filled = int(bar_width * min(progress.pct, 100) / 100)
    bar_text = Text()
    bar_text.append("━" * filled, style="green")
    bar_text.append("━" * (bar_width - filled), style="dim")
    bar_text.append(f" {progress.pct:5.1f}%")

    return Panel(
        Group(grid, Text(""), bar_text),
        title="[bold white]Isaac Lab Training Monitor[/]",
        border_style="bright_blue",
    )


def build_rewards_table(
    df,
    groups: dict[str, MetricGroup],
) -> Panel:
    """Table of all episode reward terms with latest values and trends."""
    table = Table(
        title="Reward Terms",
        expand=True,
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("Reward", style="bold", min_width=22)
    table.add_column("Value", justify="right", min_width=12)
    table.add_column("Δ", justify="center", min_width=3)

    tags = groups.get("rewards", MetricGroup("rewards")).tags
    if not tags:
        return Panel("[dim]No reward data yet[/]", title="Reward Terms", border_style="yellow")

    values = get_latest_values(df, tags)
    # Sort by absolute value descending
    sorted_tags = sorted(values.keys(), key=lambda t: abs(values[t][0]), reverse=True)

    for tag in sorted_tags:
        val, prev = values[tag]
        name = tag_display_name(tag)
        table.add_row(name, _colored_value(val), _trend_arrow(val, prev))

    return Panel(table, border_style="green")


def build_learning_table(
    df,
    groups: dict[str, MetricGroup],
) -> Panel:
    """Table of policy, loss, learning, and episode metrics."""
    table = Table(
        title="Learning & Policy",
        expand=True,
        show_lines=False,
    )
    table.add_column("Metric", style="bold", min_width=25)
    table.add_column("Value", justify="right", min_width=12)
    table.add_column("Δ", justify="center", min_width=3)

    all_tags: list[str] = []
    for key in ("total_reward", "policy", "loss", "learning", "episode"):
        g = groups.get(key)
        if g:
            all_tags.extend(g.tags)

    if not all_tags:
        return Panel("[dim]No learning data yet[/]", title="Learning & Policy", border_style="yellow")

    values = get_latest_values(df, all_tags)

    for tag in all_tags:
        if tag not in values:
            continue
        val, prev = values[tag]
        name = tag_display_name(tag)
        table.add_row(name, _colored_value(val), _trend_arrow(val, prev))

    return Panel(table, border_style="magenta")


def build_custom_metrics_table(
    df,
    groups: dict[str, MetricGroup],
) -> Panel:
    """Table of custom metrics (success rates, etc.)."""
    tags = groups.get("custom_metrics", MetricGroup("custom_metrics")).tags
    if not tags:
        return Panel("[dim]No custom metrics[/]", title="Custom Metrics", border_style="dim")

    table = Table(title="Custom Metrics", expand=True, show_lines=False)
    table.add_column("Metric", style="bold", min_width=22)
    table.add_column("Value", justify="right", min_width=12)
    table.add_column("Δ", justify="center", min_width=3)

    values = get_latest_values(df, tags)
    for tag in tags:
        if tag not in values:
            continue
        val, prev = values[tag]
        name = tag_display_name(tag)
        table.add_row(name, _colored_value(val), _trend_arrow(val, prev))

    return Panel(table, border_style="cyan")


def build_reward_plot(df, groups: dict[str, MetricGroup], width: int = 80, height: int = 12) -> Panel:
    """Plotext chart of total reward over training steps."""
    plt.clear_figure()
    plt.theme("dark")
    plt.title("Total Reward")
    plt.xlabel("Step")

    plotted = False
    for tag_suffix, color, label in [
        ("Total reward (mean)", "green", "Mean"),
        ("Total reward (max)", "cyan", "Max"),
        ("Total reward (min)", "red", "Min"),
    ]:
        full_tag = f"Reward / {tag_suffix}"
        steps, values = get_series(df, full_tag, ema_alpha=0.8)
        if len(steps) > 0:
            plt.plot(steps.tolist(), values.tolist(), color=color, label=label)
            plotted = True

    if not plotted:
        return Panel("[dim]No reward data to plot[/]", title="Total Reward", border_style="dim")

    return Panel(_plotext_to_rich(width, height), title="[bold]Total Reward[/]", border_style="green")


def build_rewards_decomposition_plot(
    df,
    groups: dict[str, MetricGroup],
    width: int = 80,
    height: int = 12,
    max_lines: int = 6,
) -> Panel:
    """Plotext multi-line chart of top reward terms."""
    tags = groups.get("rewards", MetricGroup("rewards")).tags
    if not tags:
        return Panel("[dim]No reward decomposition data[/]", title="Reward Decomposition", border_style="dim")

    # Pick the top N by absolute latest value
    values = get_latest_values(df, tags)
    sorted_tags = sorted(values.keys(), key=lambda t: abs(values[t][0]), reverse=True)[:max_lines]

    colors = ["green", "cyan", "yellow", "red", "magenta", "blue", "orange", "white"]

    plt.clear_figure()
    plt.theme("dark")
    plt.title("Top Reward Terms")
    plt.xlabel("Step")

    plotted = False
    for i, tag in enumerate(sorted_tags):
        steps, vals = get_series(df, tag, ema_alpha=0.85)
        if len(steps) > 0:
            color = colors[i % len(colors)]
            label = tag_display_name(tag)
            plt.plot(steps.tolist(), vals.tolist(), color=color, label=label)
            plotted = True

    if not plotted:
        return Panel("[dim]No data[/]", border_style="dim")

    return Panel(_plotext_to_rich(width, height), title="[bold]Reward Decomposition[/]", border_style="yellow")


def build_policy_plot(
    df,
    groups: dict[str, MetricGroup],
    width: int = 80,
    height: int = 12,
) -> Panel:
    """Plotext chart of policy diagnostics (loss, std, learning rate)."""
    policy_tags = []
    for key in ("policy", "loss", "learning"):
        g = groups.get(key)
        if g:
            policy_tags.extend(g.tags)

    if not policy_tags:
        return Panel("[dim]No policy data[/]", title="Policy Diagnostics", border_style="dim")

    colors = ["green", "red", "cyan", "yellow", "magenta", "blue"]

    plt.clear_figure()
    plt.theme("dark")
    plt.title("Policy Diagnostics")
    plt.xlabel("Step")

    plotted = False
    for i, tag in enumerate(policy_tags[:6]):
        steps, vals = get_series(df, tag, ema_alpha=0.8)
        if len(steps) > 0:
            color = colors[i % len(colors)]
            label = tag_display_name(tag)
            plt.plot(steps.tolist(), vals.tolist(), color=color, label=label)
            plotted = True

    if not plotted:
        return Panel("[dim]No data[/]", border_style="dim")

    return Panel(_plotext_to_rich(width, height), title="[bold]Policy Diagnostics[/]", border_style="magenta")


# ---------------------------------------------------------------------------
# Performance view (GPU / CPU / memory)
# ---------------------------------------------------------------------------

def _fmt_gb(gb: float) -> str:
    return f"{gb:.1f}G" if gb < 1000 else f"{gb/1000:.1f}T"


def build_performance_panels(
    snapshot: SysSnapshot | None,
    collector: SysMetricsCollector,
    term_width: int,
    term_height: int,
) -> Group:
    """Compose the live performance dashboard (GPU + CPU/RAM + processes)."""
    if snapshot is None:
        return Group(Panel("[dim]Sampling system metrics…[/]",
                           title="Performance", border_style="dim"))

    spark_w = max(20, min(60, term_width - 30))

    # ---- GPU panel -------------------------------------------------------
    if snapshot.gpus:
        gtab = Table(expand=True, show_lines=False, pad_edge=False)
        gtab.add_column("GPU", style="bold", no_wrap=True)
        gtab.add_column("Util", justify="left", min_width=24)
        gtab.add_column("Memory", justify="left", min_width=22)
        gtab.add_column("Temp", justify="right")
        gtab.add_column("Power", justify="right")
        gtab.add_column("History (util %)", justify="left", no_wrap=True)
        for g in snapshot.gpus:
            util_cell = Text()
            util_cell.append_text(_bar(g.util_pct, width=12, color="auto"))
            util_cell.append(f" {g.util_pct:4.0f}%")
            mem_cell = Text()
            mem_cell.append_text(_bar(g.mem_pct, width=10, color="auto"))
            mem_cell.append(f" {g.mem_used_mb/1024:4.1f}/{g.mem_total_mb/1024:.0f}G")
            hist = collector.gpu_util_hist.get(g.index)
            spark = _sparkline(hist, width=spark_w, vmin=0, vmax=100) if hist else ""
            pwr = f"{g.power_w:.0f}/{g.power_limit_w:.0f}W" if g.power_limit_w else f"{g.power_w:.0f}W"
            gtab.add_row(
                f"{g.index}:{g.name.replace('NVIDIA ', '')}",
                util_cell,
                mem_cell,
                f"{g.temp_c:.0f}°C",
                pwr,
                Text(spark, style="cyan"),
            )
        gpu_panel = Panel(gtab, title="[bold]GPU[/]", border_style="green")
    else:
        note = "; ".join(e for e in snapshot.errors if "nvidia" in e.lower()) or "No GPU detected"
        gpu_panel = Panel(f"[dim]{note}[/]", title="[bold]GPU[/]", border_style="yellow")

    # ---- CPU / memory panel ---------------------------------------------
    ctab = Table.grid(expand=True, padding=(0, 1))
    ctab.add_column(style="bold cyan", justify="right", min_width=10)
    ctab.add_column()

    cpu_bar = Text()
    cpu_bar.append_text(_bar(snapshot.cpu_pct, width=20, color="auto"))
    cpu_bar.append(f" {snapshot.cpu_pct:5.1f}%  ({snapshot.cpu_count} cores)")
    ctab.add_row("CPU", cpu_bar)
    if collector.cpu_hist:
        ctab.add_row("", Text(_sparkline(collector.cpu_hist, width=spark_w, vmin=0, vmax=100), style="cyan"))

    mem_bar = Text()
    mem_bar.append_text(_bar(snapshot.mem_pct, width=20, color="auto"))
    mem_bar.append(f" {snapshot.mem_pct:5.1f}%  ({_fmt_gb(snapshot.mem_used_gb)}/{_fmt_gb(snapshot.mem_total_gb)})")
    ctab.add_row("RAM", mem_bar)
    if collector.mem_hist:
        ctab.add_row("", Text(_sparkline(collector.mem_hist, width=spark_w, vmin=0, vmax=100), style="magenta"))

    if snapshot.swap_total_gb > 0:
        swap_pct = snapshot.swap_used_gb / snapshot.swap_total_gb * 100
        swap_bar = Text()
        swap_bar.append_text(_bar(swap_pct, width=20, color="auto"))
        swap_bar.append(f" {swap_pct:5.1f}%  ({_fmt_gb(snapshot.swap_used_gb)}/{_fmt_gb(snapshot.swap_total_gb)})")
        ctab.add_row("Swap", swap_bar)

    load_txt = f"{snapshot.load1:.2f}  {snapshot.load5:.2f}  {snapshot.load15:.2f}  (1/5/15 min)"
    ctab.add_row("Load", Text(load_txt))
    host_txt = snapshot.hostname
    if snapshot.slurm_job:
        host_txt += f"   Slurm job {snapshot.slurm_job}"
    ctab.add_row("Host", Text(host_txt, style="dim"))
    cpu_panel = Panel(ctab, title="[bold]CPU / Memory[/]", border_style="magenta")

    # ---- Training processes ---------------------------------------------
    if snapshot.train_procs:
        ptab = Table(expand=True, show_lines=False, pad_edge=False)
        ptab.add_column("PID", justify="right", style="bold")
        ptab.add_column("Name")
        ptab.add_column("CPU %", justify="right")
        ptab.add_column("RSS", justify="right")
        for pr in snapshot.train_procs:
            ptab.add_row(str(pr.pid), pr.name, f"{pr.cpu_pct:.0f}", f"{pr.rss_gb:.1f}G")
        proc_panel = Panel(ptab, title="[bold]Training processes[/]", border_style="cyan")
    else:
        proc_panel = Panel(
            "[dim]No training process detected on this host.[/]\n"
            "[dim](On Alex, run the monitor on the compute node via "
            "watch_alex.sh so it sees the job.)[/]",
            title="[bold]Training processes[/]", border_style="dim",
        )

    return Group(gpu_panel, cpu_panel, proc_panel)


# ---------------------------------------------------------------------------
# Runs view (job switcher)
# ---------------------------------------------------------------------------

def _compact_age(seconds: float) -> str:
    """Largest-unit-only age for the run list: '3d', '6h', '27m', '9s'."""
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def build_runs_table(runs: list, active: RunInfo, sel_idx: int, max_rows: int = 20) -> Panel:
    """Table of discovered runs with the active + highlighted rows marked.

    When there are more runs than *max_rows* (e.g. ``--all`` with a long history)
    a scrolling window centred on the highlighted row is shown so the selection
    is always visible and the table never overflows the screen.
    """
    if not runs:
        return Panel("[dim]No runs discovered.[/]", title="Runs", border_style="yellow")

    sel_idx = max(0, min(sel_idx, len(runs) - 1))

    # Compute a scrolling window centred on the selection.
    max_rows = max(3, max_rows)
    if len(runs) <= max_rows:
        start, end = 0, len(runs)
    else:
        start = max(0, sel_idx - max_rows // 2)
        end = min(len(runs), start + max_rows)
        start = max(0, end - max_rows)

    table = Table(expand=True, show_lines=False, pad_edge=False)
    table.add_column(" ", width=2, no_wrap=True)         # active marker
    table.add_column("#", justify="right", width=4, no_wrap=True)
    table.add_column("Task", min_width=16, ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("Status", width=15, no_wrap=True)
    table.add_column("Age", justify="right", width=5, no_wrap=True)
    table.add_column("Run directory", min_width=18, ratio=3, no_wrap=True, overflow="ellipsis")

    now = time.time()
    for i in range(start, end):
        r = runs[i]
        is_active = r.run_dir == active.run_dir
        is_sel = i == sel_idx
        marker = "▶" if is_active else ("›" if is_sel else " ")

        stale = run_is_stale(r)
        status = r.status + (" (stale)" if stale else "")
        status_color = {"running": "green", "completed": "blue", "failed": "red",
                        "interrupted": "yellow"}.get(r.status, "yellow")
        if stale:
            status_color = "yellow"

        mtime = latest_event_mtime(r.run_dir) or r.mtime
        age = _compact_age(now - mtime) if mtime else "—"

        row_style = "on grey19" if is_sel else None
        table.add_row(
            f"[bold green]{marker}[/]" if is_active else f"[bold]{marker}[/]",
            str(i + 1),
            r.task,
            f"[{status_color}]{status}[/]",
            age,
            r.short_name,
            style=row_style,
        )

    scroll = f" · showing {start + 1}–{end} of {len(runs)}" if len(runs) > max_rows else ""
    title = (f"[bold]Runs[/]  ([green]▶[/] active · [white]›[/] highlighted · "
             f"{len(runs)} total{scroll})")
    return Panel(table, title=title, border_style="bright_blue")


# ---------------------------------------------------------------------------
# Run picker (interactive prompt)
# ---------------------------------------------------------------------------

def pick_run(runs: list[RunInfo], console: Console) -> RunInfo | None:
    """Display a table of runs and let the user pick one."""
    if not runs:
        console.print("[red]No training runs found.[/]")
        return None

    if len(runs) == 1:
        console.print(f"[green]Auto-selected:[/] {runs[0].short_name}")
        return runs[0]

    table = Table(title="Available Training Runs", show_lines=True)
    table.add_column("#", style="bold", justify="center", width=4)
    table.add_column("Task", min_width=20)
    table.add_column("Status", min_width=10)
    table.add_column("Algorithm", min_width=8)
    table.add_column("Steps", justify="right", min_width=12)
    table.add_column("Run Directory", min_width=30)

    for i, run in enumerate(runs):
        status_color = {"running": "green", "completed": "blue", "failed": "red"}.get(
            run.status, "yellow"
        )
        table.add_row(
            str(i + 1),
            run.task,
            f"[{status_color}]{run.status}[/]",
            run.algorithm.upper(),
            f"{run.target_timesteps:,}",
            run.short_name,
        )

    console.print(table)
    console.print()

    while True:
        try:
            choice = console.input("[bold]Select run number (or 'q' to quit): [/]")
            if choice.strip().lower() == "q":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                return runs[idx]
            console.print(f"[red]Please enter 1-{len(runs)}[/]")
        except (ValueError, EOFError):
            console.print("[red]Invalid input[/]")


# ---------------------------------------------------------------------------
# Main dashboard loop
# ---------------------------------------------------------------------------

def run_terminal_dashboard(
    run_info: RunInfo,
    interval: float = 10.0,
    runs: list[RunInfo] | None = None,
    log_roots: list | None = None,
    include_all: bool = False,
) -> None:
    """Launch the full-screen live terminal dashboard with view navigation.

    *run_info* is the initially-selected run. *runs* is the full discovered
    list (for in-TUI job switching); *log_roots* / *include_all* let the data
    loader re-discover runs so newly started/finished jobs appear live.
    """
    global _INTERVAL_HINT
    _INTERVAL_HINT = interval

    runs = runs or [run_info]
    log_roots = log_roots or []

    # force_terminal=True ensures Rich uses ANSI/colour in SSH and Warp
    # (without it Rich may silently fall back to plaintext mode, making plots
    # look like raw escape codes or just spaces).
    console = Console(force_terminal=True, force_jupyter=False)
    view_holder = [0]              # current view index, written by key reader
    data_holder = [None]           # (run_info, df, groups, progress)
    sys_holder = [None]            # latest SysSnapshot
    runs_holder = [runs]           # all discovered runs, refreshed by loader
    active_holder = [run_info]     # currently monitored run, set by key reader
    try:
        sel_holder = [next(i for i, r in enumerate(runs) if r.run_dir == run_info.run_dir)]
    except StopIteration:
        sel_holder = [0]

    stop_event = threading.Event()
    wake_event = threading.Event()    # set by key reader / data / sys loaders
    reload_event = threading.Event()  # set on job switch → immediate reload

    sys_collector = SysMetricsCollector()

    # Data is loaded in dedicated threads so the render loop never blocks on
    # disk I/O or nvidia-smi. The render loop only runs when wake_event is set,
    # so it sleeps with zero CPU between view-switches and data refreshes.
    data_thread = threading.Thread(
        target=_data_loader,
        args=(stop_event, active_holder, data_holder, runs_holder, log_roots,
              include_all, wake_event, reload_event, interval),
        daemon=True,
    )
    sys_thread = threading.Thread(
        target=_sys_loader,
        args=(stop_event, sys_collector, sys_holder, wake_event, interval),
        daemon=True,
    )
    key_thread = threading.Thread(
        target=_key_reader,
        args=(stop_event, view_holder, sel_holder, runs_holder, active_holder,
              wake_event, reload_event),
        daemon=True,
    )
    data_thread.start()
    sys_thread.start()
    key_thread.start()

    try:
        with Live(console=console, screen=True, refresh_per_second=2) as live:
            while not stop_event.is_set():
                # Block (zero CPU) until a key is pressed OR new data arrives.
                # timeout is a safety net in case both threads are quiet.
                wake_event.wait(timeout=interval + 2)
                wake_event.clear()

                view = view_holder[0]
                all_runs = runs_holder[0] or [active_holder[0]]

                # shutil.get_terminal_size() issues TIOCGWINSZ directly — the
                # most reliable cross-terminal, cross-SSH, cross-Warp method.
                ts = shutil.get_terminal_size((120, 40))
                w = max(60, ts.columns)
                h = max(20, ts.lines)
                nav = _build_nav(view, active_holder[0], len(all_runs))

                # The Runs view and Performance view do not need the scalar df,
                # so they render even while the first data load is in flight.
                try:
                    if view == 4:
                        content = _view_runs(all_runs, active_holder[0], sel_holder[0], nav, w, h)
                        live.update(content)
                        continue

                    if data_holder[0] is None:
                        # No scalar data yet — Performance can still render.
                        if view == 3:
                            live.update(_view_performance(
                                sys_holder[0], sys_collector, active_holder[0],
                                ProgressInfo(), nav, w, h))
                        else:
                            live.update(Panel(
                                "[dim]Loading data…[/]",
                                title="[bold white]Isaac Lab Training Monitor[/]",
                                border_style="bright_blue",
                            ))
                        continue

                    run_info_d, df, groups, progress = data_holder[0]

                    if view == 0:
                        content = _view_overview(df, groups, progress, run_info_d, nav, w)
                    elif view == 1:
                        content = _view_reward_charts(df, groups, progress, run_info_d, nav, w, h)
                    elif view == 2:
                        content = _view_policy(df, groups, progress, run_info_d, nav, w, h)
                    elif view == 3:
                        content = _view_performance(
                            sys_holder[0], sys_collector, run_info_d, progress, nav, w, h)
                    else:
                        content = _view_runs(all_runs, active_holder[0], sel_holder[0], nav, w, h)

                    live.update(content)
                except Exception as e:
                    import traceback
                    live.update(Panel(
                        f"[red]Error rendering:[/] {e}\n\n[dim]{traceback.format_exc()}[/]",
                        title="[red]Dashboard Error[/]",
                        border_style="red",
                    ))
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        wake_event.set()    # unblock render loop
        reload_event.set()  # unblock data_thread if it's sleeping
        console.print("\n[yellow]Monitor stopped.[/]")


def _build_full_dashboard(run_info: RunInfo, console: Console) -> Group:
    """Build the complete dashboard as a Rich Group of panels."""
    # Reload data
    df = load_scalars(run_info.run_dir)
    groups = categorize_tags(df)
    progress = compute_progress(df, run_info)

    # Detect terminal width for plot sizing
    term_width = console.width or 120
    plot_width = max(60, term_width - 6)
    plot_height = 14

    # Header with progress
    header = build_header(run_info, progress)

    # Tables side by side — we'll stack them vertically for reliability
    rewards_panel = build_rewards_table(df, groups)
    learning_panel = build_learning_table(df, groups)
    custom_panel = build_custom_metrics_table(df, groups)

    # Plots
    reward_plot = build_reward_plot(df, groups, width=plot_width, height=plot_height)
    decomp_plot = build_rewards_decomposition_plot(
        df, groups, width=plot_width, height=plot_height
    )
    policy_plot = build_policy_plot(df, groups, width=plot_width, height=plot_height)

    # Timestamp
    import datetime
    timestamp = Text(
        f"  Last refresh: {datetime.datetime.now().strftime('%H:%M:%S')}  •  "
        f"Ctrl+C to stop",
        style="dim",
    )

    # Compose a two-column layout for tables using a grid
    tables_grid = Table.grid(expand=True)
    tables_grid.add_column(ratio=1)
    tables_grid.add_column(ratio=1)

    # If custom metrics exist, stack rewards + custom on left, learning on right
    if groups.get("custom_metrics"):
        left_group = Group(rewards_panel, custom_panel)
    else:
        left_group = rewards_panel

    tables_grid.add_row(left_group, learning_panel)

    return Group(
        timestamp,
        header,
        tables_grid,
        reward_plot,
        decomp_plot,
        policy_plot,
    )

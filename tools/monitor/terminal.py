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
    load_scalars,
    tag_display_name,
)


# ---------------------------------------------------------------------------
# View names and keyboard navigation
# ---------------------------------------------------------------------------

_VIEW_NAMES = ["Overview", "Reward Charts", "Policy"]


def _key_reader(
    stop_event: threading.Event,
    view_holder: list,
    wake_event: threading.Event,
) -> None:
    """Background thread: read raw keypresses and update view_holder[0].

    Uses a SURGICAL termios change: only disables ICANON + ECHO on the
    INPUT flags.  Output flags (OPOST / ONLCR) are left untouched so Rich's
    \\n characters still translate to CR+LF — preventing garbled rendering.
    tty.setraw() must NOT be used here because it clears OPOST.

    Sets `wake_event` on every view change so the render loop updates
    immediately without waiting for the next data-reload cycle.
    """
    if not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()
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
            if ch == '\x1b':  # escape sequence (arrow keys) — consume & ignore
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    os.read(fd, 2)
            elif ch in ('\x03', 'q', 'Q'):  # Ctrl+C or Q
                stop_event.set()
                wake_event.set()  # unblock main loop immediately
            elif ch == '1':
                view_holder[0] = 0
                wake_event.set()  # instant view switch
            elif ch == '2':
                view_holder[0] = 1
                wake_event.set()
            elif ch == '3':
                view_holder[0] = 2
                wake_event.set()
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def _data_loader(
    stop_event: threading.Event,
    run_info: RunInfo,
    data_holder: list,
    wake_event: threading.Event,
    interval: float,
) -> None:
    """Background thread: reload TensorBoard scalars every *interval* seconds.

    Stores the latest (df, groups, progress) tuple in data_holder[0] and
    sets wake_event so the render loop redraws immediately on new data.
    Uses stop_event.wait(timeout=interval) instead of time.sleep so it
    wakes up immediately when the dashboard is quit.
    """
    while not stop_event.is_set():
        try:
            df = load_scalars(run_info.run_dir)
            groups = categorize_tags(df)
            progress = compute_progress(df, run_info)
            data_holder[0] = (df, groups, progress)
        except Exception:
            pass
        wake_event.set()  # signal render loop: new data ready
        stop_event.wait(timeout=interval)  # sleep, but wake early on quit


def _build_nav(current_view: int) -> Group:
    """Three-row navigation bar: Rule / key hints / Rule."""
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
    line.append("    Q / Ctrl+C: Quit", style="dim")
    return Group(Rule(style="bright_blue"), line, Rule(style="bright_blue"))


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
    available = max(16, term_height - 9)  # 3 nav + 3 status + 3 timestamp rows
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
    available = max(16, term_height - 9)
    plot_h = max(12, available - 4)
    pplot = build_policy_plot(df, groups, width=term_width - 4, height=plot_h)
    ts = Text(
        f"  Last refresh: {datetime.datetime.now().strftime('%H:%M:%S')}  "
        f"│  Auto-refresh every {_INTERVAL_HINT}s",
        style="dim",
    )
    return Group(nav, status, pplot, ts)


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
) -> None:
    """Launch the full-screen live terminal dashboard with view navigation."""
    global _INTERVAL_HINT
    _INTERVAL_HINT = interval

    # force_terminal=True ensures Rich uses ANSI/colour in SSH and Warp
    # (without it Rich may silently fall back to plaintext mode, making plots
    # look like raw escape codes or just spaces).
    console = Console(force_terminal=True, force_jupyter=False)
    view_holder = [0]    # current view index, written by key reader
    data_holder = [None] # (df, groups, progress), written by data loader
    stop_event = threading.Event()
    wake_event = threading.Event()  # set by key reader OR data loader

    # Data is loaded in a dedicated thread so the render loop never blocks
    # on disk I/O. The render loop only runs when wake_event is set, so
    # it sleeps with zero CPU between view-switches and data refreshes.
    data_thread = threading.Thread(
        target=_data_loader,
        args=(stop_event, run_info, data_holder, wake_event, interval),
        daemon=True,
    )
    key_thread = threading.Thread(
        target=_key_reader, args=(stop_event, view_holder, wake_event), daemon=True
    )
    data_thread.start()
    key_thread.start()

    try:
        with Live(console=console, screen=True, refresh_per_second=2) as live:
            while not stop_event.is_set():
                # Block (zero CPU) until a key is pressed OR new data arrives.
                # timeout is a safety net in case both threads are quiet.
                wake_event.wait(timeout=interval + 2)
                wake_event.clear()

                if data_holder[0] is None:
                    live.update(Panel(
                        "[dim]Loading data…[/]",
                        title="[bold white]Isaac Lab Training Monitor[/]",
                        border_style="bright_blue",
                    ))
                    continue

                try:
                    df, groups, progress = data_holder[0]

                    # shutil.get_terminal_size() issues TIOCGWINSZ directly —
                    # the most reliable cross-terminal, cross-SSH, cross-Warp
                    # method.  Falls back to (120, 40) if the ioctl fails.
                    ts = shutil.get_terminal_size((120, 40))
                    w = max(60, ts.columns)
                    h = max(20, ts.lines)
                    nav = _build_nav(view_holder[0])

                    if view_holder[0] == 0:
                        content = _view_overview(df, groups, progress, run_info, nav, w)
                    elif view_holder[0] == 1:
                        content = _view_reward_charts(df, groups, progress, run_info, nav, w, h)
                    else:
                        content = _view_policy(df, groups, progress, run_info, nav, w, h)

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
        wake_event.set()  # unblock data_thread if it's sleeping
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

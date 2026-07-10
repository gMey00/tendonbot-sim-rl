"""CLI entry point for the Isaac Lab Training Monitor.

Usage:
    # Terminal dashboard (default) — auto-detect latest active run
    python -m tools.monitor

    # Terminal dashboard with specific run
    python -m tools.monitor --run 2026-03-30_16-26-16_ppo_torch

    # Include completed/failed runs in the picker
    python -m tools.monitor --all

    # Set refresh interval (seconds)
    python -m tools.monitor --interval 5

    # Launch Streamlit web dashboard instead
    python -m tools.monitor --web

    # Web dashboard with custom options
    python -m tools.monitor --web --interval 20 --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure workspace root is on path
_WORKSPACE = Path(__file__).resolve().parent.parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isaac Lab Training Monitor — live dashboard for SKRL training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Launch the Streamlit web dashboard instead of the terminal UI.",
    )
    parser.add_argument(
        "--run", "-r",
        type=str,
        default=None,
        help="Specific run directory (name or path) to monitor. "
             "If omitted, auto-detects the latest active run or shows a picker.",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=10.0,
        help="Refresh interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Include completed and failed runs (not just 'running').",
    )
    parser.add_argument(
        "--pick", "-p",
        action="store_true",
        help="Show interactive run picker instead of auto-selecting the latest run.",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default=None,
        help="Override the log root directory (default: auto-detect logs/skrl/).",
    )
    args = parser.parse_args()

    if args.web:
        _launch_web(args)
    else:
        _launch_terminal(args)


def _launch_web(args: argparse.Namespace) -> None:
    """Launch the Streamlit web dashboard as a subprocess."""
    from tools.monitor.core import find_log_roots

    web_script = str(Path(__file__).parent / "web.py")

    # Resolve log roots so the web process gets the right paths
    if args.log_root:
        log_root_arg = args.log_root
    else:
        roots = find_log_roots(_WORKSPACE)
        # Pass all roots as comma-separated string
        log_root_arg = ",".join(str(r) for r in roots) if roots else ""

    cmd = [
        sys.executable, "-m", "streamlit", "run", web_script,
        "--server.headless", "true",
        "--server.address", "127.0.0.1",  # localhost only — do NOT use 0.0.0.0
        "--browser.gatherUsageStats", "false",
        "--",  # Separator for streamlit args vs our args
    ]
    if args.run:
        cmd.extend(["--run", args.run])
    if args.all:
        cmd.append("--all")
    if log_root_arg:
        cmd.extend(["--log-root", log_root_arg])
    cmd.extend(["--interval", str(int(args.interval))])

    print(f"Launching Streamlit dashboard...")
    print(f"  Open http://localhost:8501 in your browser")
    print(f"  (localhost only — for remote access use: ssh -L 8501:localhost:8501 robot@<host>)")
    print(f"  Press Ctrl+C to stop\n")

    try:
        subprocess.run(cmd, cwd=str(_WORKSPACE))
    except KeyboardInterrupt:
        print("\nStopped.")


def _launch_terminal(args: argparse.Namespace) -> None:
    """Launch the Rich terminal dashboard."""
    from rich.console import Console

    from tools.monitor.core import discover_runs, find_log_roots
    from tools.monitor.terminal import pick_run, run_terminal_dashboard

    console = Console()

    # Discover log roots (may be multiple directories)
    if args.log_root:
        log_roots = [args.log_root]
    else:
        log_roots = find_log_roots(_WORKSPACE)

    if not log_roots:
        console.print("[red]No log directories found.[/]")
        sys.exit(1)

    for root in log_roots:
        console.print(f"[dim]Scanning: {root}[/]")

    # Discover runs across all roots
    runs = discover_runs(log_roots, include_all=args.all)

    if not runs:
        if args.all:
            console.print("[red]No training runs found in:[/] " + ", ".join(str(r) for r in log_roots))
        else:
            console.print(
                "[yellow]No active (running) training runs found.[/]\n"
                "  Use [bold]--all[/] to include completed/failed runs."
            )
        sys.exit(1)

    # Select a run
    if args.run:
        # Match by name or path substring
        matched = [r for r in runs if args.run in r.run_dir or args.run in r.short_name]
        if not matched:
            console.print(f"[red]No run matching '{args.run}' found.[/]")
            sys.exit(1)
        run_info = matched[0]
    elif args.pick:
        run_info = pick_run(runs, console)
        if run_info is None:
            sys.exit(0)
    else:
        # Auto-select the most recently modified active run
        run_info = runs[0]
        console.print(
            f"[green]Auto-selected most recent run:[/] [bold]{run_info.short_name}[/] "
            f"([cyan]{run_info.task}[/])\n"
            f"[dim]  {run_info.run_dir}[/]\n"
            f"[dim]  Use --pick to choose from {len(runs)} active run(s), or --run NAME to specify one.[/]"
        )

    # Launch dashboard — pass the full run list + roots so the user can switch
    # between running jobs inside the TUI and newly started runs appear live.
    run_terminal_dashboard(
        run_info,
        interval=args.interval,
        runs=runs,
        log_roots=log_roots,
        include_all=args.all,
    )


if __name__ == "__main__":
    main()

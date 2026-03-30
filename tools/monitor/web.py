"""Streamlit web dashboard with Plotly interactive charts.

Launch with:
    streamlit run tools/monitor/web.py -- [--run DIR] [--all] [--log-root PATH] [--interval N]
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit must be imported at top-level
import streamlit as st

# Allow imports from workspace root
_WORKSPACE = Path(__file__).resolve().parent.parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from tools.monitor.core import (
    MetricGroup,
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
# Streamlit config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Isaac Lab Training Monitor",
    page_icon="🤖",
    layout="wide",
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI args passed after `--` in `streamlit run ... -- --args`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--log-root", type=str, default=None)
    parser.add_argument("--interval", type=int, default=15)
    # streamlit passes extra args; ignore them
    args, _ = parser.parse_known_args()
    return args


args = _parse_args()


# ---------------------------------------------------------------------------
# Run discovery & selection
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def _discover(log_roots_csv: str, include_all: bool):
    roots = [r for r in log_roots_csv.split(",") if r]
    return discover_runs(roots, include_all=include_all)


def _load_data(run_dir: str):
    """Load scalars (not cached — always fresh)."""
    return load_scalars(run_dir)


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(size=12),
)


def _reward_chart(df: pd.DataFrame) -> go.Figure | None:
    fig = go.Figure()
    plotted = False
    for tag_suffix, color, name in [
        ("Total reward (min)", "rgba(255,80,80,0.15)", "Min"),
        ("Total reward (max)", "rgba(80,200,255,0.15)", "Max"),
        ("Total reward (mean)", "#00CC96", "Mean"),
    ]:
        full_tag = f"Reward / {tag_suffix}"
        steps, values = get_series(df, full_tag, ema_alpha=0.85)
        if len(steps) > 0:
            fill = "tonexty" if plotted and "min" not in name.lower() and "mean" not in name.lower() else None
            fig.add_trace(go.Scatter(
                x=steps, y=values, name=name, mode="lines",
                line=dict(color=color, width=2 if name == "Mean" else 1),
                fill=fill,
            ))
            plotted = True

    if not plotted:
        return None
    fig.update_layout(title="Total Reward", xaxis_title="Step", yaxis_title="Reward", **_PLOTLY_LAYOUT)
    return fig


def _reward_decomposition_chart(df: pd.DataFrame, groups: dict[str, MetricGroup]) -> go.Figure | None:
    tags = groups.get("rewards", MetricGroup("rewards")).tags
    if not tags:
        return None

    values = get_latest_values(df, tags)
    sorted_tags = sorted(values.keys(), key=lambda t: abs(values[t][0]), reverse=True)[:10]

    fig = go.Figure()
    for tag in sorted_tags:
        steps, vals = get_series(df, tag, ema_alpha=0.85)
        if len(steps) > 0:
            fig.add_trace(go.Scatter(
                x=steps, y=vals, name=tag_display_name(tag), mode="lines",
            ))

    fig.update_layout(
        title="Reward Decomposition",
        xaxis_title="Step",
        yaxis_title="Value",
        **_PLOTLY_LAYOUT,
    )
    return fig


def _policy_chart(df: pd.DataFrame, groups: dict[str, MetricGroup]) -> go.Figure | None:
    policy_tags = []
    for key in ("policy", "loss", "learning"):
        g = groups.get(key)
        if g:
            policy_tags.extend(g.tags)

    if not policy_tags:
        return None

    n = len(policy_tags)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=[tag_display_name(t) for t in policy_tags],
        vertical_spacing=0.05,
    )

    for i, tag in enumerate(policy_tags):
        steps, vals = get_series(df, tag, ema_alpha=0.8)
        if len(steps) > 0:
            fig.add_trace(
                go.Scatter(x=steps, y=vals, name=tag_display_name(tag), mode="lines"),
                row=i + 1, col=1,
            )

    fig.update_layout(
        title="Policy Diagnostics",
        height=250 * n,
        showlegend=False,
        **_PLOTLY_LAYOUT,
    )
    return fig


def _episode_chart(df: pd.DataFrame, groups: dict[str, MetricGroup]) -> go.Figure | None:
    tags = groups.get("episode", MetricGroup("episode")).tags
    custom_tags = groups.get("custom_metrics", MetricGroup("custom_metrics")).tags
    all_tags = tags + custom_tags
    if not all_tags:
        return None

    fig = go.Figure()
    for tag in all_tags:
        steps, vals = get_series(df, tag, ema_alpha=0.8)
        if len(steps) > 0:
            fig.add_trace(go.Scatter(
                x=steps, y=vals, name=tag_display_name(tag), mode="lines",
            ))

    fig.update_layout(
        title="Episode & Custom Metrics",
        xaxis_title="Step",
        **_PLOTLY_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🤖 Isaac Lab Training Monitor")

    if args.log_root:
        log_roots_csv = args.log_root  # Already comma-separated from __main__
    else:
        roots = find_log_roots(_WORKSPACE)
        log_roots_csv = ",".join(str(r) for r in roots) if roots else ""

    runs = _discover(log_roots_csv, args.all)

    if not runs:
        st.warning("No training runs found. Use `--all` to include completed runs.")
        st.stop()

    # Sidebar: run selector
    st.sidebar.header("Run Selection")
    run_labels = [f"{r.task} — {r.short_name} [{r.status}]" for r in runs]

    # Pre-select specific run if --run was given
    default_idx = 0
    if args.run:
        for i, r in enumerate(runs):
            if args.run in r.run_dir or args.run in r.short_name:
                default_idx = i
                break

    selected_idx = st.sidebar.selectbox(
        "Training Run",
        range(len(runs)),
        format_func=lambda i: run_labels[i],
        index=default_idx,
    )
    run_info = runs[selected_idx]

    refresh_interval = st.sidebar.slider("Refresh interval (s)", 5, 120, args.interval)

    # Auto-refresh
    st.sidebar.markdown(f"*Next refresh in {refresh_interval}s*")
    # Use st.empty and rerun for auto-refresh
    import time as _time
    placeholder = st.empty()

    # Load data
    df = _load_data(run_info.run_dir)
    groups = categorize_tags(df)
    progress = compute_progress(df, run_info)

    # Header metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Task", run_info.task)
    col2.metric("Steps", f"{progress.current_step:,} / {progress.target_steps:,}")
    col3.metric("Progress", f"{progress.pct:.1f}%")
    col4.metric("Speed", f"{progress.steps_per_sec:,.0f} st/s" if progress.steps_per_sec > 0 else "N/A")
    col5.metric("Elapsed", format_duration(progress.elapsed_secs))
    col6.metric("ETA", format_duration(progress.eta_secs))

    # Progress bar
    st.progress(min(progress.pct / 100, 1.0))

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = _reward_chart(df)
        if fig:
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No total reward data yet")

    with chart_col2:
        fig = _reward_decomposition_chart(df, groups)
        if fig:
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No reward decomposition data yet")

    # Tables side by side
    tab_col1, tab_col2, tab_col3 = st.columns(3)

    with tab_col1:
        st.subheader("Reward Terms")
        reward_tags = groups.get("rewards", MetricGroup("rewards")).tags
        if reward_tags:
            vals = get_latest_values(df, reward_tags)
            rows = []
            for tag in sorted(vals.keys(), key=lambda t: abs(vals[t][0]), reverse=True):
                v, prev = vals[tag]
                trend = "▲" if prev is not None and v > prev else "▼" if prev is not None and v < prev else "─"
                rows.append({"Reward": tag_display_name(tag), "Value": f"{v:.4f}", "Δ": trend})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.info("No reward data")

    with tab_col2:
        st.subheader("Learning & Policy")
        learn_tags = []
        for key in ("total_reward", "policy", "loss", "learning", "episode"):
            g = groups.get(key)
            if g:
                learn_tags.extend(g.tags)
        if learn_tags:
            vals = get_latest_values(df, learn_tags)
            rows = []
            for tag in learn_tags:
                if tag in vals:
                    v, prev = vals[tag]
                    trend = "▲" if prev is not None and v > prev else "▼" if prev is not None and v < prev else "─"
                    rows.append({"Metric": tag_display_name(tag), "Value": f"{v:.4f}", "Δ": trend})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.info("No learning data")

    with tab_col3:
        st.subheader("Custom Metrics")
        custom_tags = groups.get("custom_metrics", MetricGroup("custom_metrics")).tags
        if custom_tags:
            vals = get_latest_values(df, custom_tags)
            rows = []
            for tag in custom_tags:
                if tag in vals:
                    v, prev = vals[tag]
                    trend = "▲" if prev is not None and v > prev else "▼" if prev is not None and v < prev else "─"
                    rows.append({"Metric": tag_display_name(tag), "Value": f"{v:.4f}", "Δ": trend})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.info("No custom metrics")

    # Policy diagnostics chart
    st.subheader("Policy Diagnostics")
    fig = _policy_chart(df, groups)
    if fig:
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No policy data yet")

    # Episode & custom metrics chart
    fig = _episode_chart(df, groups)
    if fig:
        st.plotly_chart(fig, width='stretch')

    # Footer
    st.caption(f"Last refresh: {datetime.datetime.now().strftime('%H:%M:%S')}  •  "
               f"Run: {run_info.short_name}  •  Algorithm: {run_info.algorithm.upper()}")

    # Auto-refresh via rerun
    _time.sleep(refresh_interval)
    st.rerun()


if __name__ == "__main__":
    main()
else:
    # When imported by streamlit
    main()

# Isaac Lab Training Monitor

Live training dashboard for SKRL training runs. Automatically discovers all
training runs in the workspace, reads TensorBoard event files, and displays
all reward terms, learning metrics, and training progress — without any
pre-configuration.

Works with any task (Reach, Cube Place, Cube Sort, …) and any robot variant.

---

## Requirements

Install once into the `env_isaaclab` conda environment:

```bash
conda run -n env_isaaclab pip install rich plotext streamlit
```

---

## Usage

All commands are run from the **workspace root**
(`/home/robot/studentische-arbeiten`).

> **Important:** Always use `conda run --no-capture-output` (not plain
> `conda run`). Without `--no-capture-output`, conda buffers all output
> internally and you see nothing until the process exits.
>
> **Quickest option:** use the provided wrapper script:
> ```bash
> bash tools/monitor/monitor.sh          # auto-select latest run
> bash tools/monitor/monitor.sh --pick   # interactive run picker
> bash tools/monitor/monitor.sh -i 5     # faster 5s refresh
> ```

### Terminal Dashboard (recommended — works over SSH)

```bash
# Auto-detect and monitor the most recent active training run
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor

# Show interactive picker to choose from all active runs
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor --pick

# Monitor a specific run by directory name
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor --run 2026-03-30_20-15-05_ppo_torch

# Include completed/failed runs (use with --pick or --run)
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor --all --pick

# Faster refresh (every 5 seconds instead of 10)
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor -i 5
```

Alternatively, activate the environment first (then `--no-capture-output` is not needed):
```bash
conda activate env_isaaclab
python -m tools.monitor
python -m tools.monitor --pick
python -m tools.monitor -i 5
```

### Web Dashboard (browser, interactive Plotly charts)

> **Security:** The web dashboard binds to `127.0.0.1` (localhost) only.
> It is **not reachable from the network** by any direct URL.
> The only supported way to access it from another machine is SSH port
> forwarding (see below).

```bash
# Launch Streamlit web dashboard — then open http://localhost:8501
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor --web

# Web dashboard including completed runs, 20s refresh
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor --web --all -i 20

# Web dashboard for a specific run
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor --web --run 2026-03-30_20-15-05_ppo_torch
```

#### Accessing from another machine (SSH port forwarding)

Run this **on your local machine** before opening the browser:

```bash
ssh -L 8501:localhost:8501 robot@pve-robotik-humble
```

Then open `http://localhost:8501` in your local browser.
The connection is tunnelled through SSH — encrypted and authenticated.
No other remote-access method is supported or should be attempted.

### Manual log root override

If logs are in a non-standard location:

```bash
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor \
    --log-root /path/to/custom/logs/skrl

# Web mode with custom log root
conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor --web \
    --log-root /path/to/custom/logs/skrl
```

---

## All flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--web` | `-w` | off | Launch Streamlit web dashboard |
| `--run NAME` | `-r` | auto | Specific run directory name or path substring |
| `--interval N` | `-i` | `10` | Refresh interval in seconds |
| `--all` | `-a` | off | Include completed/failed runs (not just running) |
| `--pick` | `-p` | off | Show interactive picker instead of auto-selecting latest run |
| `--log-root PATH` | | auto | Override log scan root directory |

---

## What is displayed

### Terminal dashboard (Rich)

| Panel | Content |
|-------|---------|
| **Header** | Task name, algorithm, run directory, status |
| **Progress bar** | Current step / target steps, %, speed (steps/s), elapsed time, ETA |
| **Reward Terms** | All `Episode_Reward/*` tags — latest value + trend arrow (▲/▼/─) |
| **Learning & Policy** | Total reward (mean/min/max), policy std, losses, learning rate, episode length |
| **Custom Metrics** | `Info/Metrics/*` tags — grasp rate, success rate, etc. |
| **Total Reward plot** | plotext line chart of mean ± min/max over training steps |
| **Reward Decomposition** | plotext multi-line chart of top reward terms over steps |
| **Policy Diagnostics** | plotext chart of policy loss, value loss, learning rate |

### Web dashboard (Streamlit + Plotly)

- Sidebar run selector (all discovered runs with status indicator)
- Metric cards: steps, progress, speed, elapsed, ETA
- Interactive Plotly charts: zoom, pan, hover for exact values
- Reward decomposition with toggleable legend
- Policy diagnostics subplot grid
- Auto-refresh every N seconds

---

## Architecture

```
tools/monitor/
    __init__.py         Package marker
    __main__.py         CLI entry point
    core.py             Data layer: run discovery, TensorBoard parsing, ETA
    terminal.py         Rich Live terminal dashboard with plotext charts
    web.py              Streamlit web dashboard with Plotly charts
    monitor.sh          Convenience wrapper (uses --no-capture-output)
```

The monitor scans **both** of these log directories automatically:
- `logs/skrl/` (workspace root)
- `src/tensegrity_pick/logs/skrl/` (package-relative, used by train scripts)

No metric names are hardcoded — all tags are auto-discovered from the
TensorBoard event file on every refresh.

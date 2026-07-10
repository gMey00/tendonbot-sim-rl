# Isaac Lab Training Monitor

Live training dashboard for SKRL training runs. Automatically discovers all
training runs in the workspace, reads TensorBoard event files, and displays
all reward terms, learning metrics, training progress, **live GPU/CPU/memory
performance**, and a **job switcher** to flip between running jobs — without any
pre-configuration.

Works with any task (Reach, Cube Place, Cube Sort, Shirt Present, …) and any
robot variant, on both the **FAPS workstation** and the **Alex cluster** (run it
on the compute node via `tools/watch_alex.sh` — see below).

---

## Requirements

Install once into the `env_isaaclab` conda environment:

```bash
conda run -n env_isaaclab pip install rich plotext psutil tbparse streamlit
```

- `rich` + `plotext` + `tbparse` — terminal dashboard and TensorBoard parsing.
- `psutil` — CPU / RAM / process metrics for the **Performance** tab. The GPU
  panel shells out to `nvidia-smi` (already present wherever a GPU is), so a
  missing `psutil` only disables the CPU/RAM section, never crashes the monitor.
- `streamlit` — optional, only for the `--web` browser dashboard.

On Alex these are installed automatically by `tools/install_IsaacLab_Alex.sh`.

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

## Keyboard navigation (terminal dashboard)

The terminal dashboard is a live full-screen TUI with **five views** and an
in-place **job switcher** — no need to restart to look at another run.

| Key | Action |
|-----|--------|
| `1` … `5` | Switch view: **1** Overview · **2** Reward Charts · **3** Policy · **4** Performance · **5** Runs |
| `[` / `]` (or `,` / `.`) | Switch the active job to the **previous / next** discovered run — works from any view |
| `←` / `→` | Same as `[` / `]` (quick job cycling) |
| `↑` / `↓` | Move the highlight in the **Runs** view |
| `Enter` | Jump to the highlighted run (Runs view) |
| `q` / `Ctrl+C` | Quit |

The active job is shown in the nav bar; the **Runs** tab lists every discovered
run (task, status, age) with the active row marked `▶` and scrolls to keep the
selection visible when there are many runs.

## What is displayed

### Terminal dashboard (Rich)

| View / Panel | Content |
|-------|---------|
| **1 · Overview** | Header (task, algorithm, run dir, status), progress bar, and the tables below |
| **Reward Terms** | All `Episode_Reward/*` tags — latest value + trend arrow (▲/▼/─) |
| **Learning & Policy** | Total reward (mean/min/max), policy std, losses, learning rate, episode length |
| **Custom Metrics** | `Metrics/*` tags — grasp rate, success rate, etc. |
| **2 · Reward Charts** | plotext line chart of total reward (mean ± min/max) + reward decomposition of top terms |
| **3 · Policy** | plotext chart of policy loss, value loss, learning rate |
| **4 · Performance** | Live **GPU** (util %, memory, temp, power, util sparkline per GPU), **CPU/RAM** (usage bars + sparklines, load avg, swap), and detected **training processes** (PID, CPU %, RSS) |
| **5 · Runs** | Job switcher: all discovered runs with status + age, active/highlighted markers |

> **Reward/metric tags:** both the old `Info / Episode_Reward/*` /
> `Info / Metrics/*` naming (skrl ≤ 2.0) **and** the current
> `Episode_Reward/*` / `Metrics/*` naming are recognised, so historical and
> new runs both decompose correctly. (The dropped `Info / ` prefix was why
> per-term rewards stopped showing.)

### Performance tab on Alex

`nvidia-smi` and `psutil` report the resources of **whatever host the monitor
runs on**. On Alex the login node has no GPU, so launch the monitor **on the
compute node** through the documented overlap attach:

```bash
./tools/watch_alex.sh              # auto-pick your running job → monitor (train mode)
./tools/watch_alex.sh 1234567_3    # a specific array task / seed
```

There the Performance tab shows that job's GPU, the node's CPU/RAM, and the
training process. (`SLURM_JOB_ID`, when set, is shown in the CPU/Memory panel.)

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
    sysmetrics.py       GPU (nvidia-smi) + CPU/RAM (psutil) sampling for Performance tab
    terminal.py         Rich Live terminal dashboard: 5 views + job switcher + plotext charts
    web.py              Streamlit web dashboard with Plotly charts
    monitor.sh          Convenience wrapper (uses --no-capture-output)
```

The monitor scans **both** of these log directories automatically:
- `logs/skrl/` (workspace root)
- `src/tensegrity_pick/logs/skrl/` (package-relative, used by train scripts)

No metric names are hardcoded — all tags are auto-discovered from the
TensorBoard event file on every refresh.

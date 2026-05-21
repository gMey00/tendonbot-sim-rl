// Appendix A.6 — Training Monitor
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-figure

// ═══════════════════════════════════════════════════════════════════
// Training Monitor Tool
// ═══════════════════════════════════════════════════════════════════

= Appendix: Training Monitor <appendix:training_monitor>

To observe training progress in real time, a custom monitoring tool was developed as part of this work. #ac("RL") experiments in Isaac Lab using the SKRL library produce TensorBoard event files containing all logged scalar metrics. While TensorBoard itself provides a web-based visualization, it requires a graphical browser and does not present task-specific summaries such as training speed, estimated time of arrival, or reward decomposition in a consolidated view.

The monitor is implemented as a Python package (`tools.monitor`) with four modules. The *data layer* (`core.py`) handles run discovery and TensorBoard parsing. It recursively scans the log directories for `run_status.json` files that the training scripts write at startup, and uses the `tbparse` library to read scalar events into a Pandas DataFrame. All metric tag names are auto-discovered on every refresh cycle. Tags are categorized by their prefix (e.g.\ `Episode_Reward/`, `Loss/`, `Metrics/`) so the dashboard can group them semantically. The module also computes training speed in steps per second and estimates the remaining time from wall-clock timestamps.

The *terminal frontend* (`terminal.py`) builds a Rich Live dashboard rendered directly in the terminal. It uses `plotext` to draw inline line charts of reward curves and policy diagnostics using Unicode block characters. Three switchable views are provided — *Overview*, *Reward Charts*, and *Policy* — navigateable with the number keys 1–3. A background thread reads raw keypresses via `termios` to allow view switching without interrupting the Rich render loop. @fig:monitor_tui_overview shows the overview with tabular reward terms, learning metrics, and custom task metrics. @fig:monitor_tui_rewards shows the reward chart view with total reward (mean/max/min) and the top reward term decomposition. @fig:monitor_tui_policy shows the policy diagnostics view with standard deviation, losses, and learning rate.

The *web frontend* (`web.py`) provides an alternative browser-based dashboard using Streamlit with interactive Plotly charts. It offers the same information but with zoom, pan, and hover capabilities. For security, the web server binds only to localhost. Access from other machines is only supported via SSH port forwarding. @fig:monitor_webapp shows the web dashboard.

The monitor auto-refreshes at a configurable interval (default 10~seconds) and supports command-line flags for run selection, inclusion of completed runs, and custom log root directories. A convenience shell script wraps the invocation with the required `conda run --no-capture-output` flags to prevent output buffering.

#faps-figure(
  image("../../../assets/figures/monitor/monitor_tui_overview.png", width: 90%),
  caption: [Terminal monitor overview: header with task name, progress bar, and ETA. Tables of all reward terms with trend indicators, learning and policy statistics, and custom task metrics (grasp rate, success rate).],
  short-caption: [Terminal monitor — overview],
) <fig:monitor_tui_overview>

#faps-figure(
  image("../../../assets/figures/monitor/monitor_tui_rewards.png", width: 90%),
  caption: [Terminal monitor reward charts: total reward with mean, max, and min envelopes (top) and decomposition of the top individual reward terms over training steps (bottom). Charts are rendered inline using plotext Unicode block characters.],
  short-caption: [Terminal monitor — reward charts],
) <fig:monitor_tui_rewards>

#faps-figure(
  image("../../../assets/figures/monitor/monitor_tui_policy.png", width: 90%),
  caption: [Terminal monitor policy diagnostics: policy standard deviation, entropy loss, policy loss, value loss, and learning rate over training steps.],
  short-caption: [Terminal monitor — policy diagnostics],
) <fig:monitor_tui_policy>

#faps-figure(
  image("../../../assets/figures/monitor/monitor_webapp.png", width: 90%),
  caption: [Streamlit web dashboard with interactive Plotly charts. The sidebar provides run selection and refresh interval controls. Metric cards show steps, progress, speed, elapsed time, and ETA. The main area displays total reward with a filled min/max envelope and reward decomposition with a toggleable legend.],
  short-caption: [Web-based training monitor dashboard],
) <fig:monitor_webapp>

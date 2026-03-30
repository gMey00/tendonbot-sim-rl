#!/usr/bin/env bash
# Wrapper to launch the Isaac Lab Training Monitor.
# Uses --no-capture-output so Rich's live terminal output is streamed
# directly to the terminal instead of being buffered by conda run.
#
# Usage (from workspace root):
#   bash tools/monitor/monitor.sh
#   bash tools/monitor/monitor.sh --pick
#   bash tools/monitor/monitor.sh --run 2026-03-30_20-15-05_ppo_torch
#   bash tools/monitor/monitor.sh --web
#   bash tools/monitor/monitor.sh -i 5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$WORKSPACE"

exec conda run --no-capture-output -n env_isaaclab python -u -m tools.monitor "$@"

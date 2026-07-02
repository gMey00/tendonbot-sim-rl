#!/usr/bin/env bash
# ============================================================================
#  sync_checkpoints.sh  —  pull trained reach checkpoints Alex -> workstation
# ============================================================================
#  Run this ON THE WORKSTATION to copy the trained reach logs/checkpoints from
#  the Alex cluster into the local repo, so render_reach_workstation.sh can find
#  them.  The reach logs are small (~68 MB total) and are gitignored, so they
#  are not carried by a normal git pull.
#
#  USAGE
#    ./sync_checkpoints.sh <alex_ssh_host> [remote_repo_path]
#  EXAMPLE
#    ./sync_checkpoints.sh iwfa131h@alex.nhr.fau.de
#    ./sync_checkpoints.sh myuser@alex.nhr.fau.de /home/hpc/iwfa/iwfa131h/studentische-arbeiten
#
#  Only the reach subtree is transferred:
#    <remote>/src/tensegrity_pick/logs/skrl/reach/  ->  <local>/logs/skrl/reach/
# ============================================================================
set -euo pipefail

REMOTE_HOST="${1:-}"
REMOTE_REPO="${2:-/home/hpc/iwfa/iwfa131h/studentische-arbeiten}"
if [[ -z "${REMOTE_HOST}" ]]; then
    echo "Usage: $0 <alex_ssh_host> [remote_repo_path]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # scripts/rendering -> tensegrity_pick
DEST="${PKG_DIR}/logs/skrl/reach/"
SRC="${REMOTE_HOST}:${REMOTE_REPO}/src/tensegrity_pick/logs/skrl/reach/"

mkdir -p "${DEST}"
echo "[sync] ${SRC}"
echo "[sync]   -> ${DEST}"
rsync -avh --progress "${SRC}" "${DEST}"
echo "[sync] done. Variants now local:"
ls -1 "${DEST}" 2>/dev/null

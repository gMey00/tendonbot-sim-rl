#!/bin/bash -l
# ============================================================================
#  build_container.sh  —  build the Isaac Sim render container on an Alex login
# ============================================================================
#  Builds isaac_render.sif from isaac_render.def. Run on a *login* node (they
#  have internet + fakeroot); do NOT run inside a GPU job. The .sif lands on
#  $HPCVAULT (large quota) — never in $HOME (small, snapshot-doubled).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../../../.config/env_vars.sh" 2>/dev/null || \
    source "$(cd "${SCRIPT_DIR}/../../../.." && pwd)/../.config/env_vars.sh"

DEF="${SCRIPT_DIR}/isaac_render.def"
OUT_DIR="${INSTALL_PATH}/containers"
SIF="${OUT_DIR}/isaac_render.sif"

# Keep apptainer's build cache + tmp on $HPCVAULT, not $HOME.
export APPTAINER_CACHEDIR="${INSTALL_PATH}/.cache/apptainer"
export APPTAINER_TMPDIR="${INSTALL_PATH}/.cache/apptainer_tmp"
mkdir -p "${OUT_DIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "[build] def : ${DEF}"
echo "[build] out : ${SIF}"
echo "[build] cache: ${APPTAINER_CACHEDIR}"
apptainer build --fakeroot --force "${SIF}" "${DEF}"
echo "[build] done -> ${SIF}"
ls -lh "${SIF}"

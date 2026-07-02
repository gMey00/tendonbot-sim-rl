#!/usr/bin/env bash
# ============================================================================
#  render_reach_workstation.sh  —  render the reach grid locally (no Slurm)
# ============================================================================
#  For rendering the trained reach policies on a workstation with a GPU whose
#  driver Isaac Sim 5.1 supports (e.g. the FAPS workstation).  The Alex cluster
#  cannot render: its driver (610.43) crashes Isaac Sim 5.1's RTX renderer (see
#  doc / container/ notes), so videos are produced off-cluster instead.
#
#  This is a simple SEQUENTIAL loop over the reach variants that calls
#  render_play.py directly (one Isaac Sim process per variant).  The trained
#  checkpoints must be present locally under
#     src/tensegrity_pick/logs/skrl/reach/<variant>/<run>/checkpoints/
#  (copy them from Alex first — see sync_checkpoints.sh or the README).
#
#  PREREQUISITE: activate the Isaac Lab env first, e.g.
#     source <repo>/.config/env_vars.sh && conda activate "$ISAACLAB_ENV_NAME"
#  (or set PYTHON=/path/to/isaaclab/python to point at the right interpreter).
#
#  USAGE
#    ./render_reach_workstation.sh [opts]
#  OPTIONS (defaults shown)
#    --arms    "A B ..."   subset of the 6 arms            (default: all 6)
#    --spaces  "x y ..."   subset of {joint ik ikabs osc}  (default: all 4)
#    --num-envs N          robot copies                    (default: 4)
#    --env-spacing M       spacing between copies (m)      (default: 3.0)
#    --video-length N      steps @30fps (1800 = 60s)       (default: 1800)
#    --cam-eye "X Y Z"     camera world position           (default: 6.5 -4.5 3.5)
#    --cam-lookat "X Y Z"  camera world look-at            (default: 0.75 1.0 1.0)
#    --resolution "W H"    video resolution                (default: 1280 720)
#    --list                print the tasks, don't render
#    -h, --help            show this header
#
#  Videos land in each variant's own log dir:
#     logs/skrl/reach/<variant>/<run>/videos/play/rl-video-step-0.mp4
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/rendering -> tensegrity_pick
PKG_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

ALL_ARMS=(Kinova-F140 Kinova-Frankenstein UR10-F140 UR10-Frankenstein UR5e-F140 UR5e-Frankenstein)
ALL_SPACES=(joint ik ikabs osc)
space_suffix() { case "$1" in joint) echo "";; ik) echo "-IK-Rel";; ikabs) echo "-IK-Abs";; osc) echo "-OSC";; *) echo "__INVALID__";; esac; }

ARMS=("${ALL_ARMS[@]}"); SPACES=("${ALL_SPACES[@]}")
NUM_ENVS=4; ENV_SPACING=3.0; VIDEO_LENGTH=1800
CAM_EYE="6.5 -4.5 3.5"; CAM_LOOKAT="0.75 1.0 1.0"; RESOLUTION="1280 720"
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)         read -ra ARMS   <<< "$2"; shift 2 ;;
        --spaces)       read -ra SPACES <<< "$2"; shift 2 ;;
        --num-envs)     NUM_ENVS="$2"; shift 2 ;;
        --env-spacing)  ENV_SPACING="$2"; shift 2 ;;
        --video-length) VIDEO_LENGTH="$2"; shift 2 ;;
        --cam-eye)      CAM_EYE="$2"; shift 2 ;;
        --cam-lookat)   CAM_LOOKAT="$2"; shift 2 ;;
        --resolution)   RESOLUTION="$2"; shift 2 ;;
        --list)         LIST_ONLY=1; shift ;;
        -h|--help)      sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Error: unknown option '$1'. See --help." >&2; exit 1 ;;
    esac
done

valid_member() { local x="$1"; shift; local m; for m in "$@"; do [[ "$m" == "$x" ]] && return 0; done; return 1; }
for a in "${ARMS[@]}"; do valid_member "$a" "${ALL_ARMS[@]}" || { echo "Error: unknown arm '$a'." >&2; exit 1; }; done
for s in "${SPACES[@]}"; do [[ "$(space_suffix "$s")" == "__INVALID__" ]] && { echo "Error: unknown space '$s'." >&2; exit 1; }; done

TASKS=()
for arm in "${ARMS[@]}"; do for sp in "${SPACES[@]}"; do TASKS+=("Template-Reach-${arm}$(space_suffix "$sp")-Play-v0"); done; done

echo "Workstation reach-render: ${#ARMS[@]} arm(s) × ${#SPACES[@]} space(s) = ${#TASKS[@]} variant(s)"
printf '  %s\n' "${TASKS[@]}"
[[ "${LIST_ONLY}" -eq 1 ]] && exit 0

read -ra CAM_EYE_ARR    <<< "${CAM_EYE}"
read -ra CAM_LOOKAT_ARR <<< "${CAM_LOOKAT}"
read -ra RESOLUTION_ARR <<< "${RESOLUTION}"

cd "${PKG_DIR}"
echo "[render] cwd=${PKG_DIR}  python=$(command -v "${PYTHON}" || echo "${PYTHON}")"
fail=0
for task in "${TASKS[@]}"; do
    echo "============================================================"
    echo ">> ${task}   ($(date '+%H:%M:%S'))"
    echo "============================================================"
    if "${PYTHON}" scripts/rendering/render_play.py \
        --task "${task}" \
        --headless --video --video_length "${VIDEO_LENGTH}" \
        --num_envs "${NUM_ENVS}" --env_spacing "${ENV_SPACING}" \
        --cam_eye "${CAM_EYE_ARR[@]}" --cam_lookat "${CAM_LOOKAT_ARR[@]}" \
        --cam_resolution "${RESOLUTION_ARR[@]}"; then
        echo "[ok] ${task}"
    else
        echo "[FAIL] ${task} (continuing)"; fail=$((fail+1))
    fi
done
echo "============================================================"
echo "Done. ${fail} variant(s) failed of ${#TASKS[@]}."
echo "Videos -> logs/skrl/reach/<variant>/<run>/videos/play/rl-video-step-0.mp4"
find logs/skrl/reach -path '*/videos/play/*.mp4' -newermt '-1 day' -printf '  %p\n' 2>/dev/null | sort

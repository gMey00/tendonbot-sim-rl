#!/usr/bin/env bash
# ============================================================================
#  render_reach_alex.sh  —  render play-videos for the full reach grid on Alex
# ============================================================================
#  Sibling of tools/train_reach_alex.sh: instead of *training* the 24 reach
#  variants it submits one SLURM job per variant that loads the trained
#  checkpoint and renders a ~1-min play video with a framed camera
#  (scripts/rendering/render_play.py).  It reuses the proven tools/train_alex.sh
#  sbatch wrapper via its --script option, so the SLURM/conda/proxy boilerplate
#  stays in exactly one place.
#
#  GRID (default: all of it = 6 arms × 4 action spaces = 24 jobs)
#    arms    : Kinova-F140 Kinova-Frankenstein
#              UR10-F140   UR10-Frankenstein
#              UR5e-F140   UR5e-Frankenstein
#    spaces  : joint  -> Template-Reach-<ARM>-Play-v0
#              ik     -> Template-Reach-<ARM>-IK-Rel-Play-v0
#              ikabs  -> Template-Reach-<ARM>-IK-Abs-Play-v0
#              osc    -> Template-Reach-<ARM>-OSC-Play-v0
#  Each job writes its video under the variant's own log dir:
#    logs/skrl/reach/<variant>/<run>/videos/play/rl-video-step-0.mp4
#
#  USAGE
#  -----
#    ./render_reach_alex.sh [wrapper opts] [-- <train_alex.sh opts ...>]
#
#  WRAPPER OPTIONS
#    --arms    "A B ..."   subset of the arms above           (default: all 6)
#    --spaces  "x y ..."   subset of {joint ik ikabs osc}     (default: all 4)
#    --num-envs N          robot copies to render             (default: 4)
#    --env-spacing M       spacing (m) between copies         (default: 3.0)
#    --video-length N      video length in steps (30fps)      (default: 1800 = 60s)
#    --cam-eye "X Y Z"     camera world position              (default: 6.5 -4.5 3.5)
#    --cam-lookat "X Y Z"  camera world look-at               (default: 0.75 1.0 1.0)
#    --resolution "W H"    video resolution                   (default: 1280 720)
#    --time HH:MM:SS       wall time per job                  (default: 00:30:00)
#    --list                print the task IDs that would be submitted, then exit
#    -h, --help            show this header
#
#  Everything after `--` is forwarded VERBATIM to every tools/train_alex.sh call
#  (placed before the task, where train_alex.sh expects its own options), e.g.
#  for --dry-run / -m mail / --stage:
#
#  EXAMPLES
#  --------
#    # render all 24 reach play-videos
#    ./render_reach_alex.sh
#
#    # only the two Kinova arms, only OSC, preview the sbatch scripts
#    ./render_reach_alex.sh --arms "Kinova-F140 Kinova-Frankenstein" \
#        --spaces "osc" -- --dry-run
#
#    # just list the play task IDs the grid expands to
#    ./render_reach_alex.sh --list
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tools/train_alex.sh lives at <repo>/tools (this script is at
# <repo>/src/tensegrity_pick/scripts/rendering).
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAIN_ALEX="${REPO_ROOT}/tools/train_alex.sh"
# Path passed to train_alex.sh --script, relative to src/tensegrity_pick
# (train_alex.sh cd's into ${PROJECT_PATH}/src/tensegrity_pick before running).
RENDER_SCRIPT="scripts/rendering/render_play.py"

ALL_ARMS=(Kinova-F140 Kinova-Frankenstein UR10-F140 UR10-Frankenstein UR5e-F140 UR5e-Frankenstein)
ALL_SPACES=(joint ik ikabs osc)

# space key -> play task-id suffix (mirrors the trained-task suffixes + -Play)
space_suffix() {
    case "$1" in
        joint) echo "" ;;
        ik)    echo "-IK-Rel" ;;
        ikabs) echo "-IK-Abs" ;;
        osc)   echo "-OSC" ;;
        *)     echo "__INVALID__" ;;
    esac
}

# space key -> log-dir/job-name suffix (mirrors logs/skrl/reach/<variant>)
space_logkey() {
    case "$1" in
        joint) echo "" ;;
        ik)    echo "_ik" ;;
        ikabs) echo "_ikabs" ;;
        osc)   echo "_osc" ;;
        *)     echo "__INVALID__" ;;
    esac
}

# ------------------------------------------------------------ arg parsing ----
ARMS=("${ALL_ARMS[@]}")
SPACES=("${ALL_SPACES[@]}")
NUM_ENVS=4
ENV_SPACING=3.0
VIDEO_LENGTH=1800
CAM_EYE="6.5 -4.5 3.5"
CAM_LOOKAT="0.75 1.0 1.0"
RESOLUTION="1280 720"
TIME="00:30:00"
LIST_ONLY=0
FORWARD=()   # forwarded verbatim to each train_alex.sh call

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
        --time)         TIME="$2"; shift 2 ;;
        --list)         LIST_ONLY=1; shift ;;
        -h|--help)      sed -n '2,70p' "${BASH_SOURCE[0]}"; exit 0 ;;
        --)             shift; FORWARD+=("$@"); break ;;
        *)              echo "Error: unknown wrapper option '$1' (did you mean to put it after '--'?)" >&2
                        echo "       See --help." >&2; exit 1 ;;
    esac
done

# ------------------------------------------------------------- validation ----
valid_member() { local x="$1"; shift; local m; for m in "$@"; do [[ "$m" == "$x" ]] && return 0; done; return 1; }

for a in "${ARMS[@]}"; do
    valid_member "$a" "${ALL_ARMS[@]}" || { echo "Error: unknown arm '$a'. Valid: ${ALL_ARMS[*]}" >&2; exit 1; }
done
for s in "${SPACES[@]}"; do
    [[ "$(space_suffix "$s")" == "__INVALID__" ]] && { echo "Error: unknown space '$s'. Valid: ${ALL_SPACES[*]}" >&2; exit 1; }
done

# --------------------------------------------------------------- expand grid -
# arm "Kinova-F140" -> log key "kinova_f140"
arm_logkey() { echo "$1" | tr 'A-Z-' 'a-z_'; }

TASKS=()
JOBNAMES=()
for arm in "${ARMS[@]}"; do
    for sp in "${SPACES[@]}"; do
        TASKS+=("Template-Reach-${arm}$(space_suffix "$sp")-Play-v0")
        JOBNAMES+=("render_$(arm_logkey "$arm")$(space_logkey "$sp")")
    done
done

echo "Reach render grid: ${#ARMS[@]} arm(s) × ${#SPACES[@]} space(s) = ${#TASKS[@]} job(s)"
for i in "${!TASKS[@]}"; do printf '  %-44s -> %s\n' "${TASKS[$i]}" "${JOBNAMES[$i]}"; done
if [[ "${LIST_ONLY}" -eq 1 ]]; then
    exit 0
fi
[[ -f "${TRAIN_ALEX}" ]] || { echo "Error: ${TRAIN_ALEX} not found." >&2; exit 1; }

# Per-task extra args appended AFTER the task (forwarded to render_play.py).
read -ra CAM_EYE_ARR    <<< "${CAM_EYE}"
read -ra CAM_LOOKAT_ARR <<< "${CAM_LOOKAT}"
read -ra RESOLUTION_ARR <<< "${RESOLUTION}"
RENDER_EXTRA=(
    --headless --video
    --video_length "${VIDEO_LENGTH}"
    --num_envs "${NUM_ENVS}"
    --env_spacing "${ENV_SPACING}"
    --cam_eye "${CAM_EYE_ARR[@]}"
    --cam_lookat "${CAM_LOOKAT_ARR[@]}"
    --cam_resolution "${RESOLUTION_ARR[@]}"
)

# ------------------------------------------------------------------ submit ----
echo "------------------------------------------------------------"
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    jobname="${JOBNAMES[$i]}"
    # train_alex.sh option order: [wrapper opts] <TASK> [forwarded train.py args].
    # --script + -t + -j are train_alex.sh wrapper opts (before the task);
    # RENDER_EXTRA are render_play.py args (after the task).
    cmd=(bash "${TRAIN_ALEX}" --script "${RENDER_SCRIPT}" -t "${TIME}" -j "${jobname}")
    [[ ${#FORWARD[@]} -gt 0 ]] && cmd+=("${FORWARD[@]}")
    cmd+=("${task}")
    cmd+=("${RENDER_EXTRA[@]}")
    echo ">> ${cmd[*]}"
    "${cmd[@]}"
    echo "------------------------------------------------------------"
done
echo "Submitted ${#TASKS[@]} render job(s). Track with:  squeue --me"
echo "Videos will land under: logs/skrl/reach/<variant>/<run>/videos/play/"

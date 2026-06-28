#!/usr/bin/env bash
# ============================================================================
#  train_reach_alex.sh  —  submit the full reach action-space comparison
# ============================================================================
#  Thin loop on top of ./train_alex.sh that submits one Slurm job per
#  (robot arm × action space) reach variant, so the whole comparison study can
#  be launched with a single command.
#
#  GRID (default: all of it = 6 arms × 4 action spaces = 24 jobs)
#    arms    : Kinova-F140 Kinova-Frankenstein
#              UR10-F140   UR10-Frankenstein
#              UR5e-F140   UR5e-Frankenstein
#    spaces  : joint  (EMA joint-position)      -> Template-Reach-<ARM>-v0
#              ik     (Differential-IK relative)-> Template-Reach-<ARM>-IK-Rel-v0
#              ikabs  (Differential-IK absolute)-> Template-Reach-<ARM>-IK-Abs-v0
#              osc    (Operational-Space Ctrl)  -> Template-Reach-<ARM>-OSC-v0
#  Each task already logs to its own dir (logs/skrl/reach/<variant>[_ik|_ikabs|_osc]/).
#
#  USAGE
#  -----
#    ./train_reach_alex.sh [wrapper opts] [-- <train_alex.sh opts ...>]
#
#  WRAPPER OPTIONS
#    --arms   "A B ..."   subset of the arms above           (default: all 6)
#    --spaces "x y ..."   subset of {joint ik ikabs osc}     (default: all 4)
#    --list               print the task IDs that would be submitted, then exit
#    --headless 0|1       pass --headless to train.py         (default: 1)
#    -h, --help           show this header
#
#  Everything after `--` is forwarded VERBATIM to every ./train_alex.sh call
#  (i.e. placed before the task, where train_alex.sh expects its own options).
#  Use it for seeds / wall-time / mail / staging / dry-run, e.g.:
#
#  EXAMPLES
#  --------
#    # submit all 24 reach jobs (headless), default wall-time
#    ./train_reach_alex.sh
#
#    # only the two Kinova arms, only joint-space + OSC, 5-seed study, 12 h, email
#    ./train_reach_alex.sh --arms "Kinova-F140 Kinova-Frankenstein" \
#        --spaces "joint osc" -- -s "0 1 2 3 4" -t 12:00:00 -m me@fau.de --stage
#
#    # preview the generated sbatch scripts without submitting
#    ./train_reach_alex.sh --spaces "osc" -- --dry-run
#
#    # just list the task IDs the grid expands to
#    ./train_reach_alex.sh --list
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ALEX="${SCRIPT_DIR}/train_alex.sh"

ALL_ARMS=(Kinova-F140 Kinova-Frankenstein UR10-F140 UR10-Frankenstein UR5e-F140 UR5e-Frankenstein)
ALL_SPACES=(joint ik ikabs osc)

# space key -> task-id suffix
space_suffix() {
    case "$1" in
        joint) echo "" ;;
        ik)    echo "-IK-Rel" ;;
        ikabs) echo "-IK-Abs" ;;
        osc)   echo "-OSC" ;;
        *)     echo "__INVALID__" ;;
    esac
}

# ------------------------------------------------------------ arg parsing ----
ARMS=("${ALL_ARMS[@]}")
SPACES=("${ALL_SPACES[@]}")
LIST_ONLY=0
HEADLESS=1
FORWARD=()   # forwarded verbatim to each train_alex.sh call

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)     read -ra ARMS   <<< "$2"; shift 2 ;;
        --spaces)   read -ra SPACES <<< "$2"; shift 2 ;;
        --list)     LIST_ONLY=1; shift ;;
        --headless) HEADLESS="$2"; shift 2 ;;
        -h|--help)  sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
        --)         shift; FORWARD+=("$@"); break ;;
        *)          echo "Error: unknown wrapper option '$1' (did you mean to put it after '--'?)" >&2
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
TASKS=()
for arm in "${ARMS[@]}"; do
    for sp in "${SPACES[@]}"; do
        TASKS+=("Template-Reach-${arm}$(space_suffix "$sp")-v0")
    done
done

echo "Reach comparison grid: ${#ARMS[@]} arm(s) × ${#SPACES[@]} space(s) = ${#TASKS[@]} job(s)"
printf '  %s\n' "${TASKS[@]}"
if [[ "${LIST_ONLY}" -eq 1 ]]; then
    exit 0
fi
[[ -f "${TRAIN_ALEX}" ]] || { echo "Error: ${TRAIN_ALEX} not found." >&2; exit 1; }

# Per-task extra args appended AFTER the task (forwarded to train.py).
TRAINPY_EXTRA=()
[[ "${HEADLESS}" -eq 1 ]] && TRAINPY_EXTRA+=(--headless)

# ------------------------------------------------------------------ submit ----
echo "------------------------------------------------------------"
for task in "${TASKS[@]}"; do
    # Build the argv explicitly so empty arrays don't inject a stray "" that
    # train_alex.sh would mistake for the (empty) task positional.
    # Invoke via `bash` so submission does not depend on train_alex.sh's +x bit.
    cmd=(bash "${TRAIN_ALEX}")
    [[ ${#FORWARD[@]}       -gt 0 ]] && cmd+=("${FORWARD[@]}")
    cmd+=("${task}")
    [[ ${#TRAINPY_EXTRA[@]} -gt 0 ]] && cmd+=("${TRAINPY_EXTRA[@]}")
    echo ">> ${cmd[*]}"
    "${cmd[@]}"
    echo "------------------------------------------------------------"
done
echo "Submitted ${#TASKS[@]} reach job(s). Track with:  squeue --me"

#!/usr/bin/env bash
# ============================================================================
#  render_reach_container.sh  —  render the reach grid inside the Apptainer
#                                container on a Vulkan-capable partition (a40)
# ============================================================================
#  The rtxpro6k (Blackwell) nodes have a COMPUTE-ONLY NVIDIA driver — no
#  OpenGL/Vulkan userspace — so Isaac Sim's RTX renderer cannot run there
#  (bare-metal or containerised).  The a40 nodes DO have the full graphics
#  driver, and the trained reach policies are hardware-portable, so we render
#  there instead: inside isaac_render.sif (Ubuntu 22.04 / glibc 2.35 + Vulkan
#  loader), with the host NVIDIA driver supplied via `apptainer --nv` and a
#  relative-path Vulkan ICD (see nvidia_icd_relative.json).
#
#  Submits one a40 job per reach variant (default: all 24). Each renders a
#  ~1-min video via scripts/rendering/render_play.py into the variant's own
#  log dir:  logs/skrl/reach/<variant>/<run>/videos/play/rl-video-step-0.mp4
#
#  USAGE
#    ./render_reach_container.sh [opts]
#  OPTIONS (defaults shown)
#    --arms    "A B ..."   subset of the 6 arms            (default: all 6)
#    --spaces  "x y ..."   subset of {joint ik ikabs osc}  (default: all 4)
#    --partition NAME      Slurm partition (must have GPU graphics driver + Vulkan)
#                                                          (default: a40)
#    --gres    STRING      gres request                    (default: gpu:a40:1)
#    --num-envs N          robot copies                    (default: 4)
#    --env-spacing M       spacing between copies (m)      (default: 3.0)
#    --video-length N      steps @30fps (1800 = 60s)       (default: 1800)
#    --cam-eye "X Y Z"     camera world position           (default: 6.5 -4.5 3.5)
#    --cam-lookat "X Y Z"  camera world look-at            (default: 0.75 1.0 1.0)
#    --resolution "W H"    video resolution                (default: 1280 720)
#    --time HH:MM:SS       wall time per job               (default: 00:25:00)
#    --list                print the tasks, don't submit
#    --dry-run             print the generated sbatch for the first task, don't submit
#    -h, --help            show this header
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "${SCRIPT_DIR}/../../../../.." && pwd)/.config/env_vars.sh"

SIF="${INSTALL_PATH}/containers/isaac_render.sif"
ICD="${SCRIPT_DIR}/nvidia_icd_relative.json"
ENV_PREFIX="${INSTALL_PATH}/conda_envs/${ISAACLAB_ENV_NAME}"
RENDER_ENTRY="${SCRIPT_DIR}/render_in_container.sh"

ALL_ARMS=(Kinova-F140 Kinova-Frankenstein UR10-F140 UR10-Frankenstein UR5e-F140 UR5e-Frankenstein)
ALL_SPACES=(joint ik ikabs osc)
space_suffix() { case "$1" in joint) echo "";; ik) echo "-IK-Rel";; ikabs) echo "-IK-Abs";; osc) echo "-OSC";; *) echo "__INVALID__";; esac; }
space_logkey() { case "$1" in joint) echo "";; ik) echo "_ik";; ikabs) echo "_ikabs";; osc) echo "_osc";; *) echo "__INVALID__";; esac; }
arm_logkey()   { echo "$1" | tr 'A-Z-' 'a-z_'; }

# --------------------------------------------------------------- defaults ---
ARMS=("${ALL_ARMS[@]}"); SPACES=("${ALL_SPACES[@]}")
PARTITION="a40"; GRES="gpu:a40:1"
NUM_ENVS=4; ENV_SPACING=3.0; VIDEO_LENGTH=1800
CAM_EYE="6.5 -4.5 3.5"; CAM_LOOKAT="0.75 1.0 1.0"; RESOLUTION="1280 720"
TIME="00:25:00"; LIST_ONLY=0; DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)         read -ra ARMS   <<< "$2"; shift 2 ;;
        --spaces)       read -ra SPACES <<< "$2"; shift 2 ;;
        --partition)    PARTITION="$2"; shift 2 ;;
        --gres)         GRES="$2"; shift 2 ;;
        --num-envs)     NUM_ENVS="$2"; shift 2 ;;
        --env-spacing)  ENV_SPACING="$2"; shift 2 ;;
        --video-length) VIDEO_LENGTH="$2"; shift 2 ;;
        --cam-eye)      CAM_EYE="$2"; shift 2 ;;
        --cam-lookat)   CAM_LOOKAT="$2"; shift 2 ;;
        --resolution)   RESOLUTION="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --list)         LIST_ONLY=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Error: unknown option '$1'. See --help." >&2; exit 1 ;;
    esac
done

# ------------------------------------------------------------- validation ----
valid_member() { local x="$1"; shift; local m; for m in "$@"; do [[ "$m" == "$x" ]] && return 0; done; return 1; }
for a in "${ARMS[@]}"; do valid_member "$a" "${ALL_ARMS[@]}" || { echo "Error: unknown arm '$a'." >&2; exit 1; }; done
for s in "${SPACES[@]}"; do [[ "$(space_suffix "$s")" == "__INVALID__" ]] && { echo "Error: unknown space '$s'." >&2; exit 1; }; done
[[ -f "${SIF}" ]] || { echo "Error: container ${SIF} not found. Build it with container/build_container.sh" >&2; exit 1; }

# --------------------------------------------------------------- expand grid -
TASKS=(); JOBNAMES=()
for arm in "${ARMS[@]}"; do
    for sp in "${SPACES[@]}"; do
        TASKS+=("Template-Reach-${arm}$(space_suffix "$sp")-Play-v0")
        JOBNAMES+=("render_$(arm_logkey "$arm")$(space_logkey "$sp")")
    done
done

echo "Container reach-render grid: ${#ARMS[@]} arm(s) × ${#SPACES[@]} space(s) = ${#TASKS[@]} job(s) on '${PARTITION}'"
for i in "${!TASKS[@]}"; do printf '  %-46s -> %s\n' "${TASKS[$i]}" "${JOBNAMES[$i]}"; done
[[ "${LIST_ONLY}" -eq 1 ]] && exit 0

mkdir -p "${SLURM_LOG_DIR}"
read -ra CAM_EYE_ARR    <<< "${CAM_EYE}"
read -ra CAM_LOOKAT_ARR <<< "${CAM_LOOKAT}"
read -ra RESOLUTION_ARR <<< "${RESOLUTION}"

# ------------------------------------------------------------------ submit ----
submit_one() {
    local task="$1" jobname="$2"
    local job_script
    job_script="$(mktemp "${SLURM_LOG_DIR}/${jobname}_XXXX.sbatch")"
    cat > "${job_script}" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${jobname}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=${TIME}
#SBATCH --output=${SLURM_LOG_DIR}/${jobname}_%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
source "${PROJECT_PATH}/.config/env_vars.sh"

echo "=== ${jobname} on \$(hostname) | \$(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

apptainer exec --nv \\
  --bind "${ICD}:/etc/vulkan/icd.d/nvidia_icd_relative.json" \\
  --bind "${HPCVAULT}" \\
  --bind "${HOME}" \\
  --bind "\${TMPDIR}" \\
  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_relative.json \\
  --env ISAAC_ENV_PREFIX="${ENV_PREFIX}" \\
  --env PROJECT_PATH="${PROJECT_PATH}" \\
  "${SIF}" bash "${RENDER_ENTRY}" \\
    --task ${task} \\
    --headless --video --video_length ${VIDEO_LENGTH} \\
    --num_envs ${NUM_ENVS} --env_spacing ${ENV_SPACING} \\
    --cam_eye ${CAM_EYE_ARR[@]} --cam_lookat ${CAM_LOOKAT_ARR[@]} \\
    --cam_resolution ${RESOLUTION_ARR[@]}

echo "=== done \$(date) ==="
find "${PROJECT_PATH}/src/tensegrity_pick/logs/skrl/reach" -path '*/videos/play/*.mp4' -newermt '-30 minutes' 2>/dev/null | grep -F "\$(echo ${task} | sed 's/Template-Reach-//;s/-Play-v0//')" || true
EOF
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "----- generated sbatch (${job_script}) -----"; cat "${job_script}"; echo "----- (dry run) -----"; return 0
    fi
    sbatch "${job_script}"
}

echo "------------------------------------------------------------"
for i in "${!TASKS[@]}"; do
    submit_one "${TASKS[$i]}" "${JOBNAMES[$i]}"
    [[ "${DRY_RUN}" -eq 1 ]] && break
    echo "------------------------------------------------------------"
done
[[ "${DRY_RUN}" -eq 0 ]] && echo "Submitted ${#TASKS[@]} render job(s) on '${PARTITION}'. Track: squeue --me"
[[ "${DRY_RUN}" -eq 0 ]] && echo "Videos -> logs/skrl/reach/<variant>/<run>/videos/play/"

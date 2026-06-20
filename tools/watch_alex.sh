#!/bin/bash -l
# ============================================================================
#  watch_alex.sh  —  attach monitoring tools to a running Alex job
# ============================================================================
#  Attaches to the compute node of a running job (without consuming a second
#  GPU allocation) using the NHR@FAU documented mechanism:
#      srun --jobid=<jobID> --overlap --pty /bin/bash -l
#  (https://doc.nhr.fau.de/batch-processing/batch_system_slurm/#attach-to-a-running-job)
#
#  USAGE
#  -----
#    ./watch_alex.sh [jobID] [mode]
#
#  jobID : Slurm job ID. If omitted and you have exactly one running job,
#          it is selected automatically. For array jobs pass e.g. 12345_2.
#  mode  : what to attach (default: train)
#            train   -> the project TensorBoard monitor (monitor.sh)
#            btop    -> btop system + GPU resource monitor
#            gpu     -> watch nvidia-smi (lightweight, no btop needed)
#            shell   -> plain interactive shell on the node
#
#  EXAMPLES
#  --------
#    ./watch_alex.sh                 # auto-pick job, show training metrics
#    ./watch_alex.sh 1234567 btop    # performance monitor on that job's node
#    ./watch_alex.sh 1234567_3 gpu   # GPU utilisation for array task seed 3
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.config/env_vars.sh"

JOBID="${1:-}"
MODE="${2:-train}"

# Allow "./watch_alex.sh btop" (mode only) by detecting a non-numeric arg 1.
if [[ -n "${JOBID}" && ! "${JOBID}" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
    MODE="${JOBID}"
    JOBID=""
fi

# ----------------------------------------------------- resolve the job ID ----
if [[ -z "${JOBID}" ]]; then
    mapfile -t RUNNING < <(squeue --me --states=RUNNING --noheader --format='%i')
    if [[ "${#RUNNING[@]}" -eq 0 ]]; then
        echo "No running jobs found. Current queue:"
        squeue --me
        exit 1
    elif [[ "${#RUNNING[@]}" -eq 1 ]]; then
        JOBID="${RUNNING[0]}"
        echo "Auto-selected running job: ${JOBID}"
    else
        echo "Multiple running jobs — specify which one:"
        squeue --me --states=RUNNING
        exit 1
    fi
fi

ATTACH=(srun --jobid="${JOBID}" --overlap --pty)

echo "Attaching to job ${JOBID} (mode: ${MODE}). Ctrl-C / 'q' to detach."
echo "------------------------------------------------------------"

case "${MODE}" in
    train)
        # Rich TUI training monitor (python -m tools.monitor).
        "${ATTACH[@]}" /bin/bash -l -c "
            source '${PROJECT_PATH}/.config/env_vars.sh'
            module load gcc/15.2.0 2>/dev/null || true
            if module load python 2>/dev/null && command -v conda >/dev/null 2>&1; then
                eval \"\$(conda shell.bash hook)\"
                export CONDA_PKGS_DIRS='${INSTALL_PATH}/.cache/conda_pkgs'
            else
                source '${CONDA_PATH}/etc/profile.d/conda.sh'
            fi
            conda activate '${ISAACLAB_ENV_NAME}'
            cd '${PROJECT_PATH}'
            python -m tools.monitor
        "
        ;;
    btop)
        # btop ships in the conda env (installed by install_IsaacLab.sh).
        # Expose libnvidia-ml.so.1 so btop can enable GPU monitoring at runtime.
        "${ATTACH[@]}" /bin/bash -l -c "
            source '${PROJECT_PATH}/.config/env_vars.sh'
            module load gcc/15.2.0 2>/dev/null || true
            if module load python 2>/dev/null && command -v conda >/dev/null 2>&1; then
                eval \"\$(conda shell.bash hook)\"
                export CONDA_PKGS_DIRS='${INSTALL_PATH}/.cache/conda_pkgs'
            else
                source '${CONDA_PATH}/etc/profile.d/conda.sh'
            fi
            conda activate '${ISAACLAB_ENV_NAME}'
            export LD_LIBRARY_PATH=/usr/lib64:\${LD_LIBRARY_PATH:-}
            command -v btop >/dev/null 2>&1 && exec btop
            echo 'btop not found in env; falling back to top'; exec top
        "
        ;;
    gpu)
        "${ATTACH[@]}" /bin/bash -l -c \
            "watch -n 2 nvidia-smi"
        ;;
    shell)
        "${ATTACH[@]}" /bin/bash -l
        ;;
    *)
        echo "Unknown mode '${MODE}'. Use: train | btop | gpu | shell"
        exit 1
        ;;
esac
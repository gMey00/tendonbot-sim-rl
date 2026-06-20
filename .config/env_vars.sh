# ============================================================================
#  env_vars.sh  —  Multi-machine environment for tensegrity_pick
# ============================================================================
#  Source this once per shell *before* running any setup / training helper:
#
#      source .config/env_vars.sh
#
#  Machine profile is auto-detected from the hostname.
#  To force a specific profile, set MACHINE before sourcing:
#
#      MACHINE=alex source .config/env_vars.sh
#
#  Available profiles: faps_workstation, alex
#  To add a new machine: copy one of the case blocks below, adjust the
#  hostname pattern in the detection section, and set the paths accordingly.
# ============================================================================


# ============================================================================
#  Machine detection  (override by setting MACHINE in your environment)
# ============================================================================
if [[ -z "${MACHINE:-}" ]]; then
    _hn="$(hostname -s 2>/dev/null || echo unknown)"
    # $HPCVAULT is exported by Alex login nodes; use it as a reliable signal.
    if [[ -n "${HPCVAULT:-}" || "${_hn}" == *alex* || "${_hn}" == login* ]]; then
        MACHINE="alex"
    elif [[ "${_hn}" == faps* ]]; then
        MACHINE="faps_workstation"
    else
        # Unknown host — default to workstation-style layout.
        MACHINE="faps_workstation"
    fi
    unset _hn
fi
echo "env_vars.sh: loading profile '${MACHINE}'"


# ============================================================================
#  Per-machine configuration
# ============================================================================
case "${MACHINE}" in

# ----------------------------------------------------------------------------
#  FAPS Workstation
# ----------------------------------------------------------------------------
faps_workstation)
    export HPC="/home/robot"
    export VAULT="/home/robot"
    export WORKDIR="${HOME}/work"

    export INSTALL_PATH="/home/robot/Isaac"
    export CONDA_PATH="/home/robot/miniconda3"
    ;;

# ----------------------------------------------------------------------------
#  Alex cluster (NHR@FAU)
#
#  Filesystem layout (https://doc.nhr.fau.de/data/filesystems/):
#    $HOME      small, frequently backed up   -> code lives here ($HPC)
#    $HPCVAULT  large, mid/long-term storage  -> Isaac Sim + Lab + conda ($VAULT)
#    $WORK      general scratch / logs        -> training outputs
#    $TMPDIR    node-local NVMe (per job)      -> staged assets at runtime
# ----------------------------------------------------------------------------
alex)
    # Provide fallbacks so the file can also be sourced on a workstation.
    : "${HPCVAULT:=${HOME}/hpcvault}"
    : "${WORK:=${HOME}/work}"

    export HPC="${HOME}"
    export VAULT="${HPCVAULT}"
    export WORKDIR="${WORK}"

    export INSTALL_PATH="${VAULT}/Isaac"
    # Conda lives inside INSTALL_PATH to keep it off the small $HOME quota.
    export CONDA_PATH="${INSTALL_PATH}/miniconda3"

    # --- Slurm defaults ---
    # GPU resource string for the RTX PRO 6000 partition.
    export ALEX_GRES="${ALEX_GRES:-gpu:rtxpro6k:1}"
    # Host cores allocated per GPU on Alex (DefCpuPerGPU=32 → must use 32 to get 1 task).
    export ALEX_CPUS_PER_GPU="${ALEX_CPUS_PER_GPU:-32}"
    # Default wall time (Alex hard limit is 24:00:00).
    export ALEX_TIME="${ALEX_TIME:-24:00:00}"
    # Where Slurm job logs land (created on demand).
    export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${WORKDIR}/slurm_logs}"
    ;;

# ----------------------------------------------------------------------------
#  Add new machines here following the pattern above.
# ----------------------------------------------------------------------------

*)
    echo "env_vars.sh: unknown MACHINE='${MACHINE}'. Valid profiles: faps_workstation, alex" >&2
    ;;
esac


# ============================================================================
#  Common derived paths  (computed after per-machine block sets the roots)
# ============================================================================
# Repository name — override before sourcing if you cloned to a different path.
export REPO_NAME="${REPO_NAME:-studentische-arbeiten}"
export PROJECT_PATH="${HPC}/${REPO_NAME}"

export ISAACSIM_PATH="${INSTALL_PATH}/IsaacSim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export ISAACLAB_PATH="${INSTALL_PATH}/IsaacLab"
export ISAACLAB_ENV_NAME="${ISAACLAB_ENV_NAME:-env_isaaclab}"
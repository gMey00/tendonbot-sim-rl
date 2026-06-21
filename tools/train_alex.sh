#!/bin/bash -l
# ============================================================================
#  train_alex.sh  —  sbatch wrapper for tensegrity_pick training on Alex
# ============================================================================
#  Submits any training script as a batch job on the RTX PRO 6000 partition,
#  following the NHR@FAU Alex job-script conventions
#  (https://doc.nhr.fau.de/clusters/alex/#batch-processing):
#    * #!/bin/bash -l                  -> initializes the module system
#    * --export=NONE + unset SLURM_EXPORT_ENV  -> clean, reproducible env
#    * --gres=gpu:rtxpro6k:1           -> one RTX PRO 6000 (NHR-provided gres)
#    * module load python + conda activate
#
#  USAGE
#  -----
#    ./train_alex.sh <TASK> [extra train.py args...]
#
#  COMMON OPTIONS (parsed by this wrapper, must come *before* the task)
#    -s, --seeds   "0 1 2 3 4"   submit one array task per seed (5-seed study)
#    -t, --time    HH:MM:SS      wall time (default $ALEX_TIME, max 24:00:00)
#    -n, --num-envs N            override --num_envs passed to train.py
#    -i, --iters   N             --max_iterations passed to train.py
#    -j, --job-name NAME         Slurm job name (default derived from task)
#    -m, --mail    you@fau.de    email for BEGIN/END/FAIL notifications
#        --algorithm  PPO|IPPO|MAPPO   RL algorithm (default PPO)
#        --script   PATH         training script (default scripts/skrl/train.py)
#        --stage                 copy res/ assets to node-local $TMPDIR first
#        --dry-run               print the generated job script, do not submit
#
#  EXAMPLES
#  --------
#    # single run
#    ./train_alex.sh Template-Tensegrity-Reach-v0 --headless --num_envs 4096
#
#    # 5-seed robustness study as a Slurm array (supervisor requirement)
#    ./train_alex.sh -s "0 1 2 3 4" Template-Tensegrity-Reach-v0 --headless
#
#    # MAPPO multi-agent cloth run, 12 h, with email
#    ./train_alex.sh -t 12:00:00 --algorithm MAPPO -m me@fau.de \
#        Template-Tensegrity-Shirt-Sort-v0 --headless
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.config/env_vars.sh"

# ---------------------------------------------------------------- defaults ---
SEEDS=""
TIME="${ALEX_TIME}"
NUM_ENVS=""
ITERS=""
JOB_NAME=""
MAIL=""
ALGO="PPO"
TRAIN_SCRIPT="scripts/skrl/train.py"
STAGE=0
DRY_RUN=0

# ------------------------------------------------------------ arg parsing ----
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--seeds)     SEEDS="$2"; shift 2 ;;
        -t|--time)      TIME="$2"; shift 2 ;;
        -n|--num-envs)  NUM_ENVS="$2"; shift 2 ;;
        -i|--iters)     ITERS="$2"; shift 2 ;;
        -j|--job-name)  JOB_NAME="$2"; shift 2 ;;
        -m|--mail)      MAIL="$2"; shift 2 ;;
        --algorithm)    ALGO="$2"; shift 2 ;;
        --script)       TRAIN_SCRIPT="$2"; shift 2 ;;
        --stage)        STAGE=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      sed -n '2,48p' "${BASH_SOURCE[0]}"; exit 0 ;;
        --)             shift; POSITIONAL+=("$@"); break ;;
        *)              POSITIONAL+=("$1"); shift ;;
    esac
done
set -- "${POSITIONAL[@]:-}"

TASK="${1:-}"
if [[ -z "${TASK}" ]]; then
    echo "Error: no task given. See --help." >&2
    exit 1
fi
shift || true
EXTRA_ARGS="$*"     # everything else forwarded verbatim to train.py

# Build the forwarded train.py argument string.
TRAIN_ARGS="--task=${TASK} --algorithm=${ALGO}"
[[ -n "${NUM_ENVS}" ]] && TRAIN_ARGS+=" --num_envs=${NUM_ENVS}"
[[ -n "${ITERS}"    ]] && TRAIN_ARGS+=" --max_iterations=${ITERS}"
[[ -n "${EXTRA_ARGS}" ]] && TRAIN_ARGS+=" ${EXTRA_ARGS}"

# Job name + log dir.
[[ -z "${JOB_NAME}" ]] && JOB_NAME="$(echo "${TASK}" | sed 's/^Template-Tensegrity-//; s/-v0$//')"
mkdir -p "${SLURM_LOG_DIR}"

# Array directive for multi-seed studies.
ARRAY_DIRECTIVE=""
SEED_LINE='SEED="${SEED:-}"'
if [[ -n "${SEEDS}" ]]; then
    read -ra SEED_ARR <<< "${SEEDS}"
    ARRAY_DIRECTIVE="#SBATCH --array=$(IFS=,; echo "${SEED_ARR[*]}")"
    # In an array job, SLURM_ARRAY_TASK_ID *is* the seed.
    SEED_LINE='SEED="${SLURM_ARRAY_TASK_ID}"'
    LOG_PATTERN="${SLURM_LOG_DIR}/${JOB_NAME}_%A_seed%a.out"
else
    LOG_PATTERN="${SLURM_LOG_DIR}/${JOB_NAME}_%j.out"
fi

MAIL_DIRECTIVES=""
if [[ -n "${MAIL}" ]]; then
    MAIL_DIRECTIVES=$'#SBATCH --mail-type=BEGIN,END,FAIL\n'"#SBATCH --mail-user=${MAIL}"
fi

# Optional staging of simulation assets to the fast node-local NVMe.
STAGE_BLOCK=""
if [[ "${STAGE}" -eq 1 ]]; then
    STAGE_BLOCK=$(cat <<'STAGE'
# --- stage assets to node-local NVMe ($TMPDIR) for faster I/O -------------
echo "[stage] copying res/ to $TMPDIR"
mkdir -p "$TMPDIR/res"
cp -r "${PROJECT_PATH}/res/." "$TMPDIR/res/"
export TENSEGRITY_ASSET_ROOT="$TMPDIR/res"
STAGE
)
fi

# ---------------------------------------------------- generate job script ----
JOB_SCRIPT="$(mktemp "${SLURM_LOG_DIR}/${JOB_NAME}_XXXX.sbatch")"
cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=$(echo "${ALEX_GRES}" | cut -d: -f2)
#SBATCH --gres=${ALEX_GRES}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${ALEX_CPUS_PER_GPU}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_PATTERN}
#SBATCH --export=NONE
${ARRAY_DIRECTIVE}
${MAIL_DIRECTIVES}

# Clean, reproducible environment (NHR@FAU recommended boilerplate).
unset SLURM_EXPORT_ENV

# Project + Isaac environment.
source "${PROJECT_PATH}/.config/env_vars.sh"
# Isaac Sim 5.x native libs require GLIBCXX_3.4.30+ — load GCC 15 before activating conda.
module load gcc/15.2.0 2>/dev/null || true
# Activate conda: prefer cluster module, fall back to private Miniconda on \$VAULT.
# Note: 'module load python' overwrites CONDA_PKGS_DIRS; re-export our writable path after.
if module load python 2>/dev/null && command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)"
    export CONDA_PKGS_DIRS="${INSTALL_PATH}/.cache/conda_pkgs"
else
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${ISAACLAB_ENV_NAME}"

# NHR@FAU: compute nodes have no direct internet access — route Isaac Sim's
# startup network calls (asset-root check, extension registry) through the
# cluster HTTP proxy so they don't hang indefinitely.
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
export no_proxy=localhost,127.0.0.1,.nhr.fau.de,.fau.de
export NO_PROXY=localhost,127.0.0.1,.nhr.fau.de,.fau.de

cd "${PROJECT_PATH}/src/tensegrity_pick"

${STAGE_BLOCK}

# Seed handling (set per array task, or empty for a single run).
${SEED_LINE}
SEED_ARG=""
[[ -n "\${SEED}" ]] && SEED_ARG="--seed=\${SEED}"

echo "============================================================"
echo " Job ${JOB_NAME}  |  \$(date)"
echo " Node: \$(hostname)   GPU: \${CUDA_VISIBLE_DEVICES:-?}"
echo " Task: ${TASK}   Seed: \${SEED:-<none>}"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun --ntasks=1 python ${TRAIN_SCRIPT} ${TRAIN_ARGS} \${SEED_ARG}

echo "[done] \$(date)"
EOF

# ------------------------------------------------------------- submit/dry ----
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "----- generated job script (${JOB_SCRIPT}) -----"
    cat "${JOB_SCRIPT}"
    echo "----- (dry run, not submitted) -----"
    exit 0
fi

echo "Submitting ${JOB_NAME} (${ALEX_GRES}, ${TIME})..."
[[ -n "${SEEDS}" ]] && echo "  seeds: ${SEEDS}"
sbatch "${JOB_SCRIPT}"
echo "Logs -> ${LOG_PATTERN}"
echo "Watch with:  squeue --me   |   ./watch_alex.sh <jobID>"
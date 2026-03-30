#!/usr/bin/env bash
# run_validation.sh — Full tensegrity model validation pipeline
#
# Runs all three step-response data-generation scripts sequentially, then
# generates all plots.  Progress and timing are printed to stdout.
#
# Usage:
#   cd /home/robot/studentische-arbeiten/src/tensegrity_pick
#   bash scripts/model_validation/run_validation.sh [--pd-only] [--tendon-only]
#                                                    [--tendon-physical-only]
#                                                    [--base-only] [--plots-only]
#                                                    [--damping-sweep]
#
# Options:
#   --pd-only        Run only the PD arm validation
#   --tendon-only    Run only the tendon (elbow_approx) arm validation
#   --tendon-physical-only  Run only the tendon (physical linkage) validation
#   --base-only      Run only the base joint validation
#   --plots-only     Skip data generation, regenerate plots from existing data
#   --damping-sweep  Include damping sweep phases in PD and base scripts

set -euo pipefail

# ── Unbuffered Python output ─────────────────────────────────────────────────
export PYTHONUNBUFFERED=1

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${PROJ_ROOT}/outputs/model_validation"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
MASTER_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"

# ── Argument parsing ──────────────────────────────────────────────────────────
RUN_PD=true
RUN_TENDON=true
RUN_TENDON_PHYSICAL=true
RUN_BASE=true
RUN_PLOTS=true
DAMPING_SWEEP=false

for arg in "$@"; do
    case "$arg" in
        --pd-only)     RUN_TENDON=false; RUN_TENDON_PHYSICAL=false; RUN_BASE=false ;;
        --tendon-only) RUN_PD=false; RUN_TENDON_PHYSICAL=false; RUN_BASE=false ;;
        --tendon-physical-only) RUN_PD=false; RUN_TENDON=false; RUN_BASE=false ;;
        --base-only)   RUN_PD=false; RUN_TENDON=false; RUN_TENDON_PHYSICAL=false ;;
        --plots-only)  RUN_PD=false; RUN_TENDON=false; RUN_TENDON_PHYSICAL=false; RUN_BASE=false ;;
        --damping-sweep) DAMPING_SWEEP=true ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--pd-only|--tendon-only|--tendon-physical-only|--base-only|--plots-only] [--damping-sweep]" >&2
            exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log_step() {
    local msg="$1"
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  ${msg}${NC}"
    echo -e "${BLUE}  $(date '+%H:%M:%S')${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
}

log_ok() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

log_err() {
    echo -e "${RED}  ✗ $1${NC}" >&2
}

run_isaac_script() {
    local label="$1"
    local script="$2"
    shift 2
    local extra_args=("$@")
    local step_log="${LOG_DIR}/${label}_${TIMESTAMP}.log"

    log_step "Step: ${label}"
    echo "  Script: scripts/model_validation/${script}"
    echo "  Log:    ${step_log}"
    echo ""

    local start_s=$SECONDS

    if timeout 1800 conda run --no-capture-output -n env_isaaclab \
            python3 -u "${SCRIPT_DIR}/${script}" \
            --headless --num_envs 1 "${extra_args[@]}" \
            2>&1 | tee "${step_log}"; then
        local elapsed=$(( SECONDS - start_s ))
        log_ok "${label} completed in ${elapsed}s"
        return 0
    else
        local rc=$?
        local elapsed=$(( SECONDS - start_s ))
        log_err "${label} FAILED (exit ${rc}) after ${elapsed}s — see ${step_log}"
        return $rc
    fi
}

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Tensegrity Model Validation — Full Pipeline               ║"
echo "║   $(date '+%Y-%m-%d %H:%M:%S')                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Project:  ${PROJ_ROOT}"
echo "  Outputs:  ${OUTPUT_ROOT}"
echo "  Log dir:  ${LOG_DIR}"
echo ""
echo "  Steps:"
$RUN_PD    && echo "    [x] PD arm step response" \
           || echo "    [ ] PD arm step response (skipped)"
$RUN_TENDON && echo "    [x] Tendon arm step response (elbow_approx)" \
            || echo "    [ ] Tendon arm step response — elbow_approx (skipped)"
$RUN_TENDON_PHYSICAL && echo "    [x] Tendon arm step response (physical)" \
                     || echo "    [ ] Tendon arm step response — physical (skipped)"
$RUN_BASE   && echo "    [x] Base joint step response" \
            || echo "    [ ] Base joint step response (skipped)"
$RUN_PLOTS  && echo "    [x] Plot generation" \
            || echo "    [ ] Plot generation (skipped)"
$DAMPING_SWEEP && echo "    [x] Damping sweep phases included" \
              || echo "    [ ] Damping sweep phases (add --damping-sweep to include)"

PIPELINE_START=$SECONDS
FAILURES=0

# ── Step 1: PD arm ───────────────────────────────────────────────────────────
if $RUN_PD; then
    SWEEP_FLAG=()
    $DAMPING_SWEEP && SWEEP_FLAG=(--damping_sweep)
    run_isaac_script "pd_arm" "run_step_response_pd.py" "${SWEEP_FLAG[@]}" || FAILURES=$(( FAILURES + 1 ))
fi

# ── Step 2: Tendon arm (elbow_approx) ────────────────────────────────────────
if $RUN_TENDON; then
    run_isaac_script "tendon_arm" "run_step_response_tendon.py" --variant elbow_approx || FAILURES=$(( FAILURES + 1 ))
fi

# ── Step 2b: Tendon arm (physical) ───────────────────────────────────────────
if $RUN_TENDON_PHYSICAL; then
    run_isaac_script "tendon_arm_physical" "run_step_response_tendon.py" --variant physical || FAILURES=$(( FAILURES + 1 ))
fi

# ── Step 3: Base joints ──────────────────────────────────────────────────────
if $RUN_BASE; then
    SWEEP_FLAG=()
    $DAMPING_SWEEP && SWEEP_FLAG=(--damping_sweep)
    run_isaac_script "base_joints" "run_step_response_base.py" "${SWEEP_FLAG[@]}" || FAILURES=$(( FAILURES + 1 ))
fi

# ── Step 4: Plots ─────────────────────────────────────────────────────────────
if $RUN_PLOTS; then
    log_step "Step: plots"
    echo "  Script: scripts/model_validation/plot_validation.py"
    echo ""
    PLOT_FLAGS=()
    ! $DAMPING_SWEEP && PLOT_FLAGS=(--no_sweep)
    plot_start=$SECONDS
    if conda run --no-capture-output -n env_isaaclab python3 -u \
            "${SCRIPT_DIR}/plot_validation.py" "${PLOT_FLAGS[@]}"; then
        log_ok "Plots completed in $(( SECONDS - plot_start ))s"
    else
        log_err "Plot generation FAILED"
        FAILURES=$(( FAILURES + 1 ))
    fi
fi

# ── Final summary ─────────────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - PIPELINE_START ))
echo ""
echo "════════════════════════════════════════════════════════════════"
if [[ $FAILURES -eq 0 ]]; then
    echo -e "${GREEN}  VALIDATION COMPLETE — all steps passed  (${ELAPSED}s total)${NC}"
    echo ""
    echo "  Plots:  ${OUTPUT_ROOT}/plots/"
    echo "  Data:   ${OUTPUT_ROOT}/{pd,tendon,tendon_physical,base}/data/"
    echo "  Logs:   ${LOG_DIR}/"
else
    echo -e "${RED}  VALIDATION FINISHED WITH ${FAILURES} FAILURE(S)  (${ELAPSED}s total)${NC}"
    echo "  Check per-step logs in: ${LOG_DIR}/"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""

exit $FAILURES

#!/usr/bin/env bash
# train_reach.sh — Train reach task variants and regenerate plots + reports.
#
# USAGE
#   cd src/tensegrity_pick
#   ./scripts/train_reach.sh                            # train all 4 variants
#   ./scripts/train_reach.sh tensegrity ur10e           # specific variants only
#   ./scripts/train_reach.sh --skip-train tensegrity    # plot only (no training)
#   ./scripts/train_reach.sh --allow-incomplete         # allow partial run plots
#
# OPTIONS
#   --skip-train         Skip training; only run plotting/report for each variant
#   --allow-incomplete   Pass --allow-incomplete to the plot script
#   --curriculum-step N  Override the curriculum marker step (default: 4500)
#
# VARIANTS
#   tensegrity                  Template-Reach-Tensegrity-v0                  (5-DOF PD)
#   tensegrity_tendon           Template-Reach-Tensegrity-Tendon-v0           (5-DOF tendon)
#   tensegrity_physical_tendon  Template-Reach-Tensegrity-Physical-Tendon-v0  (5-DOF physical tendon)
#   ur10e                       Template-Reach-UR10e-v0                       (6-DOF PD)
#   kinova                      Template-Reach-Kinova-v0                      (7-DOF PD)

set -uo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALL_VARIANTS=(tensegrity tensegrity_tendon tensegrity_physical_tendon ur10e kinova)

declare -A TASK_IDS=(
    [tensegrity]="Template-Reach-Tensegrity-v0"
    [tensegrity_tendon]="Template-Reach-Tensegrity-Tendon-v0"
    [tensegrity_physical_tendon]="Template-Reach-Tensegrity-Physical-Tendon-v0"
    [ur10e]="Template-Reach-UR10e-v0"
    [kinova]="Template-Reach-Kinova-v0"
)

CONDA_ENV="env_isaaclab"
SKIP_TRAIN=false
ALLOW_INCOMPLETE=""
CURRICULUM_STEP=""
VARIANTS=()

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-train)        SKIP_TRAIN=true ;;
        --allow-incomplete)  ALLOW_INCOMPLETE="--allow-incomplete" ;;
        --curriculum-step)   shift; CURRICULUM_STEP="--curriculum-step $1" ;;
        --curriculum-step=*) CURRICULUM_STEP="--curriculum-step ${1#*=}" ;;
        --*)
            echo "ERROR: Unknown option: $1"
            echo "       Run with --help or see script header for usage."
            exit 1
            ;;
        *)
            VARIANTS+=("$1")
            ;;
    esac
    shift
done

# Default: all variants
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    VARIANTS=("${ALL_VARIANTS[@]}")
fi

# Validate variant names
for v in "${VARIANTS[@]}"; do
    if [[ -z "${TASK_IDS[$v]+x}" ]]; then
        echo "ERROR: Unknown variant '$v'"
        echo "       Valid variants: ${ALL_VARIANTS[*]}"
        exit 1
    fi
done

# Must be run from src/tensegrity_pick/
if [[ ! -f "scripts/skrl/train.py" ]]; then
    echo "ERROR: Run this script from src/tensegrity_pick/"
    echo "       cd src/tensegrity_pick && ./scripts/train_reach.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FAILED_TRAIN=()
FAILED_PLOT=()

print_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print_header "Reach task pipeline  |  variants: ${VARIANTS[*]}"
echo "  skip-train : $SKIP_TRAIN"
echo "  conda env  : $CONDA_ENV"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    TASK="${TASK_IDS[$VARIANT]}"
    TRAIN_OK=true

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if [[ "$SKIP_TRAIN" == false ]]; then
        print_header "TRAIN  $VARIANT  ($TASK)"

        conda run --no-capture-output -n "$CONDA_ENV" \
            python3 scripts/skrl/train.py \
            --task "$TASK" \
            --headless
        TRAIN_EXIT=$?

        if [[ $TRAIN_EXIT -ne 0 ]]; then
            echo "  ✗ Training FAILED for $VARIANT (exit $TRAIN_EXIT)"
            FAILED_TRAIN+=("$VARIANT")
            TRAIN_OK=false
        else
            echo "  ✓ Training complete for $VARIANT"
        fi
    fi

    # ------------------------------------------------------------------
    # Plotting + report (runs even if training was skipped; skipped only
    # if this variant's training just failed)
    # ------------------------------------------------------------------
    if [[ "$TRAIN_OK" == true ]]; then
        print_header "PLOT   $VARIANT"

        # shellcheck disable=SC2086
        conda run --no-capture-output -n "$CONDA_ENV" \
            python3 scripts/plot_reach_training_results.py \
            --variant "$VARIANT" \
            $ALLOW_INCOMPLETE \
            $CURRICULUM_STEP
        PLOT_EXIT=$?

        if [[ $PLOT_EXIT -ne 0 ]]; then
            echo "  ✗ Plotting FAILED for $VARIANT (exit $PLOT_EXIT)"
            FAILED_PLOT+=("$VARIANT")
        else
            echo "  ✓ Figures + report generated for $VARIANT"
        fi
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_header "SUMMARY"

ALL_OK=true

if [[ ${#FAILED_TRAIN[@]} -gt 0 ]]; then
    echo "  ✗ Training failed  : ${FAILED_TRAIN[*]}"
    ALL_OK=false
fi

if [[ ${#FAILED_PLOT[@]} -gt 0 ]]; then
    echo "  ✗ Plotting failed  : ${FAILED_PLOT[*]}"
    ALL_OK=false
fi

if [[ "$ALL_OK" == true ]]; then
    echo "  ✓ All variants completed successfully"
fi

echo ""

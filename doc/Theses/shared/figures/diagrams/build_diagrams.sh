#!/usr/bin/env bash
# build_diagrams.sh — Render all .mmd files in this directory to PDF.
#
# Usage:
#   cd doc/Theses/shared/figures/diagrams
#   bash build_diagrams.sh            # build all
#   bash build_diagrams.sh foo.mmd    # build one specific file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
CONFIG="${SCRIPT_DIR}/mmdc_config.json"

mkdir -p "${OUTPUT_DIR}"

build_one() {
    local src="$1"
    local name
    name="$(basename "${src}" .mmd)"
    echo "  ${name}.mmd → output/${name}.pdf"
    mmdc --input "${src}" \
         --output "${OUTPUT_DIR}/${name}.pdf" \
         --configFile "${CONFIG}" \
         --pdfFit \
         --quiet
}

if [[ $# -gt 0 ]]; then
    for file in "$@"; do
        build_one "${file}"
    done
else
    shopt -s nullglob
    files=("${SCRIPT_DIR}"/*.mmd)
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No .mmd files found in ${SCRIPT_DIR}"
        exit 0
    fi
    echo "Building ${#files[@]} diagram(s)…"
    for file in "${files[@]}"; do
        build_one "${file}"
    done
fi

echo "Done. PDFs are in ${OUTPUT_DIR}/"

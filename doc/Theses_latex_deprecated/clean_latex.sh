#!/usr/bin/env bash
# clean_latex.sh — Remove LaTeX temporary/build files from all 4 thesis projects
# Usage:  ./clean_latex.sh          (dry-run: lists what would be deleted)
#         ./clean_latex.sh --delete  (actually deletes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extensions produced by pdflatex / bibtex / biber / makeglossaries / latexmk
TEMP_EXTENSIONS=(
  # Core LaTeX
  aux log out toc lof lot
  # BibTeX / Biber
  bbl blg bcf run.xml
  # Glossaries / Acronyms
  glo gls glg acn acr alg
  # Index
  idx ind ilg
  # Nomenclature
  nlo nls
  # Beamer / misc
  nav snm vrb
  # Latexmk
  fdb_latexmk fls
  # SyncTeX (editor cursor sync)
  synctex.gz synctex.gz\(busy\)
  # Makeindex
  ist
)

# Directories that contain compilable LaTeX projects
LATEX_DIRS=(
  "$SCRIPT_DIR/project_thesis/guideline"
  "$SCRIPT_DIR/project_thesis/thesis"
  "$SCRIPT_DIR/master_thesis/guideline"
  "$SCRIPT_DIR/master_thesis/thesis"
)

DRY_RUN=true
[[ "$1" == "--delete" ]] && DRY_RUN=false

found=0

for dir in "${LATEX_DIRS[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "WARNING: directory not found: $dir"
    continue
  fi

  for ext in "${TEMP_EXTENSIONS[@]}"; do
    # Use -maxdepth 3 so we also catch files in chapters/ subdirectories
    while IFS= read -r -d '' file; do
      if $DRY_RUN; then
        echo "[dry-run] would remove: ${file#$SCRIPT_DIR/}"
      else
        rm -f "$file"
        echo "removed: ${file#$SCRIPT_DIR/}"
      fi
      ((found++))
    done < <(find "$dir" -maxdepth 3 -name "*.$ext" -not -path "*/shared/*" -print0 2>/dev/null)
  done
done

echo ""
if $DRY_RUN; then
  echo "Dry-run complete. $found file(s) would be deleted."
  echo "Run with --delete to actually remove them:  ./clean_latex.sh --delete"
else
  echo "Done. $found file(s) removed."
fi

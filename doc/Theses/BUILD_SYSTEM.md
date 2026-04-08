# Build System — FAPS Typst Theses

This document explains the directory layout, the `--root` requirement, symlinks, and all
`make` targets for building the PA and MA theses.

---

## Directory Layout

```
doc/Theses_typst/
├── Makefile                    ← top-level build (builds all 4 documents)
├── shared/                     ← shared assets & formatting (single source of truth)
│   ├── bibliography/
│   │   └── literature.bib
│   ├── coversheet/
│   │   ├── deckblatt.typ       ← cover page template
│   │   ├── LogoFAPS.png        ← high-res FAPS logo (600px, from EPS)
│   │   └── titelbild_wip.png  ← title image (replace when ready)
│   ├── CV/
│   │   └── curriculum_vitae.typ
│   ├── erklaerung.typ          ← declaration of authenticity
│   ├── figures/                ← shared figures (diagrams, photos)
│   └── formatting/
│       ├── acronyms.typ        ← abbreviation & symbol database + rendering
│       ├── colors.typ          ← FAPS color palette
│       ├── diagrams.typ        ← native fletcher 0.5.8 diagram library
│       ├── macros.typ          ← helper macros (todo, abb, tab-ref, …)
│       └── template.typ        ← page layout, typography, heading styles
│
├── project_thesis/
│   ├── Makefile                ← per-thesis build (PA only)
│   ├── shared -> ../shared     ← symlink (for IDE navigation & editor paths)
│   ├── guideline/
│   │   ├── PA-Guideline.typ
│   │   └── chapters/
│   └── thesis/
│       ├── PA-Thesis.typ       ← main entry point
│       └── chapters/
│           ├── 1_0_Introduction.typ
│           ├── 2_0_Background.typ
│           ├── 3_0_ResearchGap.typ
│           ├── 4_0_Methodology.typ
│           ├── 5_0_Results.typ
│           ├── 6_0_Discussion.typ
│           ├── 7_0_Conclusion.typ
│           ├── A_Appendices.typ
│           ├── background/     ← background sub-chapters (2_1 … 2_4)
│           └── methodology/    ← methodology sub-chapters
│               ├── 4_1_Overview.typ
│               ├── 4_2_PhysicalRobot.typ
│               ├── 4_3_SimulationModel.typ
│               ├── 4_4_TendonActuation.typ
│               ├── 4_5_Validation.typ
│               ├── 4_6_RLTasks.typ
│               └── 4_7_Infrastructure.typ
│
└── master_thesis/
    ├── Makefile                ← per-thesis build (MA only)
    ├── shared -> ../shared     ← symlink
    ├── guideline/
    └── thesis/
        ├── MA-Thesis.typ
        └── chapters/
```

---

## Why `--root` is Required

Typst restricts file access to a **root directory** to prevent path traversal. All chapter
files use paths like `../../shared/formatting/template.typ` — these go *above* the `thesis/`
directory to reach the `shared/` folder.

Without `--root ..`, Typst cannot resolve these paths. The `--root ..` flag sets the project
root to the `project_thesis/` (or `master_thesis/`) level, so:

- From `project_thesis/thesis/PA-Thesis.typ`, `../../shared/` resolves to `project_thesis/../shared/` = `Theses_typst/shared/` ✓

The `shared -> ../shared` symlink inside each thesis directory serves only as a convenience
for IDE navigation and editor "go to file" features — it is **not** used by Typst at compile
time because Typst resolves symlinks before checking root confinement.

---

## Make Targets

### Top-level (`doc/Theses_typst/`)

```bash
make          # build all 4 documents (PA thesis + PA guideline + MA thesis + MA guideline)
make pa       # build PA thesis + PA guideline
make ma       # build MA thesis + MA guideline
make pa-thesis     # build PA-Thesis.pdf only
make pa-guideline  # build PA-Guideline.pdf only
make ma-thesis     # build MA-Thesis.pdf only
make ma-guideline  # build MA-Guideline.pdf only
make watch-pa      # watch-mode: rebuild PA-Thesis.pdf on file changes
make watch-ma      # watch-mode: rebuild MA-Thesis.pdf on file changes
make clean         # delete all generated PDFs
```

### Project Thesis (`doc/Theses_typst/project_thesis/`)

```bash
make           # build PA-Thesis.pdf + PA-Guideline.pdf
make thesis    # build thesis/PA-Thesis.pdf only
make guideline # build guideline/PA-Guideline.pdf only
make watch-thesis    # live-reload PA thesis
make watch-guideline # live-reload PA guideline
make clean     # delete generated PDFs
```

### Master Thesis (`doc/Theses_typst/master_thesis/`)

```bash
make           # build MA-Thesis.pdf + MA-Guideline.pdf
make thesis    # build thesis/MA-Thesis.pdf only
make guideline # build guideline/MA-Guideline.pdf only
make watch-thesis    # live-reload MA thesis
make watch-guideline # live-reload MA guideline
make clean     # delete generated PDFs
```

---

## Manual `typst` Commands

If `make` is not available, compile directly:

```bash
# From doc/Theses_typst/ (top level)
cd doc/Theses_typst

# PA thesis
cd project_thesis
typst compile thesis/PA-Thesis.typ --root .. thesis/PA-Thesis.pdf

# PA guideline
typst compile guideline/PA-Guideline.typ --root .. guideline/PA-Guideline.pdf

# MA thesis
cd ../master_thesis
typst compile thesis/MA-Thesis.typ --root .. thesis/MA-Thesis.pdf

# MA guideline
typst compile guideline/MA-Guideline.typ --root .. guideline/MA-Guideline.pdf
```

Watch mode (live rebuild on save):
```bash
cd project_thesis
typst watch thesis/PA-Thesis.typ --root .. thesis/PA-Thesis.pdf
```

---

## Output Files

| Document | Output path |
|---|---|
| PA Thesis | `project_thesis/thesis/PA-Thesis.pdf` |
| PA Guideline | `project_thesis/guideline/PA-Guideline.pdf` |
| MA Thesis | `master_thesis/thesis/MA-Thesis.pdf` |
| MA Guideline | `master_thesis/guideline/MA-Guideline.pdf` |

---

## Adding a New Chapter

1. Create `thesis/chapters/N_0_ChapterName.typ`
2. Import shared files at the top:
   ```typst
   #import "../../../shared/formatting/template.typ": *
   #import "../../../shared/formatting/colors.typ": *
   #import "../../../shared/formatting/macros.typ": *
   #import "../../../shared/formatting/acronyms.typ": *
   #import "../../../shared/formatting/diagrams.typ": *
   ```
3. Add `#include "chapters/N_0_ChapterName.typ"` to `PA-Thesis.typ` (or `MA-Thesis.typ`).

> **Import path depth:** From `thesis/chapters/`, use `../../../shared/`. From
> `thesis/chapters/subfolder/`, use `../../../../shared/`.

---

## Adding a New Abbreviation or Symbol

Edit `shared/formatting/acronyms.typ`, adding an entry to the `acronyms` dictionary:

```typst
// Abbreviation
"MyAbbrev": (short: "MA", long: "My Abbreviation", tag: "abbrev"),

// Symbol
"sym:myvar": (short: $alpha$, long: "My variable description", tag: "symbol"),
```

Use `#ac("MyAbbrev")` in the text. The entry will **automatically appear** in the List of
Abbreviations / List of Symbols **only if used** in the document.

---

## Regenerating the FAPS Logo PNG

The `LogoFAPS.png` was generated from the original EPS:
```bash
cd shared/coversheet
sips -s format png -Z 600 LogoFAPS.eps --out LogoFAPS.png
```

The EPS is the authoritative source; the PNG is a build artifact. If the EPS changes, rerun
the above command.

---

## Typst Version

Built and tested with **Typst 0.14.2**. The `fletcher` diagram package requires version
≥ 0.5.8 (included via `#import "@preview/fletcher:0.5.8": *` in `diagrams.typ`).

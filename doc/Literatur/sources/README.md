# Source PDFs

This directory contains downloaded PDFs and saved web pages for all literature sources,
organized by topic and source type to match the literature overview files.

## Structure

```
sources/
  cloth_manipulation/
    Conference_&_Journal_Papers/  (16 PDFs, 0 HTML, 0 missing / 16 total)
    Conference_Papers/  (4 PDFs, 0 HTML, 0 missing / 4 total)
    Review_Articles/  (4 PDFs, 0 HTML, 0 missing / 4 total)
  control_kinematics/
    Books/  (5 PDFs, 0 HTML, 0 missing / 5 total)
    Conference_Papers/  (1 PDFs, 0 HTML, 0 missing / 1 total)
    Journal_Articles/  (7 PDFs, 0 HTML, 4 missing / 11 total)
  grasping_manipulation/
    Books/  (1 PDFs, 0 HTML, 0 missing / 1 total)
    Conference_&_Journal_Papers/  (14 PDFs, 0 HTML, 2 missing / 16 total)
    Journal_&_Conference_Papers/  (7 PDFs, 0 HTML, 2 missing / 9 total)
  linkage_mechanisms/
    Books/  (3 PDFs, 0 HTML, 0 missing / 3 total)
    Conference_&_Journal_Papers/  (4 PDFs, 0 HTML, 1 missing / 5 total)
    Journal_&_Conference_Papers/  (3 PDFs, 0 HTML, 1 missing / 4 total)
    Seminal_Papers/  (1 PDFs, 0 HTML, 0 missing / 1 total)
  reinforcement_learning/
    Book_Chapters/  (1 PDFs, 0 HTML, 0 missing / 1 total)
    Books/  (3 PDFs, 0 HTML, 0 missing / 3 total)
    Conference_Papers/  (11 PDFs, 0 HTML, 0 missing / 11 total)
    Journal_Articles/  (1 PDFs, 0 HTML, 0 missing / 1 total)
    Journal_Articles_&_Preprints/  (7 PDFs, 0 HTML, 0 missing / 7 total)
    Preprints/  (8 PDFs, 0 HTML, 0 missing / 8 total)
    Preprints_&_Conference_Papers/  (4 PDFs, 0 HTML, 0 missing / 4 total)
    Reward_Machines/  (3 PDFs, 0 HTML, 1 missing / 4 total)
    Staged_&_Composite_Rewards/  (2 PDFs, 0 HTML, 0 missing / 2 total)
    Survey_Articles/  (4 PDFs, 0 HTML, 0 missing / 4 total)
  sim_to_real/
    Conference_&_Journal_Papers/  (4 PDFs, 0 HTML, 0 missing / 4 total)
    Conference_Papers/  (5 PDFs, 0 HTML, 0 missing / 5 total)
    Conference_Papers_&_Preprints/  (3 PDFs, 0 HTML, 0 missing / 3 total)
    Preprints/  (2 PDFs, 0 HTML, 0 missing / 2 total)
  simulation/
    Books_Resources/  (0 PDFs, 1 HTML, 1 missing / 2 total)
    Conference_Papers/  (3 PDFs, 0 HTML, 0 missing / 3 total)
    Documentation/  (0 PDFs, 9 HTML, 0 missing / 9 total)
    Documentation_&_Blog_Posts/  (0 PDFs, 6 HTML, 0 missing / 6 total)
    Documentation_(Online)/  (0 PDFs, 45 HTML, 0 missing / 45 total)
    Journal_Articles/  (8 PDFs, 0 HTML, 0 missing / 8 total)
    Journal_Articles_&_Preprints/  (3 PDFs, 0 HTML, 0 missing / 3 total)
    Preprints/  (1 PDFs, 0 HTML, 0 missing / 1 total)
    Seminal_Papers/  (4 PDFs, 0 HTML, 0 missing / 4 total)
  tendon_robots/
    Books/  (2 PDFs, 1 EPUB, 0 HTML, 1 missing / 4 total)
    Conference_Papers/  (5 PDFs, 0 HTML, 1 missing / 6 total)
    Foundational_Works/  (3 PDFs, 0 HTML, 2 missing / 5 total)
    Journal_&_Conference_Papers/  (11 PDFs, 0 HTML, 0 missing / 11 total)
    Journal_Articles/  (13 PDFs, 0 HTML, 2 missing / 15 total)
    Research_Contributions/  (1 PDFs, 0 HTML, 0 missing / 1 total)
    Review_Articles/  (2 PDFs, 0 HTML, 0 missing / 2 total)
    Structure_Matrix_&_Force_Distribution/  (7 PDFs, 0 HTML, 1 missing / 8 total)
    Surveys/  (4 PDFs, 0 HTML, 0 missing / 4 total)
    Theses/  (4 PDFs, 0 HTML, 0 missing / 4 total)
    Workspace_Analysis_(CDPM)/  (5 PDFs, 0 HTML, 0 missing / 5 total)
  tensegrity_robots/
    Books/  (3 PDFs, 0 HTML, 3 missing / 6 total)
    Books_&_Surveys/  (2 PDFs, 0 HTML, 0 missing / 2 total)
    Conference_&_Journal_Papers/  (24 PDFs, 0 HTML, 3 missing / 27 total)
    Foundational_Papers/  (4 PDFs, 0 HTML, 0 missing / 4 total)
    Journal_Articles/  (10 PDFs, 0 HTML, 0 missing / 10 total)
    Patents_&_Historical/  (3 PDFs, 0 HTML, 0 missing / 3 total)
    Review_Articles/  (3 PDFs, 0 HTML, 0 missing / 3 total)
    Standards/  (0 PDFs, 0 HTML, 2 missing / 2 total)
    Survey_Articles/  (0 PDFs, 0 HTML, 1 missing / 1 total)
  workspace_analysis/
    Documentation/  (0 PDFs, 0 HTML, 1 missing / 1 total)
    Journal_&_Conference_Papers/  (4 PDFs, 0 HTML, 1 missing / 5 total)
    Journal_Articles/  (6 PDFs, 0 HTML, 0 missing / 6 total)
```

## Summary

- **Total sources tracked:** 355
- **PDFs downloaded:** 263
- **EPUBs downloaded:** 1
- **HTML saved:** 61
- **Needs alternative source:** 30

Each subfolder contains a `README.md` listing which sources were downloaded,
which are saved as HTML, and which need an alternative source (with DOI links where available).

## Note on DOI verification

A number of DOI links listed for "missing" sources were found to point to the wrong paper
(unrelated journal articles / proceedings volumes). The affected per-category README files
flag these with a `⚠` warning. A full list of wrong DOIs is included in `missing_sources.html`.

30 entries remain with no downloadable full-text PDF. Their bibkeys are listed
in the per-folder READMEs under "Needs Alternative Source" with current best-known DOIs (and
warnings where the DOI is known to be wrong).

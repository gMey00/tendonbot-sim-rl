// PA-Thesis.typ — Project Thesis Main Document
// Georg Meyer, FAPS, FAU Erlangen-Nürnberg

#import "../../shared/formatting/template.typ": *
#import "../../shared/formatting/colors.typ": *
#import "../../shared/formatting/macros.typ": *
#import "../../shared/formatting/acronyms.typ": *
#import "../../shared/coversheet/deckblatt.typ": faps-coversheet
#import "../../shared/erklaerung.typ": erklaerung

// ── Metadata ──────────────────────────────────────────────────────
#let thesis-title = "Simulative Construction and Verification of a Tendon-Driven Robot for Conveyor-Based Pick-and-Place Tasks Using Isaac Lab"
#let thesis-type = "Project Thesis"
#let thesis-program = "Computational Engineering M. Sc."
#let thesis-author = "Georg Meyer"
#let thesis-student-id = "22791103"
#let thesis-deadline = "01.07.2026"
#let thesis-duration = "x"

// ── Apply FAPS template ──────────────────────────────────────────
#show: faps-thesis.with(
  title: thesis-title,
  author: thesis-author,
)

// ── Cover page ───────────────────────────────────────────────────
#faps-coversheet(
  title: thesis-title,
  thesis-type: thesis-type,
  program: thesis-program,
  author: thesis-author,
  student-id: thesis-student-id,
  supervisors: (
    [Prof. Dr.-Ing. J. Franke],
    [M.Sc. A. Schlosser],
  ),
  deadline: thesis-deadline,
  duration: thesis-duration,
  title-image: "../../shared/coversheet/titelbild_pt.png",
)

// ── Declaration of authenticity ──────────────────────────────────
#erklaerung(name: thesis-author)

// ── Front matter (roman numerals) ────────────────────────────────
#show: faps-frontmatter

#faps-toc()
#faps-lof()
#faps-lot()
#faps-loa()

// ── Abbreviations and symbols ────────────────────────────────────
#print-abbreviations()
#print-symbols()

// ── Main matter (arabic numerals, header/footer) ─────────────────
#show: faps-header-footer
#counter(page).update(1)

// ── Chapters ─────────────────────────────────────────────────────
#include "chapters/1_0_Introduction.typ"
#include "chapters/2_0_Background.typ"
#include "chapters/3_0_ResearchGap.typ"
#include "chapters/4_0_Methodology.typ"
#include "chapters/5_0_Results.typ"
#include "chapters/6_0_Discussion.typ"
#include "chapters/7_0_Conclusion.typ"

// ── Bibliography ─────────────────────────────────────────────────
#bibliography("../../shared/bibliography/literature.bib", style: "../../shared/bibliography/iso690-numeric-alphabetical.csl")

// ── Appendices ───────────────────────────────────────────────────
#_in-appendix.update(true)
#include "chapters/A_Appendices.typ"

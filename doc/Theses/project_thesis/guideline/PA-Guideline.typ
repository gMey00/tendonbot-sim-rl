// PA-Guideline — Project Thesis Guideline Document
#import "../../shared/formatting/template.typ": *
#import "../../shared/formatting/macros.typ": *
#import "../../shared/formatting/acronyms.typ": *
#import "../../shared/formatting/colors.typ": *
#import "../../shared/coversheet/deckblatt.typ": faps-coversheet
#import "../../shared/erklaerung.typ": erklaerung

// ──────────────────────────────────────────────────────
// Metadata
// ──────────────────────────────────────────────────────
#let title = "Simulative Construction and Verification of a Tendon-Driven Robot for Conveyor-Based Pick-and-Place Tasks Using Isaac Lab"
#let thesis-type = "Project Thesis"
#let program = "Computational Engineering M. Sc."
#let author = "Georg Meyer"
#let student-id = "22791103"
#let duration = "x"
#let deadline = "01.07.2026"
#let title-image = "../../shared/coversheet/titelbild_wip.png"

// ──────────────────────────────────────────────────────
// Apply template
// ──────────────────────────────────────────────────────
#show: faps-thesis.with(
  title: title,
  author: author,
)

// ──────────────────────────────────────────────────────
// Cover page and declaration
// ──────────────────────────────────────────────────────
#faps-coversheet(
  title: title,
  thesis-type: thesis-type,
  program: program,
  author: author,
  student-id: student-id,
  duration: duration,
  deadline: deadline,
  title-image: title-image,
)

#erklaerung(name: author)

// ──────────────────────────────────────────────────────
// Front matter
// ──────────────────────────────────────────────────────
#show: faps-frontmatter
#faps-toc()

#show: faps-header-footer
#counter(page).update(1)

// ──────────────────────────────────────────────────────
// Guideline and Chapter Outline
// ──────────────────────────────────────────────────────
#include "chapters/0_0_Guideline.typ"
#include "chapters/0_1_PA_ChapterOutline.typ"
#include "chapters/0_2_BackgroundCitationLog.typ"

// ──────────────────────────────────────────────────────
// Bibliography
// ──────────────────────────────────────────────────────
#bibliography(("../../shared/bibliography/literature.bib", "../../shared/bibliography/sources.bib"), style: "ieee")

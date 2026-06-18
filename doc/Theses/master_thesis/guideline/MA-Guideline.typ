// MA-Guideline — Master Thesis Guideline Document
#import "../../shared/formatting/template.typ": *
#import "../../shared/formatting/macros.typ": *
#import "../../shared/formatting/acronyms.typ": *
#import "../../shared/formatting/colors.typ": *
#import "../../shared/coversheet/deckblatt.typ": faps-coversheet
#import "../../shared/erklaerung.typ": erklaerung

// ──────────────────────────────────────────────────────
// Metadata
// ──────────────────────────────────────────────────────
#let title = "Multi-Agent Reinforcement Learning for Condition-Based Textile Sorting with a Tendon-Driven and Collaborative Robot System"
#let thesis-type = "Master Thesis"
#let program = "Computational Engineering M. Sc."
#let author = "Georg Meyer"
#let student-id = "22791103"
#let duration = "x"
#let deadline = "TBD"
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
#include "chapters/0_1_MA_ChapterOutline.typ"

// ──────────────────────────────────────────────────────
// Bibliography
// ──────────────────────────────────────────────────────
#bibliography("../thesis/bibliography.bib", style: "ieee")

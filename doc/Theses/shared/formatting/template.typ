// FAPS Thesis Template for Typst
// Recreates the FAPS Word/LaTeX template formatting
// ============================================================

#import "colors.typ": *
#import "macros.typ": *

// ── Short/long caption state ─────────────────────────────────────
// Stores short captions keyed by "<kind>:<number>" for LOF/LOT use.
// When a figure/table is created with faps-figure()/faps-table() and a
// short-caption is provided, the long caption appears in-text while
// the short caption is used in the List of Figures / List of Tables.
#let _short-captions = state("short-captions", (:))

// Main thesis template function
#let faps-thesis(
  title: "",
  thesis-type: "Project Thesis",
  program: "Computational Engineering M. Sc.",
  author: "",
  student-id: "",
  deadline: "",
  duration: "x",
  supervisors: (),
  title-image: none,
  body,
) = {
  // Page layout matching FAPS template
  set page(
    paper: "a4",
    margin: (
      top: 2.9cm,
      bottom: 2.5cm,
      left: 2.5cm,
      right: 2.5cm,
    ),
    header-ascent: 1.1cm,
    footer-descent: 0.8cm,
  )

  // Typography — matches LaTeX 12pt helvet scaled 0.92 = ~11pt effective
  set text(
    font: ("Helvetica Neue", "Helvetica", "Nimbus Sans", "Arial"),
    size: 11pt,
    lang: "en",
  )
  set par(
    leading: 0.65em * 1.25,  // baselinestretch 1.25
    spacing: 0.65em * 1.25,
    first-line-indent: 0pt,
    justify: true,
  )

  // Heading styles — sizes match FAPS LaTeX template
  // LaTeX 12pt + helvet 0.92: \LARGE≈19pt, \Large≈16pt, \large≈13pt
  // \titlespacing{\chapter}{0pt}{0pt}{75pt}
  set heading(numbering: "1.1.1")
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(0pt)
    block(
      below: 40pt,
      text(weight: "bold", size: 19pt)[
        #if it.numbering != none {
          counter(heading).display(it.numbering)
          h(1em)
        }
        #it.body
      ],
    )
  }
  show heading.where(level: 2): it => {
    block(
      above: 24pt,
      below: 12pt,
      text(weight: "bold", size: 16pt)[
        #if it.numbering != none {
          counter(heading).display(it.numbering)
          h(0.5em)
        }
        #it.body
      ],
    )
  }
  show heading.where(level: 3): it => {
    block(
      above: 18pt,
      below: 8pt,
      text(weight: "bold", size: 13pt)[
        #if it.numbering != none {
          counter(heading).display(it.numbering)
          h(0.5em)
        }
        #it.body
      ],
    )
  }

  // Level 4: paragraph-level headers — unnumbered, own line, no trailing dot
  show heading.where(level: 4): it => {
    block(
      above: 12pt,
      below: 6pt,
      text(weight: "bold", size: 11pt)[#it.body],
    )
  }

  // Lists with FAPS green bullets
  set list(
    marker: (
      text(fill: fapsgruen, size: 0.8em)[■],
      text(fill: fapsgruen, size: 0.6em)[▪],
    ),
    indent: 0pt,
    body-indent: 0.5em,
    spacing: 5pt,
  )
  set enum(indent: 0pt, body-indent: 0.5em, spacing: 5pt)

  // ── Figure/table caption formatting ────────────────────────────
  // Matches LaTeX: format=hang, font=footnotesize, labelfont=bf
  // Caption separator is ": " (bold label, then colon, then body).
  // Hanging indent: continuation lines align with the start of the body text.
  set figure(gap: 0.8em)
  set figure.caption(separator: [ ])
  show figure.caption: it => {
    set text(size: 9pt)
    set par(hanging-indent: 0pt)
    let label-text = text(weight: "bold")[#it.supplement #context it.counter.display(it.numbering):]
    // Measure label width for hanging indent
    context {
      let label-width = measure(label-text).width + 0.3em
      pad(left: label-width, top: 0pt, bottom: 0pt,
        place(dx: -label-width, label-text)
        + it.body
      )
    }
  }

  // ── Figure/table visual separation from body text ──────────────
  // Extra spacing and a thin rule above/below figures for clear separation
  // from the body text so captions and tables are not confused with reading text.
  show figure: it => {
    v(1.2em)
    line(length: 100%, stroke: 0.4pt + dunkelgrau)
    v(0.4em)
    it
    v(0.4em)
    line(length: 100%, stroke: 0.4pt + dunkelgrau)
    v(1.2em)
  }

  // Table styling
  set table(
    stroke: 0.5pt + dunkelgrau,
    inset: 6pt,
  )

  // Math equations left-aligned
  set math.equation(numbering: "(1)")

  // Links
  show link: it => text(fill: fapsblau, it)

  body
}

// Header/footer for main content pages
#let faps-header-footer(body) = {
  set page(
    header: context {
      let headings = query(selector(heading.where(level: 1)).before(here()))
      let sections = query(selector(heading.where(level: 2)).before(here()))
      if headings.len() > 0 {
        let current-heading = headings.last()
        set text(size: 8pt)
        if sections.len() > 0 {
          let current-section = sections.last()
          upper[#current-section.body]
        } else {
          upper[#current-heading.body]
        }
        v(-0.5em)
        line(length: 100%, stroke: 0.75pt)
      }
    },
    footer: {
      line(length: 100%, stroke: 0.75pt)
      v(-0.3em)
      align(right, context {
        set text(size: 10pt)
        counter(page).display("1")
      })
    },
  )
  body
}

// Front matter pages (roman numerals, no chapter headers)
#let faps-frontmatter(body) = {
  set page(
    header: none,
    footer: {
      align(right, context {
        set text(size: 10pt)
        counter(page).display("i")
      })
    },
  )
  counter(page).update(1)
  body
}

// ── Table of Contents ────────────────────────────────────────────
// Matches LaTeX: dot leaders, chapter entries bold with 12pt/8pt spacing,
// indent levels: ch=1em, sec=2.79em, subsec=5.387em.
#let faps-toc() = {
  show outline.entry.where(level: 1): it => {
    v(12pt, weak: true)
    strong(it)
    v(4pt, weak: true)
  }
  outline(
    title: [Table of Contents],
    depth: 3,
    indent: auto,
  )
}

// ── List of Figures (with short caption support) ─────────────────
// Renders all image figures. If a figure was created with faps-figure()
// and a short-caption was provided, that short text is displayed here
// instead of the full in-text caption.
#let faps-lof() = {
  pagebreak()
  heading(numbering: none)[List of Figures]
  v(12pt)
  context {
    let shorts = _short-captions.final()
    let figs = query(figure.where(kind: image))
    for f in figs {
      let n = counter(figure.where(kind: image)).at(f.location()).at(0)
      let key = "fig:" + str(n)
      let cap-text = shorts.at(key, default: none)
      if cap-text == none and f.caption != none {
        cap-text = f.caption.body
      }
      let pg = counter(page).at(f.location()).first()
      set text(size: 11pt)
      grid(
        columns: (auto, 1fr, auto),
        column-gutter: 4pt,
        link(f.location())[Figure #n: #cap-text],
        align(bottom, box(width: 100%, repeat[.#h(4pt)])),
        link(f.location())[#pg],
      )
      v(4pt)
    }
  }
}

// ── List of Tables (with short caption support) ──────────────────
#let faps-lot() = {
  pagebreak()
  heading(numbering: none)[List of Tables]
  v(12pt)
  context {
    let shorts = _short-captions.final()
    let figs = query(figure.where(kind: table))
    for f in figs {
      let n = counter(figure.where(kind: table)).at(f.location()).at(0)
      let key = "tab:" + str(n)
      let cap-text = shorts.at(key, default: none)
      if cap-text == none and f.caption != none {
        cap-text = f.caption.body
      }
      let pg = counter(page).at(f.location()).first()
      set text(size: 11pt)
      grid(
        columns: (auto, 1fr, auto),
        column-gutter: 4pt,
        link(f.location())[Table #n: #cap-text],
        align(bottom, box(width: 100%, repeat[.#h(4pt)])),
        link(f.location())[#pg],
      )
      v(4pt)
    }
  }
}

// ── Figure wrapper with short/long caption support ───────────────
// Use this instead of bare #figure() when you need separate captions for
// the in-text display vs the List of Figures.
//
// Parameters:
//   body:          The figure content (image, diagram, etc.)
//   caption:       The LONG caption displayed under the figure in text.
//   short-caption: (optional) Short caption for the List of Figures.
//                  If omitted, the full caption is used in the LOF.
//   All other named arguments are forwarded to figure().
//
// Example:
//   #faps-figure(
//     image("path.png", width: 80%),
//     caption: [Detailed explanation of the workspace analysis results
//               showing the reachable volume for all eight tendon configurations.],
//     short-caption: [Workspace analysis results],
//   ) <fig:workspace>
#let faps-figure(body, caption: none, short-caption: none, ..args) = {
  // Embed short-caption state update inside the caption content so the
  // figure remains the sole returned element (labels attach correctly).
  let final-caption = if short-caption != none and caption != none {
    [#context {
      let n = counter(figure.where(kind: image)).get().at(0)
      _short-captions.update(d => { d.insert("fig:" + str(n), short-caption); d })
    }#caption]
  } else {
    caption
  }
  figure(body, caption: final-caption, ..args)
}

// ── Table wrapper with short/long caption support ────────────────
// Same as faps-figure but for tables.
//
// Example:
//   #faps-table(
//     table(columns: 3, ..data),
//     caption: [Full comparison of simulation parameters across all
//               configurations tested during the validation phase.],
//     short-caption: [Simulation parameter comparison],
//   ) <tab:params>
#let faps-table(body, caption: none, short-caption: none, ..args) = {
  let final-caption = if short-caption != none and caption != none {
    [#context {
      let n = counter(figure.where(kind: table)).get().at(0)
      _short-captions.update(d => { d.insert("tab:" + str(n), short-caption); d })
    }#caption]
  } else {
    caption
  }
  figure(body, caption: final-caption, kind: table, ..args)
}

// FAPS Thesis Template for Typst
// Recreates the FAPS Word/LaTeX template formatting
// ============================================================

#import "colors.typ": *
#import "macros.typ": *

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

  // Heading styles — sizes proportional to Word template
  set heading(numbering: "1.1.1")
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(0pt)
    block(
      below: 24pt,
      text(weight: "bold", size: 14pt)[
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
      above: 18pt,
      below: 8pt,
      text(weight: "bold", size: 12pt)[
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
      above: 12pt,
      below: 6pt,
      text(weight: "bold", size: 11pt)[
        #if it.numbering != none {
          counter(heading).display(it.numbering)
          h(0.5em)
        }
        #it.body
      ],
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

  // Figure captions
  set figure(gap: 0.8em)
  show figure.caption: it => {
    set text(size: 9pt)
    it
  }
  set figure.caption(separator: [ ])
  show figure.caption: it => {
    align(left)[
      #text(weight: "bold")[#it.supplement #context it.counter.display(it.numbering):]
      #it.body
    ]
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

  // Store metadata for later use
  let thesis-metadata = (
    title: title,
    thesis-type: thesis-type,
    program: program,
    author: author,
    student-id: student-id,
    deadline: deadline,
    duration: duration,
    supervisors: supervisors,
    title-image: title-image,
  )

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

// Table of Contents
#let faps-toc() = {
  show outline.entry.where(level: 1): it => {
    v(12pt, weak: true)
    strong(it)
  }
  outline(
    title: [Table of Contents],
    depth: 3,
    indent: auto,
  )
}

// List of Figures
#let faps-lof() = {
  pagebreak()
  outline(
    title: [List of Figures],
    target: figure.where(kind: image),
  )
}

// List of Tables
#let faps-lot() = {
  pagebreak()
  outline(
    title: [List of Tables],
    target: figure.where(kind: table),
  )
}

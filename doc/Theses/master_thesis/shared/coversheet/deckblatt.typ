// FAPS Cover Page (Deckblatt)
// Recreates the LaTeX Deckblatt.tex layout faithfully
// Coordinates derived from LaTeX textblock* absolute positions

#import "../formatting/colors.typ": *

#let faps-coversheet(
  title: "",
  thesis-type: "",
  program: "",
  author: "",
  student-id: "",
  supervisors: (),
  deadline: "",
  duration: "",
  title-image: none,
) = {
  // Cover page uses its own margins to allow absolute-like placement
  // LaTeX textblock* coordinates are from page top-left edge
  // We set zero margins and place everything absolutely
  set page(
    header: none,
    footer: none,
    numbering: none,
    margin: (top: 0cm, bottom: 0cm, left: 0cm, right: 0cm),
  )

  // FAPS green bar (top left) — drawn as filled rectangle, not image
  // LaTeX: (1.3cm, 2.92cm), size 1.5cm × 0.47cm
  place(
    top + left,
    dx: 1.3cm,
    dy: 2.92cm,
    rect(width: 1.5cm, height: 3cm, fill: fapsgruen),
  )

  // FAPS Logo (top right)
  // LaTeX: (paperwidth - 1.9cm - 2.5cm, 1.4cm) = (16.6cm, 1.4cm)
  place(
    top + left,
    dx: 16.6cm,
    dy: 1.4cm,
    image("LogoFAPS.png", width: 3cm, height: 3cm),
  )

  // Title
  // LaTeX: (3.0cm, 2.25cm)
  place(
    top + left,
    dx: 3.0cm,
    dy: 2.25cm,
    block(width: 13cm)[
      #set text(size: 17pt, weight: "bold")
      #set par(leading: 0.5em, justify: true)
      #title
    ],
  )

  // Thesis type and program
  // LaTeX: (3cm, 6.5cm)
  place(
    top + left,
    dx: 3.0cm,
    dy: 6.5cm,
    block(width: 13cm)[
      #set text(weight: "bold", size: 11pt)
      #set par(leading: 0.5em)
      #thesis-type in the program #program
    ],
  )

  // University info
  // LaTeX: (3cm, 8.5cm)
  place(
    top + left,
    dx: 3.0cm,
    dy: 8.5cm,
    block(width: 16cm)[
      #set text(weight: "bold", size: 11pt)
      #set par(leading: 0.5em)
      Friedrich-Alexander-Universität Erlangen-Nürnberg \
      Lehrstuhl für Fertigungsautomatisierung und Produktionssystematik \
      Prof. Dr.-Ing. J. Franke
    ],
  )

  // Title image
  // LaTeX: (3cm, 11.65cm), 16cm × 9cm
  if title-image != none {
    place(
      top + left,
      dx: 3.0cm,
      dy: 11.65cm,
      image(title-image, width: 16cm, height: 9cm),
    )
  }

  // Metadata table
  // LaTeX: (3cm, 21.1cm)
  place(
    top + left,
    dx: 3.0cm,
    dy: 21.1cm,
    block(width: 16cm)[
      #set text(size: 11pt)
      #set par(leading: 0.5em)
      #grid(
        columns: (3.5cm, 7cm, 1fr),
        row-gutter: 0.6em,
        [Author:], [#author], align(right)[Student ID: #student-id],
        [], [], [],
        [Supervisors:], supervisors.at(0, default: []), [],
        ..if supervisors.len() > 1 { supervisors.slice(1).map(s => ([], s, [])).flatten() },
        [], [], [],
        [Deadline:], [#deadline], [],
        [Duration:], [#duration months], [],
      )
    ],
  )

  pagebreak()
  // Blank back of cover
  page(header: none, footer: none, numbering: none, margin: auto)[]
}

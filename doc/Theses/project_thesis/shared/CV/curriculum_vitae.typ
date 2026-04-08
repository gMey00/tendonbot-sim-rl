// Curriculum Vitae Template
// Port of the LaTeX CurriculumVitae.tex + Format_CV.tex

#let curriculum-vitae(
  name: "",
  street: "",
  city: "",
  phone: "",
  email: "",
  birthdate: "",
  birthplace: "",
  education: (),
) = {
  // CV-specific formatting
  set text(font: "Charter", size: 12pt)
  set par(leading: 0.65em)

  // No header, footer with contact info
  set page(
    header: none,
    footer: context {
      line(length: 100%, stroke: 0.4pt)
      v(0.3em)
      set text(size: 9pt)
      align(center)[#name, #street, #city, #phone]
    },
  )

  // Section heading style for CV
  show heading.where(level: 1): it => {
    set text(size: 17pt, style: "normal")
    smallcaps(it.body)
    v(0.5em)
  }

  show heading.where(level: 2): it => {
    set text(size: 14pt, weight: "bold")
    smallcaps(it.body)
    v(-0.5em)
    line(length: 100%, stroke: 0.5pt)
    v(0.3em)
  }

  // Title
  align(right, heading(level: 1, numbering: none)[Curriculum Vitae])

  heading(level: 2, numbering: none)[Personal Information]

  grid(
    columns: (0.2fr, 0.07fr, 0.65fr),
    row-gutter: 0.6em,
    [Name], [], [#name],
    [Address], [], [#street],
    [], [], [#city],
    [], [], [],
    [Date of birth], [], [#birthdate],
    [Place of birth], [], [#birthplace],
    [], [], [],
    [Phone], [], [#phone],
    [E-Mail], [], [#email],
  )

  v(-1.3cm)
  heading(level: 2, numbering: none)[Education]

  grid(
    columns: (0.2fr, 0.07fr, 0.65fr),
    row-gutter: 0.6em,
    ..education.map(entry => (entry.at(0), [], entry.at(1))).flatten(),
  )
}

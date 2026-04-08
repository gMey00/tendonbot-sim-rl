// Declaration of Authenticity (Erklärung)
// German legal declaration required for FAPS theses

#let erklaerung(name: "") = {
  heading(numbering: none)[Erklärung]

  set text(lang: "de")

  [Ich versichere, dass ich die vorliegende Arbeit ohne fremde Hilfe und ohne Benutzung anderer als
  der angegebenen Quellen angefertigt habe und dass die Arbeit in gleicher oder
  ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegen hat und von dieser
  als Teil einer Prüfungsleistung angenommen wurde. Alle Ausführungen, die
  wörtlich oder sinngemäß übernommen wurden, sind als solche gekennzeichnet.]

  v(2cm)

  let today = datetime.today()
  let date-str = today.display("[day].[month].[year]")

  grid(
    columns: (1fr, 1fr),
    align(left)[Erlangen, den #date-str],
    align(right)[
      #line(length: 6cm, stroke: 0.4pt)
      #v(-0.3em)
      #align(center)[#name]
    ],
  )

  pagebreak()
  // Blank page
  page(header: none, footer: none, numbering: none)[]
}

// Declaration of Authenticity (Erklärung)
// German legal declaration required for FAPS theses

#let erklaerung(name: "") = {
  heading(numbering: none)[Erklärung]

  set text(lang: "de")

  [Hiermit versichere ich, Georg Meyer (Name) 22791103 (Matrikelnummer), die vorgelegte Arbeit selbstständig und ohne unzulässige Hilfe Dritter sowie ohne die Hinzuziehung nicht offengelegter und insbesondere nicht zugelassener Hilfsmittel angefertigt zu haben. Die Arbeit hat in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegen und wurde auch von keiner anderen Prüfungsbehörde bereits als Teil einer Prüfung angenommen.

Die Stellen der Arbeit, die anderen Quellen im Wortlaut oder dem Sinn nach entnommen wurden, sind durch Angaben der Herkunft kenntlich gemacht (siehe Zitationsrichtlinien). Dies gilt auch für Zeichnungen, Skizzen, bildliche Darstellungen sowie für Quellen aus dem Internet.

Bei der Auswahl von KI-Tools ist auf Datenschutzkonformität (z. B. DSGVO, HAWKI) zu achten; Tools ohne ausreichende Datenschutzgarantien sollen vermieden werden. Es dürfen keine personenbezogenen oder vertraulichen Daten in KI-Systeme eingegeben werden.

Der Bearbeitende trägt die Verantwortung für eine eigenständige Leistungserbringung, für die Korrektheit der dargestellten Informationen sowie einen reflektierten, fachlich angemessenen und ethisch vertretbaren KI-Einsatz. 

*Verstöße gegen die o.g. Regeln sind als Täuschung bzw. Täuschungsversuch zu qualifizieren und führen zu einer Bewertung der Prüfung mit „nicht bestanden“.*

Ich, Georg Meyer, erkläre hiermit, in welchem Umfang ich Künstliche Intelligenz (KI), einschließlich Tools wie ChatGPT, Co-Pilot, HAWKI, DeepL Write, Grammarly oder vergleichbare Werkzeuge, bei der Erstellung der vorliegenden Prüfungsleistung genutzt habe. Ich habe KI-gestützte Werkzeuge wie folgt eingesetzt (bitte alle zutreffenden Optionen ankreuzen):

-	Gar nicht – Die Arbeit wurde vollständig ohne den Einsatz von KI-Werkzeugen erstellt.
-	Zur Recherche & Ideengenerierung – KI wurde genutzt, um erste Anregungen, Konzepte oder Fragestellungen zu entwickeln (muss in der Arbeit dokumentiert werden).
-	Zur Erstellung von Gliederungen / Strukturierung von Texten – KI wurde als Hilfsmittel zur Strukturierung der Arbeit eingesetzt, eigene Anpassungen wurden vorgenommen.
-	Zur Formulierungshilfe (z. B. Verbesserung von Satzbau, Stil) – KI wurde zur Verbesserung von Stil und Lesbarkeit genutzt, jedoch nicht zur vollständigen Textgenerierung.
-	Zur Korrektur von Grammatik und Rechtschreibung – KI wurde zur sprachlichen Optimierung genutzt, jedoch nicht zur inhaltlichen Veränderung oder Textgenerierung.
-	Zur Übersetzung von Texten – KI wurde für Übersetzungen genutzt, wobei alle übersetzten Passagen überarbeitet und auf inhaltliche Richtigkeit geprüft wurden.
-	Zur Erstellung oder Optimierung von Grafiken, Diagrammen oder visuellen Darstellungen – KI wurde für bildliche Darstellungen genutzt, die in der Arbeit interpretiert und eingeordnet wurden.
-	Zur Programmierung / Code-Erstellung – KI wurde zur Generierung von Code verwendet, wobei alle generierten Inhalte überprüft und nachvollziehbar kommentiert wurden.
-	Zur Erstellung von KI-gestützten Zusammenfassungen wissenschaftlicher Texte KI-Tools wurden zur Erstellung von Zusammenfassungen genutzt, deren Inhalte nachvollziehbar kommentiert wurden.


Ich versichere, alle Nutzungen vollständig und wahrheitsgemäß angegeben zu haben. Mir ist bewusst, dass das Verschweigen oder falsche Angaben zur Nutzung von KI-Tools als Täuschungsversuch gewertet werden können und prüfungsrechtliche Konsequenzen haben.
]

  
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

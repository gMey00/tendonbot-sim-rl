// Chapter 0: Guideline
#import "../../../shared/formatting/macros.typ": *
#import "../../../shared/formatting/acronyms.typ": *

#heading(numbering: none)[Guideline] <ch:guideline>

== Format and Style

=== Page Layout and Typography

- Paper format: white DIN A4 paper
- Font: Arial, 12pt (in this Typst template, Helvetica achieves this)
- Line spacing: 1.2 (the template uses 1.25)
- Text alignment: justified (Blocksatz) with automatic hyphenation enabled
- Font emphasis: use only in exceptional cases; headings follow the template style and must not be underlined
- Bullet points: only square bullet points (FAPS green), never round ones; use the template's built-in list style
- FAPS color scheme: FAPS-Grün (RGB 151, 193, 57) and white

=== Page Numbering and Binding

- Page numbering starts with the introduction (Arabic numerals)
- Directories (table of contents, list of figures, etc.) receive separate Roman numeral numbering; the cover page is not counted
- Print single-sided unless otherwise agreed with the supervisor
- Two bound copies required: adhesive binding with clear cover foil and green (FAPS green) back sheet
- Each copy must include a CD/DVD containing:
  - Digital version of the thesis (PDF)
  - All digital sources (papers, scanned documents, websites)
  - All images, tables, measurement data, models, and other data

=== Section and Chapter Formatting

- Chapters are numbered numerically with at most three levels (e.g., 1.1.1); no trailing dot after the last digit
- If a subsection X.1 exists, there must also be at least an X.2 (applies to all levels)
- Main chapters should have at least 3 subchapters; aim for a balanced structure (e.g., 5 chapters at level 1, each with \~4 at level 2, each with \~3 at level 3)
- Headings in the table of contents must match the headings in the text verbatim
- Every heading must be followed by actual content (including introductory text at the start of main chapters)

=== Paragraphs, Spacing, and Whitespace

- Each paragraph must consist of at least two sentences
- No blank line between paragraphs within a chapter; one blank line before the next heading at the end of a chapter
- No double spaces anywhere in the document; use search-and-replace to eliminate them before submission
- Use non-breaking spaces (#sym.space.nobreak) between:
  - Titles and names: "Prof.~Dr.~Beispielhausen"
  - Date expressions: "19.~September~2007"
  - Abbreviations: "z.#h(0.16em)B.", "u.#h(0.16em)a.", "d.#h(0.16em)h."
  - Numbers and units: "44~mm"
  - Numbers and words: "1.350~Personen", "Nr.~5"
  - Citation references
- Check automatic hyphenation carefully for errors

=== Figures

- Every figure receives a caption below it (Unterschrift), formulated so the figure is understandable without the surrounding text
- All figures are numbered consecutively with Arabic numerals and listed in the list of figures
- Every figure must be referenced in the body text
- Title image: no caption, no numbering, not included in the list of figures
- Appendix figures: numbered within each appendix separately; not included in the list of figures
- Ensure sufficient image quality; recreate figures if originals are blurry, low-resolution, or scanned
- Use the same font in figures as in body text (Arial); font size 10pt is acceptable but must be consistent across all figures
- All figure components must be fully labeled: photo components, axes, parameters, legends with complete process parameters
- Fill areas with color only if the color conveys meaningful information; otherwise keep figures clean and minimal
- For recreated/modified figures from sources, cite with "in Anlehnung an [xy]" (based on [xy])
- Use a consistent color scheme across all figures; preferably FAPS colors unless industry cooperation requires otherwise
- Use a self-created graphic on the first page of the thesis

=== Tables

- Every table receives a caption above it (Überschrift) with a meaningful, descriptive title
- Tables are numbered consecutively with Arabic numerals and listed in the table directory
- Every table must be referenced in the body text
- Use the same font as in body text; maintain consistent font size across all tables
- Preferred alignment inside tables: left-aligned or centered (not justified)
- If a table spans a page break, do not break within a row; repeat the table header on the new page
- Extensive tables belong in the appendix

=== Diagrams

- Follow DIN 461 for axis labeling in diagrams
- Label axes with an arrow in the direction of increase (except for trial numbers etc.)
- Replace the axis label at the second-to-last tick mark with the unit
- Display all relevant information within the diagram
- Place a frame around diagrams
- Use Arial font inside diagrams with a legible font size
- Diagrams are captioned and numbered as figures (no separate numbering type)

=== Formulas

- Formula numbers are enclosed in round parentheses
- Numbering is chapter-based in the form X.Y (X = chapter number, Y = formula number within the chapter)
- Numbering restarts within each main chapter
- Variables used in a formula must be explained directly below the formula in tabular form

=== Appendix

- Technical drawings, extensive tables, supplementary documents, etc. go into appendices
- Each appendix starts on a new page and may span multiple pages
- Appendices are numbered with consecutive uppercase Latin letters (Appendix A, Appendix B, etc.)

=== Eidesstattliche Erklärung (Statutory Declaration)

- A signed declaration of independent authorship must be included on a separate page without a page number, placed after the cover page
- It must be signed by hand

== Thesis Structure and Content

=== Required Document Order

+ Deckblatt (Cover Page)
+ Erklärung des Kandidaten (Statutory Declaration)
+ Inhaltsverzeichnis (Table of Contents)
+ Abbildungsverzeichnis (List of Figures)
+ Tabellenverzeichnis (List of Tables)
+ Abkürzungsverzeichnis (List of Abbreviations)
+ Einleitung und Zielstellung (Introduction and Objective)
+ Grundlagen / Stand der Technik (Background / State of the Art)
+ Handlungsbedarf (Research Gap / Need for Action)
+ Methodenteil (Methodology)
+ Ergebnisteil (Results)
+ Diskussionsteil (Discussion)
+ Zusammenfassung und Ausblick (Summary and Outlook)
+ Literaturverzeichnis (Bibliography)
+ Anhang (Appendix)
+ Lebenslauf (Curriculum Vitae)

=== Page Count Guidelines

- Bachelorarbeit: ca. 60--80 pages
- Projekt-/Studienarbeit: ca. 60--80 pages
- Master-/Diplomarbeit: ca. 80--120 pages
- These counts exclude appendices and directories

=== Red Thread (Roter Faden)

- A clear red thread must run through the entire thesis, starting from the central research question in the introduction
- The reader must always be able to see how any part of the thesis relates to the research question
- The thesis must proceed purposefully and not lose itself in tangential topics
- All claims and statements must be causally justified; results must not merely be listed but convincingly explained
- Each main chapter must begin with a short introduction motivating the content and end with a transition to the next
- Verify the red thread by checking all headings alone (without reading the text)---the logical flow should be recognizable

=== Chapter: Einleitung und Zielstellung (Introduction)

- Length: approximately 2 pages, maximum 3 pages
- Purpose: establish contact with the reader; provide orientation ("What can the reader expect?")
- Content requirements:
  - Introduce the topic and its general background; explain why it is relevant
  - Present the specific research question
  - Justify the scope and boundaries of the thesis (what is included, what is excluded, and why)
  - Optionally describe the state of research (to show what exists and where gaps are)
  - Explain the methodological approach
  - Outline the structure of the thesis for the reader

=== Chapter: Grundlagen und Stand der Technik (Background and State of the Art)

- Clarify central concepts and define relevant terms
- Present the current state of the art objectively and without judgment (evaluation happens in the next chapter)
- Only cover theories, definitions, and knowledge directly relevant to the research question; avoid overly broad coverage
- If multiple competing theories exist, provide an overview
- Definitions are only needed for terms not commonly known by domain experts
- Present content in your own words; near-verbatim copying or translation from sources counts as insufficient
- Use primarily English-language sources where possible
- Prefer journal articles and conference papers over websites
- Avoid popular-science sources; only quote newspapers for current events
- Internet sources are acceptable only if no current literature exists elsewhere (e.g., software documentation); verify their credibility

=== Chapter: Handlungsbedarf (Research Gap)

- Critically discuss the literature from the previous chapter regarding unsolved problems and errors
- Derive current deficits and research gaps; formulate the specific research question
- Explain who considers the problem significant and what its broader impact is
- Describe the contribution the thesis makes to both research (theory, models, methods, facts) and practice
- Narrow the problem statement into a concrete, achievable objective with clear expected outcomes

=== Chapter: Methodenteil (Methodology)

- Together with Results and Discussion, this forms the main body (\~2/3 of the thesis)
- Describe the investigation plan, procedures, and all steps and tools in sufficient detail for reproducibility
- Justify why specific methods were chosen over alternatives (pros/cons analysis)
- For experimental studies: describe the population and sample in all relevant characteristics; justify sample adequacy
- Scientific criteria for methods: objectivity, reliability, and validity
- The description must enable other researchers to reproduce the investigation
- Only describe analysis procedures in detail if they are novel or non-standard

=== Chapter: Ergebnisteil (Results)

- Present results of the investigation objectively and without interpretation
- Structure the chapter logically, aligned with the theory and methodology sections
- Use figures, tables, and diagrams to visualize results
- No interpretation or evaluation at this stage

=== Chapter: Diskussionsteil (Discussion)

- Summarize and critically discuss the results
- Evaluate and interpret results in relation to the original problem and objective from the introduction
- Argue the significance for scientific discourse and practical implications

=== Chapter: Zusammenfassung und Ausblick (Summary and Outlook)

- Maximum 3 pages
- Reconnect to the central problem and objective from the introduction
- Compactly summarize the most important results
- Place results in a broader context
- Address:
  - What are the key findings with respect to the research questions?
  - How can the results be interpreted or evaluated?
  - What questions remain open, and why?
  - What do the results mean for future work on this topic?
  - Where are opportunities for further research?
- Discuss the limitations of the investigation
- Provide suggestions for future work and improvements

== Scientific Writing Style

=== General Principles

- Write in present tense (Präsens) and nominal style (Nominalstil)
- Use impersonal form throughout; never use "ich" (I) or "man" (one/you)
- Develop the red thread of argumentation before formulating sentences
- Every sentence and every claim must be precise and well-placed
- Write result-oriented: "Die Einflüsse führen zu Veränderungen." not "Im Rahmen der Arbeit wurden untersucht..."

=== Sentence Construction

- Use passive voice sparingly---prefer active formulations
- Avoid sentences with "man" entirely
- Be careful with infinitive constructions
- Avoid chaining too many nouns together ("Substantivaneinanderreihungsproblem")
- No nested or overly complex sentences (Schachtelsätze)

=== Grammar

- Avoid relative time references ("kürzlich", "später", "modern") since the thesis should remain valid for years
- Do not use subjunctive mood (könnte, sollte, müsste, dürfte)

=== Technical Language

- Use precise adjectives; avoid vague terms like "innovativ" or "intelligent"
- No pleonasms (redundant word pairs)
- Do not artificially inflate terms
- Be cautious with superlatives and comparative forms
- Do not invent words
- Use anglicisms and foreign words sparingly; only when no equivalent German term exists
- Use precise technical terminology; avoid colloquial language
- Use technical terms consistently throughout

=== Expression and Phrasing

- Spell out numbers from zero to twelve; use digits from 13 onward
- Avoid colloquialisms and journalistic clichés
- Do not use rhetorical questions; make clear statements instead
- Avoid vague qualifiers: "im Allgemeinen", "in der Regel", "beispielsweise", "teilweise"
- Avoid filler words: "nämlich", "also", "so", "eben"
- Avoid repetition of both words and arguments
- Always formulate concretely and precisely; define unambiguously; enumerate completely; avoid generalizations
- Use nominal style

=== Content Guidelines

- Do not document iterative development processes or suboptimal intermediate solutions; present only the final approach
- Do not describe device/hardware development in excessive detail; focus on scientifically relevant, generalizable, transferable aspects
- Use direct quotes rarely and only pointedly; prefer evaluating and developing ideas further in your own words
- Introduce abbreviations only when necessary for space; spell out on first use and include in the abbreviation index
- Emphasize scientific quality rather than engineering challenges
- Emphasize fundamental, generalizable, and transferable findings rather than application-specific details
- Instead of giving examples, enumerate completely, structure, and evaluate
- Make all enumerations and representations as complete and non-overlapping as possible (MECE principle)
- Always support claims with your own analyses or cited sources
- No implied personal opinion and no subjective assessments except in the Outlook section
- Where possible, summarize each chapter's key message in a figure for quick reference

=== Introductions and Transitions Between Sections

- Every main chapter or main section must begin with a brief introduction and end with an outlook/transition to the next section
- When introducing detailed explanations, name all relevant points in the introductory sentence

=== Consistency and Formatting Rules

- Introduce each abbreviation exactly once and then use it consistently
- Never write a paragraph consisting of only one sentence (minimum two sentences)
- Never write a list with only one bullet point (minimum two)
- Use nominal style for bullet-point items
- Maintain consistent spelling of expressions throughout

== Citations and Bibliography

=== Citation Style

- Citation style: IEEE (adapted FAPS style)
- Reference sources in the text using numbers in square brackets: [xy]
- Multiple consecutive citations use separate brackets
- The bibliography is ordered chronologically and numbered consecutively
- Only actually cited works appear in the bibliography

=== Citation Placement

- A citation before the period refers to the specific sentence.
- A citation after the period of the last sentence refers to the entire paragraph.

=== Direct Quotes

- Must be letter-perfect and character-perfect reproductions
- Enclose in quotation marks; cite the source at the end

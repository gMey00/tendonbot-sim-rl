# FAPS Thesis Formatting Reference

This document records all formatting specifications extracted from the FAPS Word/LaTeX template
(`doc/Theses/shared/formatting/`) and their Typst equivalents in this repository.

---

## 1. Page Layout

| Property | LaTeX value | Typst implementation |
|---|---|---|
| Paper | A4 | `paper: "a4"` |
| Print layout | One-sided (`oneside`) | One-sided |
| Top margin | `topmargin = -1in + 1.2cm` + `headheight 0.7cm` + `headsep 1.1cm` → effective text top ≈ **2.9 cm** | `margin.top: 2.9cm` |
| Bottom margin | `textheight 23.8cm` → `297 - 2.9 - 23.8 = ~2.5 cm` | `margin.bottom: 2.5cm` |
| Left margin | `evensidemargin -0.5cm` → effective ≈ **2.5 cm** (LaTeX adds 1in) | `margin.left: 2.5cm` |
| Right margin | same as left | `margin.right: 2.5cm` |
| Text width | 16 cm | (implicit: A4 21cm − 2×2.5cm = 16cm ✓) |
| Header height | 0.7 cm | `header-ascent: 1.1cm` |
| Header sep | 1.1 cm | (built into `header-ascent`) |
| Footer skip | 1.5 cm | `footer-descent: 0.8cm` |

---

## 2. Typography

| Property | LaTeX value | Typst implementation |
|---|---|---|
| Base font | Helvetica (`helvet`, scaled 0.92) | `("Helvetica Neue", "Helvetica", "Nimbus Sans", "Arial")` |
| Font scale | 0.92 × 12pt = **~11 pt effective** | `size: 11pt` |
| Line spacing | `baselinestretch 1.25` | `leading: 0.65em * 1.25` (~0.8125 em extra) |
| Paragraph spacing | `parskip 1ex ± 0.2ex` | `spacing: 0.65em * 1.25` |
| Paragraph indent | `parindent 0pt` (no indent) | `first-line-indent: 0pt` |
| Justification | `\sloppy` (justified, loose) | `justify: true` |
| Table row stretch | `\arraystretch{1.5}` | `inset: 6pt` in table style |
| Math alignment | Left-aligned (`fleqn`) | `math.equation(numbering: "(1)")` |

### Font fallback chain
The system uses: `"Helvetica Neue"` → `"Helvetica"` → `"Nimbus Sans"` → `"Arial"`.
On macOS, `"Helvetica Neue"` is always available. On Linux/CI, `"Nimbus Sans"` (from TeX Live) is
used as fallback — Typst will emit a font-fallback warning which is harmless.

---

## 3. Heading Styles

| Level | LaTeX command/format | Typst size | Spacing above | Spacing below |
|---|---|---|---|---|
| H1 (Chapter) | `\bfseries\LARGE` (~20.74pt × 0.92 ≈ 19pt) | **19 pt bold** | page break | 40 pt |
| H2 (Section) | `\bfseries\Large` (~17.28pt × 0.92 ≈ 16pt) | **16 pt bold** | 24 pt | 12 pt |
| H3 (Subsection) | `\bfseries\large` (~14.4pt × 0.92 ≈ 13pt) | **13 pt bold** | 18 pt | 8 pt |
| H4 (Paragraph) | — (unnumbered) | **11 pt bold** | 12 pt | 6 pt |

Heading numbering depth: 3 levels (`secnumdepth 3`).

### H1 rule
- A page break is inserted before every Chapter-level heading.
- No horizontal rule — the LaTeX template does NOT add a rule under chapter titles.
- The `titlespacing` `{0pt}{0pt}{75pt}` means 75pt (≈ 2.65 cm) space *below* the chapter title.
  In the Typst implementation this is set to 40pt as a design compromise.

### H4 (Paragraph-level) headings
- Level 4 headings are **unnumbered** and rendered as bold standalone lines.
- No trailing dot is added after the heading text.
- Use `==== Heading Text` in Typst source.
- These are suitable for named paragraphs within a subsection.

### Heading spacing note
The `above:` values on H2 and H3 provide automatic visual separation before any
numbered heading. This ensures paragraphs preceding a new section/subsection are
clearly separated without requiring manual `#v()` calls.

> **LaTeX sizes note:** The LaTeX template uses 12pt base + `\usepackage[scaled=0.92]{helvet}`.
> Font size commands like `\LARGE`, `\Large`, `\large` select nominal sizes which are
> then rendered at 92% scale. The Typst sizes match the *effective visual output*.

---

## 4. Colors

All colors are defined in `shared/formatting/colors.typ`.

| Name | RGB | Hex | Use |
|---|---|---|---|
| `fapsgruen` | RGB(151, 193, 57) | `#97C139` | Document accent: list bullets, dividers, **cover page bar** |
| `fapsblau` | RGB(41, 97, 147) | `#296193` | Links, (optionally headings) |
| `fapsgraudunkel` | RGB(95, 95, 95) | `#5F5F5F` | Secondary text |
| `dunkelgrau` | RGB(204, 204, 204) | `#CCCCCC` | Borders, table strokes |
| `hellgrau` | RGB(230, 230, 230) | `#E6E6E6` | Table header backgrounds |
| `mittelgrau` | RGB(128, 128, 128) | `#808080` | Medium-gray elements |

> **Note:** Both the FAPS logo and the cover page green bar use `fapsgruen`
> (`rgb(151, 193, 57)`) for consistency across the document.

---

## 5. Lists

| Property | LaTeX value | Typst implementation |
|---|---|---|
| Level 1 marker | `\color{fapsgruen}$\medblacksquare$` | `text(fill: fapsgruen, "■")` (0.8em) |
| Level 2 marker | `\color{fapsgruen}$\smallblacksquare$` | `text(fill: fapsgruen, "▪")` (0.6em) |
| Item spacing | `itemsep=5pt` | `spacing: 5pt` |
| Body indent | `labelsep=5pt` | `body-indent: 0.5em` |
| Left margin | 0pt (template default) | `indent: 0pt` |

---

## 6. Header and Footer

| Property | LaTeX rule | Typst implementation |
|---|---|---|
| Header rule | `\headrulewidth{0.75pt}` | `line(stroke: 0.75pt)` |
| Footer rule | `\footrulewidth{0.75pt}` | `line(stroke: 0.75pt)` |
| Header content | Section/subsection title (uppercase, `\scriptsize` ≈ 8pt) | `text(size: 8pt, upper[...])` |
| Footer content | Page number, right-aligned | `align(right, counter(page).display("1"))` |
| Page number size | default | `text(size: 10pt)` |
| Front matter | Roman numerals, no chapter header in header | `faps-frontmatter()` wrapper |

---

## 7. Figures and Captions

| Property | LaTeX value | Typst implementation |
|---|---|---|
| Caption format | `format=hang, font=footnotesize, labelfont=bf` | Hanging indent, 9pt, bold label |
| Caption separator | "**Figure N:**" bold, then normal text | Bold label with colon, hanging indent for continuation lines |
| Caption alignment | Left | Left-aligned with hanging indent |
| Caption size | `\footnotesize` (≈8–9pt in 11pt doc) | `text(size: 9pt)` |
| Caption gap | default `\abovecaptionskip` (10pt) | `gap: 0.8em` |
| Figure separation | `\intextsep 12pt` | Thin 0.4pt `dunkelgrau` rules above/below with `v(1.2em)` outer and `v(0.4em)` inner spacing |

### Short/long captions

The template supports separate short captions (for LOF/LOT listings) and long
captions (displayed in-text). Use `faps-figure()` or `faps-table()` from `template.typ`:

```typst
#faps-figure(
  image("path/to/image.png", width: 80%),
  caption: [This is the long detailed caption that appears under the figure
            in the chapter text, explaining all relevant details.],
  short-caption: [Short caption for LOF],
) <fig:label>
```

When `short-caption` is omitted, the full `caption` text is used in both places.

For tables:
```typst
#faps-table(
  table(columns: 3, ..data),
  caption: [Full detailed table caption.],
  short-caption: [Short caption for LOT],
) <tab:label>
```

Plain `#figure()` still works for figures that need the same caption everywhere.
The custom LOF/LOT functions (`faps-lof()`, `faps-lot()`) automatically resolve
short captions from an internal state dictionary.

Reference with `@fig:label` or `#abb(<fig:label>)`.

---

## 8. Tables

| Property | LaTeX value | Typst implementation |
|---|---|---|
| Row stretch | `\arraystretch{1.5}` | `inset: 6pt` |
| Border color | `dunkelgrau` | `stroke: 0.5pt + dunkelgrau` |
| Header style | bold, `hellgrau` fill | `fill: hellgrau, text(weight: "bold", ...)` |

Standard table pattern:
```typst
#faps-table(
  table(
    columns: (auto, 1fr, auto),
    table.header(
      [*Column 1*], [*Column 2*], [*Column 3*],
    ),
    [row 1], [data], [value],
  ),
  caption: [Detailed table caption.],
  short-caption: [Short caption for LOT],
) <tab:label>
```

Plain `#figure(..., kind: table)` still works but won't register a short caption.

---

## 9. Cover Page (Deckblatt)

All coordinates are absolute from the **top-left corner of the A4 page** (zero margins).
Defined in `shared/coversheet/deckblatt.typ`.

| Element | x (from left) | y (from top) | Size / notes |
|---|---|---|---|
| FAPS green bar | 1.3 cm | 2.92 cm | 1.5 cm wide × **0.47 cm** tall, `fapsgruen` fill |
| FAPS logo | 16.6 cm | 1.4 cm | 3 cm × 3 cm, `LogoFAPS.png` (600px high-res) |
| Title text | 3.0 cm | **2.92 cm** | 17pt bold, 13cm wide block (aligned with bar top) |
| Thesis type + program | 3.0 cm | 6.5 cm | 11pt bold |
| University info | 3.0 cm | 8.5 cm | 11pt bold |
| Title image | 3.0 cm | 11.65 cm | 16cm × 9cm |
| Metadata table | 3.0 cm | 21.1 cm | Author, supervisors, deadline, duration |

### Cover page color note

The green bar uses `fapsgruen` (`rgb(151, 193, 57)`), matching the document's
accent color for visual consistency. The title text starts at the same vertical
position as the bar (dy: 2.92cm), aligned with the bar's top edge.

Logo file: `shared/coversheet/LogoFAPS.png` — generated from `LogoFAPS.eps` at 600px.
To regenerate: `sips -s format png -Z 600 LogoFAPS.eps --out LogoFAPS.png`

---

## 10. Abbreviations and Symbols

The Typst system in `shared/formatting/acronyms.typ` provides:

| Function | Description |
|---|---|
| `#ac("KEY")` | First use: "Long Form (Short)"; subsequent: "Short" |
| `#acs("KEY")` | Always short form |
| `#acl("KEY")` | Always long form |
| `#acp("KEY")` | Plural form (first use expanded) |
| `#ac-reset()` | Reset all first-use tracking |
| `#print-abbreviations()` | Print only abbreviations **used** in the document |
| `#print-symbols()` | Print only symbols **used** in the document |

**Auto-filtering:** The lists only include entries that were referenced via `#ac()`, `#acs()`,
`#acl()`, or `#acp()` in the document. Unused entries are automatically excluded.

All acronyms are declared in `shared/formatting/acronyms.typ` under the `acronyms` dictionary.
Tags: `"abbrev"` for abbreviations, `"symbol"` for mathematical symbols.

---

## 11. Mathematics

| Property | LaTeX value | Typst |
|---|---|---|
| Equation numbering | Right-aligned `(1)` | `set math.equation(numbering: "(1)")` |
| Inline math | `$...$` | `$...$` |
| Display math | `\[ ... \]` or `equation` env | `$ ... $` (blank line before/after) |
| Vector bold | `\VEC{x}` → `\mathbf{x}` | `#vecbold(x)` → `math.bold(x)` |
| Vector symbol | `\VECSYM{x}` → `\boldsymbol{x}` | `#vecsym(x)` → `math.bold(math.italic(x))` |
| Unit | `\einheit{N}` | `#einheit("N")` → `h(0.16em) + math.upright("N")` |

---

## 12. Macro Reference

All macros are in `shared/formatting/macros.typ`.

| Typst macro | LaTeX equivalent | Output |
|---|---|---|
| `#todo("text")` | `\TODO{text}` | Red bold text |
| `#evtl("text")` | `\EVTL{text}` | Blue text |
| `#significant("text")` | `\significant{text}` | Bold text |
| `#english("term")` | `\english{term}` | *Italic (engl. term)* |
| `#name("Author")` | `\name{Author}` | SMALL CAPS |
| `#abb(<label>)` | `\abb{label}` | `Figure~@label` |
| `#tab-ref(<label>)` | `\tab{label}` | `Table~@label` |
| `#glg(<label>)` | `\glg{label}` | `Equation~@label` |

---

## 13. Bibliography

Style: **DIN ISO 690 (numeric, alphabetical)** — alphabetically sorted bibliography with numeric `[1]` citations.
File: `shared/bibliography/literature.bib`
CSL:  `shared/bibliography/iso690-numeric-alphabetical.csl`

In Typst:
```typst
#bibliography("../../shared/bibliography/literature.bib", style: "../../shared/bibliography/iso690-numeric-alphabetical.csl")
```

Key properties:
- **Citation format**: Numeric `[1]`, `[2]` in text (square brackets)
- **Bibliography order**: Sorted alphabetically by author last name, then date
- **Standard**: DIN ISO 690:2013-10
- **Custom CSL**: Based on `iso690-numeric-en` from the CSL repository, modified to sort alphabetically instead of by citation order

The previous `iso-690-author-date` style was replaced because the thesis should use numeric
`[1]`-style citations while keeping alphabetical bibliography ordering. No built-in Typst
style combines both; hence a custom CSL file.

Citations: `@key` inline, or `@key[p. 42]` with page numbers.

---

## 14. Plotly Figure Integration

Kinematic notebook plots (antiparallelogram elbow, cable-driven wrist) are exported from
Plotly as SVG and included in the thesis via `image()`.

**Workflow:**
```bash
cd doc/Tensegrity_robot
conda run -n env_isaaclab python ../Theses_typst/tools/export_plotly_figures.py
```

**Export script:** `tools/export_plotly_figures.py`
**Requirements:** `plotly`, `kaleido` (+ Chrome via `kaleido.get_chrome_sync()`)

**Output files** (in `shared/figures/`):
- `antiparallelogram_static.svg` — 5-pose elbow linkage overview
- `wrist_static_3d.svg` — 3D cable-driven wrist mechanism
- `elbow_comparison_static.svg` — antiparallelogram vs disc approximation

To update figures after changing notebook code, re-run the export script and rebuild.

---

## 15. Section Numbering and Depth

- Numbered heading levels: 3 (Chapter / Section / Subsection)
- TOC depth: 3 levels
- Appendices use letter prefixes (A, B, C) — configured per appendix with `#set heading(numbering: "A.1")`

---

## 16. Document Structure Order

1. Cover page (`faps-coversheet`)
2. Declaration (`erklaerung`)
3. Front matter (`faps-frontmatter`): roman page numbers, no header
   - Table of Contents (`faps-toc`)
   - List of Figures (`faps-lof`)
   - List of Tables (`faps-lot`)
   - List of Abbreviations (`print-abbreviations`) — *auto-filtered to used only*
   - List of Symbols (`print-symbols`) — *auto-filtered to used only*
4. Main matter (`faps-header-footer`): arabic page numbers from 1, running headers
   - Chapters 1–7
   - Bibliography
   - Appendices (A, B, C, …)

---

## 17. Compilation Notes

### Typst version
Tested with **Typst 0.14.2** (b33de9de). Install via: `snap install typst`.

### cetz / fletcher compatibility (Typst 0.14+)

The fletcher 0.5.8 package (used for diagrams) depends on cetz 0.3.4 which has a
breaking incompatibility with Typst 0.14+. The `float()` function in cetz 0.3.4
(`src/util.typ:149`) receives a `relative length` type but only accepts
`float|boolean|integer|decimal|ratio|string`.

**Fix:** Patch the cached package at `~/.cache/typst/packages/preview/cetz/0.3.4/src/util.typ`.
Add handling for `relative` type in `resolve-number`:

```typst
#let resolve-number(ctx, num) = {
  return if type(num) == length {
    float(num.to-absolute() / ctx.length)
  } else if type(num) == ratio {
    num
  } else if type(num) == relative {
    // Typst 0.14+: relative length = length + ratio; resolve length part
    float(num.length.to-absolute() / ctx.length)
  } else {
    float(num)
  }
}
```

This patch is needed until fletcher releases a version that uses cetz 0.4+.

### Font fallback

On Linux without Helvetica Neue/Helvetica/Arial, Typst falls back to Nimbus Sans
(from TeX Live). This produces harmless font-family warnings during compilation.
The output is visually identical.

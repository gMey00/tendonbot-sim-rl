// FAPS Macros — Typst equivalents of LaTeX Makros.tex
// ============================================================

#import "colors.typ": *

// Highlight macros
#let todo(content) = text(fill: red, weight: "bold", content)
#let evtl(content) = text(fill: blue, content)
#let significant(content) = text(weight: "bold", content)
#let english(content) = emph[(engl. #content)]

// Name formatting (small caps)
#let name(content) = smallcaps(content)

// Cross-reference helpers
#let glg(label) = [Equation~@#label]
#let abb(label) = [Figure~@#label]
#let tab-ref(label) = [Table~@#label]

// Math macros
#let vecbold(x) = math.bold(x)
#let vecsym(x) = math.bold(math.italic(x))
#let intd = math.upright("d")
#let ind(x) = math.attach(none, b: math.upright(x))
#let einheit(x) = [#h(0.16em)#math.upright(x)]
#let const-math(x) = math.upright(x)

.PHONY: all thesis guideline watch-thesis watch-guideline clean

all: thesis guideline

thesis:
	typst compile thesis/MA-Thesis.typ --root .. thesis/MA-Thesis.pdf

guideline:
	typst compile guideline/MA-Guideline.typ --root .. guideline/MA-Guideline.pdf

watch-thesis:
	typst watch thesis/MA-Thesis.typ --root .. thesis/MA-Thesis.pdf

watch-guideline:
	typst watch guideline/MA-Guideline.typ --root .. guideline/MA-Guideline.pdf

clean:
	rm -f thesis/MA-Thesis.pdf guideline/MA-Guideline.pdf

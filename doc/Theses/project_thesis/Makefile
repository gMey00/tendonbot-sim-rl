.PHONY: all thesis guideline watch-thesis watch-guideline clean

all: thesis guideline

thesis:
	typst compile thesis/PA-Thesis.typ --root .. thesis/PA-Thesis.pdf

guideline:
	typst compile guideline/PA-Guideline.typ --root .. guideline/PA-Guideline.pdf

watch-thesis:
	typst watch thesis/PA-Thesis.typ --root .. thesis/PA-Thesis.pdf

watch-guideline:
	typst watch guideline/PA-Guideline.typ --root .. guideline/PA-Guideline.pdf

clean:
	rm -f thesis/PA-Thesis.pdf guideline/PA-Guideline.pdf

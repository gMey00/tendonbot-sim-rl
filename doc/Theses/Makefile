.PHONY: all pa ma pa-thesis pa-guideline ma-thesis ma-guideline watch-pa watch-ma clean

all: pa ma

pa: pa-thesis pa-guideline
ma: ma-thesis ma-guideline

pa-thesis:
	cd project_thesis && typst compile thesis/PA-Thesis.typ --root .. thesis/PA-Thesis.pdf

pa-guideline:
	cd project_thesis && typst compile guideline/PA-Guideline.typ --root .. guideline/PA-Guideline.pdf

ma-thesis:
	cd master_thesis && typst compile thesis/MA-Thesis.typ --root .. thesis/MA-Thesis.pdf

ma-guideline:
	cd master_thesis && typst compile guideline/MA-Guideline.typ --root .. guideline/MA-Guideline.pdf

watch-pa:
	cd project_thesis && typst watch thesis/PA-Thesis.typ --root .. thesis/PA-Thesis.pdf

watch-ma:
	cd master_thesis && typst watch thesis/MA-Thesis.typ --root .. thesis/MA-Thesis.pdf

clean:
	rm -f project_thesis/thesis/PA-Thesis.pdf project_thesis/guideline/PA-Guideline.pdf
	rm -f master_thesis/thesis/MA-Thesis.pdf master_thesis/guideline/MA-Guideline.pdf

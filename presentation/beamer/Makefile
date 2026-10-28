# Build the TU Dresden Beamer presentation (SPEC 0055 insight-first talk).
#
# Prerequisite: assets and fonts must exist first, built reproducibly from
# the official pptx via:  python3 ../../scripts/prepare_presentation_assets.py
#
# The report figures under ../../report-shorten/figs/ are read-only
# dependencies: the deck reads them in place through \graphicspath and never
# writes to the report tree. They are listed as prerequisites so a changed
# report figure triggers a rebuild.
#
# Targets:
#   make        build main.pdf with LuaLaTeX
#   make clean  remove intermediate files (keeps main.pdf)

.PHONY: all clean

SLIDES := $(wildcard slides/*.tex)
REPORT_FIGURES := $(wildcard ../../report-shorten/figs/*.pdf) \
                  $(wildcard ../../report-shorten/figs/*.png)

all: main.pdf

main.pdf: main.tex beamerthemeTUD.sty $(SLIDES) $(REPORT_FIGURES) \
          $(wildcard assets/*.pdf) $(wildcard assets/*.jpg) \
          $(wildcard assets/*.png) $(wildcard fonts/*.ttf)
	latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -c main.tex

# Build the documentation.
#
#   make html        build into _build/html, failing on any warning
#   make livehtml    serve on http://127.0.0.1:8000 and rebuild on changes
#   make clean       remove the build and the generated API pages
#   make <builder>   any other Sphinx builder, e.g. make linkcheck

SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)" api/_autosummary examples

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) \
		--watch ../src --ignore "*/examples/*" --ignore "*/api/_autosummary/*"

.PHONY: help clean livehtml Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

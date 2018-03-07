# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SPHINXPROJ    = terrainbento
SOURCEDIR     = .
BUILDDIR      = _build
SPHINXEXCLUDE = $(shell find ../terrainbento/* -name tests -or -name examples)

ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) .

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

pre-hook:
	$(SPHINXAPIDOC) -f --module-first -o source/ ../terrainbento $(SPHINXEXCLUDE)

html: pre-hook
		$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
		@echo
		@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

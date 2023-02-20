# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
APIDOCDIR     = apisource
PACKAGENAME   = l2co

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
	
html-open:
	make html
	xdg-open ./$(BUILDDIR)/html/index.html

apidoc:
	-rm -rf ./$(SOURCEDIR)/$(APIDOCDIR)/*
	sphinx-apidoc -o ./$(SOURCEDIR)/$(APIDOCDIR)/ ../src/$(PACKAGENAME)
	@echo
	@echo "Created API documentation using sphinx-apidoc. The source files are in $(APIDOCDIR)."


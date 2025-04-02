# Makefile for Sphinx documentation
#

# Docs variables: You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build
APIDOCDIR     = source
PACKAGENAME   = l2co

# Build package variables
PACKAGEDIR    = dist

.PHONY: help test test-html build upload build-upload panel

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       	to make standalone documentation HTML files"
	@echo "  latexpdf   	to make documentation LaTeX files and run them through pdflatex"
	@echo "  init   	to install the requirements.txt"
	@echo "  init-dev   	to install the requirements_dev.txt"
	@echo "  test   	to run the tests with pytest"
	@echo "  test-html   	to run the tests with pytest and to create a HTML coverage report"
	@echo "  build		to build the package"
	@echo "  upload	to upload the package to the PyPi TEST index"
	@echo "  upload-testpypi	to upload the package to the Test PyPI index"
	@echo "  panel      	to run the panel server for the app"

doc-clean:
	-rm -rf $(BUILDDIR)/*

html:
	cd docs/;
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	cd ..
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

test:
	pytest
	@echo
	@echo "Test finished"

test-smoke:
	pytest -v -s -m smoke
	@echo
	@echo "Smoke test finished"
	
test-smoke-html:
	pytest -v -s -m smoke --cov-report html
	@echo
	@echo "Smoke test finished. The coverage report HTML pages are in ./htmlcov/index.html"
	xdg-open ./htmlcov/index.html
	

test-html:
	pytest --cov-report html
	@echo
	@echo "Test finished. The coverage report HTML pages are in ./htmlcov/index.html"
	xdg-open ./htmlcov/index.html
	
build:
	@echo "Removing previous build"
	-rm -rf $(PACKAGEDIR)/*
	@echo "Building package"
	python -m build
	
upload-testpypi:
	make build
	@echo "Uploading the package to Test PyPI via Twine ..."
	twine upload -r testpypi $(PACKAGEDIR)/* --verbose

upload:
	@echo "Uploading the package to PyPI via Twine ..."
	twine upload $(PACKAGEDIR)/* --verbose

panel:
	panel serve src/${PACKAGENAME}/app.py --dev --show
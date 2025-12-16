.PHONY: help build test lint format docs docs-serve

# Default target
help:
	@echo "Available targets:"
	@echo "  build        - Build the package distribution"
	@echo "  test         - Run tests with pytest"
	@echo "  lint         - Lint code with ruff"
	@echo "  docs         - Build documentation with mkdocs"

# Build targets
build:
	python -m build

# Testing targets
test:
	pytest

# Linting and formatting targets
lint:
	ruff check

# Documentation targets
docs:
	mkdocs build

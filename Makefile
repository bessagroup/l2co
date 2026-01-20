.PHONY: help build test lint format docs

# Default target
help:
	@echo "Available targets:"
	@echo "  build        - Build the package distribution"
	@echo "  test         - Run tests"
	@echo "  lint         - Lint code"
	@echo "  docs         - Build documentation"

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

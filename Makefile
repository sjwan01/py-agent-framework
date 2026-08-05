# py-agent development Makefile
# Usage: make lint / make typecheck / make test / make check / make build ...

PYTHON ?= python
VENV   ?= venv
BIN    := $(VENV)/bin

.PHONY: bootstrap install dev lint typecheck test check build clean publish publish-test

## Create the virtualenv and install everything
bootstrap:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev,postgres]"

## Install the package in editable mode (no extras)
install:
	$(PYTHON) -m pip install -e .

## Install with dev + postgres extras (uses existing env)
dev:
	$(PYTHON) -m pip install -e ".[dev,postgres]"

## Lint with ruff
lint:
	$(BIN)/ruff check py_agent/ tests/

## Type check with mypy (strict)
typecheck:
	$(BIN)/mypy py_agent/

## Run the test suite
test:
	$(BIN)/pytest tests/ -q

## Run all checks: lint + typecheck + test
check: lint typecheck test

## Build sdist + wheel into dist/
build: clean
	$(PYTHON) -m build

## Publish to TestPyPI (dry run before the real release)
publish-test: build
	$(BIN)/twine upload --repository testpypi dist/*

## Publish to PyPI
publish: build
	$(BIN)/twine upload dist/*

## Remove build artifacts and caches
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

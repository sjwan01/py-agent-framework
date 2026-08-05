# Contributing to py-agent

Thanks for wanting to contribute! This file is for **developers**. Users should
read the [README](README.md).

## Setup

Requires **Python ≥ 3.11**.

```bash
git clone git@github.com:sjwan01/py-agent.git
cd py-agent
make bootstrap        # creates venv/ and installs the package with dev + postgres extras
```

If you already have a `venv/`, just refresh it:

```bash
make dev              # pip install -e ".[dev,postgres]"
```

## Make targets

| Target | Runs |
|---|---|
| `make lint` | `ruff check py_agent/ tests/` |
| `make typecheck` | `mypy py_agent/` (strict) |
| `make test` | `pytest tests/ -q` |
| `make check` | lint + typecheck + test (what CI enforces) |
| `make build` | build sdist + wheel |
| `make clean` | remove build artifacts |

Always run `make check` before opening a PR — CI runs the same command.

## Tests

Most tests run with no external services. The Postgres backend tests are
skipped unless `PG_TEST_URL` is set:

```bash
export PG_TEST_URL="postgresql://user:pass@localhost:5432/pyagent_test"
make test
```

In CI, Postgres runs in a `postgres:16` service container, so those tests are
always executed there.

The MCP integration tests spin up an in-process FastMCP server; the needed
dependencies come from the `dev` extra (`fastmcp-slim[server]`).

## Pull requests

1. Open an issue to discuss non-trivial changes before coding.
2. Create a focused branch (e.g. `fix/typo-in-load-history`).
3. Run `make check` locally — CI enforces it, so fix everything it finds.
4. Keep one logical change per PR; write a clear commit message.

## Code style

- Follow the existing style — ruff (`import` order, pyflakes, module docstrings)
  and mypy strict are the contract.
- No Chinese characters in source files.
- New public API must be added to `__all__` in the relevant `__init__.py`.

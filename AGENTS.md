# py_agent — Agent Instructions

## Project

Python agent framework built on [Pydantic AI](https://ai.pydantic.dev/).
Orchestrates multi-turn conversations with context-window management,
tool registration, and session persistence (SQLite / PostgreSQL / single-turn).

## Spec-Driven Minimal Implementation

Implement the smallest solution that fully satisfies the functional requirements
documented in `CONTEXT.md` or the relevant spec. Robustness and completeness of
the required behavior are not optional: handle errors, validate inputs, and cover
the normal path, error path, boundaries, and integration with tests.

Do not add speculative features, extra abstraction layers, or optimizations that
are not required by the spec. If a requirement is unclear, ask for clarification
instead of building a more general solution.

## Build & Verify

```bash
make check    # canonical gate — exactly what CI runs
```

`make check` runs, in order: `ruff check --fix py_agent/ tests/` (import
order, dead code, pyflakes), `mypy py_agent/` (strict, `mypy.ini`), then
`pytest tests/ -q` (`asyncio_mode = auto`). Run the individual commands
from `Makefile` when you need just one step. Always run the full gate after
any change.

If you touch `types.py` (ABCs), also verify that all `SessionManager`
implementations still satisfy the interface.

### Postgres tests

The Postgres backend tests skip locally unless `PG_TEST_URL` is set; CI runs
them in a `postgres:16` service container. When you change
`py_agent/session/_postgres.py`, the real verification is the CI run — do not
treat local skips as a pass.

## Dependencies & Packaging

- Dependencies are declared **only** in `pyproject.toml` (`dependencies`).
  There is no `requirements.txt` — do not recreate one.
- Optional functionality goes in `[project.optional-dependencies]` extras:
  `postgres` (psycopg), `dev` (ruff/mypy/pytest/fastmcp-slim[server]).
- Optional backends must import their third-party deps lazily — module
  import must succeed without them. Pattern (see `session/_postgres.py`):
  type-only imports under `if TYPE_CHECKING:`, runtime import inside the
  method that needs it, with a helpful `ImportError` naming the extra.
- `__version__` lives in `py_agent/__init__.py` (hatchling reads it via
  `[tool.hatch.version]`); bump it there when preparing a release.

---

## Type Safety

### File header — every `.py` without exception

```python
"""One-line module docstring."""
from __future__ import annotations
```

`from __future__ import annotations` goes immediately after the module
docstring. This makes all annotations lazy so forward references work.

### Never bare `list` / `dict`

```python
# ✗
def foo(items: list) -> dict: ...

# ✓
def foo(items: list[Any]) -> dict[str, Any]: ...
```

mypy strict mode rejects bare generics. Always provide type arguments.

### `Any` is a deliberate boundary, not a default

Type hints exist for two reasons: **the type checker** (catch real bugs) and
**the reader** (the signature documents the contract). A precise type that
expresses the contract beats `Any`; `Any` is only acceptable where the type
is genuinely open. When you write `Any`, you should be able to name why.

**Prefer the precise type when one exists:**

```python
# ✗  message lists are not "anything"
def save(messages: list[Any]) -> None: ...

# ✓  pydantic-ai has a real type
from pydantic_ai.messages import ModelMessage
def save(messages: list[ModelMessage]) -> None: ...
```

**Deliberate `Any` is fine — it falls into one of these buckets:**

| Bucket | Example | Why it stays `Any` |
|--------|---------|--------------------|
| SDK callback params | `ctx: Any, call: Any, tool_def: Any` in hooks | pydantic-ai hook types are version-fragile; `Any` marks "SDK passthrough" |
| Generic deps parameters | `Tool[Any]`, `AbstractToolset[Any]`, `RunContext[Any]` | the framework never cares about `AgentDepsT`; `[Any]` is correct instantiation |
| External data boundaries | psycopg rows, event payload `dict[str, Any]`, JSONB dict-or-str | the data is genuinely dynamic at that interface |
| SDK shims | `SimpleNamespace(usage=RunUsage())` | constructing an object the SDK's type system doesn't expose |

**Signals that an `Any` is lazy rather than deliberate:**

- `list[Any]` where a message/model type exists → use it (e.g. `ModelMessage`).
- `self: Any` on class-attribute-bound methods → use `TYPE_CHECKING` +
  the real class (no circular import at runtime, `from __future__ import
  annotations` makes the annotation lazy).
- A parameter with a known contract typed `Any` (e.g. callbacks, managers)
  → give it a `Callable[...]` / `Protocol` / concrete type.
- `Any` on an internal list or dict whose elements are all one type
  → name the element type.

If you cannot name the reason for an `Any`, find a more precise type. If you
find a lazy `Any` during review, flag it — it hides real bugs and gives the
reader no contract.

### mypy configuration (`mypy.ini`)

```ini
[mypy]
strict = True
warn_unused_ignores = True
warn_return_any = True
warn_unreachable = True
```

Do not relax these without a documented reason. Prefer fixing the type
over adding `# type: ignore`.

---

## Docstrings — Google Style, English Only

### Format

```python
class Something:
    """One-line summary of what this class is.

    When to use it, what problem it solves. Optional second paragraph.

    Args:
        param1: What it controls. Defaults to X.
        param2: What it controls. If None, behavior is Y.

    Attributes:
        attr1: Description (for Pydantic BaseModel fields / public attrs).
    """

    def method(self, x: int, *, flag: bool = False) -> str:
        """One-line summary.

        Args:
            x: Description.
            flag: Description. Defaults to False.

        Returns:
            Description of return value.

        Raises:
            SomeError: When this is raised.
        """
```

### Rules

1. **Types live in the signature, not in the docstring.**
   ```python
   # ✓  describe semantics only — types live in the signature
   #    model: The Pydantic AI model to use for the agent. Required.

   # ✗  do not write types in Args
   #    model (Model): The Pydantic AI model to use.
   ```

2. **Pydantic `BaseModel` fields** → document under `Attributes:` in the class
   docstring. Reference: `py_agent/models.py`.

3. **`__init__` parameters** → document under `Args:` in the class docstring.

4. **Methods** → own docstring with `Args:` / `Returns:` / `Raises:`.

5. **Every public class and method** gets a docstring. Internal helpers
   (functions not in `__all__`, or inside `_`-prefixed modules) can be lighter.

6. **English only.** No non-English characters in source files. Verify with:
   ```bash
   rg '[\x{4e00}-\x{9fff}]' py_agent/
   ```

---

## Project Structure & API Boundaries

```
py_agent/
├── __init__.py           # public exports only + __version__ (dynamic version source)
├── types.py              # ABCs, Protocols, enums — zero implementation
├── models.py             # Pydantic data models
├── _context.py           # _prepare_context (watermark truncation, pure function)
├── _compaction.py        # HarnessSummarizer wrapper
├── runner/               # Agent lifecycle orchestration
│   ├── __init__.py       # re-exports AgentRunner
│   ├── _agent.py         # AgentRunner class
│   ├── _factory.py       # tool lifecycle init, compaction trigger
│   ├── _hooks.py         # Pydantic AI Hooks construction
│   └── _internals.py     # fire, notify, drain, get_tools, etc.
└── session/              # Persistence backends
    ├── __init__.py        # re-exports all managers
    ├── _local.py          # SQLite
    ├── _postgres.py       # PostgreSQL
    ├── _shared.py         # role inference, turn detection, serialization
    └── _single_turn.py    # no-persistence fallback
```

### Rules

1. **`_`-prefixed modules are private.** External code imports only from
   `py_agent` or `py_agent.session` (via `__init__.py`). Never import
   `py_agent.runner._agent` or similar directly.

2. **`__all__` in every `__init__.py`** lists the public API surface.
   When adding a new public class, update `__all__`.

3. **`types.py` stays zero-implementation.** ABCs, Protocols, and enums only.
   Move any logic to the appropriate implementation module.

4. **Separation of concerns:**
   - Data shapes → `models.py`
   - Context management → `_context.py`
   - Session I/O → `session/`
   - Orchestration → `runner/`

### `types.py` vs `models.py` — the critical boundary

These two files define WHAT things are. Everything that is a *type definition*
must live in exactly one of them — never scattered elsewhere.

#### `types.py` — Behavioral contracts (zero implementation)

| Goes here | Example from this project |
|-----------|--------------------------|
| `ABC` (abstract base class) | `SessionManager` |
| `Protocol` (structural interface) | `Extension` |
| `StrEnum` / `Enum` | `AgentRunnerEvent`, `MessageRole` |

Rules:
- No implementation code. No method bodies, no logic, no imports of
  third-party SDKs beyond what the signatures need.
- Exception: **optional Protocol methods** may carry a default body that
  declares their default behavior as part of the interface — e.g.
  `register_capabilities` returning `[]` ("contributes nothing") or
  `on_agent_runner_event_stream`'s unreachable generator. This is interface
  declaration, not business logic; anything with real behavior belongs in
  the implementation modules.
- Every new ABC/Protocol/enum/alias you create belongs here. If you find
  one defined in another file, move it here.

#### `models.py` — Data shapes (Pydantic `BaseModel`)

| Goes here | Example from this project |
|-----------|--------------------------|
| `BaseModel` subclass | `RunResult`, `ContextConfig`, `SummarizerConfig` |

Rules:
- **Every `BaseModel` goes in `models.py`.** No exceptions. If a model is
  "tightly coupled" to a module, that's a sign the module should import
  from `models.py`, not a reason to define the model locally.
- Models may have default values, `model_config`, and `@model_validator`.
- Even internal models (not in `__all__`) live here.

## Naming Conventions

| What | Convention | Example |
|------|-----------|---------|
| Public classes | PascalCase | `AgentRunner`, `LocalSessionManager` |
| Private instance attrs | `_` prefix | `self._model`, `self._protect_turns` |
| Private modules | `_` prefix | `_agent.py`, `_hooks.py` |
| Functions in private modules | `_` prefix | `_compute_diff`, `_inject_diff` |
| "Private" methods on public classes | `_` prefix | `_setup_run`, `_build_agent` |
| Constants / enums | UPPER_SNAKE_CASE | `AgentRunnerEvent.TOOL_START` |
| Test files | `test_<module>.py` | `test_compaction.py` |
| Test classes | `Test<Feature>` | `TestCompactionLoading` |

---

## Architecture Patterns

### Design decision rule of thumb

When choosing between implementation options, evaluate in this order:

1. **Semantic correctness first** — the option whose model of the problem is
   actually correct. A column that faithfully classifies its data beats a
   query that works around a mislabeled column.
2. **Natively correct over hack** — prefer the platform's native mechanisms
   (schema columns, indexes, types) over workarounds (parsing serialized
   blobs, magic offsets). A hack may be "correct today" but breaks when the
   underlying format changes.
3. **Performance last** — only after 1 and 2 are satisfied, pick the option
   with the best performance. Never trade semantic correctness or a native
   design for speed.

### ABC + multiple implementations

`SessionManager` (ABC in `types.py`) → `SingleTurnSessionManager`,
`LocalSessionManager`, `PostgresSessionManager`. New backends implement
the same ABC and are registered in `session/__init__.py`.

### Event-driven lifecycle

Events are `StrEnum` values (`AgentRunnerEvent`).
Extensions subscribe via `Extension` Protocol methods. Two dispatch modes:

- **Chain mode** (`_fire`): each extension sees the previous one's modifications.
  Used for message mutation (`BEFORE_AGENT_RUN`, `SESSION_SAVE`).
- **Notify mode** (`_fire_notify`): every extension gets the same snapshot.
  Used for cancellation votes (`COMPACTION_TRIGGER`).

### Class-attribute binding

Module-level functions in `_internals.py` / `_hooks.py` / `_factory.py` take
`self` as their first parameter and are bound as class attributes on
`AgentRunner`:

```python
class AgentRunner:
    _fire = _internals.fire
    _build_hooks = _hooks.build_hooks
```

This keeps files small without multiple inheritance. When adding a new
internal helper, follow the same pattern — put the function in the appropriate
`_*.py` module and bind it in `_agent.py`.

---

## Error Handling

### Fail-open for non-critical paths

```python
try:
    result = await extension.do_thing()
except Exception as exc:  # pragma: no cover - fail-open
    self._on_warning(f"Extension {name} failed: {exc}", exc)
```

Compaction, extension hooks, and streaming handlers must not crash the main
agent loop. Catch broadly, warn, continue.

### Fail-fast for invalid config

```python
if max_tool_calls_per_turn <= 0:
    raise ValueError(
        f"max_tool_calls_per_turn must be > 0, got {max_tool_calls_per_turn}"
    )
```

Constructor arguments that are logically invalid raise immediately.

### `# pragma: no cover`

Use sparingly — only for genuinely untestable code (fail-open exception
handlers, unreachable `if False:` branches in protocol generators).
Always add a comment explaining why: `# pragma: no cover - fail-open`.

---

## Code Style Details

### Numeric literals — underscore separator

```python
context_window: int = 128_000     # ✓
truncate_chars: int = 1_000       # ✓
max_tokens: int = 32_768          # ✓
```

### `None` semantics

`None` consistently means "disabled / use default". Every config model follows
this: `context_config` is `None` → single-turn runs have no context
management; `SummarizerConfig.model` is `None` → reuse main model.

### Boolean checks — explicit comparison, especially `True` / `False` / `None`

Two kinds of boolean checks exist and must not be mixed:

- **Contract booleans** — values whose documented contract is a literal
  `True` / `False`, e.g. `{block: true}` / `{cancel: true}` returned by
  extensions. Compare strictly with `is True` / `is False`. Never rely on
  truthiness: `1`, `"yes"`, and other truthy values are NOT the documented
  contract and must not trigger it (the writer is external and not under our
  control).

  ```python
  # ✓  only an explicit True blocks
  if call_result.get("block") is True:
  # ✗  {"block": 1} or {"block": "yes"} would also trigger this
  if call_result.get("block"):
  ```

- **Emptiness checks** — checking for "has content" on lists, strings, or
  dicts that we own. Truthiness is acceptable for these, but when the check
  tests a *specific value*, write the comparison explicitly — the reader
  should not have to know Python truthiness rules.

  ```python
  # ✓  non-empty list
  if messages:
  # ✓  empty/blank string
  if not active_sp:
  # ✓  None is always an explicit comparison
  if frozen_sp is None:
  ```

Rules:

1. `None` checks are always `is None` / `is not None`, never truthiness.
2. When a check tests a specific value (`True`, `False`, `None`, a literal),
   write it explicitly. If it can be written explicitly, it should be.
3. Contract booleans received from extensions (or any external input) must
   use strict identity comparison (`is True` / `is False`).

### Pydantic v2 `model_config`

When a model field holds a non-Pydantic type (like `pydantic_ai.models.Model`):

```python
class SummarizerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Model | None = None
```

### Imports order

Standard library → third-party → first-party, each group separated by a blank
line. Within each group, `import X` before `from X import Y`.

---

## Testing

### Config (`pytest.ini`)

```ini
[pytest]
addopts = -p no:logfire
asyncio_mode = auto
```

### Conventions

- Test files mirror source: `test_validation.py` covers `ContextConfig` +
  `AgentRunner` constructor invariants.
- Test classes group related scenarios: `TestCompactionLoading`,
  `TestContextConfigValidation`.
- Every test method has a one-line docstring.
- Fixtures are defined with docstrings and type hints.
- Cover: (1) valid-input normal path, (2) invalid-input raises, (3) boundary
  conditions, (4) integration between components.

---

## Release

Publishing is split — trial is automatic, production is manual-only:

- **Trial (TestPyPI):** any push/merge to `main` runs the publish workflow
  and uploads to TestPyPI automatically (`TEST_PYPI_API_TOKEN` secret;
  skipped when the secret is unset). This is the dry run.
- **Production (PyPI):** manual only — Actions → Publish to PyPI →
  Run workflow. Never triggered by tags or pushes.

The workflow runs `make check` first and refuses to publish on failure.
Version lives in `py_agent/__init__.py` — bump it there before a release.

---

## What NOT to Do

- ❌ Non-English characters in source code
- ❌ Bare `list` / `dict` type annotations
- ❌ Direct imports from `_`-prefixed modules outside the package
- ❌ Skipping `from __future__ import annotations`
- ❌ Relaxing mypy strictness without a documented reason
- ❌ Letting extension failures crash the agent loop
- ❌ Adding public API without an `__all__` entry
- ❌ Truthy checks on contract booleans: `if result.get("block"):` — truthy
  values like `1` / `"yes"` would trigger behavior outside the documented
  `{block: true}` contract; use `is True`
- ❌ Recreating `requirements.txt` — dependencies live in `pyproject.toml` only
- ❌ Implementation logic in `types.py`

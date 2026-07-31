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
ruff check --fix py_agent/ tests/    # import order, dead code, pyflakes
mypy py_agent/                       # strict mode (mypy.ini)
pytest tests/ -q                     # asyncio_mode = auto
```

Always run all three after any change. Order matters: ruff first (auto-fix),
then mypy (type-check), then pytest (correctness). If you touch `types.py`
(ABCs), also verify that all `SessionManager` implementations still satisfy
the interface.

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
   # ✓  Args: 里只描述语义
   #    model: The Pydantic AI model to use for the agent. Required.

   # ✗  不要在 Args 里写类型
   #    model (Model): The Pydantic AI model to use.
   ```

2. **Pydantic `BaseModel` fields** → document under `Attributes:` in the class
   docstring. Reference: `py_agent/models.py`.

3. **`__init__` parameters** → document under `Args:` in the class docstring.

4. **Methods** → own docstring with `Args:` / `Returns:` / `Raises:`.

5. **Every public class and method** gets a docstring. Internal helpers
   (functions not in `__all__`, or inside `_`-prefixed modules) can be lighter.

6. **English only.** No Chinese characters in source files. Verify with:
   ```bash
   rg '[\x{4e00}-\x{9fff}]' py_agent/
   ```

---

## Project Structure & API Boundaries

```
py_agent/
├── __init__.py           # public exports only (AgentRunner, configs)
├── types.py              # ABCs, Protocols, enums — zero implementation
├── models.py             # Pydantic data models
├── tools.py              # ToolSource implementations + ToolLifecycle
├── context.py            # ContextManager (watermark truncation)
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
   - Tool registration → `tools.py`
   - Context management → `context.py`
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
| `StrEnum` / `Enum` | `AgentRunnerEvent`, `ToolLifecycleEvent`, `MessageRole` |
| Type alias (`SomeName = Callable[[...], ...]`) | `ToolEventHandler` |

Rules:
- No implementation code. No method bodies, no logic, no imports of
  third-party SDKs beyond what the signatures need.
- Every new ABC/Protocol/enum/alias you create belongs here. If you find
  one defined in another file, move it here.

#### `models.py` — Data shapes (Pydantic `BaseModel`)

| Goes here | Example from this project |
|-----------|--------------------------|
| `BaseModel` subclass | `RunResult`, `ContextManagerConfig`, `SummarizerConfig`, `BaselineState` |

Rules:
- **Every `BaseModel` goes in `models.py`.** No exceptions. If a model is
  "tightly coupled" to a module, that's a sign the module should import
  from `models.py`, not a reason to define the model locally.
- Models may have default values, `model_config`, and `@model_validator`.
- Even internal models (not in `__all__`) live here.

#### Current violations

| File | What | Problem |
|------|------|---------|
| `context.py` | `PreparedContext(BaseModel)` | BaseModel → should be in `models.py` |
| `context.py` | `ContextConfig(BaseModel)` | BaseModel → should be in `models.py` |

These are temporary. When touching `context.py`, move them to `models.py`.

---

## Naming Conventions

| What | Convention | Example |
|------|-----------|---------|
| Public classes | PascalCase | `AgentRunner`, `ContextManager` |
| Private instance attrs | `_` prefix | `self._model`, `self._protect_turns` |
| Private modules | `_` prefix | `_agent.py`, `_hooks.py` |
| Functions in private modules | `_` prefix | `_compute_diff`, `_inject_diff` |
| "Private" methods on public classes | `_` prefix | `_setup_run`, `_build_agent` |
| Constants / enums | UPPER_SNAKE_CASE | `AgentRunnerEvent.TOOL_START` |
| Test files | `test_<module>.py` | `test_compaction.py` |
| Test classes | `Test<Feature>` | `TestCompactionLoading` |

---

## Architecture Patterns

### ABC + multiple implementations

`SessionManager` (ABC in `types.py`) → `SingleTurnSessionManager`,
`LocalSessionManager`, `PostgresSessionManager`. New backends implement
the same ABC and are registered in `session/__init__.py`.

### Event-driven lifecycle

Events are `StrEnum` values (`AgentRunnerEvent`, `ToolLifecycleEvent`).
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
this: `ContextManagerConfig` is `None` → no context management;
`SummarizerConfig.model` is `None` → reuse main model.

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

- Test files mirror source: `test_validation.py` covers `ContextManager` +
  `AgentRunner` constructor invariants.
- Test classes group related scenarios: `TestCompactionLoading`,
  `TestContextManagerValidation`.
- Every test method has a one-line docstring.
- Fixtures are defined with docstrings and type hints.
- Cover: (1) valid-input normal path, (2) invalid-input raises, (3) boundary
  conditions, (4) integration between components.

---

## What NOT to Do

- ❌ Chinese characters in source code
- ❌ Bare `list` / `dict` type annotations
- ❌ Direct imports from `_`-prefixed modules outside the package
- ❌ Skipping `from __future__ import annotations`
- ❌ Relaxing mypy strictness without a documented reason
- ❌ Letting extension failures crash the agent loop
- ❌ Adding public API without an `__all__` entry
- ❌ Implementation logic in `types.py`

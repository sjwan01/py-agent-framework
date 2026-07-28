# SPEC: Public API Docstrings — Hover-Over Quality

## Scope

Only these public surfaces. Skip everything else — internals don't need thorough docs.

```
py_agent                        # top-level
py_agent.types                  # ABCs, Protocols, enums
py_agent.models                 # config & result models
py_agent.tools                  # tool registry
py_agent.session                # session backends
```

## Target

A developer who knows Pydantic AI but has never seen this framework. They `import py_agent`, hover over a class in their IDE, and immediately understand:

- What this class **is** (one sentence)
- When they would **use** it
- Every constructor parameter: **what it controls**, its **default**, what happens if they **omit** it
- Every public method: **what it does**, its **return value**
- Don't explain Pydantic AI concepts — assume they know `Model`, `Agent`, `Tool`, `Hooks`
- Don't explain internal mechanics — only what they need to customize

## Required Format

Google-style, consistent throughout. No exceptions.

```python
class Something:
    """One-line summary of what this class is for.

    When to use it, what problem it solves. Optional second paragraph
    if context is needed.

    Args:
        param1: What param1 controls. Defaults to X.
        param2: What param2 controls. If None, behavior is Y.

    Attributes:
        attr1: Public attribute description (if any public attrs exist).
    """

    def method(self, x: int, *, flag: bool = False) -> str:
        """One-line summary of what this method does.

        Args:
            x: Description.
            flag: Description. Defaults to False.

        Returns:
            Description of return value.

        Raises:
            SomeError: When raised.
        """
```

### Rules

- **Class docstring**: one-line summary. If the class has a constructor, list `Args:` inside the class docstring unless the `__init__` itself has a docstring. Either way, every parameter must be documented.
- **Pydantic `BaseModel` fields**: document in the class docstring under `Attributes:`.
- **`__init__` parameters**: document under `Args:` in the class docstring.
- **Methods**: document in the method docstring with `Args:`, `Returns:`, `Raises:`.
- **Type in parentheses**: `param_name (str):` NOT `param_name: (str)`. Wait, actually Google style does NOT put types in the description. Types are in the signature. The description just describes. Example:

  ```python
  Args:
      model: The Pydantic AI model to use for the agent. Required.
      system_prompt: System-level instructions injected every turn. Defaults to "".
  ```
  
  No `(Model)` or `(str)` in the description — the type is in the signature.

## File-by-File Spec

### 1. `py_agent/__init__.py`
No changes needed.

---

### 2. `py_agent/types.py` — Where developers go to understand the extension system

| Class | What docstring must cover |
|-------|--------------------------|
| `SessionManager` (ABC) | Class: "Interface that all session backends must implement." Then `Attributes:` listing each abstract method: `create_session` (returns new session id), `load_history` (loads messages with optional compaction-aware protect_turns), `save_messages` (persists a batch of messages), `apply_compaction` (stores a compaction summary at a boundary), `get_max_message_seq` (returns highest seq or -1), `ensure_session` (creates session row if missing). One sentence each. |
| `Extension` (Protocol) | Class: "Protocol for user-provided extensions that hook into the agent lifecycle." `Attributes:` — `register_tool_sources` (called at init; returns list of ToolSource objects discovered by this extension), `on_tool_event` (called during tool registration; receives lifecycle events like TOOL_CONFLICT; return dict to resolve), `on_agent_runner_event` (called during run; receives AgentRunnerEvents; return dict to modify data), `on_agent_runner_event_stream` (optional async generator for streaming extensions). |
| `ToolSource` (ABC) | Class: "Interface for discovering tools from a source." `Attributes:` — `discover` (returns list of Pydantic AI Tools), `source_type` (returns "local"/"mcp"/"subagent"), `source_id` (unique identifier string), `scope` (visibility: "all", "main", or "subagent"; defaults to "all"). |
| `MessageRole` (StrEnum) | StrEnum values: USER, ASSISTANT, TOOL, UNKNOWN. Used in DB schema. |
| `ToolLifecycleEvent` (StrEnum) | Events during tool registration: TOOL_DISCOVERED (new tool found), TOOL_CONFLICT (name collision), TOOL_REGISTERED (accepted after resolution), TOOL_REMOVED (rejected). |
| `AgentRunnerEvent` (StrEnum) | Events during a run. Keep the ASCII tree diagram in the module body, already translated. |
| `ToolEventHandler` | "Type alias for tool lifecycle event handlers: Callable[[str, dict], Awaitable[dict | None]]." |

---

### 3. `py_agent/models.py` — Configuration and result types

| Class | What docstring must cover |
|-------|--------------------------|
| `RunResult` | Class: "Return value of AgentRunner.run()." `Attributes:` — `output` (final text from the agent), `session_id` (persists across turns; pass to next run()), `new_messages` (messages added this turn, persisted to DB), `usage` (Pydantic AI usage stats: input/output tokens, tool calls). |
| `ContextManagerConfig` | Class: "Configuration for automatic context window management." `Attributes:` — `context_window` (token budget, default 128000), `low_watermark_ratio` (start truncating old tool results when usage exceeds this fraction of context_window, default 0.6), `high_watermark_ratio` (flag for async compaction when usage exceeds this fraction, default 0.75), `protect_turns` (keep this many most recent turns intact, default 5), `truncate_tool_result_chars` (max characters per tool result after truncation, default 1000). Pass to AgentRunner; None skips context management. |
| `SummarizerConfig` | Class: "Configuration for LLM-powered context compaction." `Attributes:` — `model` (model for summarization; None falls back to the main agent model), `max_output_tokens` (max tokens for the summary; None computes min(32768, max(context_window*0.1, 8192))), `summary_prompt` (custom prompt template; None uses the built-in harness default with 6 sections). Pass to AgentRunner; None disables compaction. |

---

### 4. `py_agent/tools.py` — Tool registration for extension authors

| Class | What docstring must cover |
|-------|--------------------------|
| `LocalToolSource` | Class: "Wraps raw Python functions as Pydantic AI Tools." `Args:` — `tools` (list of callables or Pydantic Tool objects), `scope` (visibility; defaults to "all"). |
| `MCPServerSource` | Class: "Wraps an MCP server client for tool discovery." `Args:` — `server_name` (identifier for this server), `client_factory` (callable that returns an MCP client; called lazily on first discover()), `scope` (defaults to "all"). |
| `SubagentToolSource` | Class: "Wraps a callable or LangGraph graph as a single tool." `Args:` — `name` (tool name exposed to the agent), `runnable` (async callable or object with `ainvoke`), `description` (tool description; auto-generated if None), `scope` (defaults to "subagent"). |
| `ToolLifecycle` | Class: "Central tool registry with conflict resolution." `Args:` — `on_warning` (callback for non-fatal errors). Methods: `on(event, handler)` subscribes a handler to a lifecycle event. `add_source(source)` discovers tools from a ToolSource, fires events, resolves conflicts. `get_for_scope(scope)` returns tools filtered by scope. Built-in dedup: local tools win over MCP tools. |

---

### 5. `py_agent/session` — Session persistence backends

(Top-level docs in `session/__init__.py` are fine. Focus on the three classes.)

| Class | What docstring must cover |
|-------|--------------------------|
| `SingleTurnSessionManager` | Class: "No-persistence backend. Every run() is a fresh session." Implements SessionManager. load_history returns [], save_messages is no-op, apply_compaction raises NotImplementedError. |
| `LocalSessionManager` | Class: "SQLite-backed multi-turn session." `Args:` — `db_path` (path to SQLite file). Creates tables on first use. Supports compaction with boundary_seq logic. |
| `PostgresSessionManager` | Class: "PostgreSQL-backed multi-turn session with connection pooling." `Args:` — `pg_url` (PostgreSQL connection URL), `pool_size` (min connections, default 5), `max_overflow` (extra connections beyond pool_size, default 10). Lazily creates pool on first use. `close()` explicitly shuts down the pool. |

---

## Verification

```bash
pytest tests/ -q          # 17 passed
rg '[\x{4e00}-\x{9fff}]' py_agent/  # zero matches
```

## Do Not Touch

- `py_agent/_compaction.py`
- `py_agent/context.py`
- `py_agent/runner/` (all `_*` modules)
- `py_agent/session/_shared.py`
- `py_agent/session/_local.py` internals (just fix the class docstring)
- `py_agent/session/_postgres.py` internals (just fix the class docstring)
- Any function signatures, logic, imports, `__all__`, `# pragma: no cover`

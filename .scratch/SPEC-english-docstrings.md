# SPEC: Rewrite All Docstrings & Comments in English

## What to Do

Replace **every Chinese character** in every `.py` file under `py_agent/` with professional,
developer-friendly English. No Chinese left behind.

## Style: Google-style Docstrings

```python
class Something:
    """Single-line summary of what this class does."""

    def method(self, x: int, *, flag: bool = False) -> str:
        """Single-line summary of what this method does.

        Additional detail paragraph if the behavior is non-obvious.

        Args:
            x: What x represents.
            flag: What flag controls. Defaults to False.

        Returns:
            Description of the return value.

        Raises:
            SomeError: When this happens.
        """
```

### Rules

- **First line**: imperative mood, one sentence, no trailing period unless multi-sentence.
- **Blank line** after summary before details/Args.
- **`Args:`** — every parameter, type in `()`, colon, description. Indent continuation lines by 4 spaces.
- **`Returns:` / `Yields:` / `Raises:`** — only when the function actually returns/yields/raises something noteworthy.
- **Module docstring** (first line of file): one-line summary.
- **Class docstring**: one-line summary. Add details paragraph if the class has complex behavior.
- **Inline comments**: keep them short, lowercase, no punctuation. Delete comments that just repeat the code. Translate comments that explain *why* (algorithm choices, edge cases).
- **No emoji, no ASCII art dividers** like `──`. Keep the AgentRunnerEvent lifecycle tree diagram but translate it.
- **Preserve** `# pragma: no cover` markers exactly as-is.
- **Preserve** `from __future__ import annotations` exactly as-is.
- **Preserve** `__all__` exports exactly as-is.

## What NOT to Change

- Function/class **signatures** (names, parameters, types, defaults)
- **Logic** — conditionals, loops, arithmetic, SQL, string templates
- **Import statements**
- **`from __future__ import annotations`**
- **`__all__`**
- **`# pragma: no cover`**

## Before/After Examples

### Before (Chinese):
```python
"""AgentRunner — 框架的核心编排器。

入口：run(prompt) 和 run_stream(prompt)。两个方法共享完全相同的
生命周期流程，唯一的区别是 run() 把结果打包成一个 RunResult 返回，
run_stream() 把事件逐个 yield 给外部消费者。
"""
```

### After (English):
```python
"""Orchestrator that manages the full agent lifecycle.

Two entry points share the same internal pipeline:
``run()`` returns a ``RunResult``; ``run_stream()`` yields events
one at a time for streaming consumers.
"""
```

### Before (Chinese inline):
```python
# 超限：不执行工具，返回一句说明给模型看
if tool_calls > max_tool_calls:
    return f"Tool call limit ({max_tool_calls}) reached for this turn."
```

### After (English inline):
```python
# Enforce per-turn tool call limit.
if tool_calls > max_tool_calls:
    return f"Tool call limit ({max_tool_calls}) reached for this turn."
```

## File-by-File Context

Each entry gives you the module's role so you can write accurate docstrings
without deep-reading every line. Do NOT change the code.

---

### 1. `py_agent/__init__.py`
No changes — already English.

---

### 2. `py_agent/types.py` — Public ABCs, Protocols, enums

| Class | Role |
|-------|------|
| `SessionManager` | ABC that all session backends implement. 6 abstract methods: create/ensure session, load/save messages, compaction apply, max seq. Implemented by `SingleTurnSessionManager` (no-op), `LocalSessionManager` (SQLite), `PostgresSessionManager` (PG). |
| `Extension` | Protocol for user-provided extensions. Three hooks: `register_tool_sources` returns tool sources discovered at init time; `on_tool_event` intercepts tool registration events (conflict resolution); `on_agent_runner_event` intercepts runtime events in chain mode; `on_agent_runner_event_stream` is an optional async generator for streaming. |
| `ToolSource` | ABC for tool discovery. `discover()` returns a list of Pydantic AI `Tool` objects. `source_type` returns "local"/"mcp"/"subagent". `source_id` is a unique string. `scope` controls visibility ("all", "main", "subagent"). |
| `MessageRole` | StrEnum: USER, ASSISTANT, TOOL, UNKNOWN. Used in DB schema and `_infer_role`. |
| `ToolLifecycleEvent` | Events fired during tool registration: TOOL_DISCOVERED, TOOL_CONFLICT, TOOL_REGISTERED, TOOL_REMOVED. |
| `AgentRunnerEvent` | Events fired during a `run()` call. Has an ASCII art lifecycle diagram in the module body — translate it to English, keep the tree structure. |
| `ToolEventHandler` | Type alias: `Callable[[str, dict], Awaitable[dict | None]]`. |

---

### 3. `py_agent/models.py` — Public Pydantic config/result models

| Class | Role |
|-------|------|
| `RunResult` | Return value of `AgentRunner.run()`. Fields: `output` (text), `session_id` (for multi-turn), `new_messages` (delta persisted to DB), `usage` (token stats). |
| `ContextManagerConfig` | Configuration for context truncation/compaction. `context_window` (token cap), `low_watermark_ratio` (trigger truncation), `high_watermark_ratio` (trigger compaction), `protect_turns` (recent turns kept intact), `truncate_tool_result_chars` (max chars per tool result). Passed to `AgentRunner`; `None` means skip context management entirely. |
| `SummarizerConfig` | Configuration for LLM compaction. `model` — None falls back to the main agent model. `max_output_tokens` — None computes `min(32768, max(context_window * 0.1, 8192))`. `summary_prompt` — None uses the built-in harness default (6-section format). Passed to `AgentRunner`; `None` means no compaction. |

---

### 4. `py_agent/_compaction.py` — Internal, wraps `SummarizingCompaction`

| Class | Role |
|-------|------|
| `HarnessSummarizer` | Created internally by `AgentRunner`, never by users. Wraps `pydantic_ai_harness.compaction.SummarizingCompaction`. Configured to keep 0 messages, preserve no first user message — total replacement mode. `summarize(messages)` returns a single summary string (empty string on failure). |

---

### 5. `py_agent/context.py` — Context window management

| Class/Function | Role |
|----------------|------|
| `PreparedContext` | Output of `ContextManager.prepare()`. `messages` (list for `Agent.message_history`), `needs_compaction` (bool), `tokens_used` (estimate via chars/4). |
| `BaselineState` | Snapshot of skills, tools, and context at freeze time. Used by `_compute_diff` to inject "[System config changed]" messages. |
| `ContextManager` | Watermark-based truncation. `prepare()` freezes the system prompt as baseline on first call, then diffs subsequent states. If total tokens exceed the low watermark, truncates old tool results. Sets `needs_compaction` when over the high watermark. |
| `_estimate_and_find_boundary` | Single-pass traversal: counts total chars and finds the N-th user turn boundary from the end. |
| `_truncate_and_estimate` | Truncates `ToolReturnPart` content older than boundary, re-estimates tokens. |
| `_compute_diff` | Computes a human-readable diff between two `BaselineState` snapshots. |
| `_inject_diff` | Inserts a "[System config changed]" message before the most recent user turn. |

---

### 6. `py_agent/tools.py` — Public tool registry for extension authors

| Class | Role |
|-------|------|
| `LocalToolSource` | Wraps raw Python functions/callables as Pydantic AI `Tool` objects. `scope` defaults to "all". |
| `MCPServerSource` | Wraps an MCP client factory. Lazily connects on first `discover()`. `scope` defaults to "all". |
| `SubagentToolSource` | Wraps a callable or LangGraph runnable. Detects `ainvoke` presence for LangGraph compatibility. `scope` defaults to "subagent". |
| `ToolLifecycle` | Central tool registry with conflict resolution. `on()` subscribes handlers to lifecycle events. `add_source()` discovers tools, fires TOOL_DISCOVERED → TOOL_CONFLICT → TOOL_REGISTERED/TOOL_REMOVED. Built-in dedup: local tools beat MCP tools. `get_for_scope()` filters by scope. |

---

### 7. `py_agent/runner/__init__.py`
Minor: the module-level Chinese comment should become English. Export is `AgentRunner`.

---

### 8. `py_agent/runner/_agent.py` — AgentRunner, the main entry point

The public API. `AgentRunner` orchestrates: load history → context prepare → extension hooks → Pydantic AI agent run → save messages → async compaction.

- **Module docstring**: the ASCII lifecycle tree — translate to English, keep the tree shape.
- **`__init__`**: Document every parameter. `model` is the only required one. `system_prompt` defaults to `""`. `session_manager` defaults to `SingleTurnSessionManager`. `context_manager_config=None` skips context management. `summarizer_config=None` skips compaction. `thinking_enabled`/`thinking_level` control model thinking mode. `max_tool_calls_per_turn` limits tool invocations. `parallel_tool_calls` enables parallel execution. `hooks`/`capabilities` are forwarded to Pydantic AI's `Agent`. `on_warning` is a callback for non-fatal errors.
- **`_setup_run`**: 6-step internal pipeline. Docstring explains the sequence.
- **`_build_agent`**: Assembles Pydantic AI Agent with hooks, capabilities, model settings.
- **`_finalize_run`**: Post-run: fire events, save delta, trigger compaction, yield `run_end`.
- **`run()`**: Public synchronous-feeling API. Internally consumes the agent's stream, returns `RunResult`.
- **`run_stream()`**: Public async generator. Yields lifecycle events + token chunks.

---

### 9. `py_agent/runner/_factory.py` — Tool lifecycle init & compaction trigger

| Function | Role |
|----------|------|
| `ensure_tool_lifecycle` | Lazy-creates a `ToolLifecycle`, subscribes extension handlers, registers raw tools + extension-discovered sources. Called once, then cached. |
| `trigger_compaction` | Runs in background (`asyncio.create_task`). Gets max message_seq as boundary, loads full history, calls `summarizer.summarize()`, writes compaction record. Skips if summarizer is None. Fail-open. |

---

### 10. `py_agent/runner/_hooks.py` — Pydantic AI hook construction

| Function | Role |
|----------|------|
| `build_hooks` | Builds a `pydantic_ai.capabilities.Hooks` instance with three stages: `before_tool_execute` fires TOOL_START; `tool_execute` enforces per-turn tool call limit, fires TOOL_CALL (extensions can block or modify args); `after_tool_execute` fires TOOL_RESULT (extensions can modify result) then TOOL_END. |

---

### 11. `py_agent/runner/_internals.py` — Runtime helpers

| Function | Role |
|----------|------|
| `messages_to_persist` | Computes delta between original loaded history and agent output. Uses `id()`-based fast path, falls back to content-key dedup for copied messages. |
| `notify_streamers` | Pushes runtime events to streaming extensions' `on_agent_runner_event_stream` async generators. Fail-open per extension. |
| `drain_pending` | Async generator that yields and clears items from `pending` list one at a time. |
| `build_capabilities` | Assembles Pydantic AI capabilities list: user capabilities first, then user hooks. |
| `fire` | Chain-mode event dispatch: each extension sees the previous one's modifications. |
| `fire_notify` | Notify-mode: all extensions see the same snapshot. Supports cancel semantics (any extension returning `{cancel_key: True}` cancels). |
| `get_tools` | Returns tools for this run. Prefers ToolLifecycle if initialized; falls back to raw tools. |

---

### 12. `py_agent/session/__init__.py`
No changes — already English.

---

### 13. `py_agent/session/_shared.py` — Session utilities

| Item | Role |
|------|------|
| `_MessageAdapter` | Pydantic `TypeAdapter[ModelMessage]` for JSON serialize/deserialize of single messages. |
| `_infer_role` | Infers `MessageRole` from a message's type and first part (ModelRequest + tool_name → TOOL, ModelRequest → USER, ModelResponse → ASSISTANT). |
| `_is_turn_start` | True if message is a `ModelRequest` whose first part is `UserPromptPart` — signals a new user turn. |

---

### 14. `py_agent/session/_local.py` — SQLite session backend

| Class | Role |
|-------|------|
| `LocalSessionManager` | SQLite-backed multi-turn session. Creates tables on first connection (`sessions`, `messages`, `compactions`). `load_history` supports compaction: if a compaction record exists and enough turns have passed the boundary, prepend a summary message. `save_messages` uses auto-incrementing `message_seq`. |

---

### 15. `py_agent/session/_postgres.py` — PG session backend

| Class | Role |
|-------|------|
| `PostgresSessionManager` | PostgreSQL-backed multi-turn session. Uses `psycopg_pool.AsyncConnectionPool` with lazy init. Same compaction logic as SQLite. `close()` explicitly shuts down the pool. |

---

### 16. `py_agent/session/_single_turn.py` — No-persistence backend

| Class | Role |
|-------|------|
| `SingleTurnSessionManager` | No persistence. `load_history` always returns `[]`. `save_messages` is a no-op. `create_session` returns a UUID. `apply_compaction` raises `NotImplementedError`. This is the default when no `session_manager` is passed to `AgentRunner`. |

---

## Verification

After finishing each file, and at the end:

```bash
pytest tests/ -q          # must show 17 passed
rg '[\x{4e00}-\x{9fff}]' py_agent/  # must show zero matches (no Chinese left)
```

# CONTEXT.md — Project Background & Core Design

## Project Background

`py_agent` is a minimal agent framework built on top of Pydantic AI. Pydantic AI provides
the foundational building blocks — `Agent`, `Tool`, `Hooks` — but does not address
session-level multi-turn persistence. `py_agent` fills this gap, delivering session-level
long-term memory with the smallest possible footprint.

### Design Philosophy

Inspired by Pi Agent's architecture, but deliberately not a feature-complete port. Core
principles:

- **No built-in tools.** Capabilities like filesystem access or MCP connections are
  intentionally absent from the framework itself. In security-sensitive or sandboxed
  deployments these should be disabled by default. All tools are injected through the
  Extension system.
- **Minimal viable layer.** The framework does not reinvent the Agent — Pydantic AI
  already provides that. It layers on exactly three things: session persistence, context
  window management, and lifecycle extensibility.

### Three Pillars

| Pillar | Responsibility |
|--------|---------------|
| Context persistence | Session-level message storage across three backends, each with a distinct purpose (see below) |
| Context management | Watermark-based truncation + LLM summarization, designed to keep long conversations within the context window |
| Extension system | A Protocol-defined set of hooks spanning the full agent and tool lifecycle; the sole mechanism for injecting capabilities |

---

### Three Session Backends — Why Not One?

The framework provides three persistence backends, each for a different stage of the
agent lifecycle:

| Backend | Target | Rationale |
|---------|--------|-----------|
| Single-turn (stateless) | Quick exploration, one-off prompts | Zero setup. Every `run()` is a fresh session with empty history. No database needed. |
| SQLite | Local development, lightweight agents | File-based, zero-infrastructure. Developers can prototype agents locally with full multi-turn memory without standing up a server. |
| PostgreSQL | Production deployment | Connection pooling, JSONB queries, concurrent access. PostgreSQL's native JSON support also lets developers run custom queries over agent context — join against their own tables, inspect tool call patterns, or build analytics pipelines directly on the messages store. |

## Core Design: Context Management

### Overarching Goal

**Keep long conversations within the context window while preserving recent context.**

Context management is not a cache-optimization layer. The framework cannot control or
verify provider-side cache-key computation. Instead, it focuses on two concrete,
observable goals:

1. Avoid exceeding the configured context-window cap by truncating old tool results.
2. When truncation alone is no longer enough, replace the distant past with a compact
   LLM-generated summary so the model still has coarse-grained history.

### Dual Watermarks

Context-window management operates at two thresholds with asymmetric costs:

| Threshold | Trigger | Action | Cache impact |
|-----------|---------|--------|--------------|
| Low watermark | token count > context_window × low_ratio | Truncate old tool results outside the protected turn region | Cheap — only shortens content strings |
| High watermark | token count > context_window × high_ratio | Flag for asynchronous LLM compaction | Expensive — restructures message-list history |

Design intent:

```
token count growth →→→
  Below low watermark:     do nothing
  Low ↔ High watermark:    truncate old tool results (space reclaimed, minimal disruption)
  Above high watermark:    trigger compaction (history replaced by summary, deferred)
```

- A single-watermark design triggers compaction as soon as a threshold is crossed →
  frequent, expensive summarization.
- Dual watermarks buy extra headroom between low and high via truncation alone,
  pushing compaction as far out as possible.

### Protecting the Last N Turns

Both watermark truncation and compaction boundary selection respect a single parameter:
`protect_turns`. It guarantees that the most recent N user-assistant exchanges are never
cut or summarized away, preserving the immediate conversational context the model needs for
coherent replies.

**At the low watermark** — when tool-result truncation kicks in, only messages *before*
the protected region are affected. Tool results within the last N turns remain at full
length. The model still sees the complete output of tools it called three turns ago; old
tool noise from fifty turns ago gets trimmed.

**At the high watermark** — when selecting a compaction boundary, `load_history` walks
backwards from the latest message until it counts N turn starts. Any compaction whose
boundary falls inside this protected zone is skipped in favor of an older one. The summary
covers the distant past; the last N turns remain as verbatim messages.

Both mechanisms share the same `protect_turns` value, keeping the protection boundary
consistent: what gets truncated and what gets summarized use the same cutoff.

## Core Design: Context Persistence

### Data Model

- `messages` table: one row per message, `message_seq` monotonically increasing
- `compactions` table: one row per compaction, `(boundary_seq, summary_text)`. `boundary` records which `message_seq` the summary covers up to.
- `system_prompts` table: one row per system prompt write, ordered by write time. The
  latest row is used to reconnect to a session without re-supplying the prompt.

All three tables are scoped by `session_id`.

### History Loading ("Compass" Navigation)

How `load_history(session_id, protect_turns=N)` decides what to return:

```
1. Read max(message_seq) for the session
2. Walk backwards from the newest message, counting user turns,
   until N turns are found. Record the earliest message_seq in that region → cutoff_seq
3. Query compactions for the latest row where boundary_seq < cutoff_seq
   ├─ Found → return [compaction summary as a synthetic message]
   │          + [all complete messages after boundary]
   └─ Not found → return all messages (no compaction applies)
```

Critical detail: the query does not simply take the most recent compaction. If the latest
compaction's boundary falls inside the protected region of N recent turns, it is skipped
in favor of an older one. This is what "protect the last N turns from compaction" actually
means at read time.

### Message Saving: Delta-Only

At the end of each turn, `original_history` (after context prepare, before extension
injection) is compared against `all_messages` (after the Agent finishes). Only the
newly produced messages — the delta — are written to the `messages` table. Because
extension-injected messages happen after `original_history` is captured, they are
included in the delta and persisted; messages produced by Pydantic AI during the run
(such as SystemPromptPart) are excluded from the delta and are not persisted.

---

## Core Design: Cross-Session Isolation

### Problem

A single `AgentRunner` instance can serve multiple sessions via
`run(prompt, session_id=...)`. Any in-memory session-scoped state will leak between
them — session A's data contaminating session B's requests.

### Solution: Zero Instance State

All session-scoped state lives in the database, addressed by `session_id`:

- System prompts live in the `system_prompts` table → `load_system_prompt(session_id)` reads only
  that session's row
- Compaction records live in the `compactions` table → `load_history` queries only that
  session's boundaries
- `_prepare_context()` is a pure function accepting `(messages, config)` and returning
  `(prepared_messages, needs_compaction)`. It holds no state between invocations.

There is no concept of a "current session" — every `_setup_run` loads the stored system
prompt fresh from the database for the given session, with no dependence on memory
retained from the previous run.

---

## Core Design: Compaction Deduplication

### Why Async?

Compaction calls an LLM to produce a summary — a potentially expensive operation that can
take seconds. If it ran synchronously inside the current turn, the user would wait for the
summary to finish before seeing the agent's response. By dispatching compaction as a
background task (`asyncio.create_task`), the agent's reply is returned immediately, and the
summary is generated on the side. The user perceives no latency from context management.

### Problem: Duplicate Tasks

Because compaction is async, a compaction from the previous turn may still be executing
when the next turn flags another one. Without protection, two concurrent tasks would write
to the `compactions` table — wasting LLM tokens on duplicate work.

### Solution: In-Flight Set

An `_compaction_pending: set[session_id]` tracks currently executing compactions. Before
launching a new task:

- session_id **not** in set → add it, create the background task
- session_id **already** in set → skip; the previous task is still running

On task completion (normal or exception), the session_id is removed from the set. This
guarantees at most one compaction per session at any time.

---

## Core Design: Extension System

### Intent

Extensions are the sole mechanism for injecting capabilities into the agent. The framework
itself ships with zero built-in tools. Everything — filesystem access, MCP connections,
sub-agents — arrives through Extensions.

### Lifecycle Coverage

Extensions implement the `Extension` Protocol, which spans three distinct phases:

---

**Phase 1: Tool registration** (once per `AgentRunner` instance, lazily on first `run()`)

Extensions discover and register their tools through two hooks:

| Hook | When | Extension's role |
|------|------|-----------------|
| `register_tool_sources()` | Called once, before the first turn | Return `ToolSource` objects (local functions, MCP servers, sub-agents) that the framework will discover tools from |
| `on_tool_event(event, data)` | Fires for each tool as it is discovered and registered | Observe or resolve conflicts. Receives `TOOL_DISCOVERED`, `TOOL_CONFLICT`, `TOOL_REGISTERED`, `TOOL_REMOVED` |

Built-in default for `TOOL_CONFLICT`: local tools override MCP tools with the same name.
Extensions can override this by returning a different resolution.

---

**Phase 2: Pre-agent** (every turn, before the model runs)

Extensions can observe and modify the message list before it enters the model:

| Event | Access | Purpose |
|-------|--------|---------|
| `SESSION_START` | Read | Session was created or reused |
| `CONTEXT_PREPARE` | Read-only | Context preparation is complete (truncation applied). Extensions see the prepared state but cannot modify it here — use the next event for that. |
| `BEFORE_AGENT_RUN` | **Writable** | The only point where extensions can inject or modify messages before they enter the model. Changes here are persisted. |
| `AGENT_START` | Read | The final prompt and messages are locked; the model is about to be called |

---

**Phase 3: During & after agent execution** (every turn)

Extensions observe and intercept the model's tool calls and output, then participate in
persistence and compaction:

| Event | Access | Purpose |
|-------|--------|---------|
| `TOKEN_STREAM` | Read | A token chunk was generated (streaming only). Extensions can forward chunks to their own consumers. |
| `TOOL_START` | Read | The model decided to call a tool. Observational only. |
| `TOOL_CALL` | **Writable** | Intercept point before tool execution. Extensions can block the call (`{block: true}`) or modify arguments. |
| `TOOL_RESULT` | **Writable** | Tool execution completed. Extensions can modify the return value before the model sees it. |
| `TOOL_END` | Read | Tool call fully resolved. Observational only. |
| `AFTER_AGENT_RUN` | Read | Agent finished generating. Output and usage stats are available. |
| `AGENT_END` | Read | Final agent state. Identical payload to `AFTER_AGENT_RUN`; exists for symmetry with `AGENT_START`. |
| `SESSION_SAVE` | **Writable** | Messages are about to be persisted. Extensions can modify the delta before it hits the database. |
| `COMPACTION_TRIGGER` | **Cancellable** | Compaction was flagged. Any extension returning `{cancel: true}` blocks it. Uses *notify mode* — each extension votes independently on the same snapshot. |
| `COMPACTION_APPLIED` | Read | Compaction completed (or was cancelled). The outcome is communicated. |
| `SESSION_END` | Read | The turn is complete. |

---

**Two event-dispatch modes** (applies to all phases):

| Mode | Behavior | Used for |
|------|----------|---------|
| Chain (`_fire`) | Each extension sees the previous extension's modifications. Return values are merged into the event data. | Message mutation: `BEFORE_AGENT_RUN`, `TOOL_CALL`, `TOOL_RESULT`, `SESSION_SAVE` |
| Notify (`_fire_notify`) | Every extension receives the same read-only snapshot. No extension sees another's return. | Voting / cancellation: `COMPACTION_TRIGGER` |

### Relationship with Context Management

Messages injected by extensions through `BEFORE_AGENT_RUN` are **persisted** — they fall
within the delta computation window. The `original_history` snapshot is captured after
context preparation but before `BEFORE_AGENT_RUN`, so only extension-injected messages
are persisted. The boundary between these two stages is intentional.

---

## Public API Surface

Every public name must serve one of two purposes: enable customization, or support type
hints. Nothing else is exposed. Users should never need to reach into private modules or
monkey-patch internals. The API is split across four import paths by semantic concern.

### `py_agent` — core entry points and config

| Name | Kind | Why the user needs it |
|------|------|----------------------|
| `AgentRunner` | Class | Instantiate, call `run()` / `run_stream()` |
| `RunResult` | BaseModel | Return type of `AgentRunner.run()` |
| `ContextConfig` | BaseModel | Configure watermark thresholds, protect_turns, truncation. Frozen after construction. Pass to `AgentRunner(context_config=...)`. |
| `SummarizerConfig` | BaseModel | Configure the LLM summarizer (model, token budget, custom prompt). Pass to `AgentRunner(summarizer_config=...)`. |

### `AgentRunner` — required prompt & integration config

**`system_prompt` is always required.** Every run needs a non-empty system
prompt: pass it explicitly, or omit it only when reconnecting to an existing
session that already has a stored prompt (loaded from the `system_prompts`
table). Single-turn sessions have no stored prompt, so the caller must always
provide one. Empty strings are rejected.

The remaining constructor parameters are thin pass-throughs that mirror
Pydantic AI capabilities, kept so the framework never blocks access to the
underlying SDK:

| Parameter | Effect | Kind |
|-----------|--------|------|
| `thinking_enabled`, `thinking_level` | Enable Pydantic AI thinking mode (`ModelSettings["thinking"]`) | SDK pass-through |
| `parallel_tool_calls` | Allow concurrent tool calls (`ModelSettings["parallel_tool_calls"]`) | SDK pass-through |
| `hooks` | Append a Pydantic AI `Hooks` instance to capabilities | SDK pass-through |
| `capabilities` | Append extra Pydantic AI capabilities | SDK pass-through |
| `max_tool_calls_per_turn` | Hard cap on tool invocations per turn, enforced in the tool loop | Framework logic |

### `py_agent.session` — persistence backends and related types

| Name | Kind | Why the user needs it |
|------|------|----------------------|
| `SessionManager` | ABC | Implement custom persistence backends |
| `SingleTurnSessionManager` | Class | Default backend; also a reference for implementing `SessionManager` |
| `LocalSessionManager` | Class | SQLite backend for local development |
| `PostgresSessionManager` | Class | PostgreSQL backend for production |

| `MessageRole` | StrEnum | Type for the `role` column in custom `SessionManager` implementations |

### `py_agent.tools` — tool source implementations

| Name | Kind | Why the user needs it |
|------|------|----------------------|
| `LocalToolSource` | Class | Wrap Python functions as tools — the most common case |
| `MCPServerSource` | Class | Wrap MCP server clients |
| `SubagentToolSource` | Class | Wrap sub-agents or LangGraph graphs |

### `py_agent.types` — protocols, enums, and type aliases

| Name | Kind | Why the user needs it |
|------|------|----------------------|
| `Extension` | Protocol | Implement custom extensions |
| `ToolSource` | ABC | Implement custom tool sources |
| `AgentRunnerEvent` | StrEnum | Match event names in `on_agent_runner_event` |
| `ToolLifecycleEvent` | StrEnum | Match event names in `on_tool_event` |
| `ToolEventHandler` | Type alias | Function signature for `on_tool_event` handlers |

### Not Exported

| Name | Why not |
|------|---------|

| `ToolLifecycle` | Managed internally by `AgentRunner`; users interact via `ToolSource` and events |
| `HarnessSummarizer` | Compaction implementation detail; configured through `SummarizerConfig` |
| `ContextManagerConfig` | Removed. `ContextConfig` is the sole context-management config model. Users pass `ContextConfig` directly. |

---

## Spotted Pitfalls (Future Work)

Known issues that are documented here instead of being fixed in this pass.
Each entry describes the trigger, the consequence, and the intended fix so a
later pass can pick it up without re-deriving the context.

### P1. Compaction deduplication is per-instance, not per-session

**Trigger.** Two `AgentRunner` instances (or two processes) sharing the same
database serve the same `session_id` concurrently, and both cross the high
watermark in the same turn. The in-flight deduplication set
(`_compaction_pending` in `AgentRunner.__init__`) is an instance attribute,
so each instance independently launches a background compaction for the
session.

**Consequence.**

- The LLM summarizer runs twice for the same boundary — double token cost.
- The `compactions` table receives two rows for the same boundary — duplicate
  data, though `load_history` picks the latest eligible row, so correctness is
  unaffected.

**Scope.** Single-instance behavior is correct: within one runner, at most one
compaction per session runs at any time. The gap only appears with multiple
instances sharing a session, which is an uncommon deployment pattern (sessions
are usually pinned to one worker).

**Intended fix.** Database-level protection, e.g. an advisory lock or
`INSERT ... ON CONFLICT DO NOTHING` on `(session_id, boundary_seq)` in
`apply_compaction`. Note that write-side idempotency alone cannot prevent the
duplicate `summarize` call — the expensive step happens before the write — so
a full fix requires reserving the boundary before summarization.

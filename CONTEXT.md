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
| Context management | Watermark-based truncation + LLM summarization, designed to minimize cache-miss frequency |
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

**Maximize LLM API cache-hit rate while preserving long-conversation quality.**

LLM APIs key their cache on a byte-level hash of the full message list — including the
*effective* prompt that Pydantic AI assembles from the user's instructions plus tool
names/descriptions, skills, and context files. Any perturbation to this assembled prompt
causes a full cache miss. Every design decision in context management starts from the
question: "does this operation change the effective prompt?" 

### Dual Watermarks

Context-window management operates at two thresholds with asymmetric costs:

| Threshold | Trigger | Action | Cache impact |
|-----------|---------|--------|--------------|
| Low watermark | token count > context_window × low_ratio | Truncate old tool results outside the protected turn region | **None** — only content strings are shortened |
| High watermark | token count > context_window × high_ratio | Flag for asynchronous LLM compaction | **Full miss** — message-list structure is replaced by a summary |

Design intent:

```
token count growth →→→
  Below low watermark:     do nothing
  Low ↔ High watermark:    truncate old tool results (space reclaimed, zero cache cost)
  Above high watermark:    trigger compaction (cache miss, deferred to the last possible moment)
```

- A single-watermark design triggers compaction as soon as a threshold is crossed →
  frequent cache misses.
- Dual watermarks buy extra headroom between low and high via truncation alone (cost-free),
  pushing the inevitable compaction as far out as possible.

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

### Baseline Mechanism: Stabilizing the Effective Prompt for Cache Hits

Two distinct concepts of "system prompt" are at play:

- **User system prompt** — the `system_prompt` string passed to `AgentRunner`. This is the
  developer's instructions: personality, task description, behavioral rules. Typically
  stable across turns.
- **Effective prompt** — what the model actually receives. Pydantic AI assembles this from
  the user's system prompt **plus** the list of available tools, their names and
  descriptions, registered skills, and context files. This is what the LLM API hashes for
  its cache key.

The core tension: tools, skills, and context files may change over time. If every
addition of a tool updated the effective prompt, every request would be a cache miss.
But the model must still be informed of new capabilities.

Solution: **frozen baseline + transient diff injection**.

- Each session stores a baseline pair `(user_system_prompt, BaselineState)`. The
  `BaselineState` is a structured snapshot of tools, skills, and context at baseline time —
  kept separate from the user system prompt, not baked into a single opaque blob.
- On every turn, the current live configuration is diffed against `BaselineState`:
  - **Diff exists** → inject a transient message informing the model. The effective prompt
    stays byte-for-byte identical → cache hit.
  - **No diff** → nothing is injected.
- When the user system prompt actually changes (developer modified their instructions),
  **the baseline is silently refreshed without injecting a diff**. The prompt already
  changed — cache miss is unavoidable — so there is no reason to also notify the model
  that it changed.

Three baseline-write triggers, unified by the rule "only refresh when a cache miss is already inevitable":

| Trigger | Action | Rationale |
|---------|--------|-----------|
| First access, no baseline exists | Freeze current `(user_sp, state)` | First request has no cache to hit; writing the baseline is free |
| User system prompt changed | Silently update baseline, no diff injected | Prompt change already causes cache miss; diff would be redundant noise |
| Compaction completed | Refresh baseline if state drifted | Compaction already caused a cache miss by restructuring history; absorb accumulated state changes while we're here |

Tool changes during normal turns do **not** refresh the baseline — only a transient diff
message is injected. The baseline catches up on the next compaction.

---

## Core Design: Context Persistence

### Data Model

- `messages` table: one row per message, `message_seq` monotonically increasing
- `compactions` table: one row per compaction, `(boundary_seq, summary_text)`. `boundary` records which `message_seq` the summary covers up to.
- `baselines` table: one row per baseline snapshot, `(user_system_prompt, BaselineState JSON)`. Ordered by write time; the latest row is the current baseline. 

`user_system_prompt` is the developer-supplied instructions, **not** the full effective prompt seen by the model.

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
newly produced messages — the delta — are written to the `messages` table.

Transient diff messages are injected before `original_history` is captured, so they fall
outside the delta and are never persisted.

---

## Core Design: Cross-Session Isolation

### Problem

A single `AgentRunner` instance can serve multiple sessions via
`run(prompt, session_id=...)`. Any in-memory session-scoped state will leak between
them — session A's data contaminating session B's requests.

### Solution: Zero Instance State

All session-scoped state lives in the database, addressed by `session_id`:

- Baselines live in the `baselines` table → `load_latest_baseline(session_id)` reads only
  that session's row
- Compaction records live in the `compactions` table → `load_history` queries only that
  session's boundaries
- `prepare()` is a pure function accepting `(messages, frozen_baseline, current_state, config)`
  and returning `PreparedContext`. It holds no state between invocations.

There is no concept of a "current session" — every `_setup_run` loads the baseline fresh
from the database for the given session, with no dependence on memory retained from the
previous run.

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
| `CONTEXT_PREPARE` | Read-only | Context preparation is complete (truncation applied, diff injected). Extensions see the prepared state but cannot modify it here — use the next event for that. |
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
within the delta computation window. Transient diff messages are injected earlier in the
pipeline, before `original_history` is captured, so they are **never persisted**. The
boundary between these two injection points is intentional.

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

### `py_agent.session` — persistence backends and related types

| Name | Kind | Why the user needs it |
|------|------|----------------------|
| `SessionManager` | ABC | Implement custom persistence backends |
| `SingleTurnSessionManager` | Class | Default backend; also a reference for implementing `SessionManager` |
| `LocalSessionManager` | Class | SQLite backend for local development |
| `PostgresSessionManager` | Class | PostgreSQL backend for production |
| `BaselineState` | BaseModel | Parameter type for `SessionManager.save_baseline(state=...)` |
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
| `PreparedContext` | Internal return type of `prepare()`; users never call `prepare()` directly |
| `ToolLifecycle` | Managed internally by `AgentRunner`; users interact via `ToolSource` and events |
| `HarnessSummarizer` | Compaction implementation detail; configured through `SummarizerConfig` |
| `ContextManagerConfig` | Removed. `ContextConfig` is the sole context-management config model. Users pass `ContextConfig` directly. |

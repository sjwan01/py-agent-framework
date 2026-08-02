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
| Extension system | A Protocol-defined set of hooks spanning the full agent and tool lifecycle; tools and skills are declared on `AgentRunner`, extensions own the lifecycle (plus optional capability registration) |

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
2. When truncation alone is no longer enough, present the distant past as a compact
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
  Above high watermark:    trigger compaction (history summarized at load time, deferred)
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

- `messages` table: one row per message, `message_seq` monotonically increasing. Rows are never deleted — the full conversation history is always retained (see "Compaction Is Read-Time Only" below). The `role` column is an exact classification written by `_infer_role` — `'user'` is equivalent to a turn start and drives the cutoff lookup in `load_history`; non-user non-tool requests classify as `'unknown'` and are never mistaken for `'user'`.
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

### Compaction Is Read-Time Only

Compaction changes what the model *sees*, never what is *stored*. The `messages` table
keeps the full conversation history — `apply_compaction` only inserts a row into
`compactions`, and `load_history` decides at read time whether to present the summary
plus recent messages or the raw history. No rows are ever deleted.

Retaining full history is deliberate, not an oversight. Deleting summarized rows would
make it impossible to trace back: audit a claim against the original turn, re-run
analytics over tool call patterns, or answer "what exactly was said in turn 10".
Trading that away for storage savings is bad practice. If the hot table ever grows too
large, the fix is an archival strategy (moving old rows out), not in-place deletion.

### Message Saving: Delta-Only

At the end of each turn, the delta is the union of two parts:

1. `result.new_messages()` — messages the SDK itself tracked as new (user
   prompt, tool results, model reply).
2. The extension-injected messages, captured in `_setup_run` **at injection
   time**: after `BEFORE_AGENT_RUN`, any message whose identity is not in the
   pre-injection history is recorded. Identity is stable at that moment — the
   Agent has not touched the messages yet — so this needs no diffing against
   the post-run message list and no reliance on SDK copy behavior.

The delta is written to the `messages` table. Messages produced by Pydantic AI
during the run that are not part of the conversation (such as SystemPromptPart)
are excluded by `new_messages()` and are not persisted.

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

Extensions are the sole mechanism for injecting **lifecycle behavior** — they
observe and intercept events during a run. They may also register SDK
capabilities (``register_capabilities``). Tools and skills are **declared on
``AgentRunner`` itself** (constructor parameters), not through extensions —
that would be a hack. The framework itself ships with zero built-in tools.
Everything — filesystem access, MCP connections, skill libraries — is wired
in by the application that constructs the runner.

### Lifecycle Coverage

Extensions implement the `Extension` Protocol, which spans three distinct phases:

---

**Phase 1: Capability registration** (once per `AgentRunner` instance, lazily on first `run()`)

| Hook | When | Extension's role |
|------|------|-----------------|
| `register_capabilities()` | Called once, before the first turn | Return a list of Pydantic AI `AbstractCapability` instances (e.g. `Skills`, `PrefixTools`). Optional — extensions that only observe events skip it |

Tools and skills are declared on the constructor (`tools=`, `skills=`), not
registered here. `collect_tools` validates them: every toolset must specify a
server name (`id`), server names must be unique, toolsets are wrapped in
`_ResilientToolset` (a down server degrades — both the connection phase
`__aenter__` and the catalog phase `get_tools` fail open, see
SPEC-stateful-tool-management) and, by default (`prefix_toolset_names=True`),
prefixed with their server name (`{server}_{tool}`) so identically named
tools across servers never collide. With prefixing disabled, cross-source
name conflicts are reported by pydantic-ai at assembly time. No registration
events are fired; runtime interception happens at `TOOL_CALL` (see Phase 3).

**Skills behavior (verified against pydantic-ai source).** A `Skills`
instance is a deferred capability: the model sees only each skill's name +
description (a catalog rendered into the request prefix every turn —
deliberately including already-loaded skills, to keep the provider's
prompt-cache prefix stable), and loads the full SKILL.md body on demand via
the `load_capability` tool. The full body lands in message history as a
`LoadCapabilityReturnPart` (a `ToolReturnPart` subclass) — never in the
system prompt, because `CombinedCapability.get_instructions` skips deferred
capabilities regardless of load state. Consequences:

- **Full bodies are ordinary history messages**: persisted with the delta,
  counted in context estimation, subject to compaction like any other
  message. Compaction dropping a loaded skill's body is natural forgetting —
  the catalog still lists the skill, so the model can reload it.
- **Truncation skips skill bodies by coincidence**: the truncator matches
  `type(part) is ToolReturnPart` exactly, which excludes subclasses, so
  skill bodies are never truncated until compacted. Preserve this if the
  truncation logic changes.
- **Catalog cost scales with library size and is the same class of cost as
  tool metadata**: every tool's name/description/parameters schema and every
  skill's name/description are re-exposed to the model each turn. This is a
  scale trade-off for the application (how many servers/skills to expose),
  not something the framework special-cases — pydantic-ai's native tool
  search is the on-demand counterpart for tools.

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

**Tool policy patterns (approval, toggling, auditing).** The framework ships
no built-in tool policies. All of them are business logic and are implemented
by extensions at `TOOL_CALL`:

```python
if event == AgentRunnerEvent.TOOL_CALL:
    if not approved(data["tool_name"], data["args"]):
        return {"block": True, "reason": "approval required"}   # approval
    audit_log(data)                                                 # auditing
    if disabled(data["tool_name"]):
        return {"block": True}                                     # toggling
    return {"args": sanitize(data["args"])}                       # argument rewriting
```

Approval flows are application-specific (CLI confirm, HTTP callback,
headless auto-approve) and must not be guessed by the framework.

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
| `max_tool_calls_per_turn` | Hard cap on tool invocations per turn, enforced in the tool loop | Framework logic |

Framework policy parameters (not SDK pass-throughs):

| Parameter | Effect |
|-----------|--------|
| `extensions` | Objects implementing the `Extension` protocol: lifecycle observers + optional `register_capabilities` (see Extension System) |
| `tools` | Raw callables or Pydantic AI `Tool` / `AbstractToolset` objects — the **only** tool registration path |
| `skills` | A Pydantic AI harness `Skills` instance (skill library) available for on-demand loading. Defaults to `None` |
| `session_manager` | Persistence backend; `None` uses `SingleTurnSessionManager` |
| `context_config` | Watermark thresholds, protect_turns, truncation (see Context Management) |
| `summarizer_config` | LLM compaction config (see Compaction) |
| `on_warning` | Callback for non-fatal errors (extension crashes, toolset degradation, compaction failures) |
| `toolset_failure` | Custom handler when a toolset's catalog or connection fails: return a dict to substitute tools, `None` for default warn-and-drop, raise to fail the run (see `ToolsetFailureHandler`) |
| `prefix_toolset_names` | Default `True`: prefix every toolset's tools with its server name (`{server}_{tool}`) so identically named tools across servers never collide. Disable to expose raw names (conflicts then reported by pydantic-ai at assembly) |

`AgentRunner.close()` releases the session backend (toolset connections are entered and exited by the SDK on every run, so nothing persists there).

### `py_agent.session` — persistence backends and related types

| Name | Kind | Why the user needs it |
|------|------|----------------------|
| `SessionManager` | ABC | Implement custom persistence backends |
| `SingleTurnSessionManager` | Class | Default backend; also a reference for implementing `SessionManager` |
| `LocalSessionManager` | Class | SQLite backend for local development |
| `PostgresSessionManager` | Class | PostgreSQL backend for production |

### `py_agent.types` — protocols, enums, and type aliases

| Name | Kind | Why the user needs it |
|------|------|----------------------|
| `Extension` | Protocol | Implement custom extensions |
| `AgentRunnerEvent` | StrEnum | Match event names in `on_agent_runner_event` |
| `MessageRole` | StrEnum | Type for the `role` column in custom `SessionManager` implementations |
| `ToolsetFailureHandler` | Type alias | Custom handling when a toolset's connection or catalog fails to load (e.g. a down MCP server): return a dict to substitute tools, `None` for warn-and-drop, or raise to fail the run. Handler exceptions propagate (a mistyped signature fails fast at construction) |

### Not Exported

| Name | Why not |
|------|---------|

| `HarnessSummarizer` | Compaction implementation detail; configured through `SummarizerConfig` |

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

### P2. In-memory history cache for hot sessions (not built, by design)

**Idea.** Same-process, same-session turns re-read the full history from the
database on every `run()`. Keeping decoded history in memory and only writing
to the DB — maintaining consistency — would eliminate those reads.

**Why it is not built.** It directly contradicts the Zero Instance State
design (see Cross-Session Isolation): session-scoped memory state, cache
invalidation, and multi-instance consistency are exactly what that design
eliminates. The concrete challenges:

- background compaction (`trigger_compaction`) mutates the DB between turns
  and would race with an in-memory cache;
- multiple instances sharing a session (the Postgres deployment form) each
  hold their own cache — permanent inconsistency;
- public `save_messages` / `apply_compaction` can be invoked externally,
  bypassing the cache;
- process restart or a new runner reconnecting to the session must rebuild
  from the DB anyway.

**When it would be worth it.** Only a single-process deployment with a
long-lived session and high turn frequency. For the general framework the DB
stays the single source of truth.

**If ever built.** A bounded form: cache only the decoded message list and
append per-turn deltas, with an explicit invalidation contract covering
compaction and external writes. Not a general cache layer.

### P3. Sub-agent scope and tool categories — revisit if a subagent layer is added

The framework has no sub-agent execution path, so tool `scope` (main vs.
subagent) and distinct sub-agent tool categories are absent: an agent sees
sub-agents as plain tools. If a subagent execution layer is ever added,
scope semantics and tool categorization should be designed from real
requirements (how subagents are created, how tools are handed to them), not
guessed in advance.

# SPEC: Cross-Session Contamination

## Problem

A single `AgentRunner` instance can serve multiple sessions via `run(prompt, session_id=...)`.
Two internal mechanisms carry state across sessions when they should be per-session.

### 1. ContextManager baseline poisoning

`ContextManager._frozen_baseline` and `_baseline_state` are set on the **first call** to
`prepare()` and then reused forever. If session A triggers the baseline freeze, session B's
subsequent calls will diff against A's baseline — injecting spurious "[System config changed]"
messages and missing real config changes in B.

**Impact**: incorrect diff injection, confusing the model with phantom config changes.

### 2. Compaction double-fire

When a compaction task is still running (async, background), the next turn may trigger a
second compaction for the same session. Both write to the `compactions` table; the one with
the higher `boundary_seq` wins, the other is wasted LLM work.

**Impact**: redundant LLM calls, wasted token spend. Not a correctness bug.

---

## Solution Sketch

### ContextManager baseline

The baseline must be **per-session**, not per-instance. Options:

**A. Reset on session switch.** `ContextManager` tracks which `session_id` the current
baseline belongs to. When `prepare()` sees a different session, it resets.

**B. Move baseline into AgentRunner, keyed by session_id.** AgentRunner holds
`_baselines: dict[str, tuple[str, BaselineState]]`. Passes the correct one to
`ContextManager.prepare()` or resets when missing.

**C. Don't reuse ContextManager across sessions.** Each `_setup_run` creates a fresh
`ContextManager` (or clones from a template). Simplest semantics but allocates per-turn.

### Compaction dedup

Track in-flight compaction tasks per session:

```python
self._compaction_pending: set[str] = set()
```

Before `asyncio.create_task`:

```python
if session_id not in self._compaction_pending:
    self._compaction_pending.add(session_id)
    asyncio.create_task(self._trigger_compaction(session_id))
```

In `trigger_compaction` (finally block):

```python
self._compaction_pending.discard(session_id)
```

---

## Questions to resolve

1. **For baseline: A, B, or C?** Or another approach?
2. **Scope**: should this be one combined change or two separate commits?
3. **Testing**: what test scenarios should we add to verify cross-session isolation?

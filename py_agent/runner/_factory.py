"""Lifecycle and discovery for AgentRunner.

This module contains AgentRunner's tool registration, compaction triggering,
and automatic extension discovery. Every function's first parameter ``self``
is an ``AgentRunner`` instance.
"""
from __future__ import annotations

from typing import Any

from py_agent._context import _compute_diff
from py_agent.tools import LocalToolSource, ToolLifecycle
from py_agent.types import ToolLifecycleEvent

# Lazy tool lifecycle initialization (called on the first run).

async def ensure_tool_lifecycle(self: Any) -> Any:
    """Lazily initialize ``ToolLifecycle`` and register tools from all sources.

    Called once during the first ``run()`` / ``run_stream()`` invocation and
    cached afterwards. Registration order:

    1. Create the ``ToolLifecycle`` instance.
    2. Subscribe extension ``on_tool_event`` handlers to all tool events.
    3. Register raw tools passed to the constructor as a ``LocalToolSource``.
    4. Call each extension's ``register_tool_sources()`` and register the
       returned ``ToolSource`` instances (local, MCP, or subagent).
    """
    # already initialized, return immediately
    if self._tool_lifecycle_initialized:
        return self._tool_lifecycle

    # first run: skip ToolLifecycle if there are no tools or extensions
    if self._tool_lifecycle is None:
        if self._raw_tools or self._extensions:
            self._tool_lifecycle = ToolLifecycle(on_warning=self._on_warning)
        else:
            self._tool_lifecycle_initialized = True
            return None

    # subscribe extension handlers before registering so TOOL_CONFLICT etc. can be intercepted
    for ext in self._extensions:
        handler = getattr(ext, "on_tool_event", None)
        if handler is None:
            continue
        for event in ToolLifecycleEvent:
            self._tool_lifecycle.on(event, handler)

    # register raw tools passed to the constructor
    if self._raw_tools:
        await self._tool_lifecycle.add_source(LocalToolSource(self._raw_tools))

    # let extensions expose their own tool sources
    for ext in self._extensions:
        register = getattr(ext, "register_tool_sources", None)
        if register is None:
            continue
        try:
            sources = await register()
        except Exception as exc:  # pragma: no cover - fail-open
            self._on_warning(
                f"Extension {type(ext).__name__} register_tool_sources failed: {exc}",
                exc,
            )
            continue
        for src in sources or []:
            await self._tool_lifecycle.add_source(src)

    self._tool_lifecycle_initialized = True
    return self._tool_lifecycle


# Compaction trigger.

async def trigger_compaction(self: Any, session_id: str) -> None:
    """Asynchronously compact the history for the current session.

    Triggered from ``_finalize_run`` via ``asyncio.create_task`` so it does not
    block the current turn's response. Flow:

    1. Get the current maximum ``message_seq``; this is the compaction boundary
       (messages at and before this seq are summarized/overwritten).
    2. Load full history (``protect_turns=0``; any existing compaction summary
       will be loaded).
    3. Summarize all messages (skipped if the summarizer is ``None``).
    4. Write the summary into the ``compactions`` table.
    5. If the current state has changed, refresh the persisted baseline.
       Compaction already invalidates the cache, so this is the safe moment to
       absorb accumulated state diffs into the system prompt.
    """
    try:
        # raw maximum, unaffected by the compactions table
        boundary_seq = await self._session_manager.get_max_message_seq(session_id)
        if boundary_seq < 0:
            return

        summarizer = self._compaction_summarizer
        if summarizer is None:
            return  # no SummarizerConfig configured, skip LLM compaction

        # protect_turns=0 always returns summary + full message list
        messages = await self._session_manager.load_history(session_id)
        summary = await summarizer.summarize(messages)

        await self._session_manager.apply_compaction(
            session_id,
            summary=summary,
            boundary_seq=boundary_seq,
        )

        # Compaction already invalidates the cache, so refresh the baseline if
        # the state has drifted. Normal turns keep the baseline frozen to avoid
        # cache misses from system prompt changes.
        row = await self._session_manager.load_latest_baseline(session_id)
        frozen_baseline = row[1] if row else None
        current_state = await self._build_current_state()
        if frozen_baseline is None or _compute_diff(frozen_baseline, current_state):
            await self._session_manager.save_baseline(
                session_id,
                system_prompt=self._system_prompt,
                state=current_state,
            )
    except Exception as exc:  # pragma: no cover - fail-open
        self._on_warning(f"Compaction failed for session {session_id}: {exc}", exc)
    finally:
        self._compaction_pending.discard(session_id)

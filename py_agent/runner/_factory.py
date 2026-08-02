"""Lifecycle and discovery for AgentRunner.

This module contains AgentRunner's tool collection, compaction triggering,
and automatic extension discovery. Every function's first parameter ``self``
is an ``AgentRunner`` instance.
"""
from __future__ import annotations

import inspect
from typing import Any, cast

from pydantic_ai import Tool as PydanticTool
from pydantic_ai.toolsets import AbstractToolset

from py_agent.types import ToolsetFailureHandler


def _validate_toolset_failure_handler(handler: ToolsetFailureHandler) -> None:
    """Validate a ``toolset_failure`` handler signature eagerly.

    Called at ``AgentRunner`` construction so a mistyped handler fails fast
    with a clear message instead of a cryptic ``TypeError`` at runtime.
    A handler must accept two positional arguments ``(toolset_id, exception)``.

    Args:
        handler: The handler to validate.

    Raises:
        TypeError: When the handler cannot accept the two required arguments.
    """
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        # not introspectable (e.g. some builtins); let it surface at runtime
        return

    params = list(sig.parameters.values())
    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    required = sum(p.default is inspect.Parameter.empty for p in positional)
    accepts_two = varargs or (len(positional) >= 2 and required <= 2)
    if not accepts_two:
        raise TypeError(
            "toolset_failure must be a callable accepting "
            "(toolset_id: str, exception: Exception); "
            f"got {handler} with signature {sig}"
        )


class _ResilientToolset(AbstractToolset):
    """Wraps a toolset so catalog-load failures degrade instead of crashing the run.

    pydantic-ai fails the whole run when any toolset's ``get_tools`` raises
    (all-or-nothing). This wrapper turns that into partial degradation: the
    failing server's tools are dropped (or replaced via a custom handler),
    the rest of the toolsets keep working, and the next run retries
    automatically because the SDK re-enters and re-lists every run.

    Args:
        inner: The wrapped ``AbstractToolset`` (e.g. ``MCPToolset``).
        on_warning: Failure warning callback.
        handler: Optional custom failure handler; ``None`` warns and drops.
    """

    def __init__(
        self,
        inner: AbstractToolset,
        on_warning: Any,
        handler: ToolsetFailureHandler | None = None,
    ):
        self._inner = inner
        self._on_warning = on_warning
        self._handler = handler

    @property
    def id(self) -> str:
        """Delegate the toolset identifier to the wrapped toolset."""
        return cast(str, self._inner.id)

    async def __aenter__(self) -> Any:
        """Enter the wrapped toolset."""
        return await self._inner.__aenter__()

    async def __aexit__(self, *args: Any) -> Any:
        """Exit the wrapped toolset."""
        return await self._inner.__aexit__(*args)

    async def get_tools(self, ctx: Any) -> dict[str, Any]:
        """Load the catalog; on failure degrade (or delegate to the handler)."""
        try:
            return await self._inner.get_tools(ctx)
        except Exception as exc:  # pragma: no cover - fail-open
            if self._handler is not None:
                result = self._handler(cast(str, self._inner.id), exc)
                if result is not None:
                    return result
            self._on_warning(
                f"Toolset {self._inner.id} unavailable: {exc}",
                exc,
            )
            return {}

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate tool execution to the wrapped toolset."""
        return await self._inner.call_tool(*args, **kwargs)


# Lazy tool collection (called on the first run).

async def collect_tools(self: Any) -> tuple[list[Any], list[Any]]:
    """Collect tools and toolsets from all sources (once, lazily).

    Called once during the first ``run()`` / ``run_stream()`` invocation and
    cached afterwards. Sources are the raw tools passed to the constructor
    plus each extension's ``register_tool_sources()`` output. Every returned
    item is split by type: ``AbstractToolset`` instances (e.g. ``MCPToolset``)
    go to the Agent's ``toolsets`` (wrapped in ``_ResilientToolset`` for
    partial degradation), ``Tool`` instances (or raw callables) to ``tools``.

    Name conflicts resolve by last-writer-wins: a tool registered later
    replaces an earlier one with the same name.

    Returns:
        A tuple of ``(tools, toolsets)`` — pydantic-ai objects for the Agent.
    """
    if self._tools_initialized:
        return self._tools, self._toolsets

    tools: list[Any] = []
    toolsets: list[Any] = []
    by_name: dict[str, Any] = {}

    def _add(item: Any) -> None:
        """Split a collected item and apply last-writer-wins on names."""
        if isinstance(item, AbstractToolset):
            toolsets.append(
                _ResilientToolset(
                    item,
                    self._on_warning,
                    getattr(self, "_toolset_failure", None),
                )
            )
            return
        if not isinstance(item, PydanticTool):
            item = PydanticTool(item)
        name = item.name
        if name in by_name:
            tools[tools.index(by_name[name])] = item
        else:
            tools.append(item)
        by_name[name] = item

    # raw tools passed to the constructor
    for item in self._raw_tools:
        _add(item)

    # let extensions expose their own tools
    for ext in self._extensions:
        register = getattr(ext, "register_tool_sources", None)
        if register is None:
            continue
        try:
            items = await register()
        except Exception as exc:  # pragma: no cover - fail-open
            self._on_warning(
                f"Extension {type(ext).__name__} register_tool_sources failed: {exc}",
                exc,
            )
            continue
        for item in items or []:
            try:
                _add(item)
            except Exception as exc:  # pragma: no cover - fail-open
                self._on_warning(
                    f"Invalid tool from {type(ext).__name__}: {exc}",
                    exc,
                )
                continue

    self._tools = tools
    self._toolsets = toolsets
    self._tools_initialized = True
    return tools, toolsets


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

        if not summary.strip():
            # an empty summary would inject a meaningless shell on the next load
            self._on_warning(
                f"Compaction produced an empty summary for session "
                f"{session_id}; skipping",
                None,
            )
            return

        await self._session_manager.apply_compaction(
            session_id,
            summary=summary,
            boundary_seq=boundary_seq,
        )
    except Exception as exc:  # pragma: no cover - fail-open
        self._on_warning(f"Compaction failed for session {session_id}: {exc}", exc)
    finally:
        self._compaction_pending.discard(session_id)

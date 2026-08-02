"""Lifecycle and discovery for AgentRunner.

This module contains AgentRunner's tool collection, compaction triggering,
and automatic extension discovery. Every function's first parameter ``self``
is an ``AgentRunner`` instance.
"""
from __future__ import annotations

import inspect
from typing import Any, cast

from pydantic_ai import Tool as PydanticTool
from pydantic_ai.toolsets import AbstractToolset, PrefixedToolset

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

    pydantic-ai fails the whole run when any toolset's ``__aenter__`` (the
    connection) or ``get_tools`` (the catalog) raises. This wrapper turns
    that into partial degradation: the failing server's tools are dropped
    (or replaced via a custom handler), the rest of the toolsets keep
    working, and the next run retries automatically because the SDK
    re-enters and re-lists every run.

    Args:
        inner: The wrapped ``AbstractToolset`` (e.g. ``PrefixedToolset``).
        on_warning: Failure warning callback.
        handler: Optional custom failure handler; ``None`` warns and drops.
        id: Explicit server name; defaults to ``inner.id`` (which is ``None``
            for ``PrefixedToolset``, so pass the original server name).
    """

    def __init__(
        self,
        inner: AbstractToolset,
        on_warning: Any,
        handler: ToolsetFailureHandler | None = None,
        *,
        id: str | None = None,
    ):
        self._inner = inner
        self._id = id
        self._on_warning = on_warning
        self._handler = handler
        self._enter_failed = False
        self._substitute: dict[str, Any] | None = None

    @property
    def id(self) -> str:
        """Return the explicit server name, falling back to ``inner.id``."""
        if self._id is not None:
            return self._id
        return cast(str, self._inner.id)

    async def __aenter__(self) -> Any:
        """Enter the wrapped toolset; on connection failure, degrade."""
        self._enter_failed = False
        self._substitute = None
        try:
            return await self._inner.__aenter__()
        except Exception as exc:  # pragma: no cover - fail-open
            self._enter_failed = True
            self._resolve_failure(exc)
            return self

    async def __aexit__(self, *args: Any) -> Any:
        """Exit the wrapped toolset; skip it when entry never succeeded."""
        if self._enter_failed:
            # the inner toolset was never entered; calling its __aexit__
            # would raise ("called more times than __aenter__")
            self._enter_failed = False
            return None
        return await self._inner.__aexit__(*args)

    async def get_tools(self, ctx: Any) -> dict[str, Any]:
        """Load the catalog; on failure degrade (or delegate to the handler)."""
        if self._enter_failed:
            return self._substitute or {}
        try:
            return await self._inner.get_tools(ctx)
        except Exception as exc:  # pragma: no cover - fail-open
            self._resolve_failure(exc)
            return self._substitute or {}

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate tool execution to the wrapped toolset."""
        return await self._inner.call_tool(*args, **kwargs)

    def _resolve_failure(self, exc: Exception) -> None:
        """Delegate a failure to the custom handler or the default warn-and-drop."""
        if self._handler is not None:
            result = self._handler(self.id, exc)
            if result is not None:
                self._substitute = result
                return
        self._on_warning(f"Toolset {self.id} unavailable: {exc}", exc)


# Lazy tool collection (called on the first run).

async def collect_tools(self: Any) -> tuple[list[Any], list[Any]]:
    """Collect tools and toolsets from all sources (once, lazily).

    Called once during the first ``run()`` / ``run_stream()`` invocation and
    cached afterwards. Sources are the raw tools passed to the constructor
    plus each extension's ``register_tool_sources()`` output. Every returned
    item is split by type: ``AbstractToolset`` instances (e.g. ``MCPToolset``)
    go to the Agent's ``toolsets`` (wrapped in ``_ResilientToolset`` for
    partial degradation), ``Tool`` instances (or raw callables) to ``tools``.

    Name conflicts between tools from the same source resolve by
    last-writer-wins (a tool registered later replaces an earlier one with
    the same name). Every ``AbstractToolset`` must specify a server name
    (``id``) and server names must be unique — both are configuration errors
    that fail the run. Toolsets are wrapped in ``_ResilientToolset`` for
    partial degradation and, when ``prefix_toolset_names`` is enabled,
    prefixed with their server name (``{server}_{tool}``) so identically
    named tools across servers never collide. Cross-source name conflicts
    with prefixes disabled are reported by the SDK at assembly time.

    Returns:
        A tuple of ``(tools, toolsets)`` — pydantic-ai objects for the Agent.
    """
    if self._tools_initialized:
        return self._tools, self._toolsets

    tools: list[Any] = []
    toolsets: list[Any] = []
    by_name: dict[str, Any] = {}
    seen_server_names: set[str] = set()

    def _add(item: Any) -> None:
        """Split a collected item; enforce server names; apply prefixing."""
        if isinstance(item, AbstractToolset):
            server_name = item.id
            if server_name is None:
                raise ValueError(
                    f"Toolset {item!r} must specify a server name (id) to "
                    "disambiguate tools across servers"
                )
            if server_name in seen_server_names:
                raise ValueError(
                    f"Duplicate toolset server name {server_name!r}; "
                    "server names must be unique"
                )
            seen_server_names.add(server_name)
            wrapped: AbstractToolset = item
            if getattr(self, "_prefix_toolset_names", True):
                wrapped = PrefixedToolset(wrapped, prefix=server_name)
            toolsets.append(
                _ResilientToolset(
                    wrapped,
                    self._on_warning,
                    getattr(self, "_toolset_failure", None),
                    id=server_name,
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
            if isinstance(item, AbstractToolset):
                # configuration errors (missing/duplicate server name) propagate
                _add(item)
                continue
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

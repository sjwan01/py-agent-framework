"""Runtime helpers for AgentRunner.

These functions are bound as instance methods on the ``AgentRunner`` class in
``_agent.py`` (for example ``_fire = _internals.fire``). Their first parameter
``self`` is therefore the ``AgentRunner`` instance, so the functions can access
``self._extensions``, ``self._session_manager``, and other attributes directly.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic_ai.messages import ModelRequest, ModelResponse


def messages_to_persist(
    original_history: list[Any], all_messages: list[Any]
) -> list[Any]:
    """Compute the messages that should be persisted this turn.

    Pydantic AI's ``result.new_messages()`` treats messages injected by
    extensions during ``BEFORE_AGENT_RUN`` as "old history" and excludes them.
    Extension-injected messages must be persisted so they can be restored on
    the next turn. Because the SDK does not expose a primitive to distinguish
    "originally loaded history" from "extension-injected messages", we compute
    the difference manually.

    Deduplication logic:

    1. Fast identity check via ``id()`` — Pydantic AI does not deep-copy the
       passed ``message_history``, so the same object keeps the same id.
    2. If a message was copied (different id), fall back to a stable content key
       based on ``(kind, run_id, parts summary)``.
    """
    # first pass: filter by Python object identity
    original_ids = {id(m) for m in original_history}

    def _key(m: Any) -> tuple[Any, ...]:
        """Build a stable content key for a message.

        Used when object identity differs because the message was copied.
        """
        kind = "request" if isinstance(m, ModelRequest) else (
            "response" if isinstance(m, ModelResponse) else type(m).__name__
        )
        parts: list[Any] = []
        for part in getattr(m, "parts", ()):
            pk = getattr(part, "part_kind", None)
            if pk == "user-prompt":
                parts.append(("user-prompt", part.content))
            elif pk == "tool-return":
                parts.append(("tool-return", part.tool_name, part.tool_call_id, str(part.content)))
            elif pk == "text":
                parts.append(("text", part.content))
            elif pk == "tool-call":
                parts.append(("tool-call", part.tool_name, part.tool_call_id, str(part.args)))
            else:
                parts.append((str(pk), repr(part)))
        return (kind, getattr(m, "run_id", None), tuple(parts))

    original_keys = {_key(m) for m in original_history}
    return [
        m for m in all_messages
        if id(m) not in original_ids and _key(m) not in original_keys
    ]


async def notify_streamers(
    streamers: list[Any],
    event: str,
    data: dict[str, Any],
    pending: list[Any],
    on_warning: Callable[[str, Exception | None], None],
) -> None:
    """Push runtime events to all streaming extensions.

    Streaming extensions implement ``on_agent_runner_event_stream`` as an async
    generator. Their yielded chunks are appended to ``pending`` and later
    drained by ``run_stream()`` to the external consumer.

    Extensions without ``on_agent_runner_event_stream`` are silently skipped;
    extensions that raise are warned about and skipped (fail-open).

    Args:
        streamers: Extensions that should receive streaming events.
        event: The event name being dispatched.
        data: The event payload.
        pending: Staging list that collects yielded chunks.
        on_warning: Callback for non-fatal streaming extension failures.
    """
    for s in streamers:
        stream_fn = getattr(s, "on_agent_runner_event_stream", None)
        if stream_fn is None:
            continue
        try:
            async for chunk in stream_fn(event, data):
                pending.append(chunk)
        except Exception as exc:  # pragma: no cover - fail-open
            on_warning(
                f"Streaming extension {type(s).__name__} failed for {event}: {exc}",
                exc,
            )


async def drain_pending(
    pending: list[Any]
) -> AsyncIterator[dict[str, Any]]:
    """Yield and clear every chunk in ``pending`` one at a time."""
    while pending:
        yield pending.pop(0)


def build_capabilities(self: Any) -> list[Any]:
    """Assemble the capabilities list passed to the Pydantic AI Agent.

    User-provided capabilities come first; framework hooks are appended by the
    caller.
    """
    capabilities = list(self._capabilities)
    if self._hooks is not None:
        capabilities.append(self._hooks)
    return capabilities


async def fire(
    self: Any, event: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Dispatch an event to all extensions in chain mode.

    Chain mode means extension A receives the data, returns a modified dict,
    extension B receives A's modified data, returns another modified dict, and
    the final result is returned to the caller.

    If an extension raises, a warning is logged and the next extension continues.
    """
    current = dict(data)
    for ext in self._extensions:
        try:
            r = await ext.on_agent_runner_event(event, current)
        except Exception as exc:  # pragma: no cover - fail-open
            self._on_warning(
                f"Extension {type(ext).__name__} handler for {event} failed: {exc}",
                exc,
            )
            continue
        if isinstance(r, dict):
            current.update(r)
    return current


async def fire_notify(
    self: Any,
    event: str,
    data: dict[str, Any],
    *,
    cancel_key: str = "cancel",
) -> dict[str, Any]:
    """Dispatch an event to all extensions in notify mode.

    Unlike chain mode, every extension receives the same read-only snapshot and
    does not see other extensions' modifications. Supports cancellation:
    if any extension returns ``{cancel_key: True}``, the final result is
    ``{cancel_key: True}``.
    """
    snapshot = dict(data)
    cancelled = False
    for ext in self._extensions:
        try:
            r = await ext.on_agent_runner_event(event, snapshot)
        except Exception as exc:  # pragma: no cover - fail-open
            self._on_warning(
                f"Extension {type(ext).__name__} handler for {event} failed: {exc}",
                exc,
            )
            continue
        if isinstance(r, dict) and r.get(cancel_key) is True:
            cancelled = True
    return {cancel_key: cancelled}


async def get_tools(self: Any) -> list[Any]:
    """Return the tools the Agent should use for this run.

    Prefers ``ToolLifecycle`` when extensions have registered tool sources;
    falls back to the raw tools passed to the constructor.
    """
    lifecycle = await self._ensure_tool_lifecycle()
    if lifecycle is not None:
        tools: list[Any] = lifecycle.get_for_scope(self._scope)
        return tools
    raw_tools: list[Any] = self._raw_tools
    return raw_tools



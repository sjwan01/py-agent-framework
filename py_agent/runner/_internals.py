"""Runtime helpers for AgentRunner.

These functions are bound as instance methods on the ``AgentRunner`` class in
``_agent.py`` (for example ``_fire = _internals.fire``). Their first parameter
``self`` is therefore the ``AgentRunner`` instance, so the functions can access
``self._extensions``, ``self._session_manager``, and other attributes directly.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability

from py_agent.types import AgentRunnerEvent, Extension

if TYPE_CHECKING:
    from py_agent.runner import AgentRunner


async def notify_streamers(
    streamers: list[Extension],
    event: str,
    data: dict[str, Any],
    pending: list[dict[str, Any]],
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
    pending: list[dict[str, Any]]
) -> AsyncIterator[dict[str, Any]]:
    """Yield and clear every chunk in ``pending`` one at a time."""
    while pending:
        yield pending.pop(0)


async def fire(
    self: AgentRunner, event: str, data: dict[str, Any]
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


class _EventBridge:
    """Private extension bridging chain/stream events to ``run_with_events()``.

    ``TOOL_CALL`` / ``TOOL_RESULT`` are chain-only events (dispatched via
    ``fire``, which iterates the runner's extensions), while ``TOKEN_STREAM``
    and the other streamed lifecycle events reach streaming extensions. This
    bridge implements both hooks: the chain hook queues normalized
    ``tool_call`` / ``tool_result`` events, and the stream hook drains the
    queue first (so tool events precede the next streamed token) before
    yielding ``token`` events for ``TOKEN_STREAM``. The bridge never mutates
    chain data — its chain hook always returns ``None``.

    Queued events are pre-rewrite snapshots: the bridge sits first in the
    chain, so it observes the dispatched values before user extensions modify
    them (e.g. rewritten ``args`` / ``content`` never reach the consumer).

    Ordering is safe because every tool call is followed by a ``TOOL_END``
    stream event, so the queue drains on the next stream tick; even if a
    stream notification is skipped (fail-open), the queue drains at the
    following stream event.
    """

    def __init__(self) -> None:
        self._queue: list[dict[str, Any]] = []

    async def register_capabilities(self) -> list[AbstractCapability[Any]]:
        """The bridge contributes no SDK capabilities."""
        return []

    async def on_agent_runner_event(
        self, event: str, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Queue a normalized event for TOOL_CALL / TOOL_RESULT; never mutate.

        Args:
            event: The chain-mode event name.
            data: The event payload.

        Returns:
            ``None`` — the bridge observes and never mutates chain data.
        """
        if event == AgentRunnerEvent.TOOL_CALL:
            self._queue.append(
                {
                    "type": "tool_call",
                    "tool_name": data["tool_name"],
                    "tool_call_id": data["tool_call_id"],
                    "args": data["args"],
                }
            )
        elif event == AgentRunnerEvent.TOOL_RESULT:
            self._queue.append(
                {
                    "type": "tool_result",
                    "tool_name": data["tool_name"],
                    "tool_call_id": data["tool_call_id"],
                    "content": data["content"],
                    "is_error": data["is_error"],
                }
            )
        return None

    async def on_agent_runner_event_stream(
        self, event: str, data: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Drain queued tool events, then yield a token for TOKEN_STREAM.

        Args:
            event: The streamed event name.
            data: The event payload.

        Yields:
            Queued ``tool_call`` / ``tool_result`` events first, then a
            ``token`` event for ``TOKEN_STREAM``. Other streamed events are
            ignored.
        """
        while self._queue:
            yield self._queue.pop(0)
        if event == AgentRunnerEvent.TOKEN_STREAM:
            yield {"type": "token", "chunk": data["data"]["chunk"]}


async def fire_notify(
    self: AgentRunner,
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

    Note: the result does not distinguish "no extensions voted" from "all
    extensions voted against" — both yield ``{cancel_key: False}``. Callers
    that only need the boolean decision are unaffected.
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



"""Runtime helpers for AgentRunner.

These functions are bound as instance methods on the ``AgentRunner`` class in
``_agent.py`` (for example ``_fire = _internals.fire``). Their first parameter
``self`` is therefore the ``AgentRunner`` instance, so the functions can access
``self._extensions``, ``self._session_manager``, and other attributes directly.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from py_agent.types import Extension

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



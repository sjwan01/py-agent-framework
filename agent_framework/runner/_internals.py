"""Runtime helpers for AgentRunner."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def build_model(self):
    """Return the model override or build an OpenAI-compatible model from settings.

    Pydantic AI's ``Agent`` can accept a model name string and resolve it
    automatically, but that path does not support a custom ``base_url`` or
    ``api_key`` without a pre-configured provider.  V2 keeps explicit
    provider construction so deployments with local/vLLM endpoints work out
    of the box.
    """
    if self._model is not None:
        return self._model
    return OpenAIChatModel(
        self._settings.llm_model_id,
        provider=OpenAIProvider(
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
        ),
    )


def messages_to_persist(original_history: list, all_messages: list) -> list:
    """Return messages that should be persisted for this turn.

    Pydantic AI's ``result.new_messages()`` intentionally excludes any
    message passed in ``message_history``.  V2 extensions may inject
    messages into ``message_history`` during ``CONTEXT_PREPARE`` or
    ``BEFORE_AGENT_RUN``; those messages must survive across turns, so
    they must be persisted.  There is no SDK primitive that exposes
    "messages added by the caller to message_history", so we compute the
    delta manually against the originally loaded history.
    """
    original_ids = {id(m) for m in original_history}

    def _key(m) -> tuple:
        kind = "request" if isinstance(m, ModelRequest) else (
            "response" if isinstance(m, ModelResponse) else type(m).__name__
        )
        parts: list = []
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
    streamers: list, event: str, data: dict, pending: list
) -> None:
    """Forward an event to streaming extensions and collect yielded chunks."""
    for s in streamers:
        try:
            async for chunk in s.on_agent_runner_event_stream(event, data):
                pending.append(chunk)
        except Exception as exc:  # pragma: no cover - fail-open
            logging.getLogger(__name__).warning(
                "Streaming extension %s failed: %s",
                type(s).__name__, exc,
                exc_info=True,
            )


async def drain_pending(pending: list) -> AsyncIterator[dict]:
    """Yield all chunks currently in ``pending`` and clear it."""
    while pending:
        yield pending.pop(0)


def build_capabilities(self) -> list:
    capabilities = list(self._config.capabilities)
    if self._config.hooks is not None:
        capabilities.append(self._config.hooks)
    return capabilities


async def fire(self, event: str, data: dict) -> dict:
    """Fire event to all extensions in chain mode.

    Each extension receives the current data (including any updates from
    previous extensions). Non-None dict results are shallow-merged back so
    later extensions see the accumulated output.
    """
    current = dict(data)
    for ext in self._extensions:
        try:
            r = await ext.on_agent_runner_event(event, current)
        except Exception as exc:  # pragma: no cover - fail-open
            logging.getLogger(__name__).warning(
                "Extension %s handler for %s failed: %s",
                type(ext).__name__, event, exc,
                exc_info=True,
            )
            continue
        if isinstance(r, dict):
            current.update(r)
    return current


async def fire_notify(
    self, event: str, data: dict, *, cancel_key: str = "cancel"
) -> dict:
    """Fire event in notify mode; any extension may request cancellation.

    All extensions receive the same read-only snapshot. If any extension
    returns ``{cancel_key: True}`` the aggregate result is ``{cancel_key: True}``.
    """
    snapshot = dict(data)
    cancelled = False
    for ext in self._extensions:
        try:
            r = await ext.on_agent_runner_event(event, snapshot)
        except Exception as exc:  # pragma: no cover - fail-open
            logging.getLogger(__name__).warning(
                "Extension %s handler for %s failed: %s",
                type(ext).__name__, event, exc,
                exc_info=True,
            )
            continue
        if isinstance(r, dict) and r.get(cancel_key):
            cancelled = True
    return {cancel_key: cancelled}


def boundary_id(messages: list, boundary: int) -> str:
    """Return a stable id for the compaction boundary."""
    if boundary > 0 and boundary <= len(messages):
        msg = messages[boundary - 1]
        return getattr(msg, "run_id", None) or f"msg-{boundary - 1}"
    return f"msg-{boundary}"


async def get_tools(self) -> list:
    lifecycle = await self._ensure_tool_lifecycle()
    if lifecycle is not None:
        return lifecycle.get_for_scope(self._scope)
    return self._raw_tools

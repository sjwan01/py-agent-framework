"""AgentRunner orchestration."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from agent_framework.compaction import CompactionSummarizer
from agent_framework.models import AgentConfig, RunResult, BaselineState
from agent_framework.settings import Settings
from agent_framework.types import SessionManager, AgentRunnerEvent

from . import _factory, _hooks, _internals


class AgentRunner:
    """Orchestrates load → build → run → save."""

    # --- Delegated to sub-modules -------------------------------------------------
    _build_hooks = _hooks.build_hooks
    _build_model = _internals.build_model
    _messages_to_persist = staticmethod(_internals.messages_to_persist)
    _notify_streamers = staticmethod(_internals.notify_streamers)
    _drain_pending = staticmethod(_internals.drain_pending)
    _build_capabilities = _internals.build_capabilities
    _fire = _internals.fire
    _fire_notify = _internals.fire_notify
    _boundary_id = staticmethod(_internals.boundary_id)
    _get_tools = _internals.get_tools

    _default_session_manager = _factory.default_session_manager
    _default_context_manager = _factory.default_context_manager
    _default_compaction_summarizer = _factory.default_compaction_summarizer
    _ensure_tool_lifecycle = _factory.ensure_tool_lifecycle
    _trigger_compaction = _factory.trigger_compaction
    discover_extensions = staticmethod(_factory.discover_extensions)

    def __init__(
        self,
        settings: Settings,
        config: AgentConfig,
        *,
        session_manager: SessionManager | None = None,
        model=None,
        tools: list = (),
        tool_lifecycle=None,
        context_manager=None,
        extensions: list | None = None,
        scope: str = "main",
        compaction_summarizer: CompactionSummarizer | None = None,
    ):
        self._settings = settings
        self._config = config
        self._session_manager = session_manager or self._default_session_manager()
        self._model = model
        self._raw_tools = list(tools)
        self._tool_lifecycle = tool_lifecycle
        self._context_manager = context_manager or self._default_context_manager()
        self._extensions = extensions or []
        self._scope = scope
        self._compaction_summarizer = compaction_summarizer or self._default_compaction_summarizer()
        self._tool_lifecycle_initialized = False

    async def _setup_run(
        self, prompt: str, session_id: str | None
    ) -> tuple[str, list, list, bool]:
        """Shared pre-agent setup used by ``run()`` and ``run_stream()``.

        Returns ``(session_id, history, original_history, needs_compaction)``.
        """
        if session_id is None:
            session_id = await self._session_manager.create_session()

        await self._fire(AgentRunnerEvent.SESSION_START, {"session_id": session_id})

        if self._context_manager is None:
            raise RuntimeError("ContextManager is not configured")

        history = await self._session_manager.load_history(session_id)
        original_history = list(history)

        needs_compaction = False
        try:
            prepared = await self._context_manager.prepare(
                history,
                system_prompt=self._config.instructions,
                current_state=BaselineState(),
            )
            history = prepared.messages
            needs_compaction = prepared.needs_compaction
        except Exception as exc:  # pragma: no cover - fail-open
            logging.getLogger(__name__).warning(
                "ContextManager prepare failed: %s", exc, exc_info=True
            )

        ctx_data = {
            "session_id": session_id,
            "messages": history,
            "needs_compaction": needs_compaction,
        }
        ctx_result = await self._fire(AgentRunnerEvent.CONTEXT_PREPARE, ctx_data)
        history = ctx_result.get("messages", history)
        needs_compaction = ctx_result.get("needs_compaction", needs_compaction)

        before_data = {"session_id": session_id, "messages": history}
        before_result = await self._fire(AgentRunnerEvent.BEFORE_AGENT_RUN, before_data)
        if "messages" in before_result:
            history = before_result["messages"]

        await self._fire(
            AgentRunnerEvent.AGENT_RUN,
            {"session_id": session_id, "prompt": prompt, "messages": history},
        )

        return session_id, history, original_history, needs_compaction

    async def _build_agent(
        self,
        session_id: str,
        *,
        pending: list | None = None,
        streamers: list | None = None,
    ) -> Agent:
        """Build the Pydantic AI Agent for this turn."""
        hooks = self._build_hooks(session_id, pending=pending, streamers=streamers)

        capabilities = self._build_capabilities() or []
        if hooks not in capabilities:
            capabilities.append(hooks)

        model_settings = ModelSettings(
            parallel_tool_calls=self._settings.parallel_tool_calls,
        )

        return Agent(
            model=self._build_model(),
            instructions=self._config.instructions,
            tools=await self._get_tools(),
            capabilities=capabilities or None,
            model_settings=model_settings,
        )

    async def _finalize_run(
        self,
        session_id: str,
        original_history: list,
        result,
        output: str,
        needs_compaction: bool,
        *,
        streamers: list | None = None,
        pending: list | None = None,
    ) -> AsyncIterator[dict]:
        """Shared post-agent finalization used by ``run()`` and ``run_stream()``.

        Yields any chunks produced by streaming extensions and finally a
        ``run_end`` event.
        """
        streamers = streamers if streamers is not None else []
        pending = pending if pending is not None else []

        delta_messages = self._messages_to_persist(original_history, result.all_messages())
        usage = result.usage

        payload = {"session_id": session_id, "output": output, "usage": usage}
        await self._fire(AgentRunnerEvent.AFTER_AGENT_RUN, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.AFTER_AGENT_RUN, payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        await self._session_manager.save_messages(session_id, delta_messages)
        save_payload = {"session_id": session_id, "delta_messages": delta_messages}
        await self._fire(AgentRunnerEvent.SESSION_SAVE, save_payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.SESSION_SAVE, save_payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        if needs_compaction:
            comp_result = await self._fire_notify(AgentRunnerEvent.COMPACTION_TRIGGER, {
                "session_id": session_id,
            })
            cancelled = comp_result.get("cancel", False)
            if not cancelled:
                asyncio.create_task(self._trigger_compaction(session_id))
            applied_payload = {"session_id": session_id, "cancelled": bool(cancelled)}
            await self._fire(AgentRunnerEvent.COMPACTION_APPLIED, applied_payload)
            await self._notify_streamers(streamers, AgentRunnerEvent.COMPACTION_APPLIED, applied_payload, pending)
            async for chunk in self._drain_pending(pending):
                yield chunk

        end_payload = {"session_id": session_id}
        await self._fire(AgentRunnerEvent.SESSION_END, end_payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.SESSION_END, end_payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        yield {
            "type": "run_end",
            "session_id": session_id,
            "output": output,
            "new_messages": delta_messages,
            "usage": usage,
        }

    async def run(self, prompt: str, *, session_id: str | None = None) -> RunResult:
        session_id, history, original_history, needs_compaction = await self._setup_run(
            prompt, session_id
        )
        agent = await self._build_agent(session_id)

        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text(delta=False):
                output_parts.append(text)
                await self._fire(
                    AgentRunnerEvent.AGENT_RUN,
                    {
                        "session_id": session_id,
                        "event": "on_chat_model_stream",
                        "data": {"chunk": text},
                    },
                )

        output = "".join(output_parts)
        async for event in self._finalize_run(
            session_id, original_history, result, output, needs_compaction
        ):
            if event["type"] == "run_end":
                return RunResult(
                    output=event["output"],
                    session_id=event["session_id"],
                    new_messages=event["new_messages"],
                    usage=event["usage"],
                )

        # pragma: no cover - _finalize_run always yields run_end
        raise RuntimeError("run did not produce a run_end event")

    async def run_stream(
        self, prompt: str, *, session_id: str | None = None
    ) -> AsyncIterator[dict]:
        """Run the agent and yield events from streaming extensions.

        Non-streaming extensions continue to receive events through
        ``on_agent_runner_event`` exactly as in ``run()``.  The final event
        is always ``{"type": "run_end", ...}``.
        """
        session_id, history, original_history, needs_compaction = await self._setup_run(
            prompt, session_id
        )

        streamers = list(self._extensions)
        pending: list[dict] = []
        agent = await self._build_agent(
            session_id, pending=pending, streamers=streamers
        )

        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text(delta=False):
                output_parts.append(text)
                payload = {
                    "session_id": session_id,
                    "event": "on_chat_model_stream",
                    "data": {"chunk": text},
                }
                await self._fire(AgentRunnerEvent.AGENT_RUN, payload)
                await self._notify_streamers(streamers, AgentRunnerEvent.AGENT_RUN, payload, pending)
                async for chunk in self._drain_pending(pending):
                    yield chunk

        output = "".join(output_parts)
        async for event in self._finalize_run(
            session_id,
            original_history,
            result,
            output,
            needs_compaction,
            streamers=streamers,
            pending=pending,
        ):
            yield event

"""AgentRunner — core orchestrator of the framework.

Entry points: ``run(prompt)`` and ``run_stream(prompt)``. Both methods share
the exact same lifecycle; the only difference is that ``run()`` packages the
result into a ``RunResult``, while ``run_stream()`` yields events one by one
to external consumers.

A full run lifecycle:

  SESSION_START → load_history → context_prepare
  → BEFORE_AGENT_RUN → AGENT_START
  → [Pydantic AI loop: tool calls, token generation]
  → AFTER_AGENT_RUN → AGENT_END
  → save_messages → [optional: compaction]
  → SESSION_END → run_end
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Callable, cast

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from py_agent.models import (
    ContextManagerConfig,
    RunResult,
    SummarizerConfig,
)
from py_agent._compaction import HarnessSummarizer
from py_agent.context import BaselineState, ContextManager
from py_agent.session import SingleTurnSessionManager
from py_agent.types import (
    SessionManager,
    AgentRunnerEvent,
)

from py_agent.runner import _factory, _hooks, _internals

# Pydantic AI thinking depth levels. Invalid values are ignored and fall back.
_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high", "xhigh"}


class AgentRunner:
    """Orchestrates load → build → run → save."""

    # Class-attribute bindings: wire submodule functions as instance methods.
    #
    # Technique: module-level functions take ``self`` as their first argument and are
    # bound here as class attributes, so ``self._fire(...)`` actually calls
    # ``_internals.fire(self, ...)``. This avoids multiple inheritance or putting
    # all code in a single file.

    # --- from _internals.py (runtime helpers) ---
    _build_hooks = _hooks.build_hooks
    _messages_to_persist = staticmethod(_internals.messages_to_persist)
    _notify_streamers = staticmethod(_internals.notify_streamers)
    _drain_pending = staticmethod(_internals.drain_pending)
    _build_capabilities = _internals.build_capabilities
    _fire = _internals.fire
    _fire_notify = _internals.fire_notify
    _get_tools = _internals.get_tools

    # --- from _factory.py (init / lifecycle / discovery) ---
    _ensure_tool_lifecycle = _factory.ensure_tool_lifecycle
    _trigger_compaction = _factory.trigger_compaction

    # Constructor

    def __init__(
        self,
        model: Model,
        *,
        system_prompt: str = "",
        thinking_enabled: bool = True,
        thinking_level: str | None = None,
        extensions: list[Any] | None = None,
        tools: list[Any] | tuple[()] = (),
        session_manager: SessionManager | None = None,
        context_manager_config: ContextManagerConfig | None = None,
        summarizer_config: SummarizerConfig | None = None,
        max_tool_calls_per_turn: int = 5,
        parallel_tool_calls: bool = False,
        hooks: Any = None,
        capabilities: list[Any] | None = None,
        on_warning: Callable[[str, Exception | None], None] | None = None,
    ):
        """Construct AgentRunner.

        Only ``model`` is required:

            runner = AgentRunner(model=my_model)

        All other parameters are optional: ``system_prompt``, ``extensions``, ``tools``,
        ``session_manager``, ``context_manager_config``, ``summarizer_config``, etc.
        Defaults are an empty system prompt, ``SingleTurnSessionManager``,
        no context management, and no compaction.
        """
        def _noop(msg: str, exc: Exception | None = None) -> None:
            pass

        self._model = model
        self._system_prompt = system_prompt
        self._thinking_enabled = thinking_enabled
        self._thinking_level = thinking_level
        self._scope = "main"

        self._extensions = extensions or []

        self._raw_tools = list(tools)
        self._tool_lifecycle = None
        self._tool_lifecycle_initialized = False

        self._session_manager = session_manager or SingleTurnSessionManager()

        self._context_manager: ContextManager | None = None
        if context_manager_config is not None:
            self._context_manager = ContextManager(
                context_window_cap=context_manager_config.context_window,
                low_watermark_ratio=context_manager_config.low_watermark_ratio,
                high_watermark_ratio=context_manager_config.high_watermark_ratio,
                protect_turns=context_manager_config.protect_turns,
                truncate_chars=context_manager_config.truncate_tool_result_chars,
            )
        self._protect_turns = (
            context_manager_config.protect_turns if context_manager_config else 5
        )

        self._compaction_summarizer: HarnessSummarizer | None = None
        if summarizer_config is not None:
            context_window = (
                context_manager_config.context_window
                if context_manager_config
                else 128_000
            )
            default_max = int(min(32_768, max(context_window * 0.1, 8_192)))
            self._compaction_summarizer = HarnessSummarizer(
                model=summarizer_config.model or model,
                max_output_tokens=summarizer_config.max_output_tokens
                or default_max,
                summary_prompt=summarizer_config.summary_prompt,
            )

        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._parallel_tool_calls = parallel_tool_calls

        self._hooks = hooks
        self._capabilities = capabilities or []

        self._on_warning = on_warning or _noop

    # Shared helpers for run() and run_stream()

    async def _setup_run(
        self, prompt: str, session_id: str | None
    ) -> tuple[str, list, list, bool]:
        """Shared pre-run setup for ``run()`` / ``run_stream()``.

        Returns:
            A tuple of ``(session_id, history, original_history, needs_compaction)``:

            - history: message list after ContextManager truncation and extension injection
            - original_history: raw list loaded from the DB (used to compute the turn delta)
            - needs_compaction: whether ContextManager flagged compaction as needed
        """
        # step 1: create or reuse session
        if session_id is None:
            session_id = await self._session_manager.create_session()
        else:
            await self._session_manager.ensure_session(session_id)

        # SESSION_START event
        await self._fire(AgentRunnerEvent.SESSION_START, {"session_id": session_id})

        # step 2: load history (compaction logic is already applied inside)
        history = await self._session_manager.load_history(
            session_id, protect_turns=self._protect_turns
        )
        # keep the original copy — extensions may inject messages into history,
        # and we need original_history to compute the real delta for this turn
        original_history = list(history)

        # step 3: ContextManager.prepare — truncation, baseline diff injection,
        # and compaction flag (skipped when no ContextManager is configured)
        needs_compaction = False
        if self._context_manager is not None:
            try:
                prepared = await self._context_manager.prepare(
                    history,
                    system_prompt=self._system_prompt,
                    current_state=BaselineState(),
                )
                history = prepared.messages
                needs_compaction = prepared.needs_compaction
            except Exception as exc:  # pragma: no cover - fail-open
                self._on_warning(f"ContextManager prepare failed: {exc}", exc)

        # step 4: CONTEXT_PREPARE event (read-only — extension return values are ignored,
        # the only place to modify messages is the next BEFORE_AGENT_RUN step)
        ctx_data = {
            "session_id": session_id,
            "messages": history,
            "needs_compaction": needs_compaction,
        }
        await self._fire(AgentRunnerEvent.CONTEXT_PREPARE, ctx_data)

        # step 5: BEFORE_AGENT_RUN event (extensions inject messages here)
        before_data = {"session_id": session_id, "messages": history}
        before_result = await self._fire(AgentRunnerEvent.BEFORE_AGENT_RUN, before_data)
        if "messages" in before_result:
            history = before_result["messages"]

        # step 6: AGENT_START event (read-only — informs extensions the run has started)
        await self._fire(
            AgentRunnerEvent.AGENT_START,
            {"session_id": session_id, "prompt": prompt, "messages": history},
        )

        return session_id, history, original_history, needs_compaction

    async def _build_agent(
        self,
        session_id: str,
        *,
        pending: list[Any] | None = None,
        streamers: list[Any] | None = None,
    ) -> Agent:
        """Build the Pydantic AI Agent for this turn.

        Args:
            session_id: Current session identifier.
            pending: Staging list for streaming events.
            streamers: Streaming extensions that should receive runtime events.

        Returns:
            A configured ``Agent`` instance.
        """
        # build hooks (intercept logic before/during/after tool execution)
        hooks = self._build_hooks(session_id, pending=pending, streamers=streamers)

        # assemble the capabilities list
        capabilities = self._build_capabilities()
        if hooks not in capabilities:
            capabilities.append(hooks)

        # model settings
        model_settings = ModelSettings(
            parallel_tool_calls=self._parallel_tool_calls,
        )
        # thinking configuration
        if self._thinking_enabled:
            level = self._thinking_level
            if level is not None and level not in _VALID_THINKING_LEVELS:
                self._on_warning(
                    f"Invalid thinking_level {level!r} ignored (valid: "
                    f"{', '.join(sorted(_VALID_THINKING_LEVELS))})",
                    None,
                )
                level = None
            model_settings["thinking"] = cast(Any, level if level is not None else True)

        return Agent(
            model=self._model,
            instructions=self._system_prompt,
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
        streamers: list[Any] | None = None,
        pending: list[Any] | None = None,
    ) -> AsyncIterator[dict]:
        """Shared post-run cleanup for ``run()`` / ``run_stream()``.

        After the Agent finishes, this:

        1. Fires ``AFTER_AGENT_RUN`` and ``AGENT_END``.
        2. Saves the turn's delta messages to the DB.
        3. Triggers compaction if needed and not cancelled.
        4. Fires ``SESSION_END``.
        5. Yields ``run_end`` (``{"type": "run_end", ...}``).

        This is an async generator. ``run_stream()`` yields the intermediate events to
        the external consumer; ``run()`` only consumes the final ``run_end``.
        """
        streamers = streamers if streamers is not None else []
        pending = pending if pending is not None else []

        # compute this turn's delta (original history vs. full list produced by the Agent)
        delta_messages = self._messages_to_persist(original_history, result.all_messages())
        usage = result.usage

        # Agent-end events
        payload = {"session_id": session_id, "output": output, "usage": usage}
        await self._fire(AgentRunnerEvent.AFTER_AGENT_RUN, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.AFTER_AGENT_RUN, payload, pending)
        await self._fire(AgentRunnerEvent.AGENT_END, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.AGENT_END, payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        # Save messages
        await self._session_manager.save_messages(session_id, delta_messages)
        save_payload = {"session_id": session_id, "delta_messages": delta_messages}
        await self._fire(AgentRunnerEvent.SESSION_SAVE, save_payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.SESSION_SAVE, save_payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        # Compaction (if needed and not cancelled by extensions)
        if needs_compaction:
            # ask extensions whether to cancel this compaction
            comp_result = await self._fire_notify(AgentRunnerEvent.COMPACTION_TRIGGER, {
                "session_id": session_id,
            })
            cancelled = comp_result.get("cancel", False)
            if not cancelled:
                # trigger in the background so the current turn's response is not blocked
                asyncio.create_task(self._trigger_compaction(session_id))
            # notify extensions of the compaction outcome (cancelled or triggered)
            applied_payload = {"session_id": session_id, "cancelled": bool(cancelled)}
            await self._fire(AgentRunnerEvent.COMPACTION_APPLIED, applied_payload)
            await self._notify_streamers(streamers, AgentRunnerEvent.COMPACTION_APPLIED, applied_payload, pending)
            async for chunk in self._drain_pending(pending):
                yield chunk

        # Session end
        end_payload = {"session_id": session_id}
        await self._fire(AgentRunnerEvent.SESSION_END, end_payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.SESSION_END, end_payload, pending)
        async for chunk in self._drain_pending(pending):
            yield chunk

        # Final event
        yield {
            "type": "run_end",
            "session_id": session_id,
            "output": output,
            "new_messages": delta_messages,
            "usage": usage,
        }

    # Public API

    async def run(self, prompt: str, *, session_id: str | None = None) -> RunResult:
        """Run one conversation turn and return a ``RunResult``.

        This is the most common entry point. Internal flow:

           _setup_run → _build_agent → agent.run_stream (consume text internally)
           → _finalize_run → extract run_end → return RunResult

        Streaming consumption here is an implementation detail — callers do not need to
        know that ``stream_text()`` is used underneath.
        """
        # 1. setup (load history, context handling, extension injection)
        session_id, history, original_history, needs_compaction = await self._setup_run(
            prompt, session_id
        )
        # 2. build agent
        agent = await self._build_agent(session_id)

        # 3. run agent (always uses run_stream; tokens are concatenated into full text internally)
        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text(delta=False):
                output_parts.append(text)
                # each token chunk also fires a TOKEN_STREAM event
                # (but not notify_streamers, because run() is not a streaming API)
                await self._fire(
                    AgentRunnerEvent.TOKEN_STREAM,
                    {
                        "session_id": session_id,
                        "data": {"chunk": text},
                    },
                )

        output = "".join(output_parts)

        # 4. finalize (save, compaction, events)
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

        # should never reach here — _finalize_run always yields run_end
        raise RuntimeError("run did not produce a run_end event")  # pragma: no cover

    async def run_stream(
        self, prompt: str, *, session_id: str | None = None
    ) -> AsyncIterator[dict]:
        """Run one conversation turn and yield events one by one.

        Differences from ``run()``:

        - The Agent is built with ``streamers`` so all extensions receive runtime events.
        - Token chunks, tool calls, and lifecycle events are yielded as they happen.
        - The final event is still ``{"type": "run_end", ...}``.
        """
        # 1. setup (same as run())
        session_id, history, original_history, needs_compaction = await self._setup_run(
            prompt, session_id
        )

        # 2. build agent — pass streamers to enable streaming event push
        streamers = list(self._extensions)
        pending: list[dict] = []
        agent = await self._build_agent(
            session_id, pending=pending, streamers=streamers
        )

        # 3. run agent — notify + drain for every token chunk
        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text(delta=False):
                output_parts.append(text)
                # TOKEN_STREAM event: fire for legacy extensions and
                # notify_streamers (streaming extension yields go into pending)
                payload = {
                    "session_id": session_id,
                    "data": {"chunk": text},
                }
                await self._fire(AgentRunnerEvent.TOKEN_STREAM, payload)
                await self._notify_streamers(
                    streamers, AgentRunnerEvent.TOKEN_STREAM, payload, pending,
                )
                # drain pending chunks to the external consumer
                async for chunk in self._drain_pending(pending):
                    yield chunk

        output = "".join(output_parts)

        # 4. finalize — yield all post-agent events (including final run_end)
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

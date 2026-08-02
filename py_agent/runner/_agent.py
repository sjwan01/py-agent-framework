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
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai import Tool as PydanticTool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness.skills import Skills

from py_agent._compaction import HarnessSummarizer
from py_agent._context import _prepare_context
from py_agent.models import (
    ContextConfig,
    RunResult,
    SummarizerConfig,
)
from py_agent.runner import _factory, _hooks, _internals
from py_agent.session import SingleTurnSessionManager
from py_agent.types import (
    AgentRunnerEvent,
    Extension,
    SessionManager,
    ToolsetFailureHandler,
)

# Pydantic AI thinking depth levels. Invalid values are ignored and fall back.
_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high", "xhigh"}


class AgentRunner:
    """Orchestrator that manages the full agent lifecycle.

    Only ``model`` and a ``system_prompt`` are required. Multi-turn sessions
    auto-configure context management and compaction. The minimal usage::

        runner = AgentRunner(model=my_model, system_prompt="You are...")

    A system prompt is always required: pass it explicitly, or omit it only
    when reconnecting to an existing session that already has a stored prompt.

    Two entry points:
    - ``run(prompt)`` — execute one turn, return ``RunResult``.
    - ``run_stream(prompt)`` — execute one turn, yield events as they happen.

    Args:
        model: Pydantic AI model used for the agent's main conversation loop.
        system_prompt: Instructions injected at the start of every turn.
            ``None`` means "use the stored system prompt" when reconnecting
            to an existing session; it is an error if no prompt is stored.
            Empty strings are also rejected.
        thinking_enabled: Enable Pydantic AI thinking mode for the main model.
            Defaults to ``True``.
        thinking_level: Thinking depth (``"minimal"``, ``"low"``, ``"medium"``,
            ``"high"``, ``"xhigh"``). Invalid values are ignored with a warning.
            Defaults to ``None`` (model default).
        extensions: Objects implementing the ``Extension`` protocol — they
            observe and intercept lifecycle events and may register SDK
            capabilities. Defaults to ``None``.
        tools: Raw callables or Pydantic AI ``Tool`` objects registered directly.
            Defaults to ``()``.
        skills: A Pydantic AI harness ``Skills`` instance (a skill library)
            made available to the model for on-demand loading. Defaults to
            ``None`` (no skills).
        session_manager: Backend for multi-turn persistence. ``None`` uses
            ``SingleTurnSessionManager`` (every run is a fresh session). Pass
            ``LocalSessionManager(db_path=...)`` or
            ``PostgresSessionManager(pg_url=...)`` for persistence.
        context_config: Configuration for automatic context window
            truncation and compaction detection. For multi-turn sessions, a
            default ``ContextConfig`` is created automatically if this is
            unset. Ignored when using ``SingleTurnSessionManager`` (the default).
        summarizer_config: Configuration for LLM-based context compaction.
            For multi-turn sessions, a default ``HarnessSummarizer`` reusing
            the main model is created automatically if this is unset.
            Ignored when using ``SingleTurnSessionManager`` (the default).
        max_tool_calls_per_turn: Hard limit on tool invocations per turn.
            The model receives a message and continues when exceeded.
            Defaults to ``5``.
        parallel_tool_calls: Allow the model to issue multiple tool calls
            concurrently. Defaults to ``False``.
        hooks: Removed — capabilities are assembled by the framework from
            ``skills`` and extensions' ``register_capabilities``.
        capabilities: Removed — see ``hooks``.
        on_warning: Callback for non-fatal errors (extension crashes,
            compaction failures, etc.). Defaults to a no-op.
        toolset_failure: Custom handler for a toolset whose connection or
            catalog failed to load (e.g. an MCP server is down). Return a
            dict to substitute tools, ``None`` for the default warn-and-drop,
            or raise to fail the run. Defaults to ``None`` (warn and drop —
            partial degradation, other servers keep working).
        prefix_toolset_names: Prefix every toolset's tools with its server
            name (``{server}_{tool}``) so identically named tools across
            servers never collide. When disabled, cross-source name
            conflicts are reported by pydantic-ai at assembly time.
            Defaults to ``True``.
    """

    # Class-attribute bindings: wire submodule functions as instance methods.
    #
    # Technique: module-level functions take ``self`` as their first argument and are
    # bound here as class attributes, so ``self._fire(...)`` actually calls
    # ``_internals.fire(self, ...)``. This avoids multiple inheritance or putting
    # all code in a single file.

    # --- from _internals.py (runtime helpers) ---
    _build_hooks = _hooks.build_hooks
    _notify_streamers = staticmethod(_internals.notify_streamers)
    _drain_pending = staticmethod(_internals.drain_pending)
    _fire = _internals.fire
    _fire_notify = _internals.fire_notify

    # --- from _factory.py (init / lifecycle / discovery) ---
    _collect_tools = _factory.collect_tools
    _collect_capabilities = _factory.collect_capabilities
    _trigger_compaction = _factory.trigger_compaction

    # Constructor

    def __init__(
        self,
        model: Model,
        *,
        system_prompt: str | None = None,
        thinking_enabled: bool = True,
        thinking_level: str | None = None,
        extensions: Sequence[Extension] | None = None,
        tools: Sequence[
            PydanticTool[Any] | AbstractToolset[Any] | Callable[..., Any]
        ] = (),
        skills: Skills[Any] | None = None,
        session_manager: SessionManager | None = None,
        context_config: ContextConfig | None = None,
        summarizer_config: SummarizerConfig | None = None,
        max_tool_calls_per_turn: int = 5,
        parallel_tool_calls: bool = False,
        on_warning: Callable[[str, Exception | None], None] | None = None,
        toolset_failure: ToolsetFailureHandler | None = None,
        prefix_toolset_names: bool = True,
    ):
        """See the class docstring for full parameter documentation."""
        def _noop(msg: str, exc: Exception | None = None) -> None:
            pass

        self._model = model
        self._system_prompt = system_prompt
        self._thinking_enabled = thinking_enabled
        self._thinking_level = thinking_level

        self._extensions = extensions or []

        self._raw_tools = list(tools)
        self._tools: list[PydanticTool[Any]] = []
        self._toolsets: list[AbstractToolset[Any]] = []
        self._tools_initialized = False
        self._skills = skills

        self._session_manager = session_manager or SingleTurnSessionManager()
        is_multi = not isinstance(self._session_manager, SingleTurnSessionManager)

        # resolve on_warning early so the single-turn config warning can use it
        self._on_warning = on_warning or _noop
        if toolset_failure is not None:
            _factory._validate_toolset_failure_handler(toolset_failure)
        self._toolset_failure = toolset_failure
        self._prefix_toolset_names = prefix_toolset_names

        # context config: single-turn → never; multi-turn → from config or
        # sensible defaults
        self._context_config: ContextConfig | None = None
        self._compaction_summarizer: HarnessSummarizer | None = None
        self._protect_turns = 0

        if is_multi:
            self._context_config = context_config or ContextConfig()
            self._protect_turns = self._context_config.protect_turns

            if summarizer_config is not None:
                default_max = int(
                    min(32_768, max(self._context_config.context_window_cap * 0.1, 8_192))
                )
                self._compaction_summarizer = HarnessSummarizer(
                    model=summarizer_config.model or model,
                    max_output_tokens=summarizer_config.max_output_tokens
                    or default_max,
                    summary_prompt=summarizer_config.summary_prompt,
                )
            else:
                self._compaction_summarizer = HarnessSummarizer(model=model)
        elif context_config is not None or summarizer_config is not None:
            # Explicit config on a single-turn runner is silently dropped;
            # tell the user so the loss is not invisible.
            self._on_warning(
                "context_config and summarizer_config are ignored for "
                "single-turn sessions; pass a persistent SessionManager "
                "(LocalSessionManager / PostgresSessionManager) to enable "
                "context management",
                None,
            )

        if max_tool_calls_per_turn <= 0:
            raise ValueError(
                f"max_tool_calls_per_turn must be > 0, got {max_tool_calls_per_turn}"
            )
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._parallel_tool_calls = parallel_tool_calls

        self._collected_capabilities: list[AbstractCapability[Any]] = []
        self._capabilities_initialized = False

        self._compaction_pending: set[str] = set()

    # Shared helpers for run() and run_stream()

    async def _setup_run(
        self, prompt: str, session_id: str | None
    ) -> tuple[str, list[ModelMessage], list[ModelMessage], bool, str]:
        """Shared pre-run setup for ``run()`` / ``run_stream()``.

        Returns:
            A tuple of ``(session_id, history, injected, needs_compaction, active_sp)``:

            - history: message list after context preparation and extension injection
            - injected: extension-injected messages, captured by identity before
                the Agent touches them (they are part of this turn's delta)
            - needs_compaction: whether context preparation flagged compaction as needed
            - active_sp: the system prompt to use for this turn (user-provided
                or loaded from session storage)
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

        # step 3: load stored system prompt and run prepare() for truncation
        # (skipped when no ContextConfig is configured)
        needs_compaction = False
        active_sp = self._system_prompt
        if self._context_config is not None:
            frozen_sp = await self._session_manager.load_system_prompt(session_id)

            # If the caller did not provide a system prompt, fall back to the
            # one stored for the session so reconnecting does not require
            # remembering the original prompt.
            if active_sp is None:
                active_sp = frozen_sp

        # A system prompt is always required: provided by the caller, or loaded
        # from the stored session prompt when reconnecting. Single-turn sessions
        # have no stored prompt, so the caller must always provide one.
        active_sp = (active_sp or "").strip()
        if not active_sp:
            raise ValueError(
                "system_prompt must be a non-empty string "
                f"for session {session_id!r}"
            )

        if self._context_config is not None:
            # Persist the resolved system prompt if it differs from the stored
            # value (or if there is no stored value yet). This makes reconnecting
            # to a session without re-supplying the prompt possible.
            if frozen_sp is None or frozen_sp.strip() != active_sp:
                await self._session_manager.save_system_prompt(
                    session_id, system_prompt=active_sp
                )

            try:
                history, needs_compaction = await _prepare_context(
                    history,
                    config=self._context_config,
                )
            except Exception as exc:  # pragma: no cover - fail-open
                self._on_warning(f"Context prepare failed: {exc}", exc)

        # keep the original copy after context preparation so extension-injected
        # messages (which happen after this point) can be identified by identity
        original_history = list(history)

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

        # Record extension-injected messages NOW, while the Agent has not yet
        # touched them: object identity is stable here, so id() reliably
        # separates "originally loaded history" from "extension-injected".
        # After agent.run() the SDK may copy messages and identity would be
        # unreliable — hence the snapshot before the run, not a diff afterwards.
        original_ids = {id(m) for m in original_history}
        injected = [m for m in history if id(m) not in original_ids]

        # step 6: AGENT_START event (read-only — informs extensions the run has started)
        await self._fire(
            AgentRunnerEvent.AGENT_START,
            {"session_id": session_id, "prompt": prompt, "messages": history},
        )

        return session_id, history, injected, needs_compaction, active_sp

    async def _build_agent(
        self,
        session_id: str,
        *,
        active_sp: str,
        pending: list[dict[str, Any]] | None = None,
        streamers: list[Extension] | None = None,
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

        # assemble the capabilities list: skills (constructor) first, then
        # extension-registered capabilities, then the framework's tool hooks
        capabilities: list[AbstractCapability[Any]] = []
        if self._skills is not None:
            capabilities.append(self._skills)
        capabilities.extend(await self._collect_capabilities())
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

        # assemble the tools and toolsets from all sources (cached)
        tools, toolsets = await self._collect_tools()

        # We pass the user's "system prompt" as Pydantic AI's `instructions` parameter.
        # `instructions` are attached to each model request as a separate field and are
        # NOT inserted into message_history. This matches our design: we rebuild Agent
        # per turn and manage history/load_history ourselves, so we do not want
        # Pydantic AI to persist a SystemPromptPart that could shadow a later
        # system-prompt change.
        # `active_sp` is resolved in _setup_run and is always non-empty: the
        # caller must provide a system prompt, or the stored one is reused
        # when reconnecting to an existing session.
        return Agent(
            model=self._model,
            instructions=active_sp,
            tools=tools,
            toolsets=toolsets or None,
            capabilities=capabilities or None,
            model_settings=model_settings,
        )

    async def _finalize_run(
        self,
        session_id: str,
        injected: list[ModelMessage],
        result: Any,
        output: str,
        needs_compaction: bool,
        *,
        streamers: list[Extension] | None = None,
        pending: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Shared post-run cleanup for ``run()`` / ``run_stream()``.

        After the Agent finishes, this:

        1. Fires ``AFTER_AGENT_RUN`` and ``AGENT_END``.
        2. Fires ``SESSION_SAVE`` (extensions may rewrite the delta) and saves
           the turn's delta messages to the DB.
        3. Triggers compaction if needed and not cancelled.
        4. Fires ``SESSION_END``.
        5. Yields ``run_end`` (``{"type": "run_end", ...}``).

        This is an async generator. ``run_stream()`` yields the intermediate events to
        the external consumer; ``run()`` only consumes the final ``run_end``.
        """
        streamers = streamers if streamers is not None else []
        pending = pending if pending is not None else []

        # this turn's delta: the SDK's tracked new messages (user prompt, tool
        # results, model reply) plus the extension-injected messages recorded
        # in _setup_run while their identity was still stable
        delta_messages = list(result.new_messages()) + injected
        usage = result.usage

        # Agent-end events
        payload = {"session_id": session_id, "output": output, "usage": usage}
        await self._fire(AgentRunnerEvent.AFTER_AGENT_RUN, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.AFTER_AGENT_RUN, payload, pending, on_warning=self._on_warning)
        await self._fire(AgentRunnerEvent.AGENT_END, payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.AGENT_END, payload, pending, on_warning=self._on_warning)
        async for chunk in self._drain_pending(pending):
            yield chunk

        # SESSION_SAVE event: extensions may rewrite the delta before it is persisted.
        save_data = {"session_id": session_id, "delta_messages": delta_messages}
        save_payload = await self._fire(AgentRunnerEvent.SESSION_SAVE, save_data)
        if "delta_messages" in save_payload:
            delta_messages = save_payload["delta_messages"]
        await self._session_manager.save_messages(session_id, delta_messages)
        await self._notify_streamers(streamers, AgentRunnerEvent.SESSION_SAVE, save_payload, pending, on_warning=self._on_warning)
        async for chunk in self._drain_pending(pending):
            yield chunk

        # Compaction (if needed and not cancelled by extensions)
        if needs_compaction:
            # ask extensions whether to cancel this compaction
            comp_result = await self._fire_notify(AgentRunnerEvent.COMPACTION_TRIGGER, {
                "session_id": session_id,
            })
            cancelled = comp_result.get("cancel", False)
            if not cancelled and session_id not in self._compaction_pending:
                self._compaction_pending.add(session_id)
                # trigger in the background so the current turn's response is not blocked
                asyncio.create_task(self._trigger_compaction(session_id))

            # notify extensions of the compaction outcome (cancelled or triggered)
            applied_payload = {"session_id": session_id, "cancelled": bool(cancelled)}
            await self._fire(AgentRunnerEvent.COMPACTION_APPLIED, applied_payload)
            await self._notify_streamers(streamers, AgentRunnerEvent.COMPACTION_APPLIED, applied_payload, pending, on_warning=self._on_warning)
            async for chunk in self._drain_pending(pending):
                yield chunk

        # Session end
        end_payload = {"session_id": session_id}
        await self._fire(AgentRunnerEvent.SESSION_END, end_payload)
        await self._notify_streamers(streamers, AgentRunnerEvent.SESSION_END, end_payload, pending, on_warning=self._on_warning)
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

    async def close(self) -> None:
        """Release resources held by the runner.

        Closes the session backend (e.g. the Postgres connection pool).
        Toolset connections need no explicit handling — pydantic-ai enters
        and exits every toolset on each run, so nothing persists across runs.
        """
        close = getattr(self._session_manager, "close", None)
        if close is not None:
            await close()

    async def run(self, prompt: str, *, session_id: str | None = None) -> RunResult:
        """Run one conversation turn and return a ``RunResult``.

        This is the most common entry point. Internal flow:

           _setup_run → _build_agent → agent.run (direct result)
           → _finalize_run → extract run_end → return RunResult
        """
        # 1. setup (load history, context handling, extension injection)
        session_id, history, injected, needs_compaction, active_sp = await self._setup_run(
            prompt, session_id
        )
        # 2. build agent
        agent = await self._build_agent(session_id, active_sp=active_sp)

        # 3. run agent (non-streaming — read the output directly from the result)
        result = await agent.run(prompt, message_history=history)
        output = result.output

        # 4. finalize (save, compaction, events)
        async for event in self._finalize_run(
            session_id, injected, result, output, needs_compaction
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
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one conversation turn and yield events one by one.

        Differences from ``run()``:

        - The Agent is built with ``streamers`` so all extensions receive runtime events.
        - Token chunks, tool calls, and lifecycle events are yielded as they happen.
        - The final event is still ``{"type": "run_end", ...}``.
        """
        # 1. setup (same as run())
        session_id, history, injected, needs_compaction, active_sp = await self._setup_run(
            prompt, session_id
        )

        # 2. build agent — pass streamers to enable streaming event push
        streamers = list(self._extensions)
        pending: list[dict[str, Any]] = []
        agent = await self._build_agent(
            session_id, active_sp=active_sp, pending=pending, streamers=streamers
        )

        # 3. run agent — notify + drain for every token chunk.
        # delta=True yields incremental chunks; concatenating them rebuilds the
        # full text exactly once. (delta=False yields accumulated prefixes and
        # would duplicate the output on join.)
        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text(delta=True):
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
                    on_warning=self._on_warning,
                )
                # drain pending chunks to the external consumer
                async for chunk in self._drain_pending(pending):
                    yield chunk

        output = "".join(output_parts)

        # 4. finalize — yield all post-agent events (including final run_end)
        async for event in self._finalize_run(
            session_id,
            injected,
            result,
            output,
            needs_compaction,
            streamers=streamers,
            pending=pending,
        ):
            yield event

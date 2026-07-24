"""AgentRunner — orchestrates load → build → run → save."""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from agent_framework.compaction import CompactionSummarizer, LLMCompactionSummarizer
from agent_framework.context import ContextManager
from agent_framework.models import AgentConfig, RunResult, BaselineState
from agent_framework.pg_session import PostgresSessionManager
from agent_framework.serializer import MessageSerializer
from agent_framework.settings import Settings
from agent_framework.session import LocalSessionManager
from agent_framework.tools import LocalToolSource, ToolLifecycle
from agent_framework.types import (
    SessionManager,
    AgentRunnerEvent,
    ToolLifecycleEvent,
)


class AgentRunner:
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

    def _default_context_manager(self) -> ContextManager | None:
        """Create a default ContextManager using the configured context window."""
        return ContextManager(
            context_window_cap=self._settings.context_window,
            low_watermark_ratio=self._settings.low_watermark_ratio,
            high_watermark_ratio=self._settings.high_watermark_ratio,
            protect_turns=self._settings.protect_turns,
            truncate_chars=self._settings.truncate_tool_result_chars,
        )

    def _default_compaction_summarizer(self) -> CompactionSummarizer | None:
        """Create a default LLM summarizer if a compaction model is configured."""
        if self._settings.compaction_model_id:
            return LLMCompactionSummarizer(self._settings)
        return None

    def _default_session_manager(self) -> SessionManager:
        """Pick Postgres if configured, otherwise SQLite (temp file by default), otherwise single-turn."""
        if self._settings.postgres_url:
            return PostgresSessionManager(
                pg_url=self._settings.postgres_url,
                serializer=MessageSerializer(),
                pool_size=self._settings.pg_pool_size,
                max_overflow=self._settings.pg_max_overflow,
            )
        db_path = self._settings.sqlite_path
        if db_path is None:
            fd, db_path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
        return LocalSessionManager(
            db_path=db_path,
            serializer=MessageSerializer(),
        )

    async def _ensure_tool_lifecycle(self) -> ToolLifecycle | None:
        """Lazily build ToolLifecycle and register extension sources / raw tools."""
        if self._tool_lifecycle_initialized:
            return self._tool_lifecycle

        if self._tool_lifecycle is None:
            if self._raw_tools or self._extensions:
                self._tool_lifecycle = ToolLifecycle()
            else:
                self._tool_lifecycle_initialized = True
                return None

        # Subscribe extension tool event handlers before registering sources so
        # they can influence conflicts such as TOOL_CONFLICT.
        for ext in self._extensions:
            handler = getattr(ext, "on_tool_event", None)
            if handler is None:
                continue
            for event in ToolLifecycleEvent:
                self._tool_lifecycle.on(event, handler)

        if self._raw_tools:
            await self._tool_lifecycle.add_source(LocalToolSource(self._raw_tools))

        for ext in self._extensions:
            register = getattr(ext, "register_tool_sources", None)
            if register is None:
                continue
            try:
                sources = await register()
            except Exception as exc:  # pragma: no cover - fail-open
                logging.getLogger(__name__).warning(
                    "Extension %s register_tool_sources failed: %s",
                    type(ext).__name__, exc,
                    exc_info=True,
                )
                continue
            for src in sources or []:
                await self._tool_lifecycle.add_source(src)

        self._tool_lifecycle_initialized = True
        return self._tool_lifecycle

    async def _get_tools(self) -> list:
        lifecycle = await self._ensure_tool_lifecycle()
        if lifecycle is not None:
            return lifecycle.get_for_scope(self._scope)
        return self._raw_tools

    def _build_model(self):
        if self._model is not None:
            return self._model
        return OpenAIChatModel(
            self._settings.llm_model_id,
            provider=OpenAIProvider(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url,
            ),
        )

    def _build_capabilities(self) -> list:
        capabilities = list(self._config.capabilities)
        if self._config.hooks is not None:
            capabilities.append(self._config.hooks)
        return capabilities

    async def _fire(self, event: str, data: dict) -> dict:
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

    async def _fire_notify(self, event: str, data: dict, *, cancel_key: str = "cancel") -> dict:
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

    @staticmethod
    def _boundary_id(messages: list, boundary: int) -> str:
        """Return a stable id for the compaction boundary."""
        if boundary > 0 and boundary <= len(messages):
            msg = messages[boundary - 1]
            return getattr(msg, "run_id", None) or f"msg-{boundary - 1}"
        return f"msg-{boundary}"

    @staticmethod
    def _compute_delta(history: list, all_messages: list) -> list:
        """Return messages produced in the current turn (not in input history).

        Uses identity first; if Pydantic AI ever copies history items, falls
        back to a stable content key based on kind, run_id and parts.
        """
        history_ids = {id(m) for m in history}

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

        history_keys = {_key(m) for m in history}
        delta = []
        for m in all_messages:
            if id(m) in history_ids:
                continue
            if _key(m) in history_keys:
                continue
            delta.append(m)
        return delta

    async def _trigger_compaction(self, session_id: str) -> None:
        try:
            messages = await self._session_manager.load_history(session_id)
            boundary = self._context_manager.find_compaction_boundary(messages)
            if boundary <= 0:
                return
            early_messages = messages[:boundary]
            boundary_entry_id = self._boundary_id(messages, boundary)

            summarizer = self._compaction_summarizer
            if summarizer is None:
                summary = "Context compacted to fit window."
            else:
                summary = await summarizer.summarize(early_messages, BaselineState())

            await self._session_manager.apply_compaction(
                session_id,
                summary=summary,
                boundary_entry_id=boundary_entry_id,
            )
        except Exception as exc:  # pragma: no cover - fail-open
            logging.getLogger(__name__).warning(
                "Compaction failed for session %s: %s", session_id, exc,
                exc_info=True,
            )

    @staticmethod
    def discover_extensions(paths: list[str]) -> list:
        """Discover Extension implementations from Python files in the given paths."""
        extensions = []
        for path_str in paths:
            p = Path(path_str)
            if not p.exists():
                continue
            for py_file in p.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if isinstance(attr, type) and hasattr(attr, "on_agent_runner_event"):
                            if attr_name != "Extension":
                                try:
                                    extensions.append(attr())
                                except TypeError:
                                    pass
        return extensions

    async def run(self, prompt: str, *, session_id: str | None = None) -> RunResult:
        if session_id is None:
            session_id = await self._session_manager.create_session()

        await self._fire(AgentRunnerEvent.SESSION_START, {"session_id": session_id})

        history = await self._session_manager.load_history(session_id)

        # CONTEXT_PREPARE — ContextManager first, then extensions
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

        # BEFORE_AGENT_RUN — extensions can modify messages, not system_prompt
        before_data = {"session_id": session_id, "messages": history}
        before_result = await self._fire(AgentRunnerEvent.BEFORE_AGENT_RUN, before_data)
        if "messages" in before_result:
            history = before_result["messages"]

        # AGENT_RUN
        await self._fire(
            AgentRunnerEvent.AGENT_RUN,
            {"session_id": session_id, "prompt": prompt, "messages": history},
        )

        # Framework hooks forward tool execution events to AGENT_RUN subscribers
        # and enforce per-turn tool call limits.
        hooks = Hooks()
        tool_calls = 0
        max_tool_calls = self._settings.max_tool_calls_per_turn

        @hooks.on.before_tool_execute
        async def _on_tool_start(ctx, *, call, tool_def, args):
            await self._fire(
                AgentRunnerEvent.AGENT_RUN,
                {
                    "session_id": session_id,
                    "event": "on_tool_start",
                    "name": call.tool_name,
                    "data": {"args": args},
                },
            )
            return args

        @hooks.on.tool_execute
        async def _on_tool_call(ctx, *, call, tool_def, args, handler):
            nonlocal tool_calls
            tool_calls += 1
            if tool_calls > max_tool_calls:
                return f"Tool call limit ({max_tool_calls}) reached for this turn."

            call_data = {
                "session_id": session_id,
                "tool_name": call.tool_name,
                "tool_call_id": call.tool_call_id,
                "args": dict(args),
            }
            call_result = await self._fire(AgentRunnerEvent.TOOL_CALL, call_data)
            if call_result.get("block"):
                reason = call_result.get("reason", "Blocked by extension")
                return f"Tool call blocked: {reason}"
            if "args" in call_result:
                args = call_result["args"]
            return await handler(args)

        @hooks.on.after_tool_execute
        async def _on_tool_result(ctx, *, call, tool_def, args, result):
            result_data = {
                "session_id": session_id,
                "tool_name": call.tool_name,
                "tool_call_id": call.tool_call_id,
                "content": result,
                "is_error": False,
            }
            result_data = await self._fire(AgentRunnerEvent.TOOL_RESULT, result_data)
            content = result_data.get("content", result)
            await self._fire(
                AgentRunnerEvent.AGENT_RUN,
                {
                    "session_id": session_id,
                    "event": "on_tool_end",
                    "name": call.tool_name,
                    "data": {"result": content},
                },
            )
            return content

        capabilities = self._build_capabilities() or []
        if hooks not in capabilities:
            capabilities.append(hooks)

        model_settings = ModelSettings(
            parallel_tool_calls=self._settings.parallel_tool_calls,
        )

        agent = Agent(
            model=self._build_model(),
            instructions=self._config.instructions,
            tools=await self._get_tools(),
            capabilities=capabilities or None,
            model_settings=model_settings,
        )

        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text(delta=True):
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
        new_messages = result.all_messages()
        delta_messages = self._compute_delta(history, new_messages)
        usage = result.usage

        await self._fire(AgentRunnerEvent.AFTER_AGENT_RUN, {
            "session_id": session_id, "output": output, "usage": usage,
        })

        # SESSION_SAVE
        await self._session_manager.save_messages(session_id, delta_messages)
        await self._fire(AgentRunnerEvent.SESSION_SAVE, {
            "session_id": session_id,
            "delta_messages": delta_messages,
        })

        # COMPACTION_TRIGGER — extensions can cancel
        if needs_compaction:
            comp_result = await self._fire_notify(AgentRunnerEvent.COMPACTION_TRIGGER, {
                "session_id": session_id,
            })
            cancelled = comp_result.get("cancel", False)
            if not cancelled:
                asyncio.create_task(self._trigger_compaction(session_id))
            await self._fire(
                AgentRunnerEvent.COMPACTION_APPLIED,
                {"session_id": session_id, "cancelled": bool(cancelled)},
            )

        await self._fire(AgentRunnerEvent.SESSION_END, {"session_id": session_id})

        return RunResult(
            output=output, session_id=session_id,
            new_messages=new_messages, usage=usage,
        )

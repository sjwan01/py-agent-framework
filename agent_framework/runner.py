"""AgentRunner — orchestrates load → build → run → save."""
from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent_framework.models import AgentConfig, RunResult, BaselineState
from agent_framework.settings import Settings
from agent_framework.session import SingleTurnSessionManager
from agent_framework.types import SessionManager, AgentRunnerEvent


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
    ):
        self._settings = settings
        self._config = config
        self._session_manager = session_manager or SingleTurnSessionManager()
        self._model = model
        self._raw_tools = list(tools)
        self._tool_lifecycle = tool_lifecycle
        self._context_manager = context_manager
        self._extensions = extensions or []

    def _get_tools(self) -> list:
        if self._tool_lifecycle is not None:
            return self._tool_lifecycle.get_for_scope()
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

    async def _fire(self, event: str, data: dict) -> dict | None:
        """Fire event to all extensions. Last non-None result wins."""
        result = None
        for ext in self._extensions:
            r = await ext.on_agent_runner_event(event, data)
            if r is not None:
                result = r
        return result

    async def _trigger_compaction(self, session_id: str) -> None:
        try:
            await self._session_manager.apply_compaction(
                session_id,
                summary="Context compacted to fit window.",
                boundary_entry_id="",
            )
        except Exception:
            pass

    @staticmethod
    def discover_extensions(paths: list[str]) -> list:
        """Discover Extension implementations from Python files in the given paths."""
        import importlib.util
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
        if self._context_manager is not None:
            prepared = await self._context_manager.prepare(
                history,
                system_prompt=self._config.instructions,
                current_state=BaselineState(),
            )
            history = prepared.messages
            needs_compaction = prepared.needs_compaction

        ctx_data = {"session_id": session_id, "messages": history}
        await self._fire(AgentRunnerEvent.CONTEXT_PREPARE, ctx_data)

        # BEFORE_AGENT_RUN — extensions can modify messages
        before_result = await self._fire(AgentRunnerEvent.BEFORE_AGENT_RUN, ctx_data)
        if before_result and "messages" in before_result:
            history = before_result["messages"]
        if before_result and "system_prompt" in before_result:
            self._config.instructions = before_result["system_prompt"]

        # AGENT_RUN
        agent = Agent(
            model=self._build_model(),
            instructions=self._config.instructions,
            tools=self._get_tools(),
            capabilities=self._build_capabilities() or None,
        )

        output_parts: list[str] = []
        async with agent.run_stream(prompt, message_history=history) as result:
            async for text in result.stream_text():
                output_parts.append(text)

        output = "".join(output_parts)
        new_messages = result.all_messages()
        usage = result.usage

        await self._fire(AgentRunnerEvent.AFTER_AGENT_RUN, {
            "session_id": session_id, "output": output, "usage": usage,
        })

        # SESSION_SAVE
        await self._session_manager.save_messages(session_id, new_messages)
        await self._fire(AgentRunnerEvent.SESSION_SAVE, {"session_id": session_id})

        # COMPACTION_TRIGGER — extensions can cancel
        if needs_compaction:
            comp_result = await self._fire(AgentRunnerEvent.COMPACTION_TRIGGER, {
                "session_id": session_id,
            })
            cancelled = comp_result and comp_result.get("cancel")
            if not cancelled:
                asyncio.create_task(self._trigger_compaction(session_id))
            await self._fire(AgentRunnerEvent.COMPACTION_APPLIED, {
                "session_id": session_id, "cancelled": bool(cancelled),
            })

        await self._fire(AgentRunnerEvent.SESSION_END, {"session_id": session_id})

        return RunResult(
            output=output, session_id=session_id,
            new_messages=new_messages, usage=usage,
        )

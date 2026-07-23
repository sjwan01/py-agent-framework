"""AgentRunner — orchestrates load → build → run → save."""
from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent_framework.models import AgentConfig, RunResult, BaselineState
from agent_framework.settings import Settings
from agent_framework.session import SingleTurnSessionManager
from agent_framework.types import SessionManager


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
    ):
        self._settings = settings
        self._config = config
        self._session_manager = session_manager or SingleTurnSessionManager()
        self._model = model
        self._raw_tools = list(tools)
        self._tool_lifecycle = tool_lifecycle
        self._context_manager = context_manager

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

    async def _trigger_compaction(self, session_id: str) -> None:
        try:
            await self._session_manager.apply_compaction(
                session_id,
                summary="Context compacted to fit window.",
                boundary_entry_id="",
            )
        except Exception:
            pass

    async def run(self, prompt: str, *, session_id: str | None = None) -> RunResult:
        if session_id is None:
            session_id = await self._session_manager.create_session()

        history = await self._session_manager.load_history(session_id)

        needs_compaction = False
        if self._context_manager is not None:
            prepared = await self._context_manager.prepare(
                history,
                system_prompt=self._config.instructions,
                current_state=BaselineState(),
            )
            history = prepared.messages
            needs_compaction = prepared.needs_compaction

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

        await self._session_manager.save_messages(session_id, new_messages)

        if needs_compaction:
            asyncio.create_task(self._trigger_compaction(session_id))

        return RunResult(
            output=output, session_id=session_id,
            new_messages=new_messages, usage=usage,
        )

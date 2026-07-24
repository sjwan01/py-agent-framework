"""Context compaction summarizers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict

import ujson

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from agent_framework.models import BaselineState
from agent_framework.settings import Settings


class CompactionSummarizer(ABC):
    """Abstract summarizer used by AgentRunner to compact old context."""

    @abstractmethod
    async def summarize(self, messages: list, baseline_state: BaselineState) -> str: ...


class LLMCompactionSummarizer(CompactionSummarizer):
    """OpenAI-compatible LLM summarizer producing a Pi-style structured summary."""

    def __init__(self, settings: Settings):
        self._settings = settings

    async def summarize(self, messages: list, baseline_state: BaselineState) -> str:
        model = OpenAIChatModel(
            self._settings.compaction_model_id,
            provider=OpenAIProvider(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url,
            ),
        )
        agent = Agent(
            model=model,
            system_prompt=(
                "You are a context compaction assistant. "
                "Summarize the conversation so far using exactly this format:\n\n"
                "Goal: ...\n"
                "Progress: ...\n"
                "Key Decisions: ...\n"
                "Next Steps: ..."
            ),
        )
        prompt = _build_prompt(messages, baseline_state)
        max_tokens = self._settings.compaction_max_output_tokens
        model_settings = ModelSettings(max_tokens=max_tokens) if max_tokens else None
        result = await agent.run(prompt, model_settings=model_settings)
        return str(result.data)


def _build_prompt(messages: list, baseline_state: BaselineState) -> str:
    """Serialize early messages into a prompt for the summarizer."""
    lines = [
        "Summarize the following conversation context for later reference.",
        "",
        f"Baseline skills: {list(baseline_state.skills.keys())}",
        f"Baseline tools: {list(baseline_state.tools.keys())}",
        "",
        "Messages:",
    ]
    for msg in messages:
        lines.append(_serialize_message(msg))
    return "\n".join(lines)


def _serialize_message(msg) -> str:
    """Best-effort text serialization of a Pydantic AI message."""
    if isinstance(msg, (ModelRequest, ModelResponse)):
        try:
            d = asdict(msg)
            _strip_timestamps(d)
            return ujson.dumps(d, default=str, ensure_ascii=False)
        except Exception:
            pass
    return str(msg)


def _strip_timestamps(d: dict) -> None:
    """Recursively remove timestamp fields to keep the prompt stable."""
    for key in list(d.keys()):
        if key in ("timestamp",):
            d.pop(key, None)
        elif isinstance(d[key], dict):
            _strip_timestamps(d[key])
        elif isinstance(d[key], list):
            for item in d[key]:
                if isinstance(item, dict):
                    _strip_timestamps(item)

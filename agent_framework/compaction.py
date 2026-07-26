"""Context compaction summarizers."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import SummarizingCompaction

from agent_framework.settings import Settings
from agent_framework.types import CompactionSummarizer

DEEPSEEK_OFFICIAL_URL = "https://api.deepseek.com"

_DEFAULT_SUMMARY_PROMPT = (
    "You are a context compaction assistant. "
    "Summarize the conversation so far using exactly this format:\n\n"
    "Goal: ...\n"
    "Progress: ...\n"
    "Key Decisions: ...\n"
    "Next Steps: ...\n\n"
    "Messages:\n{messages}"
)


class HarnessSummarizer(CompactionSummarizer):
    """Pydantic AI Harness ``SummarizingCompaction`` backed summarizer."""

    def __init__(self, settings: Settings):
        self._settings = settings

        # 每个字段独立回退：不配就用主 LLM 的
        model_id = settings.compaction_model_id or settings.llm_model_id
        base_url = settings.compaction_base_url or settings.llm_base_url
        api_key = settings.compaction_api_key or settings.llm_api_key

        if base_url == DEEPSEEK_OFFICIAL_URL:
            provider = DeepSeekProvider(api_key=api_key)
        else:
            provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        model = OpenAIChatModel(model_id, provider=provider)

        summary_prompt = settings.compaction_summary_prompt or _DEFAULT_SUMMARY_PROMPT

        self._strategy = SummarizingCompaction(
            model=model,
            max_messages=1,
            keep_messages=0,
            preserve_first_user_message=False,
            summary_prompt=summary_prompt,
        )

    async def summarize(self, messages: list) -> str:
        ctx = cast(Any, SimpleNamespace(usage=RunUsage()))
        compacted = await self._strategy.compact(messages, ctx)
        for msg in compacted:
            if isinstance(msg, ModelRequest):
                system_parts = [
                    part for part in msg.parts if isinstance(part, SystemPromptPart)
                ]
                if system_parts:
                    # SummarizingCompaction appends the summary after any leading
                    # system prompts, so the last system part carries the summary.
                    return system_parts[-1].content
        return ""

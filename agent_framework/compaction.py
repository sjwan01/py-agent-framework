"""Context compaction summarizers."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import SummarizingCompaction

from agent_framework.settings import Settings
from agent_framework.types import CompactionSummarizer

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
    """Pydantic AI Harness ``SummarizingCompaction`` 包装器。"""

    def __init__(self, *, model, settings: Settings):
        prompt = settings.compaction_summary_prompt or _DEFAULT_SUMMARY_PROMPT
        self._strategy = SummarizingCompaction(
            model=model,
            max_tokens=settings.compaction_max_output_tokens,
            max_messages=1,
            keep_messages=0,
            preserve_first_user_message=False,
            summary_prompt=prompt,
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
                    return system_parts[-1].content
        return ""

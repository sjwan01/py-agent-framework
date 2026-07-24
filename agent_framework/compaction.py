"""Context compaction summarizers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import SummarizingCompaction

from agent_framework.models import BaselineState
from agent_framework.settings import Settings


class CompactionSummarizer(ABC):
    """Abstract summarizer used by AgentRunner to compact old context."""

    @abstractmethod
    async def summarize(self, messages: list, baseline_state: BaselineState) -> str: ...


class HarnessSummarizer(CompactionSummarizer):
    """Pydantic AI Harness ``SummarizingCompaction`` backed summarizer."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._strategy = SummarizingCompaction(
            model=settings.compaction_model_id,
            max_messages=1,
            keep_messages=0,
            preserve_first_user_message=False,
            summary_prompt=(
                "You are a context compaction assistant. "
                "Summarize the conversation so far using exactly this format:\n\n"
                "Goal: ...\n"
                "Progress: ...\n"
                "Key Decisions: ...\n"
                "Next Steps: ...\n\n"
                "Messages:\n{messages}"
            ),
        )

    async def summarize(self, messages: list, baseline_state: BaselineState) -> str:
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

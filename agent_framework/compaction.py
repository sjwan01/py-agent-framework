"""Context compaction summarizers."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import SummarizingCompaction

from agent_framework.settings import Settings
from agent_framework.types import CompactionSummarizer


class HarnessSummarizer(CompactionSummarizer):
    """Pydantic AI Harness ``SummarizingCompaction`` backed summarizer."""

    def __init__(self, settings: Settings):
        self._settings = settings
        # TODO: 当前 summary_prompt 非常草率，只是硬编码了一个四段式模板。
        #       pi-agent 的 compaction 摘要格式包含 Goal、Constraints & Preferences、
        #       Progress（Done/In Progress/Blocked）、Key Decisions、Next Steps、
        #       Critical Context、read-files、modified-files 等结构化字段，且支持
        #       通过 Extension 的 session_before_compact 事件自定义摘要。
        #       V2 应支持可配置的 prompt 模板，并允许 Extension 覆盖摘要。
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

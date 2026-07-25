"""Context compaction summarizers."""
from __future__ import annotations

# TODO: CompactionSummarizer 这个接口应该移到 agent_framework.types.py，
#       与 SessionManager、Extension 等其他 seams 放在一起，而不是藏在业务实现里。
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
    # TODO: baseline_state 参数当前没有实现，summarize 实现里完全没用到它。
    #       要么真正利用 baseline_state 生成带 baseline diff 的摘要，要么移除该参数。
    async def summarize(self, messages: list, baseline_state: BaselineState) -> str: ...


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

    async def summarize(self, messages: list, baseline_state: BaselineState) -> str:
        # TODO: baseline_state 参数未被使用。见 CompactionSummarizer 的 TODO。
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

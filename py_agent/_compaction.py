"""Context compaction summarizers."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import SummarizingCompaction


class HarnessSummarizer:
    """Wrapper around Pydantic AI Harness ``SummarizingCompaction``."""

    def __init__(
        self,
        *,
        model: Model,
        max_output_tokens: int | None = None,
        summary_prompt: str | None = None,
    ):
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_output_tokens,
            max_messages=1,
            keep_messages=0,
            preserve_first_user_message=False,
        )
        if summary_prompt is not None:
            kwargs["summary_prompt"] = summary_prompt
        self._strategy = SummarizingCompaction(**kwargs)

    async def summarize(self, messages: list[Any]) -> str:
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

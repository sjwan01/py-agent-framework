"""Pydantic data models for the agent framework."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic_ai.models import Model


class RunResult(BaseModel):
    """Return value of ``AgentRunner.run()``.

    Attributes:
        output: Final text output from the agent.
        session_id: Identifier that persists across turns; pass to the next ``run()``.
        new_messages: Messages added this turn, persisted to the session backend.
        usage: Pydantic AI usage stats such as input/output tokens and tool calls.
    """

    output: str
    session_id: str
    new_messages: list[Any]
    usage: Any | None = None


class ContextManagerConfig(BaseModel):
    """Configuration for automatic context window management.

    Pass to ``AgentRunner`` to enable context management. ``None`` disables it.

    Attributes:
        context_window: Token budget for the context window. Defaults to 128000.
        low_watermark_ratio: Fraction of ``context_window`` that triggers truncation of old tool results. Defaults to 0.6.
        high_watermark_ratio: Fraction of ``context_window`` that flags the context for asynchronous compaction. Defaults to 0.75.
        protect_turns: Number of most recent turns to keep intact. Defaults to 5.
        truncate_tool_result_chars: Maximum characters per tool result after truncation. Defaults to 1000.
    """

    context_window: int = 128_000
    low_watermark_ratio: float = 0.6
    high_watermark_ratio: float = 0.75
    protect_turns: int = 5
    truncate_tool_result_chars: int = 1_000


class SummarizerConfig(BaseModel):
    """Configuration for LLM-powered context compaction.

    Pass to ``AgentRunner`` to enable compaction. ``None`` disables it.

    Attributes:
        model: Pydantic AI model used for summarization. ``None`` falls back to the main agent model.
        max_output_tokens: Maximum tokens for the summary. ``None`` computes ``min(32768, max(context_window * 0.1, 8192))``.
        summary_prompt: Custom prompt template for summarization. ``None`` uses the built-in six-section harness default.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Model used for summarization. None falls back to the main agent model.
    model: Model | None = None
    max_output_tokens: int | None = None
    summary_prompt: str | None = None

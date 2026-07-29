"""Pydantic data models for the agent framework."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.models import Model


class BaselineState(BaseModel):
    """Snapshot of skills, tools, and context at baseline time for diff injection."""

    skills: dict[str, str] = Field(default_factory=dict)
    tools: dict[str, str] = Field(default_factory=dict)
    context: list[str] = Field(default_factory=list)


class PreparedContext(BaseModel):
    """Output of ``py_agent._context._prepare_context()``.

    Attributes:
        messages: Messages after truncation and transient diff injection.
        needs_compaction: Whether the context size exceeds the high watermark.
        tokens_used: Rough token estimate of the returned messages.
    """

    messages: list[Any] = Field(default_factory=list)
    needs_compaction: bool = False
    tokens_used: int = 0


class ContextConfig(BaseModel):
    """Immutable watermark / truncation configuration.

    Attributes:
        low_watermark_ratio: Fraction of ``context_window_cap`` that triggers
            truncation of old tool results. Defaults to 0.6.
        high_watermark_ratio: Fraction of ``context_window_cap`` that flags
            the context for asynchronous compaction. Defaults to 0.75.
        protect_turns: Number of most recent turns to keep intact. Defaults to 5.
        truncate_chars: Maximum characters per tool result after truncation.
            Defaults to 1000.
        context_window_cap: Token budget for the context window. Defaults to 128_000.
    """

    model_config = ConfigDict(frozen=True)

    low_watermark_ratio: float = 0.6
    high_watermark_ratio: float = 0.75
    protect_turns: int = 5
    truncate_chars: int = 1_000
    context_window_cap: int = 128_000

    @model_validator(mode="after")
    def _validate_ratios(self) -> ContextConfig:
        """Ensure watermark ratios satisfy 0 < low < high < 1."""
        low = self.low_watermark_ratio
        high = self.high_watermark_ratio
        if not (0 < low < high < 1):
            raise ValueError(
                f"watermark ratios must satisfy 0 < low ({low}) "
                f"< high ({high}) < 1"
            )
        if self.protect_turns < 0:
            raise ValueError(
                f"protect_turns must be >= 0, got {self.protect_turns}"
            )
        return self


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

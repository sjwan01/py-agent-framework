"""Pydantic data models for the agent framework."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import Model


class RunResult(BaseModel):
    # Agent 运行后拼装好的文本输出。
    output: str

    # 跨轮次存活的 session 标识。
    session_id: str

    # 本轮新产生的消息（相对于 DB 中已有历史的增量）。
    new_messages: list[Any]

    # Pydantic AI 用量统计（input/output token、tool 调用次数等）。
    usage: Any | None = None


class PreparedContext(BaseModel):
    # 准备好传入 Agent.message_history 的消息列表。
    messages: list[Any] = Field(default_factory=list)

    # 是否因超过高水位线而需要触发 compaction。
    needs_compaction: bool = False

    # 截断/清理后进入 Agent 的消息的估算 token 占用（chars ÷ 4 粗略估算）。
    tokens_used: int = 0


class BaselineState(BaseModel):
    # 基线时刻的 skill 名 → description 快照。
    skills: dict[str, str] = Field(default_factory=dict)

    # 基线时刻的 tool 名 → description 快照。
    tools: dict[str, str] = Field(default_factory=dict)

    # 基线时刻的额外上下文条目（路径、标识等）。
    context: list[str] = Field(default_factory=list)


class ContextManagerConfig(BaseModel):
    """Configuration for context window truncation and watermark management.

    When passed to ``AgentRunner``, a ``ContextManager`` is created internally.
    When ``None``, context management is skipped entirely (single-turn mode).
    """

    context_window: int = 128_000
    low_watermark_ratio: float = 0.6
    high_watermark_ratio: float = 0.75
    protect_turns: int = 5
    truncate_tool_result_chars: int = 1_000


class SummarizerConfig(BaseModel):
    """Configuration for LLM-based context compaction via ``HarnessSummarizer``.

    When passed to ``AgentRunner``, a ``HarnessSummarizer`` is created
    internally and compaction is enabled. When ``None``, LLM summarization
    is skipped.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Model used for summarization. None → fall back to the main agent model.
    model: Model | None = None
    max_output_tokens: int | None = None
    summary_prompt: str | None = None

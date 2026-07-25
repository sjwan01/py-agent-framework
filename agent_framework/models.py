"""Pydantic data models for the agent framework."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    # 每轮注入 Agent 的 system prompt / instructions。
    instructions: str = ""

    # 可选的 pydantic_ai.capabilities.Hooks 实例。
    hooks: Any = None

    # 额外的 Pydantic AI capability 或 toolset。
    capabilities: list[Any] = Field(default_factory=list)


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

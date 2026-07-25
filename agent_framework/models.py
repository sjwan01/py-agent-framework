"""Pydantic data models for the agent framework."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    instructions: str = Field(
        default="",
        description="System prompt / instructions injected into every Agent turn.",
    )
    hooks: Any = Field(
        default=None,
        description="Optional pydantic_ai.capabilities.Hooks instance.",
    )
    capabilities: list[Any] = Field(
        default_factory=list,
        description="Additional Pydantic AI capabilities or toolsets.",
    )


class RunResult(BaseModel):
    output: str = Field(
        description="Assembled text output of the agent run.",
    )
    session_id: str = Field(
        description="Session identifier that survives across turns.",
    )
    new_messages: list[Any] = Field(
        description="Messages *newly produced* in this turn (delta from saved history).",
    )
    usage: Any | None = Field(
        default=None,
        description="Pydantic AI usage statistics (input/output tokens, tool calls, etc.).",
    )


class PreparedContext(BaseModel):
    messages: list[Any] = Field(
        default_factory=list,
        description="Messages ready to be fed into the Agent as message_history.",
    )
    needs_compaction: bool = Field(
        default=False,
        description="True when estimated tokens exceed the high watermark.",
    )
    tokens_used: int = Field(
        default=0,
        description="Estimated token count after ContextManager processing.",
    )


class BaselineState(BaseModel):
    skills: dict[str, str] = Field(
        default_factory=dict,
        description="Frozen snapshot of skill name → description at baseline time.",
    )
    tools: dict[str, str] = Field(
        default_factory=dict,
        description="Frozen snapshot of tool name → description at baseline time.",
    )
    context: list[str] = Field(
        default_factory=list,
        description="Frozen snapshot of additional context entries (paths, identifiers, etc.).",
    )

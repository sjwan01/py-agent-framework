"""Pydantic data models for the agent framework."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    instructions: str = ""
    hooks: Any = None
    capabilities: list[Any] = Field(default_factory=list)


class RunResult(BaseModel):
    output: str
    session_id: str
    new_messages: list[Any]
    usage: Any | None = None


class PreparedContext(BaseModel):
    messages: list[Any] = Field(default_factory=list)
    needs_compaction: bool = False
    tokens_used: int = 0


class BaselineState(BaseModel):
    skills: dict[str, str] = Field(default_factory=dict)
    tools: dict[str, str] = Field(default_factory=dict)
    context: list[str] = Field(default_factory=list)

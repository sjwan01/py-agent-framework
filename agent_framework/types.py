"""Core types — ABCs, Protocols, and enums.

Zero implementation. Pure interface definitions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Awaitable
from enum import StrEnum
from typing import Any, Protocol


# ── SessionManager (external seam) ────────────────────────────────────

class SessionManager(ABC):
    @abstractmethod
    async def create_session(self, *, metadata: dict | None = None) -> str: ...
    @abstractmethod
    async def load_history(self, session_id: str, *, protect_turns: int = 0) -> list: ...
    @abstractmethod
    async def save_messages(self, session_id: str, messages: list) -> None: ...
    @abstractmethod
    async def apply_compaction(self, session_id: str, summary: str, boundary_seq: int) -> None: ...
    @abstractmethod
    async def get_max_message_seq(self, session_id: str) -> int: ...
    @abstractmethod
    async def ensure_session(self, session_id: str, *, metadata: dict | None = None) -> str: ...


# ── CompactionSummarizer ──────────────────────────────────────────────

class CompactionSummarizer(ABC):
    """Abstract summarizer used by AgentRunner to compact old context."""

    @abstractmethod
    async def summarize(self, messages: list) -> str: ...


# ── Extension Protocol ────────────────────────────────────────────────

class Extension(Protocol):
    async def register_tool_sources(self) -> list[Any]: ...
    async def on_tool_event(self, event: str, data: dict) -> dict | None: ...
    async def on_agent_runner_event(self, event: str, data: dict) -> dict | None: ...
    async def on_agent_runner_event_stream(
        self, event: str, data: dict
    ) -> AsyncIterator[dict]:
        """Optional async generator for streaming extensions.

        If the extension does not override this, ``run()`` and ``run_stream()``
        ignore it.
        """
        if False:  # pragma: no cover
            yield {}


# ── ToolSource ─────────────────────────────────────────────────────────

class ToolSource(ABC):
    @abstractmethod
    async def discover(self) -> list: ...
    @property
    @abstractmethod
    def source_type(self) -> str: ...
    @property
    @abstractmethod
    def source_id(self) -> str: ...
    @property
    def scope(self) -> str:
        """Visibility scope for this source. Defaults to global ('all')."""
        return "all"


# ── Data enums ───────────────────────────────────────────────────────

class MessageRole(StrEnum):
    """消息角色枚举，统一 DB schema 与 _infer_role 中的角色值。"""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    UNKNOWN = "unknown"


# ── Event enums ───────────────────────────────────────────────────────
#
# ToolLifecycleEvent 发生在工具注册阶段（AgentRunner 初始化 ToolLifecycle 时）：
#
#   _ensure_tool_lifecycle()
#   ├── register_tool_sources / add_source
#   │   ├── TOOL_DISCOVERED   # 工具源发现工具
#   │   ├── TOOL_CONFLICT     # 同名工具冲突
#   │   └── TOOL_REGISTERED   # 冲突解决后正式注册
#   └── TOOL_REMOVED          # 被去重/冲突解决淘汰的工具
#
# AgentRunnerEvent 发生在单次 run() / run_stream() 运行时。
#
#   run() / run_stream()
#   ├── SESSION_START
#   ├── load_history
#   ├── CONTEXT_PREPARE          # 只读：Extension 观察 ContextManager 输出
#   ├── BEFORE_AGENT_RUN         # 可写：Extension 改 messages 的唯一入口
#   ├── AGENT_START              # 只读：prompt + 最终进入 Agent 的 messages
#   │   ├── (Pydantic AI tool loop)
#   │   │   ├── TOOL_START       # 日志/UI
#   │   │   ├── TOOL_CALL        # 拦截：block / 改 args
#   │   │   ├── TOOL_RESULT      # 拦截：改结果
#   │   │   └── TOOL_END         # 日志/UI
#   │   └── TOKEN_STREAM         # run_stream 时每个 token chunk
#   ├── AGENT_END                # 只读：output + usage
#   ├── SESSION_SAVE             # 可写：保存前改 delta_messages
#   ├── COMPACTION_TRIGGER       # 若 ContextManager 标记得 compaction；可 cancel
#   │   └── COMPACTION_APPLIED
#   └── SESSION_END
#

class ToolLifecycleEvent(StrEnum):
    TOOL_DISCOVERED = "tool_discovered"
    TOOL_CONFLICT = "tool_conflict"
    TOOL_REGISTERED = "tool_registered"
    TOOL_REMOVED = "tool_removed"


class AgentRunnerEvent(StrEnum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONTEXT_PREPARE = "context_prepare"
    BEFORE_AGENT_RUN = "before_agent_run"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOKEN_STREAM = "token_stream"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AFTER_AGENT_RUN = "after_agent_run"
    SESSION_SAVE = "session_save"
    COMPACTION_TRIGGER = "compaction_trigger"
    COMPACTION_APPLIED = "compaction_applied"


# ── Event handler type ────────────────────────────────────────────────

ToolEventHandler = Callable[[str, dict], Awaitable[dict | None]]


__all__ = [
    "SessionManager",
    "CompactionSummarizer",
    "Extension",
    "ToolSource",
    "MessageRole",
    "ToolLifecycleEvent",
    "AgentRunnerEvent",
    "ToolEventHandler",
]

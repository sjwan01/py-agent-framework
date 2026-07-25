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
    async def load_history(self, session_id: str) -> list: ...
    @abstractmethod
    async def save_messages(self, session_id: str, messages: list) -> None: ...
    @abstractmethod
    async def apply_compaction(self, session_id: str, summary: str, boundary_entry_id: str) -> None: ...


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
# AgentRunnerEvent 发生在单次 run() / run_stream() 运行时：
#
#   run() / run_stream()
#   ├── SESSION_START
#   ├── load_history
#   ├── CONTEXT_PREPARE       # ContextManager 先执行，Extension 可改 messages
#   ├── BEFORE_AGENT_RUN      # Extension 最后改 messages 的机会
#   ├── AGENT_RUN             # 携带 {prompt, messages}
#   │   ├── (Pydantic AI tool loop)
#   │   │   ├── AGENT_RUN / event=on_tool_start
#   │   │   ├── TOOL_CALL
#   │   │   ├── AGENT_RUN / event=on_tool_end
#   │   │   └── TOOL_RESULT
#   │   └── AGENT_RUN / event=on_chat_model_stream   # 每个 token chunk
#   ├── AFTER_AGENT_RUN
#   ├── SESSION_SAVE
#   ├── COMPACTION_TRIGGER    # 若 ContextManager 标记得 compaction
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
    AGENT_RUN = "agent_run"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AFTER_AGENT_RUN = "after_agent_run"
    SESSION_SAVE = "session_save"
    COMPACTION_TRIGGER = "compaction_trigger"
    COMPACTION_APPLIED = "compaction_applied"


# ── Event handler type ────────────────────────────────────────────────

ToolEventHandler = Callable[[str, dict], Awaitable[dict | None]]

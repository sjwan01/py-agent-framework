"""Core types — ABCs, Protocols, and enums.

Zero implementation. Pure interface definitions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Awaitable
from enum import StrEnum
from typing import Protocol


# SessionManager (external seam)

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


# Extension Protocol

class Extension(Protocol):
    async def register_tool_sources(self) -> list[ToolSource]: ...
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


# ToolSource

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


# Data enums

class MessageRole(StrEnum):
    """Message role enum used in the DB schema and by `_infer_role`."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    UNKNOWN = "unknown"


# Event enums
#
# ToolLifecycleEvent fires while AgentRunner initializes ToolLifecycle:
#
#   _ensure_tool_lifecycle()
#   ├── register_tool_sources / add_source
#   │   ├── TOOL_DISCOVERED   # tool source discovers a tool
#   │   ├── TOOL_CONFLICT     # name collision between tools
#   │   └── TOOL_REGISTERED   # finalized registration after conflict resolution
#   └── TOOL_REMOVED          # tools eliminated by deduplication/conflict resolution
#
# AgentRunnerEvent fires during a single run() / run_stream() execution.
#
#   run() / run_stream()
#   ├── SESSION_START
#   ├── load_history
#   ├── CONTEXT_PREPARE          # read-only: Extension observes ContextManager output
#   ├── BEFORE_AGENT_RUN         # writable: Extension's only entry point to modify messages
#   ├── AGENT_START              # read-only: prompt + final messages entering the Agent
#   │   ├── (Pydantic AI tool loop)
#   │   │   ├── TOOL_START       # log/UI
#   │   │   ├── TOOL_CALL        # intercept: block / modify args
#   │   │   ├── TOOL_RESULT      # intercept: modify result
#   │   │   └── TOOL_END         # log/UI
#   │   └── TOKEN_STREAM         # per-token chunk during run_stream
#   ├── AGENT_END                # read-only: output + usage
#   ├── SESSION_SAVE             # writable: modify delta_messages before persistence
#   ├── COMPACTION_TRIGGER       # flagged by ContextManager; cancellable
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


# Event handler type

ToolEventHandler = Callable[[str, dict], Awaitable[dict | None]]


__all__ = [
    "SessionManager",
    "Extension",
    "ToolSource",
    "MessageRole",
    "ToolLifecycleEvent",
    "AgentRunnerEvent",
    "ToolEventHandler",
]
